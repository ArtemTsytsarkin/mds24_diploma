from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from data import DEFAULT_CONFIG, DataConfig, prepare_datasets, set_global_seed
from env import EnvFactoryConfig, MOEXTradingEnvV2, MultiEpisodeEnv, build_env, make_rolling_episodes
from models import build_policy_kwargs, check_slstm_available, get_device, get_model_registry, smoke_test_encoder

EPS = 1e-8


@dataclass
class TrainConfig:
    window_size: int = 30
    batch_size: int = 256
    learning_rate: float = 1e-4
    n_steps: int = 512
    n_epochs: int = 5
    gamma: float = 0.990
    gae_lambda: float = 0.95
    clip_range: float = 0.12
    # Higher entropy helps PPO avoid collapsing the action mean to zero.
    ent_coef: float = 0.005
    target_kl: float = 0.03
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    stage1_steps: int = 0
    # Collapse appeared after ~95k in your logs; shorter default saves time.
    stage2_steps: int = 150_000
    seed: int = 42
    tensorboard_log: str = "./tb_logs"
    save_dir: str = "./models"
    initial_balance: float = 1_000_000.0
    commission: float = 0.0005
    allow_short: bool = False
    margin_ratio: float = 0.25
    max_position_pct: float = 0.25
    max_gross_leverage: float = 1.0
    max_trade_notional_pct: float = 0.04
    min_trade_action: float = 0.0
    min_order_notional_pct: float = 0.0025
    rebalance_cooldown: int = 0
    reward_scale: float = 100.0
    reward_mode: str = "upside_opportunity"
    benchmark_coef: float = 0.35
    opportunity_coef: float = 1.0
    target_exposure: float = 0.30
    low_exposure_threshold: float = 0.03
    cash_penalty_grace_steps: int = 5
    turnover_penalty_coef: float = 0.0
    downside_penalty_coef: float = 0.0
    force_liquidate_on_turbulence: bool = True
    episode_length: int = 504
    episode_stride: int = 126
    verbose: int = 1
    checkpoint_freq: int = 5_000
    checkpoint_warmup_steps: int = 20_000
    log_every_steps: int = 1_000
    csv_log_name: str = "training_log.csv"
    resume: bool = True
    early_eval_freq: int = 10_000
    early_stop_patience: int = 20
    early_min_improvement: float = 0.005
    risk_free_rate: float = 0.0
    selection_metric: str = "best_score"
    disable_ensemble: bool = False
    # Validation safeguards against no-trade checkpoint selection/collapse.
    min_val_trades: int = 20
    min_val_trade_rate: float = 0.10
    no_trade_eval_patience: int = 3


def make_env_cfg(train_cfg: TrainConfig) -> EnvFactoryConfig:
    return EnvFactoryConfig(
        window_size=train_cfg.window_size,
        initial_balance=train_cfg.initial_balance,
        commission=train_cfg.commission,
        allow_short=train_cfg.allow_short,
        margin_ratio=train_cfg.margin_ratio,
        max_position_pct=train_cfg.max_position_pct,
        max_gross_leverage=train_cfg.max_gross_leverage,
        max_trade_notional_pct=train_cfg.max_trade_notional_pct,
        min_trade_action=train_cfg.min_trade_action,
        min_order_notional_pct=train_cfg.min_order_notional_pct,
        rebalance_cooldown=train_cfg.rebalance_cooldown,
        force_liquidate_on_turbulence=train_cfg.force_liquidate_on_turbulence,
        reward_scale=train_cfg.reward_scale,
        reward_mode=train_cfg.reward_mode,
        benchmark_coef=train_cfg.benchmark_coef,
        opportunity_coef=train_cfg.opportunity_coef,
        target_exposure=train_cfg.target_exposure,
        low_exposure_threshold=train_cfg.low_exposure_threshold,
        cash_penalty_grace_steps=train_cfg.cash_penalty_grace_steps,
        turnover_penalty_coef=train_cfg.turnover_penalty_coef,
        downside_penalty_coef=train_cfg.downside_penalty_coef,
    )


def _base_env_kwargs_from_cfg(env_cfg: EnvFactoryConfig, threshold: float) -> Dict:
    return {
        "turbulence_threshold": threshold,
        "initial_balance": env_cfg.initial_balance,
        "window_size": env_cfg.window_size,
        "commission": env_cfg.commission,
        "max_shares_per_trade": env_cfg.max_shares_per_trade,
        "allow_short": env_cfg.allow_short,
        "margin_ratio": env_cfg.margin_ratio,
        "max_position_pct": env_cfg.max_position_pct,
        "max_gross_leverage": env_cfg.max_gross_leverage,
        "max_trade_notional_pct": env_cfg.max_trade_notional_pct,
        "min_trade_action": env_cfg.min_trade_action,
        "min_order_notional_pct": env_cfg.min_order_notional_pct,
        "rebalance_cooldown": env_cfg.rebalance_cooldown,
        "force_liquidate_on_turbulence": env_cfg.force_liquidate_on_turbulence,
        "reward_scale": env_cfg.reward_scale,
        "reward_mode": env_cfg.reward_mode,
        "benchmark_coef": env_cfg.benchmark_coef,
        "opportunity_coef": env_cfg.opportunity_coef,
        "target_exposure": env_cfg.target_exposure,
        "low_exposure_threshold": env_cfg.low_exposure_threshold,
        "cash_penalty_grace_steps": env_cfg.cash_penalty_grace_steps,
        "turnover_penalty_coef": env_cfg.turnover_penalty_coef,
        "downside_penalty_coef": env_cfg.downside_penalty_coef,
        "liquidation_penalty": env_cfg.liquidation_penalty,
        "atr_target": env_cfg.atr_target,
        "atr_order_scale_min": env_cfg.atr_order_scale_min,
        "atr_order_scale_max": env_cfg.atr_order_scale_max,
    }


def make_multi_episode_vec_env(
    *,
    close,
    features,
    turb,
    threshold,
    env_cfg: EnvFactoryConfig,
    episode_length: int,
    episode_stride: int,
    regime=None,
    rvi=None,
    atr_values=None,
    seed: int = 42,
):
    episodes = make_rolling_episodes(
        close=close,
        features=features,
        turbulence=turb,
        window_size=env_cfg.window_size,
        episode_length=episode_length,
        stride=episode_stride,
        regime=regime,
        rvi=rvi,
        atr_values=atr_values,
    )
    if not episodes:
        raise ValueError("No training episodes generated")
    base_kwargs = _base_env_kwargs_from_cfg(env_cfg, threshold)

    def _factory():
        return MultiEpisodeEnv(
            episodes=episodes,
            base_env_cls=MOEXTradingEnvV2,
            base_env_kwargs=base_kwargs,
            shuffle=True,
            seed=seed,
        )
    print(f"Training episodes: {len(episodes)} | episode_length≈{episode_length} | stride={episode_stride}")
    return DummyVecEnv([_factory])


def evaluate_policy(model, env, deterministic: bool = True):
    obs, _ = env.reset()
    obs_batch = obs[np.newaxis, ...]
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    done = False
    while not done:
        action_batch, lstm_states = model.predict(
            obs_batch,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        obs_raw, _, terminated, truncated, _ = env.step(action_batch[0])
        obs_batch = obs_raw[np.newaxis, ...]
        done = bool(terminated or truncated)
        episode_starts = np.array([done], dtype=bool)
    return list(env.portfolio_history)


def compute_metrics(portfolio_history, initial_balance, n_trades, risk_free_rate: float = 0.0) -> Dict[str, float]:
    values = np.asarray(portfolio_history, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        values = np.array([initial_balance, initial_balance], dtype=float)
    p_init = float(initial_balance)
    p_final = float(values[-1])
    cr = (p_final - p_init) / max(p_init, EPS) * 100.0
    mer = (float(values.max()) - p_init) / max(p_init, EPS) * 100.0
    running_max = np.maximum.accumulate(values)
    drawdowns = (values - running_max) / (running_max + EPS)
    mpb = abs(float(drawdowns.min())) * 100.0
    daily_returns = np.diff(values) / (values[:-1] + EPS)
    excess = daily_returns - risk_free_rate / 252.0
    std = float(excess.std(ddof=0)) if len(excess) else 0.0
    sr = 0.0 if std < 1e-10 else float(excess.mean() / std * np.sqrt(252.0))
    downside = excess[excess < 0]
    dstd = float(downside.std(ddof=0)) if len(downside) else 0.0
    sortino = 0.0 if dstd < 1e-10 else float(excess.mean() / dstd * np.sqrt(252.0))
    calmar = 0.0 if mpb < 1e-8 else float(cr / mpb)
    wins = daily_returns[daily_returns > 0]
    losses = daily_returns[daily_returns < 0]
    win_rate = float((daily_returns > 0).mean()) if len(daily_returns) else 0.0
    profit_factor = float(wins.sum() / (abs(losses.sum()) + EPS)) if len(daily_returns) else 0.0
    days = max(len(values) - 1, 1)
    trade_rate = int(n_trades) / days
    appt = (p_final - p_init) / max(int(n_trades), 1) / 1000.0
    return {
        "CR": float(cr),
        "MER": float(mer),
        "MPB": float(mpb),
        "APPT": float(appt),
        "SR": float(sr),
        "Sortino": float(sortino),
        "Calmar": float(calmar),
        "WinRate": float(win_rate),
        "ProfitFactor": float(profit_factor),
        "Trades": int(n_trades),
        "TradeRate": float(trade_rate),
        "Days": int(days),
        "FinalValue": float(p_final),
    }


def is_trade_eligible(m: Dict[str, float], min_trades: int = 20, min_trade_rate: float = 0.10) -> bool:
    """True when a validation run is not cash-like/no-trade."""
    return int(m.get("Trades", 0)) >= int(min_trades) and float(m.get("TradeRate", 0.0)) >= float(min_trade_rate)


def score_metrics(m: Dict[str, float]) -> float:
    """Composite validation score with a strong no-trade guard.

    Your logs showed that high-SR/low-drawdown no-trade checkpoints can be
    selected even though they are not trading policies. This score still rewards
    Sharpe and return, but heavily penalizes validation runs with almost no
    trades.
    """
    sr = float(m.get("SR", 0.0))
    cr_frac = float(m.get("CR", 0.0)) / 100.0
    dd_frac = float(m.get("MPB", 0.0)) / 100.0
    trade_rate = float(m.get("TradeRate", 0.0))
    trades = int(m.get("Trades", 0))
    score = 0.55 * sr + 0.40 * cr_frac - 0.10 * dd_frac
    if trades == 0:
        score -= 5.0
    elif trade_rate < 0.10:
        score -= 2.0 * (0.10 - trade_rate) / 0.10
    return float(score)


def checkpoint_key(result: Dict[str, float], metric: str) -> float:
    """Metric used for final model selection.

    Supported values:
      - best_score: composite score_metrics()
      - best_SR: Sharpe Ratio
      - best_CR: cumulative return
      - best_MPB: minimum drawdown
      - best_trade: trading-oriented score that prioritizes CR and rejects
        near-cash solutions by applying an activity penalty.
    """
    metric = str(metric)
    if metric == "best_score":
        return float(result.get("score", 0.0))
    if metric == "best_SR":
        return float(result.get("SR", 0.0))
    if metric == "best_CR":
        return float(result.get("CR", 0.0))
    if metric == "best_MPB":
        return -float(result.get("MPB", 0.0))
    if metric == "best_trade":
        sr = float(result.get("SR", 0.0))
        cr = float(result.get("CR", 0.0))          # percent units
        dd = float(result.get("MPB", 0.0))         # percent units
        trade_rate = float(result.get("TradeRate", 0.0))
        trades = int(result.get("Trades", 0))
        activity_penalty = 5.0 * max(0.0, 0.15 - trade_rate) / 0.15
        if trades < 20:
            activity_penalty += 5.0
        return cr + 0.25 * sr - 0.50 * dd - activity_penalty
    raise ValueError(
        f"Unknown selection_metric={metric!r}. "
        "Use one of: best_score, best_SR, best_CR, best_MPB, best_trade."
    )


def eligible_results(results: list[Dict], min_trades: int = 20, min_trade_rate: float = 0.10) -> list[Dict]:
    """Filter out cash-like checkpoints before final selection/ensemble."""
    good = [r for r in results if is_trade_eligible(r, min_trades=min_trades, min_trade_rate=min_trade_rate)]
    return good if good else list(results)


def select_checkpoint(results: list[Dict], metric: str, min_trades: int = 20, min_trade_rate: float = 0.10) -> Dict:
    if not results:
        raise ValueError("select_checkpoint: empty results")
    candidates = eligible_results(results, min_trades=min_trades, min_trade_rate=min_trade_rate)
    return max(candidates, key=lambda r: checkpoint_key(r, metric))


class TrainProgressCallback(BaseCallback):
    def __init__(self, *, model_name: str, stage_name: str, total_stage_steps: int, run_dir: str, log_every_steps: int, stage_offset_steps: int):
        super().__init__()
        self.model_name = model_name
        self.stage_name = stage_name
        self.total_stage_steps = int(total_stage_steps)
        self.run_dir = run_dir
        self.log_every_steps = int(log_every_steps)
        self.stage_offset_steps = int(stage_offset_steps)
        self.csv_path = os.path.join(run_dir, "training_log.csv")
        self.state_path = os.path.join(run_dir, "resume_state.json")
        self.start_time: Optional[float] = None

    def _stage_elapsed(self) -> int:
        return max(int(getattr(self.model, "num_timesteps", 0)) - self.stage_offset_steps, 0)

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self._write_state("running")

    def _write_state(self, status: str) -> None:
        global_ts = int(getattr(self.model, "num_timesteps", 0))
        stage_elapsed = self._stage_elapsed()
        state = {
            "model_name": self.model_name,
            "stage": self.stage_name,
            "status": status,
            "num_timesteps": global_ts,
            "stage_elapsed_steps": stage_elapsed,
            "stage_offset_steps": self.stage_offset_steps,
            "total_stage_steps": self.total_stage_steps,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if self.stage_name == "stage2":
            state["stage2_elapsed_steps"] = stage_elapsed
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _on_step(self) -> bool:
        if self.log_every_steps <= 0 or self.n_calls % self.log_every_steps != 0:
            return True
        elapsed = time.time() - (self.start_time or time.time())
        stage_elapsed = self._stage_elapsed()
        progress = min(stage_elapsed / max(self.total_stage_steps, 1), 1.0)
        rewards = self.locals.get("rewards")
        reward_mean = float(np.mean(rewards)) if rewards is not None else float("nan")
        fps = stage_elapsed / max(elapsed, EPS)
        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": self.model_name,
            "stage": self.stage_name,
            "num_timesteps": int(getattr(self.model, "num_timesteps", 0)),
            "stage_elapsed_steps": stage_elapsed,
            "stage_total_steps": self.total_stage_steps,
            "stage_progress_pct": round(progress * 100.0, 4),
            "reward_mean": reward_mean,
            "elapsed_sec": round(elapsed, 3),
            "fps": round(fps, 3),
        }
        pd.DataFrame([row]).to_csv(self.csv_path, mode="a", index=False, header=not os.path.exists(self.csv_path))
        self._write_state("running")
        if np.isfinite(reward_mean):
            self.logger.record("custom/reward_mean", reward_mean)
        self.logger.record("custom/stage_progress_pct", progress * 100.0)
        self.logger.record("custom/fps_stage", fps)
        print(
            f"[{self.model_name}][{self.stage_name}] step={int(getattr(self.model, 'num_timesteps', 0)):,} | "
            f"stage={stage_elapsed:,}/{self.total_stage_steps:,} ({progress * 100:5.1f}%) | "
            f"reward_mean={reward_mean:.5f} | fps={fps:.1f}"
        )
        return True

    def _on_training_end(self) -> None:
        self._write_state("completed")


class EarlyStoppingCallback(BaseCallback):
    def __init__(
        self,
        *,
        val_env,
        eval_freq: int,
        patience: int,
        min_improvement: float,
        best_model_path: str,
        risk_free_rate: float = 0.0,
        selection_metric: str = "best_score",
        no_trade_eval_patience: int = 3,
        min_val_trades: int = 20,
        min_val_trade_rate: float = 0.10,
        verbose: bool = True,
    ):
        super().__init__()
        self.val_env = val_env
        self.eval_freq = int(eval_freq)
        self.patience = int(patience)
        self.min_improvement = float(min_improvement)
        self.best_model_path = best_model_path
        self.risk_free_rate = float(risk_free_rate)
        self.selection_metric = str(selection_metric)
        self.no_trade_eval_patience = int(no_trade_eval_patience)
        self.min_val_trades = int(min_val_trades)
        self.min_val_trade_rate = float(min_val_trade_rate)
        self.verbose = bool(verbose)
        self.best_score = -np.inf
        self.best_sr = -np.inf
        self.no_improve_count = 0
        self.no_trade_eval_count = 0
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        step = int(self.num_timesteps)
        if step - self._last_eval_step < self.eval_freq:
            return True
        self._last_eval_step = step
        curve = evaluate_policy(self.model, self.val_env)
        metrics = compute_metrics(
            curve,
            self.val_env.initial_balance,
            getattr(self.val_env, "trade_count", 0),
            risk_free_rate=self.risk_free_rate,
        )
        metrics_for_selection = dict(metrics)
        metrics_for_selection["score"] = score_metrics(metrics_for_selection)
        score = checkpoint_key(metrics_for_selection, self.selection_metric)

        if is_trade_eligible(metrics_for_selection, self.min_val_trades, self.min_val_trade_rate):
            self.no_trade_eval_count = 0
        else:
            self.no_trade_eval_count += 1

        self.logger.record("eval/selection_score", score)
        self.logger.record("eval/composite_score", metrics_for_selection["score"])
        self.logger.record("eval/SR", metrics["SR"])
        self.logger.record("eval/CR", metrics["CR"])
        self.logger.record("eval/MPB", metrics["MPB"])
        self.logger.record("eval/Trades", metrics["Trades"])
        self.logger.record("eval/TradeRate", metrics["TradeRate"])
        self.logger.dump(step)

        if self.verbose:
            print(
                f"\n[EarlyStopping] step={step:,} | metric={self.selection_metric} score={score:+.4f} "
                f"SR={metrics['SR']:+.3f} CR={metrics['CR']:+.2f}% "
                f"MPB={metrics['MPB']:.2f}% Trades={metrics['Trades']} "
                f"TradeRate={metrics['TradeRate']:.3f} | best={self.best_score:+.4f} "
                f"| no_improve={self.no_improve_count}/{self.patience} "
                f"| no_trade={self.no_trade_eval_count}/{self.no_trade_eval_patience}"
            )

        if score > self.best_score + self.min_improvement:
            self.best_score = score
            self.best_sr = metrics["SR"]
            self.no_improve_count = 0
            self.model.save(self.best_model_path)
            if self.verbose:
                print(f"[EarlyStopping] Новый лучший {self.selection_metric}={score:+.4f}; сохранено: {self.best_model_path}")
        else:
            self.no_improve_count += 1
            if self.verbose:
                print(f"[EarlyStopping] Нет улучшения ({self.no_improve_count}/{self.patience})")
        if self.no_trade_eval_count >= self.no_trade_eval_patience:
            if self.verbose:
                print(
                    f"[EarlyStopping] no-trade collapse detected: "
                    f"{self.no_trade_eval_count}/{self.no_trade_eval_patience} validation evals below "
                    f"min trades/rate. Stopping."
                )
            return False

        if self.no_improve_count >= self.patience:
            if self.verbose:
                print(f"[EarlyStopping] Остановка. Лучший {self.selection_metric}={self.best_score:+.4f}, SR={self.best_sr:+.3f}")
            return False
        return True


class StageCheckpointCallback(CheckpointCallback):
    def __init__(self, *, stage_name: str, run_dir: str, stage_offset_steps: int, warmup_steps: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.stage_name = stage_name
        self.run_dir = run_dir
        self.stage_offset_steps = int(stage_offset_steps)
        self.warmup_steps = int(warmup_steps)
        self.state_path = os.path.join(run_dir, "resume_state.json")
        self._warmup_done = False

    def _on_step(self) -> bool:
        if self.n_calls <= self.warmup_steps:
            if not self._warmup_done and self.n_calls == self.warmup_steps:
                self._warmup_done = True
                print(f"\n[{self.stage_name}] warmup {self.warmup_steps:,} завершён; чекпоинты включены.")
            return True
        ok = super()._on_step()
        if self.save_freq > 0 and self.n_calls % self.save_freq == 0:
            global_ts = int(getattr(self.model, "num_timesteps", 0))
            elapsed = max(global_ts - self.stage_offset_steps, 0)
            state = {
                "stage": self.stage_name,
                "status": "checkpointed",
                "num_timesteps": global_ts,
                "stage_elapsed_steps": elapsed,
                "stage_offset_steps": self.stage_offset_steps,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            if self.stage_name == "stage2":
                state["stage2_elapsed_steps"] = elapsed
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        return ok


def _build_callbacks(*, model_name: str, stage_name: str, total_steps: int, run_dir: str, checkpoint_dir: str, checkpoint_freq: int, log_every_steps: int, stage_offset_steps: int, checkpoint_warmup_steps: int = 0) -> CallbackList:
    callbacks = [
        TrainProgressCallback(
            model_name=model_name,
            stage_name=stage_name,
            total_stage_steps=total_steps,
            run_dir=run_dir,
            log_every_steps=log_every_steps,
            stage_offset_steps=stage_offset_steps,
        )
    ]
    if checkpoint_freq > 0:
        callbacks.append(
            StageCheckpointCallback(
                stage_name=stage_name,
                run_dir=run_dir,
                stage_offset_steps=stage_offset_steps,
                warmup_steps=checkpoint_warmup_steps,
                save_freq=checkpoint_freq,
                save_path=checkpoint_dir,
                name_prefix=f"{model_name}_{stage_name}",
            )
        )
    return CallbackList(callbacks)


def extract_step_from_checkpoint(filename: str) -> int:
    m = re.search(r"(\d+)_steps\.zip$", filename)
    return int(m.group(1)) if m else -1


def _find_latest_checkpoint(checkpoint_dir: str, prefix: str) -> Optional[str]:
    if not os.path.isdir(checkpoint_dir):
        return None
    candidates = [f for f in os.listdir(checkpoint_dir) if f.startswith(prefix) and f.endswith(".zip")]
    if not candidates:
        return None
    candidates.sort(key=extract_step_from_checkpoint)
    return os.path.join(checkpoint_dir, candidates[-1])


def _read_resume_state(run_dir: str) -> Dict[str, object]:
    path = os.path.join(run_dir, "resume_state.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _simulate_rebalance_strategy(close: np.ndarray, initial_balance: float, commission: float, start: int, rebalance_every: int, target_fn, turb: Optional[np.ndarray] = None, turb_threshold: Optional[float] = None):
    n_days, n_assets = close.shape
    cash = float(initial_balance)
    shares = np.zeros(n_assets, dtype=np.float64)
    curve = [float(initial_balance)]
    trades = 0
    for t in range(start, n_days - 1):
        prices = close[t].astype(float)
        equity = cash + float(np.sum(shares * prices))
        if turb is not None and turb_threshold is not None and float(turb[t - 1]) > float(turb_threshold):
            target_values = np.zeros(n_assets, dtype=np.float64)
        elif (t - start) % max(rebalance_every, 1) == 0:
            target_values = np.asarray(target_fn(t, equity), dtype=np.float64)
        else:
            target_values = shares * prices
        target_shares = target_values / (prices + EPS)
        delta = target_shares - shares
        notional = float(np.sum(np.abs(delta) * prices))
        if notional > equity * 0.001:
            cost = notional * commission
            cash -= float(np.sum(delta * prices)) + cost
            shares = target_shares
            trades += int(np.sum(np.abs(delta * prices) > EPS))
        value_next = cash + float(np.sum(shares * close[t + 1]))
        curve.append(float(max(value_next, 0.0)))
    return curve, trades


def compute_baseline_metrics(close_test: np.ndarray, turb_test: np.ndarray, turb_threshold: float, initial_balance: float, window_size: int = 30, commission: float = 0.0005, risk_free_rate: float = 0.0) -> Dict[str, Dict]:
    close = np.asarray(close_test, dtype=np.float64)
    start = int(window_size)
    if start >= len(close) - 1:
        return {}
    n_assets = close.shape[1]
    days = len(close) - start
    results: Dict[str, Dict] = {}
    results["cash"] = compute_metrics([initial_balance] * days, initial_balance, 0, risk_free_rate=risk_free_rate)

    def bh_target(t, equity):
        return np.ones(n_assets) * (equity / n_assets)
    bh_curve, bh_trades = _simulate_rebalance_strategy(close, initial_balance, commission, start, 10**9, bh_target)
    results["buy_and_hold"] = compute_metrics(bh_curve, initial_balance, bh_trades, risk_free_rate=risk_free_rate)

    def ew_target(t, equity):
        return np.ones(n_assets) * (equity / n_assets)
    ew_curve, ew_trades = _simulate_rebalance_strategy(close, initial_balance, commission, start, 5, ew_target, turb_test, turb_threshold)
    results["equal_weight"] = compute_metrics(ew_curve, initial_balance, ew_trades, risk_free_rate=risk_free_rate)

    def momentum_target(t, equity):
        lookback = 20
        if t - lookback < 0:
            return np.zeros(n_assets)
        mom = close[t - 1] / (close[t - lookback] + EPS) - 1.0
        winners = mom > 0.0
        values = np.zeros(n_assets)
        if winners.any():
            values[winners] = equity * min(0.25, 1.0 / winners.sum())
        return values
    mom_curve, mom_trades = _simulate_rebalance_strategy(close, initial_balance, commission, start, 5, momentum_target, turb_test, turb_threshold)
    results["momentum_trader"] = compute_metrics(mom_curve, initial_balance, mom_trades, risk_free_rate=risk_free_rate)
    return results


def save_trade_log(env, path: str, tickers: Optional[list[str]] = None) -> None:
    trades = list(getattr(env, "trade_log", []))
    if not trades:
        pd.DataFrame().to_csv(path, index=False)
        return
    df = pd.DataFrame(trades)
    if tickers and "asset" in df.columns:
        mapping = {i: t for i, t in enumerate(tickers)}
        df["ticker"] = df["asset"].map(mapping)
    df.to_csv(path, index=False)


def _save_metrics_snapshot(run_dir: str, model_name: str, metrics: Dict, baseline_metrics: Optional[Dict] = None) -> None:
    snapshot = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "model_name": model_name}
    for k, v in (metrics or {}).items():
        snapshot[f"model_{k}"] = v
    for bname, bmet in (baseline_metrics or {}).items():
        for k, v in bmet.items():
            snapshot[f"{bname}_{k}"] = v
    pd.DataFrame([snapshot]).to_csv("metrics_history.csv", mode="a", header=not os.path.exists("metrics_history.csv"), index=False)
    with open(os.path.join(run_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"model_metrics": metrics, "baseline_metrics": baseline_metrics}, f, ensure_ascii=False, indent=2)


def train_model(model_name: str, data_cfg: Optional[DataConfig] = None, train_cfg: Optional[TrainConfig] = None, run_test: bool = True,bundle = None):
    data_cfg = data_cfg or DEFAULT_CONFIG
    train_cfg = train_cfg or TrainConfig()
    set_global_seed(train_cfg.seed)
    os.makedirs(train_cfg.save_dir, exist_ok=True)
    os.makedirs(train_cfg.tensorboard_log, exist_ok=True)
    device = get_device()
    use_slstm = check_slstm_available()
    registry = get_model_registry(use_slstm)
    if model_name not in registry:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(registry)}")
    spec = registry[model_name]
    env_cfg = make_env_cfg(train_cfg)

    if bundle is None:
        bundle = prepare_datasets(data_cfg)
    regime_all = getattr(bundle, "regime_all", None)
    rvi_all = getattr(bundle, "rvi_all", None)
    regime_calm = regime_all[bundle.calm_mask] if regime_all is not None else None
    rvi_calm = rvi_all[bundle.calm_mask] if rvi_all is not None else None

    obs_space = build_env(
        close_prices=bundle.close_test,
        features=bundle.features_test,
        turbulence=bundle.turb_test,
        turbulence_threshold=bundle.threshold,
        short=env_cfg.allow_short,
        v2=True,
        atr_values=bundle.atr_test,
        regime=bundle.regime_test,
        rvi=bundle.rvi_test,
        cfg=env_cfg,
    ).observation_space
    smoke_shape = smoke_test_encoder(spec.extractor_class, spec.extractor_kwargs, obs_space, device=device)
    print(f"[{model_name}] encoder smoke test output: {smoke_shape}")

    policy_kwargs = build_policy_kwargs(spec)
    ppo_kwargs = dict(
        policy="MlpLstmPolicy",
        learning_rate=train_cfg.learning_rate,
        n_steps=train_cfg.n_steps,
        batch_size=train_cfg.batch_size,
        n_epochs=train_cfg.n_epochs,
        gamma=train_cfg.gamma,
        gae_lambda=train_cfg.gae_lambda,
        clip_range=train_cfg.clip_range,
        ent_coef=train_cfg.ent_coef,
        target_kl=train_cfg.target_kl,
        vf_coef=train_cfg.vf_coef,
        max_grad_norm=train_cfg.max_grad_norm,
        verbose=train_cfg.verbose,
        seed=train_cfg.seed,
        tensorboard_log=train_cfg.tensorboard_log,
        device=device,
        policy_kwargs=policy_kwargs,
    )

    run_dir = os.path.join(train_cfg.save_dir, model_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    stage2_env = make_multi_episode_vec_env(
        close=bundle.close_train,
        features=bundle.features_train,
        turb=bundle.turb_train,
        threshold=bundle.threshold,
        env_cfg=env_cfg,
        episode_length=train_cfg.episode_length,
        episode_stride=train_cfg.episode_stride,
        regime=bundle.regime_train,
        rvi=bundle.rvi_train,
        atr_values=bundle.atr_train,
        seed=train_cfg.seed,
    )
    stage1_env = None
    if train_cfg.stage1_steps > 0:
        stage1_env = make_multi_episode_vec_env(
            close=bundle.close_all[bundle.calm_mask],
            features=bundle.features_all[bundle.calm_mask],
            turb=bundle.turb_aligned.values.astype(np.float32)[bundle.calm_mask],
            threshold=bundle.threshold,
            env_cfg=env_cfg,
            episode_length=min(train_cfg.episode_length, int(bundle.calm_mask.sum())),
            episode_stride=max(30, train_cfg.episode_stride),
            regime=regime_calm,
            rvi=rvi_calm,
            atr_values=bundle.atr_all[bundle.calm_mask],
            seed=train_cfg.seed + 1,
        )

    val_env = build_env(
        close_prices=bundle.close_val,
        features=bundle.features_val,
        turbulence=bundle.turb_val,
        turbulence_threshold=bundle.threshold,
        short=env_cfg.allow_short,
        v2=True,
        atr_values=bundle.atr_val,
        regime=bundle.regime_val,
        rvi=bundle.rvi_val,
        cfg=env_cfg,
    )

    resume_state = _read_resume_state(run_dir) if train_cfg.resume else {}
    model: Optional[RecurrentPPO] = None
    current_stage = "stage2" if train_cfg.stage1_steps <= 0 else "stage1"
    if train_cfg.resume:
        latest_stage2 = _find_latest_checkpoint(checkpoint_dir, f"{model_name}_stage2")
        latest_stage1 = _find_latest_checkpoint(checkpoint_dir, f"{model_name}_stage1")
        if latest_stage2:
            print(f"[{model_name}] resume stage2: {latest_stage2}")
            model = RecurrentPPO.load(latest_stage2, env=stage2_env, device=device)
            current_stage = "stage2"
        elif latest_stage1 and stage1_env is not None:
            print(f"[{model_name}] resume stage1: {latest_stage1}")
            model = RecurrentPPO.load(latest_stage1, env=stage1_env, device=device)
            current_stage = str(resume_state.get("stage", "stage1"))

    if model is None:
        model = RecurrentPPO(env=stage1_env if stage1_env is not None else stage2_env, **ppo_kwargs)

    if current_stage == "stage1" and stage1_env is not None and train_cfg.stage1_steps > 0:
        done = int(getattr(model, "num_timesteps", 0))
        remaining = max(train_cfg.stage1_steps - done, 0)
        if remaining > 0:
            print(f"\n[{model_name}] Stage1 calm-market warmup: {remaining:,}/{train_cfg.stage1_steps:,} steps")
            model.learn(
                total_timesteps=remaining,
                progress_bar=True,
                reset_num_timesteps=False,
                tb_log_name=f"{model_name}_stage1",
                callback=_build_callbacks(
                    model_name=model_name,
                    stage_name="stage1",
                    total_steps=train_cfg.stage1_steps,
                    run_dir=run_dir,
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_freq=train_cfg.checkpoint_freq,
                    log_every_steps=train_cfg.log_every_steps,
                    stage_offset_steps=0,
                ),
            )
        stage1_final = os.path.join(run_dir, f"{model_name}_stage1_final")
        model.save(stage1_final)
        print(f"[{model_name}] stage1 saved: {stage1_final}")
        model.set_env(stage2_env)
        current_stage = "stage2"

    if current_stage == "stage2":
        model.set_env(stage2_env)
        stage_offset = train_cfg.stage1_steps if train_cfg.stage1_steps > 0 else 0
        if train_cfg.resume and resume_state.get("stage") == "stage2" and int(resume_state.get("stage2_elapsed_steps", 0)) > 0:
            stage2_done = int(resume_state.get("stage2_elapsed_steps", 0))
        else:
            stage2_done = max(int(getattr(model, "num_timesteps", 0)) - stage_offset, 0)
        remaining = max(train_cfg.stage2_steps - stage2_done, 0)
        print(f"[{model_name}] Stage2 rolling-episode trading: {remaining:,}/{train_cfg.stage2_steps:,} steps left")
        if remaining > 0:
            best_model_path = os.path.join(run_dir, f"{model_name}_{train_cfg.selection_metric}")
            early = EarlyStoppingCallback(
                val_env=val_env,
                eval_freq=train_cfg.early_eval_freq,
                patience=train_cfg.early_stop_patience,
                min_improvement=train_cfg.early_min_improvement,
                best_model_path=best_model_path,
                risk_free_rate=train_cfg.risk_free_rate,
                selection_metric=train_cfg.selection_metric,
                no_trade_eval_patience=train_cfg.no_trade_eval_patience,
                min_val_trades=train_cfg.min_val_trades,
                min_val_trade_rate=train_cfg.min_val_trade_rate,
                verbose=True,
            )
            callbacks = CallbackList([
                _build_callbacks(
                    model_name=model_name,
                    stage_name="stage2",
                    total_steps=train_cfg.stage2_steps,
                    run_dir=run_dir,
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_freq=train_cfg.checkpoint_freq,
                    log_every_steps=train_cfg.log_every_steps,
                    checkpoint_warmup_steps=train_cfg.checkpoint_warmup_steps,
                    stage_offset_steps=stage_offset,
                ),
                early,
            ])
            model.learn(
                total_timesteps=remaining,
                progress_bar=True,
                callback=callbacks,
                reset_num_timesteps=False,
                tb_log_name=f"{model_name}_stage2",
            )
            last_path = os.path.join(run_dir, f"{model_name}_stage2_last")
            model.save(last_path)
            print(f"[{model_name}] stage2 last saved: {last_path}")
            # Do not load the early-stopping model here. Final selection happens
            # below after all checkpoints are evaluated with selection_metric.
        else:
            print(f"[{model_name}] Stage2 already completed/skipped.")

    selected_model_path = os.path.join(run_dir, model_name)

    ckpt_results: list = []
    ensemble_paths: list = []
    try:
        from checkpoint_selector import CheckpointSelector
        selector = CheckpointSelector(checkpoint_dir, model_name, stage="stage2", risk_free_rate=train_cfg.risk_free_rate)
        ckpt_results = selector.evaluate_all(val_env, device=str(device))
        if ckpt_results:
            top = selector.select_top(
                ckpt_results,
                k=3,
                min_trades=train_cfg.min_val_trades,
                min_trade_rate=train_cfg.min_val_trade_rate,
            )
            ensemble_paths = [r["path"] for r in top]
            with open(os.path.join(run_dir, "ensemble_models.json"), "w", encoding="utf-8") as f:
                json.dump({"ensemble_paths": ensemble_paths}, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"[{model_name}] CheckpointSelector skipped: {exc}")

    if ckpt_results:
        selected = select_checkpoint(
            ckpt_results,
            train_cfg.selection_metric,
            min_trades=train_cfg.min_val_trades,
            min_trade_rate=train_cfg.min_val_trade_rate,
        )
        print(
            f"\n[{model_name}] SELECTED MODEL by {train_cfg.selection_metric}: "
            f"step={selected.get('step', -1):,} | score={selected.get('score', float('nan')):+.3f} "
            f"SR={selected.get('SR', float('nan')):+.3f} "
            f"CR={selected.get('CR', float('nan')):+.2f}% "
            f"MPB={selected.get('MPB', float('nan')):.2f}% "
            f"Trades={selected.get('Trades', -1)}"
        )
        model = RecurrentPPO.load(selected["path"], env=stage2_env, device=device)
    else:
        early_path = os.path.join(run_dir, f"{model_name}_{train_cfg.selection_metric}")
        if os.path.exists(early_path + ".zip"):
            print(f"[{model_name}] no checkpoints; loading early-stopping model: {early_path}")
            model = RecurrentPPO.load(early_path, env=stage2_env, device=device)
        else:
            print(f"[{model_name}] no checkpoints; using last in-memory model")

    model.save(selected_model_path)
    print(f"[{model_name}] selected model saved: {selected_model_path}")

    metrics = None
    equity_curve = None
    baseline_metrics = None
    best_by_metric: Dict[str, Dict] = {}
    ensemble_metrics = None

    if run_test:
        from ensemble_policy import EnsemblePolicy

        def make_test_env():
            return build_env(
                close_prices=bundle.close_test,
                features=bundle.features_test,
                turbulence=bundle.turb_test,
                turbulence_threshold=bundle.threshold,
                short=env_cfg.allow_short,
                v2=True,
                atr_values=bundle.atr_test,
                regime=bundle.regime_test,
                rvi=bundle.rvi_test,
                cfg=env_cfg,
            )

        criteria = {
            "best_trade": (lambda r: checkpoint_key(r, "best_trade"), "max trading-oriented score"),
            "best_score": (lambda r: r["score"], "max composite score"),
            "best_SR": (lambda r: r["SR"], "max Sharpe"),
            "best_CR": (lambda r: r["CR"], "max cumulative return"),
            "best_MPB": (lambda r: -r["MPB"], "min drawdown"),
        }
        winner_paths: Dict[str, str] = {}
        sep = "=" * 72
        print(f"\n{sep}\n  ИТОГОВЫЙ ОТЧЁТ: validation → test\n{sep}")
        report_candidates = eligible_results(
            ckpt_results,
            min_trades=train_cfg.min_val_trades,
            min_trade_rate=train_cfg.min_val_trade_rate,
        )
        if ckpt_results and len(report_candidates) < len(ckpt_results):
            print(
                f"  Filtered cash-like checkpoints for report/ensemble: "
                f"{len(ckpt_results) - len(report_candidates)} removed "
                f"(min_trades={train_cfg.min_val_trades}, min_trade_rate={train_cfg.min_val_trade_rate})"
            )
        if report_candidates:
            for cname, (key_fn, label) in criteria.items():
                best = max(report_candidates, key=key_fn)
                winner_paths[cname] = best["path"]
                test_env = make_test_env()
                best_model = RecurrentPPO.load(best["path"], device=device)
                curve = evaluate_policy(best_model, test_env)
                tm = compute_metrics(curve, test_env.initial_balance, getattr(test_env, "trade_count", 0), risk_free_rate=train_cfg.risk_free_rate)
                val_m = {k: best.get(k, float("nan")) for k in ["score", "SR", "CR", "MPB", "Trades", "TradeRate"]}
                best_by_metric[cname] = {"path": best["path"], "step": best.get("step", -1), "label": label, "val": val_m, "test": tm}
                print(f"\n  [{cname}] {label} | step={best.get('step', -1):,}")
                print(f"    VAL  → score={val_m['score']:+.3f} SR={val_m['SR']:+.3f} CR={val_m['CR']:+.1f}% MPB={val_m['MPB']:.1f}% Trades={val_m['Trades']}")
                print(f"    TEST → SR={tm['SR']:+.3f} CR={tm['CR']:+.1f}% MPB={tm['MPB']:.1f}% Trades={tm['Trades']} WR={tm['WinRate']:.2f}")

        test_env = make_test_env()
        equity_curve = evaluate_policy(model, test_env)
        metrics = compute_metrics(equity_curve, test_env.initial_balance, getattr(test_env, "trade_count", 0), risk_free_rate=train_cfg.risk_free_rate)
        trade_log_path = os.path.join(run_dir, "selected_model_test_trade_log.csv")
        save_trade_log(test_env, trade_log_path, tickers=bundle.tickers)
        print(f"Trade log saved: {trade_log_path}")
        print(f"\n  [selected_model] selected by {train_cfg.selection_metric}")
        print(f"    TEST → SR={metrics['SR']:+.3f} CR={metrics['CR']:+.1f}% MPB={metrics['MPB']:.1f}% Trades={metrics['Trades']} WR={metrics['WinRate']:.2f} PF={metrics['ProfitFactor']:.2f}")

        unique_paths = list(dict.fromkeys(winner_paths.values() or ensemble_paths))
        if train_cfg.disable_ensemble:
            print(f"\n  [ENSEMBLE] disabled by config")
        elif len(unique_paths) >= 2:
            try:
                path_score = {r["path"]: max(float(r.get("score", 0.0)), 0.0) for r in ckpt_results}
                weights = np.array([path_score.get(p, 0.0) for p in unique_paths], dtype=np.float32)
                if weights.sum() <= 0:
                    weights = np.ones(len(unique_paths), dtype=np.float32) / len(unique_paths)
                else:
                    weights = weights / weights.sum()
                ens = EnsemblePolicy(unique_paths, device=str(device), weights=weights)
                ens_env = make_test_env()
                ens_curve = ens.evaluate(ens_env)
                ensemble_metrics = compute_metrics(ens_curve, ens_env.initial_balance, getattr(ens_env, "trade_count", 0), risk_free_rate=train_cfg.risk_free_rate)
                print(f"\n  [ENSEMBLE] {len(unique_paths)} checkpoint models")
                print(f"    TEST → SR={ensemble_metrics['SR']:+.3f} CR={ensemble_metrics['CR']:+.1f}% MPB={ensemble_metrics['MPB']:.1f}% Trades={ensemble_metrics['Trades']}")
            except Exception as exc:
                print(f"\n  [ENSEMBLE] error: {exc}")
        else:
            print(f"\n  [ENSEMBLE] not enough unique checkpoint models ({len(unique_paths)})")

        baseline_metrics = compute_baseline_metrics(
            bundle.close_test,
            bundle.turb_test,
            bundle.threshold,
            env_cfg.initial_balance,
            window_size=train_cfg.window_size,
            commission=train_cfg.commission,
            risk_free_rate=train_cfg.risk_free_rate,
        )
        for name, bm in baseline_metrics.items():
            print(f"  [{name}] TEST → SR={bm['SR']:+.3f} CR={bm['CR']:+.1f}% MPB={bm['MPB']:.1f}% Trades={bm['Trades']}")
        print(f"{sep}\n")

        _save_metrics_snapshot(run_dir, model_name, metrics, baseline_metrics)
        report = {
            "best_by_metric": best_by_metric,
            "selected_model_test": metrics,
            "ensemble_test": ensemble_metrics,
            "ensemble_paths": unique_paths,
            "baseline": baseline_metrics,
            "env_config": asdict(env_cfg),
            "train_config": asdict(train_cfg),
            "data_config": asdict(data_cfg),
        }
        report_path = os.path.join(run_dir, "val_test_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[{model_name}] report saved: {report_path}")

    return {
        "model_name": model_name,
        "model_path": selected_model_path,
        "metrics": metrics,
        "best_by_metric": best_by_metric,
        "ensemble_metrics": ensemble_metrics,
        "ensemble_paths": ensemble_paths,
        "baseline_metrics": baseline_metrics,
        "portfolio_history": equity_curve,
        "use_slstm": use_slstm,
        "device": str(device),
        "data_config": asdict(data_cfg),
        "train_config": asdict(train_cfg),
        "run_dir": run_dir,
        "checkpoint_dir": checkpoint_dir,
        "csv_log": os.path.join(run_dir, train_cfg.csv_log_name),
    }



def compare_models(
    model_names: list[str],
    data_cfg: Optional[DataConfig] = None,
    train_cfg: Optional[TrainConfig] = None,
    output_csv: str = "comparison_results.csv",
    output_json: str = "comparison_results.json",
) -> Dict[str, Dict]:
    """Train/evaluate several architectures and save a diploma-style table."""
    data_cfg = data_cfg or DEFAULT_CONFIG
    train_cfg = train_cfg or TrainConfig()
    
    # ← ДОБАВИТЬ: загружаем данные один раз для всех моделей
    print("Загружаем данные один раз для всех моделей...")
    from data import prepare_datasets
    bundle = prepare_datasets(data_cfg)

    rows = []
    results: Dict[str, Dict] = {}
    for name in model_names:
        print(f"\n{'=' * 72}\n  MODEL COMPARISON RUN: {name}\n{'=' * 72}")
        try:
            res = train_model(name, data_cfg=data_cfg, train_cfg=train_cfg, run_test=True, bundle=bundle)
            metrics = res.get("metrics") or {}
            baseline = res.get("baseline_metrics") or {}
            row = {
                "model": name,
                "selection_metric": train_cfg.selection_metric,
                "SR": metrics.get("SR"),
                "CR": metrics.get("CR"),
                "MPB": metrics.get("MPB"),
                "Sortino": metrics.get("Sortino"),
                "Calmar": metrics.get("Calmar"),
                "WinRate": metrics.get("WinRate"),
                "ProfitFactor": metrics.get("ProfitFactor"),
                "Trades": metrics.get("Trades"),
                "TradeRate": metrics.get("TradeRate"),
                "FinalValue": metrics.get("FinalValue"),
                "run_dir": res.get("run_dir"),
            }
            for bname, bmet in baseline.items():
                row[f"{bname}_SR"] = bmet.get("SR")
                row[f"{bname}_CR"] = bmet.get("CR")
                row[f"{bname}_MPB"] = bmet.get("MPB")
            rows.append(row)
            results[name] = res
        except Exception as exc:
            print(f"[compare_models] ERROR for {name}: {exc}")
            rows.append({"model": name, "error": str(exc)})
            results[name] = {"error": str(exc)}

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n{'=' * 72}\n  ARCHITECTURE COMPARISON TABLE\n{'=' * 72}")
    if not df.empty:
        cols = [c for c in ["model", "selection_metric", "SR", "CR", "MPB", "ProfitFactor", "Trades", "TradeRate"] if c in df.columns]
        print(df[cols].to_string(index=False))
    print(f"\nSaved: {output_csv}\nSaved: {output_json}")
    return results

def cli():
    parser = argparse.ArgumentParser(description="Train MOEX trading-RL model")
    parser.add_argument("--model", default="mlp", choices=["mlp", "last_mlp", "lstm", "transformer_causal", "tft_like", "xlstm_base", "xlstm_attn", "xlstm_large"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage1-steps", type=int, default=0)
    parser.add_argument("--stage2-steps", type=int, default=150_000)
    parser.add_argument("--save-dir", default="./models")
    parser.add_argument("--tensorboard-log", default="./tb_logs")
    parser.add_argument("--data-file", default="moex_data_v3.csv")
    parser.add_argument("--checkpoint-freq", type=int, default=5_000)
    parser.add_argument("--log-every-steps", type=int, default=1_000)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--use-hmm", action="store_true")
    parser.add_argument("--use-rvi", action="store_true")

    parser.add_argument("--selection-metric", default="best_score", choices=["best_trade", "best_score", "best_SR", "best_CR", "best_MPB"])
    parser.add_argument("--reward-mode", default="upside_opportunity", choices=["upside_opportunity", "benchmark_relative", "log_return"])
    parser.add_argument("--benchmark-coef", type=float, default=0.35)
    parser.add_argument("--opportunity-coef", type=float, default=1.0)
    parser.add_argument("--target-exposure", type=float, default=0.30)
    parser.add_argument("--low-exposure-threshold", type=float, default=0.03)
    parser.add_argument("--cash-penalty-grace-steps", type=int, default=5)
    parser.add_argument("--turnover-penalty-coef", type=float, default=0.0)
    parser.add_argument("--min-trade-action", type=float, default=0.0)
    parser.add_argument("--rebalance-cooldown", type=int, default=0)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--early-min-improvement", type=float, default=0.005)
    parser.add_argument("--disable-ensemble", action="store_true")
    parser.add_argument("--min-val-trades", type=int, default=20)
    parser.add_argument("--min-val-trade-rate", type=float, default=0.10)
    parser.add_argument("--no-trade-eval-patience", type=int, default=3)

    parser.add_argument("--compare", action="store_true", help="Train/evaluate several architectures and save comparison table")
    parser.add_argument("--compare-models", nargs="+", default=["mlp", "lstm", "transformer_causal", "tft_like"])
    parser.add_argument("--comparison-output", default="comparison_results.csv")
    args = parser.parse_args()

    data_cfg = DataConfig(**{**asdict(DEFAULT_CONFIG), "data_file": args.data_file, "use_hmm": args.use_hmm, "use_rvi": args.use_rvi})
    train_cfg = TrainConfig(
        seed=args.seed,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        save_dir=args.save_dir,
        tensorboard_log=args.tensorboard_log,
        checkpoint_freq=args.checkpoint_freq,
        log_every_steps=args.log_every_steps,
        resume=not args.no_resume,
        allow_short=args.allow_short,
        selection_metric=args.selection_metric,
        reward_mode=args.reward_mode,
        benchmark_coef=args.benchmark_coef,
        opportunity_coef=args.opportunity_coef,
        target_exposure=args.target_exposure,
        low_exposure_threshold=args.low_exposure_threshold,
        cash_penalty_grace_steps=args.cash_penalty_grace_steps,
        turnover_penalty_coef=args.turnover_penalty_coef,
        min_trade_action=args.min_trade_action,
        rebalance_cooldown=args.rebalance_cooldown,
        early_stop_patience=args.early_stop_patience,
        early_min_improvement=args.early_min_improvement,
        disable_ensemble=args.disable_ensemble,
        min_val_trades=args.min_val_trades,
        min_val_trade_rate=args.min_val_trade_rate,
        no_trade_eval_patience=args.no_trade_eval_patience,
    )

    if args.compare:
        json_path = os.path.splitext(args.comparison_output)[0] + ".json"
        compare_models(args.compare_models, data_cfg=data_cfg, train_cfg=train_cfg, output_csv=args.comparison_output, output_json=json_path)
        return

    result = train_model(args.model, data_cfg=data_cfg, train_cfg=train_cfg, run_test=True)
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
