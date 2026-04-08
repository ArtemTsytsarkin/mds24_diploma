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
from env import EnvFactoryConfig, build_env
from models import build_policy_kwargs, check_slstm_available, get_device, get_model_registry, smoke_test_encoder


@dataclass
class TrainConfig:
    window_size: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-4
    n_steps: int = 2048
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    stage1_steps: int = 300_000
    stage2_steps: int = 700_000
    seed: int = 42
    tensorboard_log: str = "./tb_logs"
    save_dir: str = "./models"
    margin_ratio: float = 0.25
    initial_balance: float = 1_000_000.0
    commission: float = 0.0005
    max_shares_per_trade: int = 100
    verbose: int = 1
    checkpoint_freq: int = 50_000
    log_every_steps: int = 1_000
    csv_log_name: str = "training_log.csv"
    resume: bool = True


class TrainProgressCallback(BaseCallback):
    """Console + CSV + TensorBoard logging with ETA and stage metadata."""

    def __init__(
        self,
        *,
        model_name: str,
        stage_name: str,
        total_stage_steps: int,
        run_dir: str,
        log_every_steps: int = 1_000,
    ):
        super().__init__()
        self.model_name = model_name
        self.stage_name = stage_name
        self.total_stage_steps = total_stage_steps
        self.run_dir = run_dir
        self.log_every_steps = log_every_steps
        self.csv_path = os.path.join(run_dir, "training_log.csv")
        self.state_path = os.path.join(run_dir, "resume_state.json")
        self.start_time: Optional[float] = None
        self.stage_start_num_timesteps: int = 0

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self.stage_start_num_timesteps = int(getattr(self.model, "num_timesteps", 0))
        self._write_state(status="running")

    def _append_csv_row(self, row: Dict[str, float]) -> None:
        df = pd.DataFrame([row])
        header = not os.path.exists(self.csv_path)
        df.to_csv(self.csv_path, mode="a", header=header, index=False)

    def _write_state(self, status: str) -> None:
        global_ts = int(getattr(self.model, "num_timesteps", 0))
        stage_elapsed = global_ts - self.stage_start_num_timesteps
        state = {
            "model_name": self.model_name,
            "stage": self.stage_name,
            "status": status,
            "num_timesteps": global_ts,
            "stage_elapsed_steps": stage_elapsed,
            "total_stage_steps": int(self.total_stage_steps),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        # Persist stage-specific elapsed steps under a stable key so resume can
        # reconstruct progress without relying on global num_timesteps arithmetic.
        if self.stage_name == "stage2":
            state["stage2_elapsed_steps"] = stage_elapsed
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _on_step(self) -> bool:
        if self.log_every_steps <= 0 or self.n_calls % self.log_every_steps != 0:
            return True

        elapsed = time.time() - (self.start_time or time.time())
        stage_elapsed_steps = int(getattr(self.model, "num_timesteps", 0) - self.stage_start_num_timesteps)
        stage_progress = stage_elapsed_steps / max(self.total_stage_steps, 1)
        eta_sec = elapsed * (1.0 - stage_progress) / max(stage_progress, 1e-8)

        rewards = self.locals.get("rewards")
        reward_mean = float(np.mean(rewards)) if rewards is not None else float("nan")
        fps = stage_elapsed_steps / max(elapsed, 1e-8)

        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": self.model_name,
            "stage": self.stage_name,
            "num_timesteps": int(getattr(self.model, "num_timesteps", 0)),
            "stage_elapsed_steps": stage_elapsed_steps,
            "stage_total_steps": int(self.total_stage_steps),
            "stage_progress_pct": round(stage_progress * 100.0, 4),
            "reward_mean": reward_mean,
            "elapsed_sec": round(elapsed, 3),
            "eta_sec": max(int(eta_sec), 0),
            "fps": round(fps, 3),
        }
        self._append_csv_row(row)
        self._write_state(status="running")

        if np.isfinite(reward_mean):
            self.logger.record("custom/reward_mean", reward_mean)
        self.logger.record("custom/stage_progress_pct", stage_progress * 100.0)
        self.logger.record("custom/eta_sec", max(float(eta_sec), 0.0))
        self.logger.record("custom/fps_stage", fps)

        reward_str = f"{reward_mean:.6f}" if np.isfinite(reward_mean) else "nan"
        print(
            f"[{self.model_name}][{self.stage_name}] "
            f"step={int(getattr(self.model, 'num_timesteps', 0)):,} | "
            f"stage={stage_elapsed_steps:,}/{self.total_stage_steps:,} ({stage_progress * 100:6.2f}%) | "
            f"reward_mean={reward_str} | elapsed={int(elapsed)}s | eta={max(int(eta_sec), 0)}s | fps={fps:.1f}"
        )
        return True

    def _on_training_end(self) -> None:
        self._write_state(status="completed")


class StageCheckpointCallback(CheckpointCallback):
    """Checkpoint saver that also mirrors resume metadata for the current stage."""

    def __init__(self, *, stage_name: str, run_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.stage_name = stage_name
        self.run_dir = run_dir
        self.state_path = os.path.join(run_dir, "resume_state.json")

    def _on_step(self) -> bool:
        ok = super()._on_step()
        if self.save_freq > 0 and self.n_calls % self.save_freq == 0:
            global_ts = int(getattr(self.model, "num_timesteps", 0))
            state = {
                "stage": self.stage_name,
                "status": "checkpointed",
                "num_timesteps": global_ts,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            # Preserve stage2_elapsed_steps in checkpoint metadata so that
            # resume logic can reconstruct progress correctly (fix #2).
            existing = {}
            if os.path.exists(self.state_path):
                try:
                    with open(self.state_path, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                except Exception:
                    pass
            if self.stage_name == "stage2":
                state["stage2_elapsed_steps"] = existing.get("stage2_elapsed_steps", 0)
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        return ok


def make_vec_env(close, features, turb, threshold, env_cfg: EnvFactoryConfig, short: bool = True):
    from functools import partial

    def _make_env(close_, features_, turb_, threshold_, env_cfg_, short_):
        return build_env(
            close_prices=close_,
            features=features_,
            turbulence=turb_,
            turbulence_threshold=threshold_,
            short=short_,
            cfg=env_cfg_,
        )

    factory = partial(_make_env, close, features, turb, threshold, env_cfg, short)
    return DummyVecEnv([factory])


def evaluate_policy(model, env):
    """
    Evaluate on a raw (non-vectorized) env.
    RecurrentPPO.predict expects observations with a batch dimension,
    so we add/remove it explicitly and keep lstm_states consistent.
    """
    obs, _ = env.reset()
    # Add batch dim: (window, features) -> (1, window, features)
    obs_batch = obs[np.newaxis, ...]
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    done = False
    while not done:
        action_batch, lstm_states = model.predict(
            obs_batch,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=True,
        )
        # action_batch shape: (1, n_assets) — pass the single action vector
        obs_raw, _, terminated, truncated, _ = env.step(action_batch[0])
        obs_batch = obs_raw[np.newaxis, ...]
        episode_starts = np.array([terminated or truncated], dtype=bool)
        done = bool(terminated or truncated)
    return env.portfolio_history


def compute_metrics(portfolio_history, initial_balance, n_trades, risk_free_rate: float = 0.16) -> Dict[str, float]:
    values = np.array(portfolio_history, dtype=float)
    p_init = initial_balance
    p_final = values[-1]
    cr = (p_final - p_init) / p_init * 100
    mer = (values.max() - p_init) / p_init * 100
    running_max = np.maximum.accumulate(values)
    mpb = abs(((values - running_max) / (running_max + 1e-8)).min()) * 100
    appt = (p_final - p_init) / max(n_trades, 1) / 1000
    daily_returns = np.diff(values) / (values[:-1] + 1e-8)
    excess_returns = daily_returns - risk_free_rate / 252
    sr = (excess_returns.mean() / (excess_returns.std() + 1e-8)) * np.sqrt(252)

    wins = daily_returns[daily_returns > 0]
    losses = daily_returns[daily_returns < 0]
    win_rate = float((daily_returns > 0).mean()) if len(daily_returns) else 0.0
    profit_factor = float(wins.sum() / (abs(losses.sum()) + 1e-8)) if len(daily_returns) else 0.0

    return {
        "CR": float(cr),
        "MER": float(mer),
        "MPB": float(mpb),
        "APPT": float(appt),
        "SR": float(sr),
        "WinRate": float(win_rate),
        "ProfitFactor": float(profit_factor),
        "Trades": int(n_trades),
        "FinalValue": float(p_final),
    }


def _extract_step_from_checkpoint(filename: str) -> int:
    m = re.search(r"_(\d+)_steps\.zip$", filename)
    return int(m.group(1)) if m else -1


def _find_latest_checkpoint(checkpoint_dir: str, prefix: str) -> Optional[str]:
    if not os.path.isdir(checkpoint_dir):
        return None
    candidates = [
        f for f in os.listdir(checkpoint_dir) if f.startswith(prefix) and f.endswith(".zip")
    ]
    if not candidates:
        return None
    candidates.sort(key=_extract_step_from_checkpoint)
    latest = candidates[-1]
    return os.path.join(checkpoint_dir, latest)


def _read_resume_state(run_dir: str) -> Dict[str, object]:
    state_path = os.path.join(run_dir, "resume_state.json")
    if not os.path.exists(state_path):
        return {}
    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_stage_callbacks(
    *,
    model_name: str,
    stage_name: str,
    total_stage_steps: int,
    run_dir: str,
    checkpoint_dir: str,
    checkpoint_freq: int,
    log_every_steps: int,
) -> CallbackList:
    callbacks = [
        TrainProgressCallback(
            model_name=model_name,
            stage_name=stage_name,
            total_stage_steps=total_stage_steps,
            run_dir=run_dir,
            log_every_steps=log_every_steps,
        )
    ]
    if checkpoint_freq > 0:
        callbacks.append(
            StageCheckpointCallback(
                stage_name=stage_name,
                run_dir=run_dir,
                save_freq=checkpoint_freq,
                save_path=checkpoint_dir,
                name_prefix=f"{model_name}_{stage_name}",
            )
        )
    return CallbackList(callbacks)


def train_model(
    model_name: str,
    data_cfg: Optional[DataConfig] = None,
    train_cfg: Optional[TrainConfig] = None,
    run_test: bool = True,
):
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

    env_cfg = EnvFactoryConfig(
        window_size=train_cfg.window_size,
        margin_ratio=train_cfg.margin_ratio,
        initial_balance=train_cfg.initial_balance,
        commission=train_cfg.commission,
        max_shares_per_trade=train_cfg.max_shares_per_trade,
    )

    bundle = prepare_datasets(data_cfg)

    obs_space = build_env(
        close_prices=bundle.close_test,
        features=bundle.features_test,
        turbulence=bundle.turb_test,
        turbulence_threshold=bundle.threshold,
        short=spec.short,
        cfg=env_cfg,
    ).observation_space
    smoke_shape = smoke_test_encoder(spec.extractor_class, spec.extractor_kwargs, obs_space, device=device)
    print(f"[{model_name}] encoder smoke test output: {smoke_shape}")

    policy_kwargs = build_policy_kwargs(spec)
    base_ppo = dict(
        policy="MlpLstmPolicy",
        learning_rate=train_cfg.learning_rate,
        n_steps=train_cfg.n_steps,
        batch_size=train_cfg.batch_size,
        n_epochs=train_cfg.n_epochs,
        gamma=train_cfg.gamma,
        gae_lambda=train_cfg.gae_lambda,
        clip_range=train_cfg.clip_range,
        ent_coef=train_cfg.ent_coef,
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

    stage1_env = make_vec_env(
        bundle.close_all[bundle.calm_mask],
        bundle.features_all[bundle.calm_mask],
        bundle.turb_aligned[bundle.calm_mask].values,
        bundle.threshold,
        env_cfg,
        short=spec.short,
    )
    stage2_env = make_vec_env(
        bundle.close_train,
        bundle.features_train,
        bundle.turb_train,
        bundle.threshold,
        env_cfg,
        short=spec.short,
    )

    model: Optional[RecurrentPPO] = None
    current_stage = "stage1"
    resume_state = _read_resume_state(run_dir) if train_cfg.resume else {}

    if train_cfg.resume:
        latest_stage2_ckpt = _find_latest_checkpoint(checkpoint_dir, f"{model_name}_stage2")
        latest_stage1_ckpt = _find_latest_checkpoint(checkpoint_dir, f"{model_name}_stage1")
        if latest_stage2_ckpt:
            print(f"[{model_name}] resume from checkpoint: {latest_stage2_ckpt}")
            model = RecurrentPPO.load(latest_stage2_ckpt, env=stage2_env, device=device)
            current_stage = "stage2"
        elif latest_stage1_ckpt:
            print(f"[{model_name}] resume from checkpoint: {latest_stage1_ckpt}")
            model = RecurrentPPO.load(latest_stage1_ckpt, env=stage1_env, device=device)
            current_stage = str(resume_state.get("stage", "stage1"))

    if model is None:
        model = RecurrentPPO(env=stage1_env, **base_ppo)
        current_stage = "stage1"

    if current_stage == "stage1":
        already_done = int(getattr(model, "num_timesteps", 0))
        remaining = max(train_cfg.stage1_steps - already_done, 0)
        if remaining > 0:
            print(
                f"\n[{model_name}] Этап 1: спокойный рынок "
                f"{data_cfg.calm_start}–{data_cfg.calm_end} "
                f"({remaining:,} из {train_cfg.stage1_steps:,} шагов осталось)..."
            )
            model.learn(
                total_timesteps=remaining,
                progress_bar=True,
                callback=_build_stage_callbacks(
                    model_name=model_name,
                    stage_name="stage1",
                    total_stage_steps=train_cfg.stage1_steps,
                    run_dir=run_dir,
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_freq=train_cfg.checkpoint_freq,
                    log_every_steps=train_cfg.log_every_steps,
                ),
                reset_num_timesteps=False,
                tb_log_name=f"{model_name}_stage1",
            )
        else:
            print(f"[{model_name}] Этап 1 уже завершён, пропускаю.")

        stage1_final_path = os.path.join(run_dir, f"{model_name}_stage1_final")
        model.save(stage1_final_path)
        print(f"[{model_name}] этап 1 сохранён: {stage1_final_path}")
        model.set_env(stage2_env)
        current_stage = "stage2"

    if current_stage == "stage2":
        # If we resumed from a stage2 checkpoint the global num_timesteps already
        # contains stage1 steps, so we must NOT subtract stage1_steps again.
        # resume_state carries authoritative stage2 elapsed count when available.
        resume_stage2_done = int(resume_state.get("stage2_elapsed_steps", 0)) if train_cfg.resume else 0
        global_ts = int(getattr(model, "num_timesteps", 0))
        if resume_state.get("stage") == "stage2" and resume_stage2_done > 0:
            stage2_done = resume_stage2_done
        else:
            stage2_done = max(global_ts - train_cfg.stage1_steps, 0)
        remaining = max(train_cfg.stage2_steps - stage2_done, 0)
        print(
            f"[{model_name}] Этап 2: полный train {data_cfg.train_start}–{data_cfg.train_end} "
            f"({remaining:,} из {train_cfg.stage2_steps:,} шагов осталось)..."
        )
        if remaining > 0:
            model.learn(
                total_timesteps=remaining,
                progress_bar=True,
                callback=_build_stage_callbacks(
                    model_name=model_name,
                    stage_name="stage2",
                    total_stage_steps=train_cfg.stage2_steps,
                    run_dir=run_dir,
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_freq=train_cfg.checkpoint_freq,
                    log_every_steps=train_cfg.log_every_steps,
                ),
                reset_num_timesteps=False,
                tb_log_name=f"{model_name}_stage2",
            )
        else:
            print(f"[{model_name}] Этап 2 уже завершён, пропускаю.")

    model_path = os.path.join(run_dir, model_name)
    model.save(model_path)
    print(f"[{model_name}] модель сохранена: {model_path}")

    metrics = None
    portfolio = None
    if run_test:
        test_env = build_env(
            close_prices=bundle.close_test,
            features=bundle.features_test,
            turbulence=bundle.turb_test,
            turbulence_threshold=bundle.threshold,
            short=spec.short,
            cfg=env_cfg,
        )
        portfolio = evaluate_policy(model, test_env)
        metrics = compute_metrics(portfolio, test_env.initial_balance, test_env.trade_count)
        print(f"[{model_name}] test metrics: {json.dumps(metrics, ensure_ascii=False, indent=2)}")

    return {
        "model_name": model_name,
        "model_path": model_path,
        "metrics": metrics,
        "portfolio_history": portfolio,
        "use_slstm": use_slstm,
        "device": str(device),
        "data_config": asdict(data_cfg),
        "train_config": asdict(train_cfg),
        "run_dir": run_dir,
        "checkpoint_dir": checkpoint_dir,
        "csv_log": os.path.join(run_dir, train_cfg.csv_log_name),
    }


def cli():
    parser = argparse.ArgumentParser(description="Train MOEX RL model")
    parser.add_argument("--model", default="xlstm_base", choices=["lstm", "xlstm_base", "xlstm_attn", "xlstm_large"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage1-steps", type=int, default=300_000)
    parser.add_argument("--stage2-steps", type=int, default=700_000)
    parser.add_argument("--save-dir", default="./models")
    parser.add_argument("--tensorboard-log", default="./tb_logs")
    parser.add_argument("--data-file", default="moex_data.csv")
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--log-every-steps", type=int, default=1_000)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    data_cfg = DataConfig(**{**asdict(DEFAULT_CONFIG), "data_file": args.data_file})
    train_cfg = TrainConfig(
        seed=args.seed,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        save_dir=args.save_dir,
        tensorboard_log=args.tensorboard_log,
        checkpoint_freq=args.checkpoint_freq,
        log_every_steps=args.log_every_steps,
        resume=not args.no_resume,
    )
    result = train_model(args.model, data_cfg=data_cfg, train_cfg=train_cfg, run_test=True)
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
