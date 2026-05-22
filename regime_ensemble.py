"""
regime_ensemble.py
==================
Ансамбль с онлайн детектором режима.

Три модели (bull / sideways / bear) обучаются каждая на своих данных.
При инференсе онлайн-детектор режима вычисляет мягкие веса по последним
20 дням и возвращает взвешенное среднее действий всех трёх моделей.

Аугментация:
  - jittering : добавляем гауссовский шум к log-returns
  - scaling   : масштабируем амплитуду движений (тренд × k)

Запуск:
  # Обучение
  python regime_ensemble.py --model lstm --stage1-steps 10000 --stage2-steps 20000 --no-test

  # Полное обучение
  python regime_ensemble.py --model xlstm_large --stage1-steps 150000 --stage2-steps 300000

  # Только оценка уже обученного ансамбля
  python regime_ensemble.py --eval-only --ensemble-dir ./models/regime_ensemble --model xlstm_large
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv

# ── Импорты из твоего проекта ─────────────────────────────────────────────────
from data import DEFAULT_CONFIG, DataConfig, prepare_datasets, set_global_seed
from env import EnvFactoryConfig, build_env, MultiEpisodeEnv
from models import (
    build_policy_kwargs,
    check_slstm_available,
    get_device,
    get_model_registry,
    smoke_test_encoder,
)

# ══════════════════════════════════════════════════════════════════════════════
#  КОНСТАНТЫ
# ══════════════════════════════════════════════════════════════════════════════

REGIME_NAMES        = ["bull", "sideways", "bear"]
MIN_EPISODE_LENGTH  = 60    # минимум дней в непрерывном эпизоде
DETECTOR_LOOKBACK   = 20    # дней для онлайн детектора
VOL_BEAR_THRESHOLD  = 0.32  # annualized vol выше → bear
TREND_BULL_THRESHOLD = 0.03 # рост за 20 дней выше → bull
DETECTOR_SHARPNESS  = 8.0   # крутизна sigmoid переключения
JITTER_STD          = 0.001 # std гауссовского шума
SCALE_RANGE         = (0.80, 1.20)


# ══════════════════════════════════════════════════════════════════════════════
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (скопированы из train.py, без зависимостей)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_policy(model: RecurrentPPO, env) -> list:
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
            deterministic=True,
        )
        obs_raw, _, terminated, truncated, _ = env.step(action_batch[0])
        obs_batch = obs_raw[np.newaxis, ...]
        episode_starts = np.array([terminated or truncated], dtype=bool)
        done = bool(terminated or truncated)
    return env.portfolio_history


def compute_metrics(
    portfolio_history, initial_balance, n_trades, risk_free_rate: float = 0.16
) -> Dict[str, float]:
    values = np.array(portfolio_history, dtype=float)
    p_init, p_final = initial_balance, values[-1]
    cr  = (p_final - p_init) / p_init * 100
    mer = (values.max() - p_init) / p_init * 100
    running_max = np.maximum.accumulate(values)
    mpb = abs(((values - running_max) / (running_max + 1e-8)).min()) * 100
    appt = (p_final - p_init) / max(n_trades, 1) / 1000
    daily_returns  = np.diff(values) / (values[:-1] + 1e-8)
    excess_returns = daily_returns - risk_free_rate / 252
    if len(excess_returns) < 2 or excess_returns.std() < 1e-6:
        sr = float("nan")
    else:
        sr = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    wins = daily_returns[daily_returns > 0]
    losses = daily_returns[daily_returns < 0]
    win_rate = float((daily_returns > 0).mean()) if len(daily_returns) else 0.0
    pf = float(wins.sum() / (abs(losses.sum()) + 1e-8)) if len(daily_returns) else 0.0
    return {
        "CR": float(cr), "MER": float(mer), "MPB": float(mpb),
        "APPT": float(appt), "SR": float(sr), "WinRate": float(win_rate),
        "ProfitFactor": float(pf), "Trades": int(n_trades),
        "FinalValue": float(p_final),
    }


def compute_baseline_metrics(
    close_test: np.ndarray, turb_test: np.ndarray,
    turb_threshold: float, initial_balance: float,
) -> Dict[str, Dict]:
    n_days, n_assets = close_test.shape
    results = {}
    prices_0 = close_test[0]
    shares = (initial_balance / n_assets) / (prices_0 + 1e-8)
    bh_portfolio = [float(np.sum(shares * close_test[t])) for t in range(n_days)]
    results["buy_and_hold"] = compute_metrics(bh_portfolio, initial_balance, 1)

    ew_portfolio = [initial_balance]
    ew_balance = initial_balance
    ew_shares  = np.zeros(n_assets, dtype=np.float64)
    commission = 0.0005
    ew_trades  = 0
    for t in range(n_days - 1):
        prices = close_test[t]
        if turb_test[t] > turb_threshold:
            ew_balance += np.sum(ew_shares * prices) * (1 - commission)
            if np.any(ew_shares != 0):
                ew_trades += 1
            ew_shares = np.zeros(n_assets, dtype=np.float64)
        else:
            pv = ew_balance + np.sum(ew_shares * prices)
            target = pv / n_assets
            new_shares = target / (prices + 1e-8)
            delta = new_shares - ew_shares
            cost = np.sum(np.abs(delta) * prices) * commission
            if cost < pv * 0.001:
                ew_balance -= np.sum(delta * prices) + cost
                ew_shares   = new_shares
                ew_trades  += 1
        ew_portfolio.append(ew_balance + np.sum(ew_shares * close_test[t + 1]))
    results["equal_weight"] = compute_metrics(ew_portfolio, initial_balance, ew_trades)
    return results


def _extract_step(filename: str) -> int:
    m = re.search(r"_(\d+)_steps\.zip$", filename)
    return int(m.group(1)) if m else -1


def _find_latest_checkpoint(checkpoint_dir: str, prefix: str) -> Optional[str]:
    if not os.path.isdir(checkpoint_dir):
        return None
    candidates = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith(prefix) and f.endswith(".zip")
    ]
    if not candidates:
        return None
    candidates.sort(key=_extract_step)
    return os.path.join(checkpoint_dir, candidates[-1])


def _read_resume_state(run_dir: str) -> Dict:
    path = os.path.join(run_dir, "resume_state.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_resume_state(run_dir: str, state: Dict) -> None:
    path = os.path.join(run_dir, "resume_state.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def make_vec_env(
    close, features, turb, threshold,
    env_cfg: EnvFactoryConfig,
    short: bool = True, v2: bool = False,
    atr_indices=None, regime=None, rvi=None,
) -> DummyVecEnv:
    """Создаёт векторизованную среду напрямую через env.py."""
    def _make(close_, features_, turb_, threshold_, cfg_, short_, v2_, atr_, regime_, rvi_):
        return build_env(
            close_prices=close_, features=features_,
            turbulence=turb_, turbulence_threshold=threshold_,
            short=short_, v2=v2_, atr_indices=atr_,
            regime=regime_, rvi=rvi_, cfg=cfg_,
        )
    factory = partial(
        _make, close, features, turb, threshold,
        env_cfg, short, v2, atr_indices, regime, rvi,
    )
    return DummyVecEnv([factory])


# ══════════════════════════════════════════════════════════════════════════════
#  CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    window_size: int         = 30
    batch_size: int          = 128
    learning_rate: float     = 3e-5
    n_steps: int             = 1024
    n_epochs: int            = 4
    gamma: float             = 0.985
    gae_lambda: float        = 0.95
    clip_range: float        = 0.1
    ent_coef: float          = 0.005
    target_kl: float         = 0.02
    vf_coef: float           = 0.5
    max_grad_norm: float     = 0.5
    stage1_steps: int        = 150_000
    stage2_steps: int        = 300_000
    seed: int                = 42
    tensorboard_log: str     = "./tb_logs"
    save_dir: str            = "./models/regime_ensemble"
    margin_ratio: float      = 0.25
    initial_balance: float   = 1_000_000.0
    commission: float        = 0.0005
    max_shares_per_trade: int = 100
    verbose: int             = 1
    checkpoint_freq: int     = 5_000
    log_every_steps: int     = 1_000


class TrainProgressCallback(BaseCallback):
    def __init__(self, *, model_name, stage_name, total_stage_steps,
                 run_dir, log_every_steps=1_000):
        super().__init__()
        self.model_name        = model_name
        self.stage_name        = stage_name
        self.total_stage_steps = total_stage_steps
        self.run_dir           = run_dir
        self.log_every_steps   = log_every_steps
        self.csv_path          = os.path.join(run_dir, "training_log.csv")
        self.start_time        = None
        self.stage_start_ts    = 0

    def _on_training_start(self):
        self.start_time     = time.time()
        self.stage_start_ts = int(getattr(self.model, "num_timesteps", 0))

    def _on_step(self) -> bool:
        if self.log_every_steps <= 0 or self.n_calls % self.log_every_steps != 0:
            return True
        elapsed = time.time() - (self.start_time or time.time())
        stage_elapsed = int(getattr(self.model, "num_timesteps", 0)) - self.stage_start_ts
        progress = stage_elapsed / max(self.total_stage_steps, 1)
        eta = elapsed * (1.0 - progress) / max(progress, 1e-8)
        rewards = self.locals.get("rewards")
        reward_mean = float(np.mean(rewards)) if rewards is not None else float("nan")
        fps = stage_elapsed / max(elapsed, 1e-8)
        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": self.model_name, "stage": self.stage_name,
            "num_timesteps": int(getattr(self.model, "num_timesteps", 0)),
            "stage_elapsed": stage_elapsed, "stage_total": self.total_stage_steps,
            "progress_pct": round(progress * 100, 4),
            "reward_mean": reward_mean, "elapsed_sec": round(elapsed, 3),
            "eta_sec": max(int(eta), 0), "fps": round(fps, 3),
        }
        df = pd.DataFrame([row])
        header = not os.path.exists(self.csv_path)
        df.to_csv(self.csv_path, mode="a", header=header, index=False)
        if np.isfinite(reward_mean):
            self.logger.record("custom/reward_mean", reward_mean)
        self.logger.record("custom/progress_pct", progress * 100)
        reward_str = f"{reward_mean:.6f}" if np.isfinite(reward_mean) else "nan"
        print(
            f"[{self.model_name}][{self.stage_name}] "
            f"step={int(getattr(self.model, 'num_timesteps', 0)):,} | "
            f"stage={stage_elapsed:,}/{self.total_stage_steps:,} "
            f"({progress*100:6.2f}%) | reward={reward_str} | "
            f"eta={max(int(eta), 0)}s | fps={fps:.1f}"
        )
        return True

    def _on_training_end(self):
        pass


class EarlyStoppingCallback(BaseCallback):
    def __init__(self, *, val_env, eval_freq=5_000, patience=40,
                 min_improvement=0.03, best_model_path="./best_model",
                 verbose=True, use_calmar=False):
        super().__init__()
        self.val_env         = val_env
        self.eval_freq       = eval_freq
        self.patience        = patience
        self.min_improvement = min_improvement
        self.best_model_path = best_model_path
        self.verbose         = verbose
        self.use_calmar      = use_calmar   # True для bull: Calmar надёжнее SR на малых выборках
        self.best_score      = -np.inf      # универсальное поле (SR или Calmar)
        self.no_improve      = 0
        self._last_eval      = 0

    @staticmethod
    def _calmar(m: Dict) -> float:
        """
        Calmar ratio = CR / MPB.
        При MPB=0 (нет просадки — идеальный рост) возвращаем CR как есть.
        При CR<=0 и MPB>0 — отрицательный Calmar сигнализирует о потерях.
        Менее чувствителен к дисперсии дневных returns чем SR —
        надёжнее на малых выборках (30 шагов val у bull).
        """
        cr  = m["CR"]       # в процентах
        mpb = m["MPB"]      # в процентах, всегда >= 0
        if mpb < 1e-6:
            return cr       # нет просадки → Calmar = CR
        return cr / mpb

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval < self.eval_freq:
            return True
        self._last_eval = self.num_timesteps

        portfolio = evaluate_policy(self.model, self.val_env)
        tc = getattr(self.val_env, "trade_count", max(len(portfolio) - 1, 1))
        m  = compute_metrics(portfolio, self.val_env.initial_balance, tc)
        sr, cr = m["SR"], m["CR"]

        values     = np.array(portfolio, dtype=float)
        n_steps    = len(values) - 1
        n_active   = int(np.sum(np.abs(np.diff(values)) > 1e-6))
        pct_active = 100.0 * n_active / max(n_steps, 1)

        # nan = портфель плоский → пропускаем без штрафа no_improve
        if not np.isfinite(sr) and n_active == 0:
            if self.verbose:
                print(
                    f"\n[EarlyStopping] step={self.num_timesteps:,} | SR=nan "
                    f"(val: {n_steps} шагов, активных {n_active} = {pct_active:.1f}%) "
                    f"— пропускаем, no_improve не меняем ({self.no_improve}/{self.patience})"
                )
            return True

        # Выбираем метрику: Calmar для bull (малый val), SR для остальных
        if self.use_calmar:
            score        = self._calmar(m)
            metric_name  = "Calmar"
            metric_val   = score
        else:
            score        = sr if np.isfinite(sr) else -10.0
            metric_name  = "SR"
            metric_val   = sr

        self.logger.record("eval/SR",         sr if np.isfinite(sr) else -10.0)
        self.logger.record("eval/CR",         cr)
        self.logger.record("eval/MPB",        m["MPB"])
        self.logger.record("eval/active_pct", pct_active)
        if self.use_calmar:
            self.logger.record("eval/Calmar", score)
        self.logger.dump(self.num_timesteps)

        if self.verbose:
            print(
                f"\n[EarlyStopping] step={self.num_timesteps:,} | "
                f"SR={sr:.4f} CR={cr:.2f}% MPB={m['MPB']:.2f}% "
                f"{'Calmar=' + f'{score:.4f}' if self.use_calmar else ''} | "
                f"val: {n_steps} шагов активных={pct_active:.1f}% | "
                f"best {metric_name}={self.best_score:.4f} | "
                f"no_improve={self.no_improve}/{self.patience}"
            )

        if score > self.best_score + self.min_improvement:
            self.best_score = score
            self.no_improve = 0
            self.model.save(self.best_model_path)
            if self.verbose:
                print(f"[EarlyStopping] Новый лучший {metric_name}={score:.4f}"
                      f" → {self.best_model_path}")
        else:
            self.no_improve += 1
            if self.verbose:
                print(f"[EarlyStopping] Нет улучшения ({self.no_improve}/{self.patience})")

        if self.no_improve >= self.patience:
            if self.verbose:
                print(f"\n[EarlyStopping] Останавливаю. "
                      f"Лучший {metric_name}={self.best_score:.4f}")
            return False
        return True

    def _on_training_end(self):
        if self.verbose and self.best_sr > -np.inf:
            print(f"[EarlyStopping] Завершено. Лучший SR={self.best_sr:.4f}")


class StageCheckpointCallback(CheckpointCallback):
    def __init__(self, *, stage_name, run_dir, warmup_steps=0, **kwargs):
        super().__init__(**kwargs)
        self.stage_name    = stage_name
        self.run_dir       = run_dir
        self.warmup_steps  = warmup_steps
        self._warmup_done  = False

    def _on_step(self) -> bool:
        if self.n_calls <= self.warmup_steps:
            if not self._warmup_done and self.n_calls == self.warmup_steps:
                self._warmup_done = True
                print(f"\n[{self.stage_name}] Warmup done ({self.warmup_steps:,} steps)")
            return True
        ok = super()._on_step()
        if self.save_freq > 0 and self.n_calls % self.save_freq == 0:
            state = {
                "stage": self.stage_name, "status": "checkpointed",
                "num_timesteps": int(getattr(self.model, "num_timesteps", 0)),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            _write_resume_state(self.run_dir, state)
        return ok


def _build_stage_callbacks(
    *, model_name, stage_name, total_stage_steps,
    run_dir, checkpoint_dir, checkpoint_freq, log_every_steps,
    checkpoint_warmup_steps=0,
) -> CallbackList:
    callbacks = [TrainProgressCallback(
        model_name=model_name, stage_name=stage_name,
        total_stage_steps=total_stage_steps,
        run_dir=run_dir, log_every_steps=log_every_steps,
    )]
    if checkpoint_freq > 0:
        callbacks.append(StageCheckpointCallback(
            stage_name=stage_name, run_dir=run_dir,
            warmup_steps=checkpoint_warmup_steps,
            save_freq=checkpoint_freq, save_path=checkpoint_dir,
            name_prefix=f"{model_name}_{stage_name}",
        ))
    return CallbackList(callbacks)


# ══════════════════════════════════════════════════════════════════════════════
#  АУГМЕНТАЦИЯ
# ══════════════════════════════════════════════════════════════════════════════

def augment_jitter(
    close: np.ndarray, features: np.ndarray,
    noise_std: float = JITTER_STD,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Добавляет гауссовский шум к log-returns цен."""
    if rng is None:
        rng = np.random.default_rng()
    log_ret = np.diff(np.log(close + 1e-8), axis=0)
    noise   = rng.normal(0, noise_std, size=log_ret.shape)
    aug_close = np.empty_like(close)
    aug_close[0] = close[0]
    aug_close[1:] = close[0] * np.exp(np.cumsum(log_ret + noise, axis=0))
    return aug_close.astype(np.float32), features.copy()


def augment_scaling(
    close: np.ndarray, features: np.ndarray,
    scale_range: Tuple[float, float] = SCALE_RANGE,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Масштабирует амплитуду движений цен."""
    if rng is None:
        rng = np.random.default_rng()
    scale   = rng.uniform(*scale_range)
    log_ret = np.diff(np.log(close + 1e-8), axis=0)
    aug_close = np.empty_like(close)
    aug_close[0] = close[0]
    aug_close[1:] = close[0] * np.exp(np.cumsum(log_ret * scale, axis=0))
    return aug_close.astype(np.float32), features.copy()


def augment_trend(
    close: np.ndarray,
    features: np.ndarray,
    drift_range: Tuple[float, float] = (0.0002, 0.0008),
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Добавляет к log-returns небольшой положительный drift — имитирует
    более выраженный bull-тренд на основе реального эпизода.

    drift_range : (min, max) дневного дрейфа в долях.
        0.0002 ≈ +5%  годовых дополнительно
        0.0008 ≈ +20% годовых дополнительно

    Фичи не меняются — только цены. Это намеренно: модель учится торговать
    в условиях роста, используя те же технические индикаторы.
    Важно: drift применяется поверх реальных returns, сохраняя волатильность
    и внутридневную структуру движений оригинального эпизода.
    """
    if rng is None:
        rng = np.random.default_rng()

    drift   = rng.uniform(*drift_range)
    log_ret = np.diff(np.log(close + 1e-8), axis=0)  # (n-1, n_assets)

    # Добавляем равномерный drift ко всем активам
    aug_log_ret = log_ret + drift

    aug_close = np.empty_like(close)
    aug_close[0] = close[0]
    aug_close[1:] = close[0] * np.exp(np.cumsum(aug_log_ret, axis=0))

    return aug_close.astype(np.float32), features.copy()


def augment_episode(
    close: np.ndarray,
    features: np.ndarray,
    n_copies: int,
    rng: Optional[np.random.Generator] = None,
    regime_name: str = "",
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Создаёт n_copies версий эпизода.

    Базовая ротация (все режимы):
      копия 1       — оригинал
      чётные        — jitter (гауссовский шум к returns)
      нечётные      — scaling (масштаб амплитуды)
      каждая 4-я    — jitter + scaling вместе

    Для bull-режима каждая 3-я копия дополнительно получает trend-drift:
      копия 3, 6, 9 — augment_trend поверх базовой аугментации
      Это даёт модели больше примеров устойчивого роста на основе
      реальной структуры волатильности, не синтетических данных.
    """
    if rng is None:
        rng = np.random.default_rng()

    is_bull = (regime_name == "bull")
    result  = [(close.copy(), features.copy())]

    for i in range(n_copies - 1):
        # Базовая аугментация
        if i % 2 == 0:
            c, f = augment_jitter(close, features, rng=rng)
        else:
            c, f = augment_scaling(close, features, rng=rng)
        if i % 4 == 3:
            c, f = augment_jitter(c, f, rng=rng)

        # Для bull: каждая 3-я копия получает дополнительный положительный drift
        # Это имитирует более выраженный тренд роста, которого мало в реальных данных
        if is_bull and (i + 1) % 3 == 0:
            c, f = augment_trend(c, f, rng=rng)

        result.append((c, f))

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  РАЗБИВКА ДАННЫХ ПО РЕЖИМАМ
# ══════════════════════════════════════════════════════════════════════════════

def find_continuous_episodes(
    indices: np.ndarray,
    min_length: int = MIN_EPISODE_LENGTH,
    max_gap: int = 5,
) -> List[np.ndarray]:
    """Находит непрерывные отрезки из массива индексов."""
    if len(indices) == 0:
        return []
    indices  = np.sort(indices)
    episodes = []
    current  = [indices[0]]
    for i in range(1, len(indices)):
        if indices[i] - indices[i - 1] <= max_gap:
            current.append(indices[i])
        else:
            if len(current) >= min_length:
                episodes.append(np.array(current))
            current = [indices[i]]
    if len(current) >= min_length:
        episodes.append(np.array(current))
    return episodes


def split_regime_data(
    bundle,
    regime_label: int,
    val_ratio: float = 0.2,
    n_aug_copies: int = 4,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """
    Разбивает данные одного режима на train/val.

    Использует ВСЕ дни режима — и длинные и короткие эпизоды.
    Короткие эпизоды не выбрасываются: MultiEpisodeEnv корректно
    сбрасывает LSTM на границе каждого эпизода через terminated=True.

    Стратегия разбивки:
      - Последний непрерывный эпизод (≥ MIN_VAL_STEPS дней) → val
      - Все остальные эпизоды (любой длины ≥ window_size+1) → train
      - Если эпизодов нет совсем — делим весь массив индексов 80/20
    """
    if rng is None:
        rng = np.random.default_rng(42)

    regime_name = REGIME_NAMES[regime_label]
    window_size = 30   # должен совпадать с TrainConfig.window_size
    MIN_VAL_STEPS = 30  # минимум торгуемых шагов в val-эпизоде

    if bundle.regime_train is None:
        raise ValueError("HMM режимы не вычислены. Установи use_hmm=True в DataConfig.")

    # Все глобальные индексы дней этого режима в train-периоде
    global_train_idx  = np.where(bundle.train_mask)[0]
    local_regime_idx  = np.where(bundle.regime_train == regime_label)[0]
    global_regime_idx = global_train_idx[local_regime_idx]

    # Ищем непрерывные эпизоды любой длины ≥ window_size+1
    # (минимум чтобы был хоть один торгуемый шаг)
    MIN_EP = window_size + 1
    all_episodes = find_continuous_episodes(global_regime_idx, min_length=MIN_EP)

    total_days = len(global_regime_idx)
    days_in_episodes = sum(len(ep) for ep in all_episodes)
    days_skipped = total_days - days_in_episodes

    print(f"\n[{regime_name}] Всего дней режима в train: {total_days}")
    print(f"  Непрерывных эпизодов (≥{MIN_EP} дней): {len(all_episodes)}")
    print(f"  Дней в эпизодах: {days_in_episodes} | Выброшено (слишком короткие): {days_skipped}")

    # ── Выбор val-эпизода ────────────────────────────────────────────────────
    # ВАЖНО: val ВСЕГДА должен быть хронологически последним.
    # Иначе модель обучается на данных которые наступают после val — look-ahead bias.
    #
    # Стратегия:
    #   1. Берём последний эпизод (хронологически) как кандидат для val.
    #   2. Если он достаточно длинный (≥ min_val_length) — используем его целиком.
    #   3. Если слишком короткий — "склеиваем" с хвостом предпоследнего эпизода
    #      чтобы набрать нужную длину.
    #   4. Все эпизоды ПЕРЕД val-эпизодом идут в train.
    min_val_length = window_size + MIN_VAL_STEPS  # = 60 дней

    # Последний эпизод хронологически — всегда val-кандидат
    last_ep     = all_episodes[-1]
    last_ep_pos = len(all_episodes) - 1

    if len(last_ep) >= min_val_length:
        # Последний эпизод достаточно длинный — берём его целиком
        val_ep      = last_ep
        val_ep_pos  = last_ep_pos
        train_eps_raw = [ep for i, ep in enumerate(all_episodes) if i != val_ep_pos]
    else:
        # Последний эпизод короткий — добираем дни из предпоследнего
        needed = min_val_length - len(last_ep)
        if len(all_episodes) >= 2:
            donor     = all_episodes[-2]
            # Берём только хвост донора, оставляя ему минимум window+1 дней
            max_borrow = len(donor) - (window_size + 1)
            borrow     = min(needed, max(max_borrow, 0))
            if borrow > 0:
                val_ep = np.concatenate([donor[-borrow:], last_ep])
                # Донор обрезается: убираем заимствованный хвост
                trimmed_donor = donor[:-borrow]
                train_eps_raw = (
                    [ep for i, ep in enumerate(all_episodes) if i < last_ep_pos - 1]
                    + ([trimmed_donor] if len(trimmed_donor) > window_size else [])
                )
                print(f"  INFO: последний val-эпизод короткий ({len(last_ep)} дней) — "
                      f"добавлен хвост предыдущего ({borrow} дней), итого {len(val_ep)} дней")
            else:
                # Донор тоже слишком маленький — берём последний как есть
                val_ep        = last_ep
                train_eps_raw = [ep for i, ep in enumerate(all_episodes) if i != last_ep_pos]
                print(f"  WARN: val-эпизод {len(val_ep)} дней — мало, "
                      f"но донор тоже слишком короткий")
        else:
            # Только один эпизод — делим его 80/20
            split         = max(int(len(last_ep) * (1 - val_ratio)), window_size + 1)
            train_eps_raw = [last_ep[:split]] if split > window_size else []
            val_ep        = last_ep[split:]
            print(f"  WARN: один эпизод — делим {split}/{len(val_ep)} дней (train/val)")

    val_ep_pos = last_ep_pos  # всегда последний

    # Выводим статистику эпизодов
    print(f"\n  Эпизоды (хронологически):")
    for i, ep in enumerate(all_episodes):
        if i == last_ep_pos:
            role = "→ VAL (последний хронологически)"
        else:
            role = "→ train"
        steps = max(len(ep) - window_size, 0)
        print(f"    [{i+1:2d}] {len(ep):4d} дней ({steps:3d} торгуемых) {role}")

    # ── Аугментация train-эпизодов ───────────────────────────────────────────
    train_episodes = []
    for ep_idx in train_eps_raw:
        ep_close    = bundle.close_all[ep_idx]
        ep_features = bundle.features_all[ep_idx]
        ep_turb     = bundle.turb_aligned.values[ep_idx].astype(np.float32)
        ep_regime   = bundle.regime_all[ep_idx] if bundle.regime_all is not None else None
        ep_rvi      = bundle.rvi_all[ep_idx]    if bundle.rvi_all    is not None else None

        # Аугментируем только если эпизод достаточно длинный (≥ 2×window)
        # Короткие эпизоды берём как есть — аугментация на них нестабильна
        if len(ep_idx) >= 2 * window_size:
            aug_pairs = augment_episode(
                ep_close, ep_features,
                n_copies=n_aug_copies,
                rng=rng,
                regime_name=regime_name,   # bull получает trend-аугментацию
            )
        else:
            aug_pairs = [(ep_close.copy(), ep_features.copy())]

        for aug_close, aug_features in aug_pairs:
            train_episodes.append({
                "close":   aug_close,
                "features": aug_features,
                "turb":    ep_turb,
                "regime":  ep_regime,
                "rvi":     ep_rvi,
            })

    # Val данные
    val_close    = bundle.close_all[val_ep]
    val_features = bundle.features_all[val_ep]
    val_turb     = bundle.turb_aligned.values[val_ep].astype(np.float32)
    val_regime_  = bundle.regime_all[val_ep] if bundle.regime_all is not None else None
    val_rvi_     = bundle.rvi_all[val_ep]    if bundle.rvi_all    is not None else None

    n_train    = sum(len(ep["close"]) for ep in train_episodes)
    n_val      = len(val_ep)
    val_steps  = max(n_val - window_size, 0)

    print(f"\n  Train: {len(train_episodes)} эпизодов ({n_train} дней после аугментации ×{n_aug_copies})")
    print(f"  Val:   {n_val} дней → {val_steps} торгуемых шагов"
          + (" ⚠ МАЛО" if val_steps < MIN_VAL_STEPS else " ✓"))

    return {
        "regime_label":   regime_label,
        "regime_name":    regime_name,
        "train_episodes": train_episodes,
        "val_close":      val_close,
        "val_features":   val_features,
        "val_turb":       val_turb,
        "val_regime":     val_regime_,
        "val_rvi":        val_rvi_,
        "n_train_days":   n_train,
        "n_val_days":     n_val,
    }


def build_concatenated_env(
    episodes: List[Dict],
    bundle,
    env_cfg: EnvFactoryConfig,
    spec,
    turbulence_threshold: Optional[float] = None,
) -> DummyVecEnv:
    """
    Создаёт DummyVecEnv из списка эпизодов через MultiEpisodeEnv.

    MultiEpisodeEnv при каждом reset() переключается на следующий эпизод
    (в случайном порядке) и возвращает terminated=True в конце каждого.
    RecurrentPPO получает правильный сигнал сбросить LSTM на границах
    несмежных временных отрезков — данные не смешиваются.

    Все эпизоды (включая короткие) попадают в обучение — ничего не теряется.
    """
    from env import MOEXTradingEnvV2, MultiEpisodeEnv

    threshold = turbulence_threshold if turbulence_threshold is not None else bundle.threshold

    # Общие kwargs для каждой внутренней среды.
    # close_prices / features / turbulence / regime / rvi
    # подставляются в MultiEpisodeEnv._make_env() для каждого эпизода отдельно.
    base_kwargs: Dict = dict(
        turbulence_threshold = threshold,
        initial_balance      = env_cfg.initial_balance,
        window_size          = env_cfg.window_size,
        commission           = env_cfg.commission,
        max_shares_per_trade = env_cfg.max_shares_per_trade,
        margin_ratio         = env_cfg.margin_ratio,
        atr_indices          = bundle.atr_indices,
    )

    total_days = sum(len(ep["close"]) for ep in episodes)
    print(f"  MultiEpisodeEnv: {len(episodes)} эпизодов, "
          f"{total_days} дней суммарно, threshold={threshold:.2f}")

    def _factory():
        return MultiEpisodeEnv(
            episodes        = episodes,
            base_env_cls    = MOEXTradingEnvV2,
            base_env_kwargs = base_kwargs,
            shuffle         = True,
        )

    return DummyVecEnv([_factory])


# ══════════════════════════════════════════════════════════════════════════════
#  ОНЛАЙН ДЕТЕКТОР РЕЖИМА
# ══════════════════════════════════════════════════════════════════════════════

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def detect_regime_weights(
    recent_close: np.ndarray,
    lookback: int             = DETECTOR_LOOKBACK,
    vol_bear_thr: float       = VOL_BEAR_THRESHOLD,
    trend_bull_thr: float     = TREND_BULL_THRESHOLD,
    sharpness: float          = DETECTOR_SHARPNESS,
) -> Dict[str, float]:
    """
    Вычисляет мягкие веса bull/sideways/bear по последним `lookback` дням.

    recent_close : (lookback, n_assets)
    Возвращает   : {"bull": w, "sideways": w, "bear": w}, сумма = 1.0

    MIN_W = 0.01: "чужие" модели почти отключаются в явные периоды.
    Это снижает количество противоречивых сделок и лишних комиссий.
    """
    prices  = recent_close[-lookback:]
    log_ret = np.diff(np.log(prices + 1e-8), axis=0)

    # Annualized realized vol (среднее по активам)
    vol = float(log_ret.std(axis=0).mean() * np.sqrt(252))

    # Суммарный тренд за период (среднее по активам)
    trend = float(((prices[-1] / (prices[0] + 1e-8)) - 1).mean())

    # Просадка от максимума за период
    idx      = prices.mean(axis=1)
    drawdown = float((idx.max() - idx[-1]) / (idx.max() + 1e-8))

    # Bear: высокая волатильность
    w_bear = _sigmoid(sharpness * (vol - vol_bear_thr))

    # Bull: положительный тренд, ослабляется при просадке
    w_bull = _sigmoid(sharpness * (trend - trend_bull_thr)) * (1.0 - drawdown)
    w_bull *= (1.0 - w_bear)

    w_side = max(1.0 - w_bear - w_bull, 0.0)

    # Минимальный вес 0.01 — почти отключает "чужие" модели в явные периоды.
    # 0.05 создавало слишком много противоречивых сделок между моделями:
    # в явный bear-день bull с весом 5% тянул в лонг → лишние комиссии → SR уходил в минус.
    MIN_W = 0.01
    w_bull = max(w_bull, MIN_W)
    w_side = max(w_side, MIN_W)
    w_bear = max(w_bear, MIN_W)

    total = w_bull + w_side + w_bear
    return {
        "bull":     w_bull / total,
        "sideways": w_side / total,
        "bear":     w_bear / total,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  РЕЖИМНЫЙ АНСАМБЛЬ (ИНФЕРЕНС)
# ══════════════════════════════════════════════════════════════════════════════

class RegimeEnsemblePolicy:
    """
    Ансамбль трёх RecurrentPPO моделей с онлайн детектором режима.

    Для каждого шага:
      1. Детектор смотрит на последние 20 дней цен
      2. Возвращает мягкие веса bull/sideways/bear
      3. Каждая модель выдаёт своё действие
      4. Финальное действие = взвешенная сумма
    """

    def __init__(
        self,
        model_paths: Dict[str, str],
        device: str           = "cpu",
        lookback: int         = DETECTOR_LOOKBACK,
        vol_bear_thr: float   = VOL_BEAR_THRESHOLD,
        trend_bull_thr: float = TREND_BULL_THRESHOLD,
        sharpness: float      = DETECTOR_SHARPNESS,
    ):
        print("[RegimeEnsemble] Загружаем модели...")
        self.models: Dict[str, RecurrentPPO] = {}
        for regime, path in model_paths.items():
            print(f"  {regime}: {path}")
            self.models[regime] = RecurrentPPO.load(path, device=device)

        self.lookback       = lookback
        self.vol_bear_thr   = vol_bear_thr
        self.trend_bull_thr = trend_bull_thr
        self.sharpness      = sharpness

        self.lstm_states: Dict[str, Optional[tuple]] = {r: None for r in REGIME_NAMES}
        self.episode_starts: Dict[str, np.ndarray]   = {
            r: np.ones((1,), dtype=bool) for r in REGIME_NAMES
        }
        self._price_buffer: List[np.ndarray] = []

    def reset(self) -> None:
        self.lstm_states    = {r: None for r in REGIME_NAMES}
        self.episode_starts = {r: np.ones((1,), dtype=bool) for r in REGIME_NAMES}
        self._price_buffer  = []

    def predict(
        self,
        obs: np.ndarray,
        current_close: np.ndarray,
        verbose_weights: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        obs           : (window_size, n_features)
        current_close : (n_assets,) — текущие цены для детектора
        """
        self._price_buffer.append(current_close.copy())
        if len(self._price_buffer) > self.lookback + 5:
            self._price_buffer.pop(0)

        if len(self._price_buffer) < self.lookback:
            weights = {"bull": 1/3, "sideways": 1/3, "bear": 1/3}
        else:
            recent  = np.stack(self._price_buffer[-self.lookback:], axis=0)
            weights = detect_regime_weights(
                recent,
                lookback=self.lookback,
                vol_bear_thr=self.vol_bear_thr,
                trend_bull_thr=self.trend_bull_thr,
                sharpness=self.sharpness,
            )

        if verbose_weights:
            print(f"  Режим: bull={weights['bull']:.2f} "
                  f"side={weights['sideways']:.2f} "
                  f"bear={weights['bear']:.2f}")

        obs_batch = obs[np.newaxis, ...]
        n_assets  = current_close.shape[0]
        action    = np.zeros(n_assets, dtype=np.float32)

        for regime_name, model in self.models.items():
            act_batch, new_state = model.predict(
                obs_batch,
                state=self.lstm_states[regime_name],
                episode_start=self.episode_starts[regime_name],
                deterministic=True,
            )
            self.lstm_states[regime_name]    = new_state
            self.episode_starts[regime_name] = np.zeros((1,), dtype=bool)
            action += weights[regime_name] * act_batch[0]

        return np.clip(action, -1.0, 1.0).astype(np.float32), weights

    def evaluate(self, env, verbose: bool = False) -> list:
        """Прогоняет ансамбль на невекторизованном env."""
        obs, _ = env.reset()
        self.reset()
        done  = False
        step  = 0

        # Диагностика — собираем по дням
        diag_weights: List[Dict]  = []
        diag_actions: List[float] = []   # среднее abs действие ансамбля
        diag_trades:  List[int]   = []   # сколько сделок на шаге

        prev_shares = np.zeros(env.n_assets, dtype=np.float32)

        while not done:
            idx           = min(env.current_step, len(env.close_prices) - 1)
            current_close = env.close_prices[idx]
            action, weights = self.predict(obs, current_close,
                                           verbose_weights=verbose)

            # Собираем диагностику до step()
            diag_weights.append(weights.copy())
            diag_actions.append(float(np.abs(action).mean()))

            obs, _, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

            # Считаем реальные сделки по изменению позиций
            cur_shares = getattr(env, "shares_held", prev_shares)
            n_trades   = int(np.any(np.abs(cur_shares - prev_shares) > 1e-6))
            diag_trades.append(n_trades)
            prev_shares = cur_shares.copy()
            step += 1

        # ── Итоговый диагностический отчёт ───────────────────────────────────
        if diag_weights:
            bull_w = np.mean([w["bull"]     for w in diag_weights])
            side_w = np.mean([w["sideways"] for w in diag_weights])
            bear_w = np.mean([w["bear"]     for w in diag_weights])
            mean_action   = float(np.mean(diag_actions))
            total_trades  = int(np.sum(diag_trades))
            trade_freq    = total_trades / max(step, 1)

            print(f"\n[RegimeEnsemble] Диагностика теста ({step} шагов):")
            print(f"  Средние веса детектора: "
                  f"bull={bull_w:.3f}  sideways={side_w:.3f}  bear={bear_w:.3f}")
            print(f"  Среднее |действие|: {mean_action:.4f}")
            print(f"  Сделок: {total_trades} ({trade_freq*100:.1f}% дней)")

            # Предупреждение если торгует слишком часто
            if trade_freq > 0.5:
                print(f"  ⚠ ВЫСОКАЯ ЧАСТОТА ТОРГОВЛИ: {trade_freq*100:.1f}% дней "
                      f"→ комиссии съедают прибыль (комиссия 0.05% за сделку)")
            if mean_action < 0.05:
                print(f"  ⚠ СЛАБЫЕ ДЕЙСТВИЯ: среднее |action|={mean_action:.4f} "
                      f"→ модели почти не торгуют, возможна недообученность")

        return env.portfolio_history


# ══════════════════════════════════════════════════════════════════════════════
#  ОБУЧЕНИЕ ОДНОЙ РЕЖИМНОЙ МОДЕЛИ
# ══════════════════════════════════════════════════════════════════════════════

def train_regime_model(
    regime_name: str,
    regime_data: Dict,
    bundle,
    spec,
    env_cfg: EnvFactoryConfig,
    train_cfg: TrainConfig,
    base_ppo: Dict,
    save_dir: str,
    device,
) -> str:
    """
    Обучает одну модель на данных конкретного режима.
    Возвращает путь к лучшей модели (по val SR через EarlyStopping).
    """
    model_name     = f"regime_{regime_name}"
    run_dir        = os.path.join(save_dir, model_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Обучаем: {model_name}")
    print(f"  Train:   {regime_data['n_train_days']} дней (с аугментацией)")
    print(f"  Val:     {regime_data['n_val_days']} дней")
    print(f"{'='*60}")

    # ── Раздельный turbulence threshold по режимам ────────────────────────────
    # bull:     стандартный — защита от кризисов внутри bull-периода
    # sideways: ×2 — боковик бывает волатильным, не мешаем торговать
    # bear:     ×5 — медвежья модель учится торговать в волатильность
    _turb_multipliers = {"bull": 1.0, "sideways": 2.0, "bear": 5.0}
    regime_threshold  = bundle.threshold * _turb_multipliers.get(regime_name, 1.0)
    print(f"  Turbulence threshold: {regime_threshold:.2f} "
          f"(базовый {bundle.threshold:.2f} × {_turb_multipliers.get(regime_name, 1.0):.1f} "
          f"для '{regime_name}')")

    # Строим среды с режимным порогом
    train_env = build_concatenated_env(
        regime_data["train_episodes"], bundle, env_cfg, spec,
        turbulence_threshold=regime_threshold,
    )
    val_env = build_env(
        close_prices=regime_data["val_close"],
        features=regime_data["val_features"],
        turbulence=regime_data["val_turb"],
        turbulence_threshold=regime_threshold,
        short=spec.short, v2=True,
        atr_indices=bundle.atr_indices,
        regime=regime_data["val_regime"],
        rvi=regime_data["val_rvi"],
        cfg=env_cfg,
    )

    # Stage1 — calm данные (базовые паттерны рынка)
    calm_regime = bundle.regime_all[bundle.calm_mask] if bundle.regime_all is not None else None
    calm_rvi    = bundle.rvi_all[bundle.calm_mask]    if bundle.rvi_all    is not None else None
    stage1_env  = make_vec_env(
        bundle.close_all[bundle.calm_mask],
        bundle.features_all[bundle.calm_mask],
        bundle.turb_aligned[bundle.calm_mask].values,
        bundle.threshold, env_cfg,
        short=spec.short, v2=True,
        atr_indices=bundle.atr_indices,
        regime=calm_regime, rvi=calm_rvi,
    )

    # Resume
    resume_state = _read_resume_state(run_dir)
    model: Optional[RecurrentPPO] = None
    current_stage = "stage1"

    latest_s2 = _find_latest_checkpoint(checkpoint_dir, f"{model_name}_stage2")
    latest_s1 = _find_latest_checkpoint(checkpoint_dir, f"{model_name}_stage1")

    if latest_s2:
        print(f"[{model_name}] resume stage2: {latest_s2}")
        model = RecurrentPPO.load(latest_s2, env=train_env, device=device)
        current_stage = "stage2"
    elif latest_s1:
        print(f"[{model_name}] resume stage1: {latest_s1}")
        model = RecurrentPPO.load(latest_s1, env=stage1_env, device=device)
        current_stage = str(resume_state.get("stage", "stage1"))

    if model is None:
        model = RecurrentPPO(env=stage1_env, **base_ppo)
        current_stage = "stage1"

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    if current_stage == "stage1":
        done  = int(getattr(model, "num_timesteps", 0))
        remaining = max(train_cfg.stage1_steps - done, 0)
        if remaining > 0:
            print(f"[{model_name}] Stage1: {remaining:,} шагов на calm данных...")
            model.learn(
                total_timesteps=remaining,
                progress_bar=True,
                callback=_build_stage_callbacks(
                    model_name=model_name, stage_name="stage1",
                    total_stage_steps=train_cfg.stage1_steps,
                    run_dir=run_dir, checkpoint_dir=checkpoint_dir,
                    checkpoint_freq=train_cfg.checkpoint_freq,
                    log_every_steps=train_cfg.log_every_steps,
                ),
                reset_num_timesteps=False,
                tb_log_name=f"{model_name}_stage1",
            )
        else:
            print(f"[{model_name}] Stage1 уже завершён.")
        model.save(os.path.join(run_dir, f"{model_name}_stage1_final"))
        model.set_env(train_env)
        current_stage = "stage2"

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    if current_stage == "stage2":
        resume_s2 = int(resume_state.get("stage2_elapsed_steps", 0))
        global_ts = int(getattr(model, "num_timesteps", 0))
        s2_done   = resume_s2 if (resume_state.get("stage") == "stage2" and resume_s2 > 0) \
                    else max(global_ts - train_cfg.stage1_steps, 0)
        remaining = max(train_cfg.stage2_steps - s2_done, 0)

        print(f"[{model_name}] Stage2: {remaining:,} шагов на данных режима '{regime_name}'...")

        best_path = os.path.join(run_dir, f"{model_name}_best")

        if remaining > 0:
            # Bull: Calmar надёжнее SR на малых val-выборках (30 шагов)
            # Sideways/Bear: SR стандартная метрика
            use_calmar_flag = (regime_name == "bull")
            patience_by_regime = {"bull": 80, "sideways": 40, "bear": 40}
            early_stopping = EarlyStoppingCallback(
                val_env=val_env,
                eval_freq=5_000,
                patience=patience_by_regime.get(regime_name, 40),
                min_improvement=0.03,
                best_model_path=best_path,
                verbose=True,
                use_calmar=use_calmar_flag,
            )
            stage2_cb = _build_stage_callbacks(
                model_name=model_name, stage_name="stage2",
                total_stage_steps=train_cfg.stage2_steps,
                run_dir=run_dir, checkpoint_dir=checkpoint_dir,
                checkpoint_freq=train_cfg.checkpoint_freq,
                log_every_steps=train_cfg.log_every_steps,
                checkpoint_warmup_steps=30_000,
            )
            model.learn(
                total_timesteps=remaining,
                progress_bar=True,
                callback=CallbackList([stage2_cb, early_stopping]),
                reset_num_timesteps=False,
                tb_log_name=f"{model_name}_stage2",
            )
        else:
            print(f"[{model_name}] Stage2 уже завершён.")

    final_path = os.path.join(run_dir, f"{model_name}_final")
    model.save(final_path)

    best_path = os.path.join(run_dir, f"{model_name}_best")
    if os.path.exists(best_path + ".zip"):
        print(f"[{model_name}] Используем best модель: {best_path}")
        return best_path
    print(f"[{model_name}] Используем final модель: {final_path}")
    return final_path


# ══════════════════════════════════════════════════════════════════════════════
#  ГЛАВНАЯ ФУНКЦИЯ
# ══════════════════════════════════════════════════════════════════════════════

def train_regime_ensemble(
    model_name: str              = "xlstm_large",
    data_cfg: Optional[DataConfig] = None,
    train_cfg: Optional[TrainConfig] = None,
    aug_copies: Optional[Dict[str, int]] = None,
    run_test: bool               = True,
    save_dir: str                = "./models/regime_ensemble",
) -> Dict:
    """
    Обучает три модели (bull / sideways / bear) и оценивает ансамбль на тесте.

    aug_copies : {"bull": N, "sideways": N, "bear": N}
                 Сколько аугментированных копий делать для каждого режима.
    """
    data_cfg   = data_cfg   or DataConfig(**{**vars(DEFAULT_CONFIG), "use_hmm": True})
    train_cfg  = train_cfg  or TrainConfig()
    aug_copies = aug_copies or {"bull": 5, "sideways": 2, "bear": 2}

    set_global_seed(train_cfg.seed)
    os.makedirs(save_dir, exist_ok=True)

    device    = get_device()
    use_slstm = check_slstm_available()
    registry  = get_model_registry(use_slstm)

    if model_name not in registry:
        raise ValueError(f"Неизвестная модель: {model_name}. Доступны: {list(registry)}")
    spec = registry[model_name]

    env_cfg = EnvFactoryConfig(
        window_size=train_cfg.window_size,
        margin_ratio=train_cfg.margin_ratio,
        initial_balance=train_cfg.initial_balance,
        commission=train_cfg.commission,
        max_shares_per_trade=train_cfg.max_shares_per_trade,
    )

    print("\n[RegimeEnsemble] Загружаем данные (use_hmm=True)...")
    bundle = prepare_datasets(data_cfg)

    if bundle.regime_train is None:
        raise ValueError(
            "HMM режимы не вычислены. Убедись что use_hmm=True в DataConfig."
        )

    print("\n[RegimeEnsemble] Распределение режимов в train:")
    for i, name in enumerate(REGIME_NAMES):
        count = (bundle.regime_train == i).sum()
        pct   = count / len(bundle.regime_train) * 100
        print(f"  {name}: {count} дней ({pct:.1f}%)")

    # Smoke test
    obs_space = build_env(
        close_prices=bundle.close_test,
        features=bundle.features_test,
        turbulence=bundle.turb_test,
        turbulence_threshold=bundle.threshold,
        short=spec.short, v2=True,
        atr_indices=bundle.atr_indices,
        regime=bundle.regime_test,
        rvi=bundle.rvi_test,
        cfg=env_cfg,
    ).observation_space
    smoke = smoke_test_encoder(spec.extractor_class, spec.extractor_kwargs, obs_space, device=device)
    print(f"\n[RegimeEnsemble] Encoder smoke test: {smoke}")

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
        target_kl=train_cfg.target_kl,
        vf_coef=train_cfg.vf_coef,
        max_grad_norm=train_cfg.max_grad_norm,
        verbose=train_cfg.verbose,
        seed=train_cfg.seed,
        tensorboard_log=train_cfg.tensorboard_log,
        device=device,
        policy_kwargs=policy_kwargs,
    )

    rng = np.random.default_rng(train_cfg.seed)

    # ── Обучаем три модели ────────────────────────────────────────────────────
    model_paths: Dict[str, str] = {}
    for regime_idx, regime_name in enumerate(REGIME_NAMES):
        n_copies    = aug_copies.get(regime_name, 2)
        regime_data = split_regime_data(
            bundle=bundle, regime_label=regime_idx,
            val_ratio=0.2, n_aug_copies=n_copies, rng=rng,
        )
        best_path = train_regime_model(
            regime_name=regime_name, regime_data=regime_data,
            bundle=bundle, spec=spec, env_cfg=env_cfg,
            train_cfg=train_cfg, base_ppo=base_ppo,
            save_dir=save_dir, device=device,
        )
        model_paths[regime_name] = best_path

    # Сохраняем пути
    paths_file = os.path.join(save_dir, "regime_model_paths.json")
    with open(paths_file, "w", encoding="utf-8") as f:
        json.dump(model_paths, f, ensure_ascii=False, indent=2)
    print(f"\n[RegimeEnsemble] Пути к моделям: {paths_file}")

    results: Dict = {"model_paths": model_paths, "save_dir": save_dir}

    # ── Оценка на тесте ───────────────────────────────────────────────────────
    if run_test:
        print(f"\n{'='*65}")
        print("  ИТОГОВЫЙ ОТЧЁТ")
        print(f"{'='*65}")

        ensemble = RegimeEnsemblePolicy(model_paths=model_paths, device=str(device))
        test_env = build_env(
            close_prices=bundle.close_test,
            features=bundle.features_test,
            turbulence=bundle.turb_test,
            turbulence_threshold=bundle.threshold,
            short=spec.short, v2=True,
            atr_indices=bundle.atr_indices,
            regime=bundle.regime_test,
            rvi=bundle.rvi_test,
            cfg=env_cfg,
        )

        portfolio = ensemble.evaluate(test_env)
        tc        = getattr(test_env, "trade_count", max(len(portfolio) - 1, 1))
        metrics   = compute_metrics(portfolio, test_env.initial_balance, tc)

        print(f"\n  [REGIME ENSEMBLE] {model_name} × 3 режима")
        print(f"    TEST → SR={metrics['SR']:+.3f}  "
              f"CR={metrics['CR']:+.1f}%  "
              f"MPB={metrics['MPB']:.1f}%  "
              f"WinRate={metrics['WinRate']:.2f}  "
              f"Trades={metrics['Trades']}")

        baseline = compute_baseline_metrics(
            bundle.close_test, bundle.turb_test,
            bundle.threshold, env_cfg.initial_balance,
        )
        bh = baseline.get("buy_and_hold", {})
        ew = baseline.get("equal_weight", {})
        print(f"\n  [buy_and_hold] SR={bh.get('SR', 0):+.3f}  "
              f"CR={bh.get('CR', 0):+.1f}%  MPB={bh.get('MPB', 0):.1f}%")
        print(f"  [equal_weight] SR={ew.get('SR', 0):+.3f}  "
              f"CR={ew.get('CR', 0):+.1f}%  MPB={ew.get('MPB', 0):.1f}%")
        print(f"\n{'='*65}\n")

        report = {
            "model_name":    model_name,
            "model_paths":   model_paths,
            "ensemble_test": metrics,
            "baseline":      baseline,
            "aug_copies":    aug_copies,
            "detector": {
                "lookback":        DETECTOR_LOOKBACK,
                "vol_bear_thr":    VOL_BEAR_THRESHOLD,
                "trend_bull_thr":  TREND_BULL_THRESHOLD,
                "sharpness":       DETECTOR_SHARPNESS,
            },
        }
        report_path = os.path.join(save_dir, "regime_ensemble_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[RegimeEnsemble] Отчёт: {report_path}")

        results["metrics"]   = metrics
        results["baseline"]  = baseline
        results["portfolio"] = portfolio

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  ОЦЕНКА УЖЕ ОБУЧЕННОГО АНСАМБЛЯ
# ══════════════════════════════════════════════════════════════════════════════

def eval_only(
    ensemble_dir: str,
    model_name: str                = "xlstm_large",
    data_cfg: Optional[DataConfig] = None,
) -> None:
    """Оценивает уже обученный ансамбль без переобучения."""
    paths_file = os.path.join(ensemble_dir, "regime_model_paths.json")
    if not os.path.exists(paths_file):
        raise FileNotFoundError(f"Не найден файл: {paths_file}")
    with open(paths_file, "r", encoding="utf-8") as f:
        model_paths = json.load(f)

    data_cfg = data_cfg or DataConfig(**{**vars(DEFAULT_CONFIG), "use_hmm": True})
    bundle   = prepare_datasets(data_cfg)
    device   = get_device()
    registry = get_model_registry(check_slstm_available())
    spec     = registry[model_name]
    env_cfg  = EnvFactoryConfig()

    ensemble = RegimeEnsemblePolicy(model_paths=model_paths, device=str(device))
    test_env = build_env(
        close_prices=bundle.close_test,
        features=bundle.features_test,
        turbulence=bundle.turb_test,
        turbulence_threshold=bundle.threshold,
        short=spec.short, v2=True,
        atr_indices=bundle.atr_indices,
        regime=bundle.regime_test,
        rvi=bundle.rvi_test,
        cfg=env_cfg,
    )
    print("\n[eval_only] Прогоняем ансамбль на тесте...")
    portfolio = ensemble.evaluate(test_env, verbose=True)
    tc = getattr(test_env, "trade_count", max(len(portfolio) - 1, 1))
    metrics = compute_metrics(portfolio, test_env.initial_balance, tc)
    print(f"\n  TEST → SR={metrics['SR']:+.3f}  "
          f"CR={metrics['CR']:+.1f}%  "
          f"MPB={metrics['MPB']:.1f}%  "
          f"Trades={metrics['Trades']}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def cli() -> None:
    parser = argparse.ArgumentParser(description="Regime Ensemble Training")
    parser.add_argument(
        "--model", default="xlstm_large",
        choices=["lstm", "xlstm_base", "xlstm_attn", "xlstm_large"],
    )
    parser.add_argument("--stage1-steps",     type=int, default=150_000)
    parser.add_argument("--stage2-steps",     type=int, default=300_000)
    parser.add_argument("--save-dir",         default="./models/regime_ensemble")
    parser.add_argument("--tensorboard-log",  default="./tb_logs")
    parser.add_argument("--seed",             type=int, default=42)
    parser.add_argument("--aug-bull",         type=int, default=5)
    parser.add_argument("--aug-sideways",     type=int, default=2)
    parser.add_argument("--aug-bear",         type=int, default=2)
    parser.add_argument("--eval-only",        action="store_true")
    parser.add_argument("--ensemble-dir",     default=None)
    parser.add_argument("--no-test",          action="store_true")
    args = parser.parse_args()

    if args.eval_only:
        eval_only(
            ensemble_dir=args.ensemble_dir or args.save_dir,
            model_name=args.model,
        )
        return

    data_cfg = DataConfig(**{**vars(DEFAULT_CONFIG), "use_hmm": True})
    train_cfg = TrainConfig(
        seed=args.seed,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        save_dir=args.save_dir,
        tensorboard_log=args.tensorboard_log,
    )
    aug_copies = {
        "bull":     args.aug_bull,
        "sideways": args.aug_sideways,
        "bear":     args.aug_bear,
    }
    results = train_regime_ensemble(
        model_name=args.model,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        aug_copies=aug_copies,
        run_test=not args.no_test,
        save_dir=args.save_dir,
    )
    if results.get("metrics"):
        print(json.dumps(results["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    cli()
