from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

EPS = 1e-8


def _as_float_2d(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={arr.shape}")
    return arr


def _as_float_1d(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={arr.shape}")
    return arr


@dataclass
class EnvFactoryConfig:
    """Risk and execution settings for the trading environment.

    Action is a signed order-intensity vector, not target portfolio weights.
    Positive values buy, negative values sell/short, near-zero values hold.
    """

    window_size: int = 30
    initial_balance: float = 1_000_000.0
    commission: float = 0.0005
    max_shares_per_trade: int = 100  # kept for compatibility
    allow_short: bool = False
    margin_ratio: float = 0.25
    max_position_pct: float = 0.25
    max_gross_leverage: float = 1.00
    max_trade_notional_pct: float = 0.04
    # Anti-collapse: do not zero small PPO actions before notional sizing.
    # Micro-orders are still filtered by min_order_notional_pct.
    min_trade_action: float = 0.0
    min_order_notional_pct: float = 0.0025
    # Anti-collapse: cooldown can trap policy near zero-action; default off.
    rebalance_cooldown: int = 0
    force_liquidate_on_turbulence: bool = True
    reward_scale: float = 100.0
    # "upside_opportunity" penalizes staying under-exposed only when the
    # equal-weight market benchmark rises. It does not reward random trading.
    reward_mode: str = "upside_opportunity"
    benchmark_coef: float = 0.35
    opportunity_coef: float = 1.0
    target_exposure: float = 0.30
    low_exposure_threshold: float = 0.03
    cash_penalty_grace_steps: int = 5
    turnover_penalty_coef: float = 0.0
    downside_penalty_coef: float = 0.0
    liquidation_penalty: float = 0.0
    atr_target: float = 0.025
    atr_order_scale_min: float = 0.35
    atr_order_scale_max: float = 1.25


class MOEXTradingEnv(gym.Env):
    """Causal multi-asset RL trading environment.

    Time convention:
      * current_step is execution day t;
      * observation contains technical features for [t-window_size, ..., t-1];
      * account/regime/RVI/turbulence state uses t-1 only;
      * action is a signed order delta executed at close_prices[t];
      * reward is based on account equity marked at close_prices[t+1].

    This is trading-RL: actions are orders that change shares held. They are not
    target portfolio weights. Risk limits may clamp orders internally.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        close_prices: np.ndarray,
        features: np.ndarray,
        turbulence: np.ndarray,
        turbulence_threshold: float,
        initial_balance: float = 1_000_000.0,
        window_size: int = 30,
        commission: float = 0.0005,
        max_shares_per_trade: int = 100,
        allow_short: bool = False,
        margin_ratio: float = 0.25,
        max_position_pct: float = 0.25,
        max_gross_leverage: float = 1.0,
        max_trade_notional_pct: float = 0.04,
        min_trade_action: float = 0.0,
        min_order_notional_pct: float = 0.0025,
        rebalance_cooldown: int = 0,
        force_liquidate_on_turbulence: bool = True,
        reward_scale: float = 100.0,
        reward_mode: str = "upside_opportunity",
        benchmark_coef: float = 0.35,
        opportunity_coef: float = 1.0,
        target_exposure: float = 0.30,
        low_exposure_threshold: float = 0.03,
        cash_penalty_grace_steps: int = 5,
        turnover_penalty_coef: float = 0.0,
        downside_penalty_coef: float = 0.0,
        liquidation_penalty: float = 0.0,
        atr_values: Optional[np.ndarray] = None,
        atr_target: float = 0.025,
        atr_order_scale_min: float = 0.35,
        atr_order_scale_max: float = 1.25,
    ):
        super().__init__()
        self.close_prices = _as_float_2d(close_prices, "close_prices")
        self.features = _as_float_2d(features, "features")
        self.turbulence = _as_float_1d(turbulence, "turbulence")
        if len(self.close_prices) != len(self.features) or len(self.close_prices) != len(self.turbulence):
            raise ValueError(
                "close_prices, features and turbulence must have same length: "
                f"{len(self.close_prices)}, {len(self.features)}, {len(self.turbulence)}"
            )
        if window_size < 2:
            raise ValueError("window_size must be >= 2")
        if len(self.close_prices) <= window_size + 2:
            raise ValueError(
                f"Not enough rows: rows={len(self.close_prices)}, window_size={window_size}. "
                "Need at least window_size + 3 rows."
            )

        self.atr_values = None
        if atr_values is not None:
            atr_arr = _as_float_2d(atr_values, "atr_values")
            if atr_arr.shape != self.close_prices.shape:
                raise ValueError(f"atr_values shape must equal close_prices shape, got {atr_arr.shape} vs {self.close_prices.shape}")
            self.atr_values = atr_arr

        self.turbulence_threshold = float(turbulence_threshold)
        self.initial_balance = float(initial_balance)
        self.window_size = int(window_size)
        self.commission = float(commission)
        self.max_shares = int(max_shares_per_trade)
        self.allow_short = bool(allow_short)
        self.margin_ratio = float(margin_ratio)
        self.max_position_pct = float(max_position_pct)
        self.max_gross_leverage = float(max_gross_leverage)
        self.max_trade_notional_pct = float(max_trade_notional_pct)
        self.min_trade_action = float(min_trade_action)
        self.min_order_notional_pct = float(min_order_notional_pct)
        self.rebalance_cooldown = int(rebalance_cooldown)
        self.force_liquidate_on_turbulence = bool(force_liquidate_on_turbulence)
        self.reward_scale = float(reward_scale)
        self.reward_mode = str(reward_mode)
        if self.reward_mode not in {"log_return", "benchmark_relative", "upside_opportunity"}:
            raise ValueError("reward_mode must be one of: 'log_return', 'benchmark_relative', 'upside_opportunity'")
        self.benchmark_coef = float(benchmark_coef)
        self.opportunity_coef = float(opportunity_coef)
        self.target_exposure = float(target_exposure)
        self.low_exposure_threshold = float(low_exposure_threshold)
        self.cash_penalty_grace_steps = int(cash_penalty_grace_steps)
        self.turnover_penalty_coef = float(turnover_penalty_coef)
        self.downside_penalty_coef = float(downside_penalty_coef)
        self.liquidation_penalty = float(liquidation_penalty)
        self.atr_target = float(atr_target)
        self.atr_order_scale_min = float(atr_order_scale_min)
        self.atr_order_scale_max = float(atr_order_scale_max)

        self.n_assets = int(self.close_prices.shape[1])
        self.n_features = int(self.features.shape[1])
        self.last_trade_step = len(self.close_prices) - 2

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.n_features),
            dtype=np.float32,
        )

        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = np.zeros(self.n_assets, dtype=np.float32)
        self.prev_account_equity = self.initial_balance
        self.account_history = [self.initial_balance]
        self.portfolio_history = self.account_history
        self.trade_count = 0
        self.trade_log: list[Dict[str, Any]] = []
        self.last_order_step = -10**9
        self.low_exposure_steps = 0
        self._high_water_mark = self.initial_balance
        self._bankrupt = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = float(self.initial_balance)
        self.shares_held = np.zeros(self.n_assets, dtype=np.float32)
        self.prev_account_equity = float(self.initial_balance)
        self.account_history = [float(self.initial_balance)]
        self.portfolio_history = self.account_history
        self.trade_count = 0
        self.trade_log = []
        self.last_order_step = -10**9
        self.low_exposure_steps = 0
        self._high_water_mark = float(self.initial_balance)
        self._bankrupt = False
        return self._get_obs(), {}

    def _signal_idx(self) -> int:
        return max(0, min(self.current_step - 1, len(self.close_prices) - 1))

    def _execution_idx(self) -> int:
        return max(0, min(self.current_step, len(self.close_prices) - 1))

    def _mark_idx_after_execution(self, exec_idx: int) -> int:
        return min(int(exec_idx) + 1, len(self.close_prices) - 1)

    def _account_equity_at(self, idx: int) -> float:
        idx = max(0, min(int(idx), len(self.close_prices) - 1))
        return float(self.balance + np.sum(self.shares_held * self.close_prices[idx]))

    def _account_equity(self) -> float:
        return self._account_equity_at(self._signal_idx())

    def _portfolio_value(self) -> float:
        return self._account_equity()

    def _get_obs(self) -> np.ndarray:
        end = min(max(self.current_step, self.window_size), len(self.features))
        start = end - self.window_size
        return self.features[start:end].copy().astype(np.float32)

    def render(self):
        equity = self._account_equity()
        print(
            f"step={self.current_step:4d} | equity={equity:,.0f} | "
            f"return={(equity / self.initial_balance - 1) * 100:+.2f}% | "
            f"cash={self.balance:,.0f} | positions={self.shares_held}"
        )

    def _trade_side(self, old_pos: float, delta: float) -> str:
        new_pos = old_pos + delta
        if abs(delta) <= EPS:
            return "hold"
        if delta > 0:
            if old_pos < 0 and new_pos <= 0:
                return "cover"
            if old_pos < 0 < new_pos:
                return "cover_and_buy"
            return "buy"
        if old_pos > 0 and new_pos >= 0:
            return "sell"
        if old_pos > 0 > new_pos:
            return "sell_and_short"
        return "short"

    def _record_trades(
        self,
        *,
        exec_idx: int,
        prices: np.ndarray,
        old_positions: np.ndarray,
        deltas: np.ndarray,
        reason: str,
        equity_after: float,
    ) -> None:
        for i, delta in enumerate(deltas):
            delta_f = float(delta)
            if abs(delta_f) <= EPS:
                continue
            self.trade_log.append(
                {
                    "step": int(exec_idx),
                    "asset": int(i),
                    "side": self._trade_side(float(old_positions[i]), delta_f),
                    "reason": reason,
                    "signed_qty": delta_f,
                    "qty": float(abs(delta_f)),
                    "price": float(prices[i]),
                    "notional": float(abs(delta_f) * prices[i]),
                    "commission": float(abs(delta_f) * prices[i] * self.commission),
                    "cash_after": float(self.balance),
                    "equity_after": float(equity_after),
                }
            )

    def _atr_action_scale(self) -> np.ndarray:
        if self.atr_values is None:
            return np.ones(self.n_assets, dtype=np.float32)
        atr = np.asarray(self.atr_values[self._signal_idx()], dtype=np.float32)
        scale = self.atr_target / (atr + EPS)
        return np.clip(scale, self.atr_order_scale_min, self.atr_order_scale_max).astype(np.float32)

    def _action_to_deltas(self, action: np.ndarray, prices: np.ndarray, equity: float) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(self.n_assets)
        action = np.clip(action, -1.0, 1.0)
        action = np.where(np.abs(action) >= self.min_trade_action, action, 0.0)
        if self.rebalance_cooldown > 0 and (self.current_step - self.last_order_step) < self.rebalance_cooldown:
            action[:] = 0.0

        action = action * self._atr_action_scale()
        order_notional = action * max(equity, 0.0) * self.max_trade_notional_pct
        deltas = order_notional / (prices + EPS)

        min_notional = max(equity, 0.0) * self.min_order_notional_pct
        deltas = np.where(np.abs(deltas * prices) >= min_notional, deltas, 0.0)
        return deltas.astype(np.float32)

    def _apply_position_limits(self, deltas: np.ndarray, prices: np.ndarray, equity: float) -> np.ndarray:
        current_values = self.shares_held.astype(np.float32) * prices
        target_values = current_values + deltas * prices

        max_long = max(equity, 0.0) * self.max_position_pct
        if self.allow_short:
            max_short_per_asset = max(equity, 0.0) * self.margin_ratio
            target_values = np.clip(target_values, -max_short_per_asset, max_long)
        else:
            target_values = np.clip(target_values, 0.0, max_long)

        gross = float(np.sum(np.abs(target_values)))
        max_gross = max(equity, 0.0) * self.max_gross_leverage
        if gross > max_gross + EPS and gross > 0:
            target_values *= max_gross / gross

        if self.allow_short:
            short_value = float(np.sum(np.maximum(-target_values, 0.0)))
            max_total_short = max(equity, 0.0) * self.margin_ratio
            if short_value > max_total_short + EPS and short_value > 0:
                short_scale = max_total_short / short_value
                target_values = np.where(target_values < 0, target_values * short_scale, target_values)

        limited = (target_values - current_values) / (prices + EPS)
        return limited.astype(np.float32)

    def _scale_buys_to_cash(self, deltas: np.ndarray, prices: np.ndarray) -> np.ndarray:
        buy_notional = float(np.sum(np.maximum(deltas, 0.0) * prices))
        if buy_notional <= EPS:
            return deltas
        sell_notional = float(np.sum(np.maximum(-deltas, 0.0) * prices))
        available = max(self.balance, 0.0) + sell_notional * (1.0 - self.commission)
        required = buy_notional * (1.0 + self.commission)
        if required <= available + EPS:
            return deltas
        scale = max(available, 0.0) / (required + EPS)
        return np.where(deltas > 0, deltas * scale, deltas).astype(np.float32)

    def _execute_deltas(self, deltas: np.ndarray, prices: np.ndarray, reason: str, exec_idx: int) -> float:
        deltas = np.asarray(deltas, dtype=np.float32)
        old_positions = self.shares_held.copy()
        notional = float(np.sum(np.abs(deltas) * prices))
        if notional <= EPS:
            return 0.0
        commission = notional * self.commission
        self.balance -= float(np.sum(deltas * prices)) + commission
        self.shares_held += deltas
        self.trade_count += int(np.sum(np.abs(deltas * prices) > EPS))
        self.last_order_step = int(exec_idx)
        equity_after_exec = self._account_equity_at(exec_idx)
        self._record_trades(
            exec_idx=exec_idx,
            prices=prices,
            old_positions=old_positions,
            deltas=deltas,
            reason=reason,
            equity_after=equity_after_exec,
        )
        return notional

    def _liquidate(self, prices: np.ndarray, exec_idx: int, reason: str) -> float:
        if np.all(np.abs(self.shares_held) <= EPS):
            return 0.0
        deltas = -self.shares_held.copy()
        return self._execute_deltas(deltas, prices, reason=reason, exec_idx=exec_idx)

    def _benchmark_log_return(self, exec_idx: int, mark_idx: int) -> float:
        """Equal-weight benchmark return over the same reward interval.

        This is the core anti-collapse change. If the tradable universe rises
        and the agent sits in cash, the benchmark-relative part is negative. If
        the universe falls and the agent avoids exposure, it is positive.
        """
        p0 = np.asarray(self.close_prices[exec_idx], dtype=np.float64)
        p1 = np.asarray(self.close_prices[mark_idx], dtype=np.float64)
        asset_log_ret = np.log((p1 + EPS) / (p0 + EPS))
        return float(np.nanmean(asset_log_ret))

    def step(self, action):
        if self.current_step > self.last_trade_step:
            return self._get_obs(), 0.0, True, False, {"already_done": True}

        exec_idx = self._execution_idx()
        signal_idx = self._signal_idx()
        mark_idx = self._mark_idx_after_execution(exec_idx)
        prices = self.close_prices[exec_idx]
        equity_before = max(float(self.prev_account_equity), 1.0)
        turbulence_signal = float(self.turbulence[signal_idx])

        executed_notional = 0.0
        forced_liquidation = False
        if self.force_liquidate_on_turbulence and turbulence_signal > self.turbulence_threshold:
            executed_notional += self._liquidate(prices, exec_idx, reason="turbulence_liquidation")
            forced_liquidation = True
        else:
            raw_deltas = self._action_to_deltas(action, prices, equity_before)
            limited = self._apply_position_limits(raw_deltas, prices, equity_before)
            limited = self._scale_buys_to_cash(limited, prices)
            executed_notional += self._execute_deltas(limited, prices, reason="policy_order", exec_idx=exec_idx)

        equity_after = self._account_equity_at(mark_idx)
        terminated = False
        if equity_after <= 0:
            self._bankrupt = True
            terminated = True
            equity_after = 0.0

        if equity_after > self._high_water_mark:
            self._high_water_mark = float(equity_after)

        log_ret = float(np.log(max(equity_after, 1.0) / max(equity_before, 1.0)))
        benchmark_log_ret = self._benchmark_log_return(exec_idx, mark_idx)
        excess_log_ret = log_ret - benchmark_log_ret

        # Mark exposure after execution at the same mark-to-market index used for reward.
        mark_prices = self.close_prices[mark_idx]
        gross_exposure = float(np.sum(np.abs(self.shares_held * mark_prices)) / max(equity_after, 1.0))
        if gross_exposure < self.low_exposure_threshold:
            self.low_exposure_steps += 1
        else:
            self.low_exposure_steps = 0

        opportunity_penalty = 0.0
        if self.reward_mode == "upside_opportunity":
            # Penalize cash only when the equal-weight benchmark rises.
            # This fixes no-trade collapse without rewarding meaningless turnover.
            market_upside = max(benchmark_log_ret, 0.0)
            exposure_gap = max(self.target_exposure - gross_exposure, 0.0) / max(self.target_exposure, EPS)
            if self.low_exposure_steps <= self.cash_penalty_grace_steps:
                exposure_gap = 0.0
            opportunity_penalty = self.opportunity_coef * market_upside * exposure_gap
            reward_core = log_ret - opportunity_penalty
        elif self.reward_mode == "benchmark_relative":
            reward_core = log_ret + self.benchmark_coef * excess_log_ret
        else:
            reward_core = log_ret

        turnover = float(executed_notional / max(equity_before, 1.0))
        downside = max(-log_ret, 0.0)
        reward = self.reward_scale * (
            reward_core
            - self.turnover_penalty_coef * turnover
            - self.downside_penalty_coef * downside
        )
        if forced_liquidation:
            reward -= self.liquidation_penalty
        if self._bankrupt:
            reward = -10.0

        self.prev_account_equity = float(equity_after)
        self.account_history.append(float(max(equity_after, 0.0)))
        self.current_step += 1

        truncated = self.current_step > self.last_trade_step
        info = {
            "account_equity": float(equity_after),
            "portfolio_value": float(equity_after),
            "balance": float(self.balance),
            "turbulence": turbulence_signal,
            "forced_liquidation": bool(forced_liquidation),
            "turnover": turnover,
            "log_return": log_ret,
            "benchmark_log_return": benchmark_log_ret,
            "excess_log_return": excess_log_ret,
            "reward_core": float(reward_core),
            "opportunity_penalty": float(opportunity_penalty),
            "gross_exposure": float(gross_exposure),
            "low_exposure_steps": int(self.low_exposure_steps),
            "trade_notional": float(executed_notional),
        }
        return self._get_obs(), float(reward), terminated, truncated, info


class MOEXTradingEnvShort(MOEXTradingEnv):
    """Compatibility wrapper that enables short selling."""

    def __init__(self, *args, margin_ratio: float = 0.25, **kwargs):
        kwargs.pop("allow_short", None)
        super().__init__(*args, allow_short=True, margin_ratio=margin_ratio, **kwargs)


class MOEXTradingEnvV2(MOEXTradingEnv):
    """Trading env with account state appended to the last token of observation."""

    def __init__(
        self,
        *args,
        regime: Optional[np.ndarray] = None,
        rvi: Optional[np.ndarray] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.regime = None if regime is None else np.asarray(regime, dtype=np.int8).reshape(-1)
        self.rvi = None if rvi is None else np.asarray(rvi, dtype=np.float32).reshape(-1)
        if self.regime is not None and len(self.regime) != len(self.close_prices):
            raise ValueError("regime length must equal close_prices length")
        if self.rvi is not None and len(self.rvi) != len(self.close_prices):
            raise ValueError("rvi length must equal close_prices length")

        self._n_regime_features = 3 if self.regime is not None else 0
        self._n_rvi_features = 1 if self.rvi is not None else 0
        self.n_account_features = self.n_assets + 4 + self._n_regime_features + self._n_rvi_features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.n_features + self.n_account_features),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        return self._get_obs_extended(), {}

    def _account_state(self) -> np.ndarray:
        idx = self._signal_idx()
        prices = self.close_prices[idx]
        equity = max(self._account_equity_at(idx), 1.0)
        state = np.zeros(self.n_account_features, dtype=np.float32)
        pos_values = self.shares_held * prices
        state[: self.n_assets] = np.clip(pos_values / equity, -1.0, 1.0)
        offset = self.n_assets
        state[offset] = float(np.clip(self.balance / equity, -2.0, 2.0)); offset += 1
        drawdown = (self._high_water_mark - equity) / (self._high_water_mark + EPS)
        state[offset] = float(np.clip(drawdown, 0.0, 1.0)); offset += 1
        state[offset] = float(np.clip(np.sum(np.abs(pos_values)) / equity, 0.0, 3.0)); offset += 1
        state[offset] = 1.0 if float(self.turbulence[idx]) > self.turbulence_threshold else 0.0; offset += 1
        if self.regime is not None:
            r = int(np.clip(self.regime[idx], 0, 2))
            state[offset + r] = 1.0
            offset += 3
        if self.rvi is not None:
            state[offset] = float(np.clip(self.rvi[idx], 0.0, 1.0))
        return state

    def _get_obs_extended(self) -> np.ndarray:
        base_obs = super()._get_obs()
        account_block = np.zeros((self.window_size, self.n_account_features), dtype=np.float32)
        account_block[-1] = self._account_state()
        return np.concatenate([base_obs, account_block], axis=1).astype(np.float32)

    def step(self, action):
        _, reward, terminated, truncated, info = super().step(action)
        return self._get_obs_extended(), reward, terminated, truncated, info


def make_rolling_episodes(
    *,
    close: np.ndarray,
    features: np.ndarray,
    turbulence: np.ndarray,
    window_size: int,
    episode_length: int = 504,
    stride: int = 126,
    regime: Optional[np.ndarray] = None,
    rvi: Optional[np.ndarray] = None,
    atr_values: Optional[np.ndarray] = None,
) -> list[Dict[str, np.ndarray]]:
    close = _as_float_2d(close, "close")
    features = _as_float_2d(features, "features")
    turbulence = _as_float_1d(turbulence, "turbulence")
    n = len(close)
    min_len = window_size + 3
    if n < min_len:
        return []

    episode_length = int(max(episode_length, min_len))
    stride = int(max(stride, 1))
    episodes: list[Dict[str, np.ndarray]] = []

    def add_slice(start: int, end: int) -> None:
        if end - start < min_len:
            return
        ep: Dict[str, np.ndarray] = {
            "close_prices": close[start:end],
            "features": features[start:end],
            "turbulence": turbulence[start:end],
        }
        if regime is not None:
            ep["regime"] = np.asarray(regime[start:end], dtype=np.int8)
        if rvi is not None:
            ep["rvi"] = np.asarray(rvi[start:end], dtype=np.float32)
        if atr_values is not None:
            ep["atr_values"] = np.asarray(atr_values[start:end], dtype=np.float32)
        episodes.append(ep)

    add_slice(0, n)
    if n > episode_length:
        for start in range(0, n - episode_length + 1, stride):
            add_slice(start, start + episode_length)
        add_slice(n - episode_length, n)
    return episodes


class MultiEpisodeEnv(gym.Env):
    """Samples one historical trading episode on each reset."""

    metadata = {"render_modes": []}

    def __init__(self, episodes: list[Dict[str, np.ndarray]], base_env_cls, base_env_kwargs: dict, shuffle: bool = True, seed: int = 42):
        super().__init__()
        if not episodes:
            raise ValueError("MultiEpisodeEnv: empty episodes list")
        self.episodes = episodes
        self.base_env_cls = base_env_cls
        self.base_env_kwargs = dict(base_env_kwargs)
        self.shuffle = bool(shuffle)
        self._rng = np.random.default_rng(seed)
        self._order = np.arange(len(episodes))
        self._cursor = len(self._order)
        self._current_env = None
        probe = self._make_env(0)
        self.observation_space = probe.observation_space
        self.action_space = probe.action_space
        self.initial_balance = probe.initial_balance
        self.trade_count = 0
        self.account_history = []
        self.portfolio_history = self.account_history
        self.trade_log = []

    def _make_env(self, episode_idx: int):
        kwargs = dict(self.base_env_kwargs)
        kwargs.update(self.episodes[episode_idx])
        return self.base_env_cls(**kwargs)

    def _next_episode_idx(self) -> int:
        self._cursor += 1
        if self._cursor >= len(self._order):
            self._order = np.arange(len(self.episodes))
            if self.shuffle:
                self._rng.shuffle(self._order)
            self._cursor = 0
        return int(self._order[self._cursor])

    def reset(self, seed=None, options=None):
        ep_idx = self._next_episode_idx()
        self._current_env = self._make_env(ep_idx)
        obs, info = self._current_env.reset(seed=seed, options=options)
        info = dict(info)
        info["episode_idx"] = ep_idx
        self.trade_count = 0
        self.account_history = list(self._current_env.account_history)
        self.portfolio_history = self.account_history
        self.trade_log = []
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._current_env.step(action)
        self.trade_count = getattr(self._current_env, "trade_count", self.trade_count)
        self.account_history = getattr(self._current_env, "account_history", self.account_history)
        self.portfolio_history = self.account_history
        self.trade_log = getattr(self._current_env, "trade_log", self.trade_log)
        return obs, reward, terminated, truncated, info

    def render(self):
        if self._current_env is not None:
            self._current_env.render()

    def close(self):
        if self._current_env is not None:
            self._current_env.close()


def _env_kwargs_from_cfg(cfg: EnvFactoryConfig) -> Dict[str, Any]:
    return {
        "initial_balance": cfg.initial_balance,
        "window_size": cfg.window_size,
        "commission": cfg.commission,
        "max_shares_per_trade": cfg.max_shares_per_trade,
        "allow_short": cfg.allow_short,
        "margin_ratio": cfg.margin_ratio,
        "max_position_pct": cfg.max_position_pct,
        "max_gross_leverage": cfg.max_gross_leverage,
        "max_trade_notional_pct": cfg.max_trade_notional_pct,
        "min_trade_action": cfg.min_trade_action,
        "min_order_notional_pct": cfg.min_order_notional_pct,
        "rebalance_cooldown": cfg.rebalance_cooldown,
        "force_liquidate_on_turbulence": cfg.force_liquidate_on_turbulence,
        "reward_scale": cfg.reward_scale,
        "reward_mode": cfg.reward_mode,
        "benchmark_coef": cfg.benchmark_coef,
        "opportunity_coef": cfg.opportunity_coef,
        "target_exposure": cfg.target_exposure,
        "low_exposure_threshold": cfg.low_exposure_threshold,
        "cash_penalty_grace_steps": cfg.cash_penalty_grace_steps,
        "turnover_penalty_coef": cfg.turnover_penalty_coef,
        "downside_penalty_coef": cfg.downside_penalty_coef,
        "liquidation_penalty": cfg.liquidation_penalty,
        "atr_target": cfg.atr_target,
        "atr_order_scale_min": cfg.atr_order_scale_min,
        "atr_order_scale_max": cfg.atr_order_scale_max,
    }


def build_env(
    close_prices: np.ndarray,
    features: np.ndarray,
    turbulence: np.ndarray,
    turbulence_threshold: float,
    short: bool = False,
    v2: bool = True,
    atr_indices: Optional[list] = None,
    atr_values: Optional[np.ndarray] = None,
    regime: Optional[np.ndarray] = None,
    rvi: Optional[np.ndarray] = None,
    cfg: Optional[EnvFactoryConfig] = None,
):
    cfg = cfg or EnvFactoryConfig()
    kwargs = _env_kwargs_from_cfg(cfg)
    kwargs.update(
        {
            "close_prices": close_prices,
            "features": features,
            "turbulence": turbulence,
            "turbulence_threshold": turbulence_threshold,
            "atr_values": atr_values,
        }
    )
    if short:
        kwargs["allow_short"] = True
    if v2:
        kwargs.update({"regime": regime, "rvi": rvi})
        return MOEXTradingEnvV2(**kwargs)
    if kwargs.get("allow_short"):
        return MOEXTradingEnvShort(**kwargs)
    return MOEXTradingEnv(**kwargs)
