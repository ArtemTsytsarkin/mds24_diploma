from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MOEXTradingEnv(gym.Env):
    """Long-only базовая среда из ноутбуков."""

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
    ):
        super().__init__()
        self.close_prices = close_prices
        self.features = features
        self.turbulence = turbulence
        self.turbulence_threshold = turbulence_threshold
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.commission = commission
        self.max_shares = max_shares_per_trade
        self.n_assets = close_prices.shape[1]
        self.n_features = features.shape[1]

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, self.n_features),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = np.zeros(self.n_assets, dtype=np.float32)
        self.prev_portfolio_value = self.initial_balance
        self.portfolio_history = [self.initial_balance]
        self.trade_count = 0
        return self._get_obs(), {}

    def step(self, action):
        turb = self.turbulence[self.current_step]
        if turb > self.turbulence_threshold:
            self.current_step += 1
            done = self.current_step >= len(self.close_prices)
            self.portfolio_history.append(self._portfolio_value())
            return self._get_obs(), -1.0, done, False, {"turbulence": turb}

        prices = self.close_prices[self.current_step]
        shares_delta = (action * self.max_shares).astype(np.float32)
        shares_delta = np.maximum(shares_delta, -self.shares_held)

        buy_cost = np.sum(np.maximum(shares_delta, 0) * prices)
        if buy_cost > self.balance:
            shares_delta = np.where(
                shares_delta > 0,
                shares_delta * self.balance / (buy_cost + 1e-8),
                shares_delta,
            )

        cost = np.sum(np.abs(shares_delta) * prices) * self.commission
        self.balance -= np.sum(shares_delta * prices) + cost
        self.shares_held += shares_delta
        self.trade_count += int(np.any(shares_delta != 0))
        self.current_step += 1

        done = self.current_step >= len(self.close_prices)
        new_value = self._portfolio_value()
        self.portfolio_history.append(new_value)
        reward = float(
            np.clip(
                (new_value - self.prev_portfolio_value) / (self.prev_portfolio_value + 1e-8),
                -1.0,
                1.0,
            )
        )
        self.prev_portfolio_value = new_value
        return self._get_obs(), reward, done, False, {"portfolio_value": new_value, "turbulence": turb}

    def render(self):
        value = self._portfolio_value()
        print(f"День {self.current_step:4d} | {value:>14,.0f} ₽ | {(value / self.initial_balance - 1) * 100:+.2f}%")

    def _get_obs(self):
        return self.features[self.current_step - self.window_size : self.current_step].copy()

    def _portfolio_value(self):
        idx = min(self.current_step, len(self.close_prices) - 1)
        value = float(self.balance + np.sum(self.shares_held * self.close_prices[idx]))
        # Guard: portfolio value should never go below zero in a long-only env;
        # for the short env a negative value signals a margin call — clamp to 0
        # so that reward shaping doesn't produce undefined behaviour.
        return max(value, 0.0)


class MOEXTradingEnvShort(MOEXTradingEnv):
    """
    Версия из improved notebooks:
    - short selling
    - margin_ratio=0.25
    - бонус за правильное направление
    - штраф за short на растущем рынке
    - автозакрытие шорта при росте >2%
    """

    def __init__(self, *args, margin_ratio: float = 0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.margin_ratio = margin_ratio
        self._prev_prices = None

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._prev_prices = None
        return obs, info

    def step(self, action):
        turb = self.turbulence[self.current_step]
        if turb > self.turbulence_threshold:
            self.current_step += 1
            done = self.current_step >= len(self.close_prices)
            self.portfolio_history.append(self._portfolio_value())
            self._prev_prices = None
            return self._get_obs(), -1.0, done, False, {"turbulence": turb}

        prices = self.close_prices[self.current_step]

        if self._prev_prices is not None:
            price_change_auto = (prices - self._prev_prices) / (self._prev_prices + 1e-8)
            for i in range(self.n_assets):
                if self.shares_held[i] < 0 and price_change_auto[i] > 0.02:
                    close_delta = -self.shares_held[i]
                    cost = close_delta * prices[i] * (1 + self.commission)
                    self.balance -= cost
                    self.shares_held[i] = 0

        self._prev_prices = prices.copy()

        shares_delta = (action * self.max_shares).astype(np.float32)
        max_short_value = self._portfolio_value() * self.margin_ratio
        for i in range(self.n_assets):
            if shares_delta[i] < 0:
                max_short_shares = max_short_value / (prices[i] + 1e-8)
                shares_delta[i] = max(shares_delta[i], -max_short_shares - self.shares_held[i])

        buy_cost = np.sum(np.maximum(shares_delta, 0) * prices)
        if buy_cost > self.balance:
            shares_delta = np.where(
                shares_delta > 0,
                shares_delta * self.balance / (buy_cost + 1e-8),
                shares_delta,
            )

        cost = np.sum(np.abs(shares_delta) * prices) * self.commission
        self.balance -= np.sum(shares_delta * prices) + cost
        self.shares_held += shares_delta
        self.trade_count += int(np.any(shares_delta != 0))
        self.current_step += 1

        done = self.current_step >= len(self.close_prices)
        new_value = self._portfolio_value()
        self.portfolio_history.append(new_value)

        # Fix #5: direction_bonus must not peek at future prices.
        # Use the price change from _prev_prices to current prices (already observed).
        if self._prev_prices is not None:
            price_change = (prices - self._prev_prices) / (self._prev_prices + 1e-8)
        else:
            price_change = np.zeros(self.n_assets)

        raw_reward = (new_value - self.prev_portfolio_value) / (self.prev_portfolio_value + 1e-8)
        direction_bonus = np.sum(np.sign(self.shares_held) * price_change) * 0.1
        raw_reward += direction_bonus

        bad_short = (
            np.sum((self.shares_held < 0) * (price_change > 0) * np.abs(self.shares_held) * prices)
            / (self.prev_portfolio_value + 1e-8)
        )
        raw_reward -= bad_short * 0.2

        reward = float(np.clip(raw_reward, -1.0, 1.0))
        self.prev_portfolio_value = new_value

        return self._get_obs(), reward, done, False, {
            "portfolio_value": new_value,
            "balance": self.balance,
            "turbulence": turb,
        }


@dataclass
class EnvFactoryConfig:
    window_size: int = 30
    margin_ratio: float = 0.25
    initial_balance: float = 1_000_000.0
    commission: float = 0.0005
    max_shares_per_trade: int = 100


def build_env(
    close_prices: np.ndarray,
    features: np.ndarray,
    turbulence: np.ndarray,
    turbulence_threshold: float,
    short: bool = True,
    cfg: Optional[EnvFactoryConfig] = None,
):
    cfg = cfg or EnvFactoryConfig()
    env_cls = MOEXTradingEnvShort if short else MOEXTradingEnv
    kwargs = dict(
        close_prices=close_prices,
        features=features,
        turbulence=turbulence,
        turbulence_threshold=turbulence_threshold,
        initial_balance=cfg.initial_balance,
        window_size=cfg.window_size,
        commission=cfg.commission,
        max_shares_per_trade=cfg.max_shares_per_trade,
    )
    if short:
        kwargs["margin_ratio"] = cfg.margin_ratio
    return env_cls(**kwargs)
