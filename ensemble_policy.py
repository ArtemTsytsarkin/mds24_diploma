from __future__ import annotations

from typing import List, Optional

import numpy as np
from sb3_contrib import RecurrentPPO


class EnsemblePolicy:
    """Action-level ensemble for RecurrentPPO policies."""

    def __init__(self, model_paths: List[str], device: str = "cpu", weights: Optional[np.ndarray] = None):
        if not model_paths:
            raise ValueError("EnsemblePolicy: пустой список моделей")
        print(f"[EnsemblePolicy] Загружаем {len(model_paths)} моделей...")
        self.models: List[RecurrentPPO] = []
        for path in model_paths:
            print(f"  {path}")
            self.models.append(RecurrentPPO.load(path, device=device))
        if weights is not None:
            w = np.asarray(weights, dtype=np.float32).reshape(-1)
            if len(w) != len(self.models):
                raise ValueError("weights length must match model_paths length")
            self.weights = w / w.sum() if w.sum() > 0 else np.ones(len(self.models), dtype=np.float32) / len(self.models)
            print("[EnsemblePolicy] Агрегация: weighted mean")
        else:
            self.weights = None
            print("[EnsemblePolicy] Агрегация: median")
        self.lstm_states: List[Optional[tuple]] = [None] * len(self.models)
        self._is_first_step = True

    def reset(self) -> None:
        self.lstm_states = [None] * len(self.models)
        self._is_first_step = True

    def predict(self, obs: np.ndarray) -> np.ndarray:
        obs_batch = obs[np.newaxis, ...]
        episode_start = np.array([self._is_first_step], dtype=bool)
        actions = []
        new_states = []
        for i, model in enumerate(self.models):
            action_batch, state = model.predict(
                obs_batch,
                state=self.lstm_states[i],
                episode_start=episode_start,
                deterministic=True,
            )
            actions.append(action_batch[0])
            new_states.append(state)
        self.lstm_states = new_states
        self._is_first_step = False
        stacked = np.stack(actions, axis=0)
        if self.weights is None:
            return np.median(stacked, axis=0).astype(np.float32)
        return np.average(stacked, axis=0, weights=self.weights).astype(np.float32)

    def evaluate(self, env) -> list:
        obs, _ = env.reset()
        self.reset()
        done = False
        while not done:
            action = self.predict(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
        return list(env.portfolio_history)
