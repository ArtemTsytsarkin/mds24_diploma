from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

from sb3_contrib import RecurrentPPO


def extract_step_from_checkpoint(filename: str) -> int:
    m = re.search(r"(\d+)_steps\.zip$", filename)
    return int(m.group(1)) if m else -1


class CheckpointSelector:
    """Evaluates stage checkpoints on validation env and selects models by one score."""

    def __init__(self, checkpoint_dir: str, model_name: str, stage: str = "stage2", risk_free_rate: float = 0.0):
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.stage = stage
        self.risk_free_rate = float(risk_free_rate)

    def _get_checkpoints(self) -> List[str]:
        if not os.path.isdir(self.checkpoint_dir):
            return []
        prefix = f"{self.model_name}_{self.stage}"
        files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith(prefix) and f.endswith(".zip")]
        files.sort(key=extract_step_from_checkpoint)
        return [os.path.join(self.checkpoint_dir, f) for f in files]

    def _evaluate_one(self, path: str, val_env, device: str) -> Optional[Dict]:
        from train import compute_metrics, evaluate_policy, score_metrics
        try:
            model = RecurrentPPO.load(path, device=device)
            curve = evaluate_policy(model, val_env)
            metrics = compute_metrics(curve, val_env.initial_balance, getattr(val_env, "trade_count", 0), risk_free_rate=self.risk_free_rate)
            step = extract_step_from_checkpoint(os.path.basename(path))
            return {
                "path": path,
                "step": step,
                "SR": metrics["SR"],
                "CR": metrics["CR"],
                "MPB": metrics["MPB"],
                "WinRate": metrics["WinRate"],
                "ProfitFactor": metrics["ProfitFactor"],
                "Trades": metrics["Trades"],
                "TradeRate": metrics["TradeRate"],
                "score": score_metrics(metrics),
            }
        except Exception as exc:
            print(f"  [CheckpointSelector] Ошибка при оценке {path}: {exc}")
            return None

    def evaluate_all(self, val_env, device: str = "cpu") -> List[Dict]:
        checkpoints = self._get_checkpoints()
        if not checkpoints:
            print(f"[CheckpointSelector] Чекпоинтов не найдено в {self.checkpoint_dir}")
            return []
        print(f"[CheckpointSelector] Оцениваем {len(checkpoints)} чекпоинтов на val_env...")
        results: List[Dict] = []
        for i, path in enumerate(checkpoints):
            step = extract_step_from_checkpoint(os.path.basename(path))
            print(f"  [{i + 1}/{len(checkpoints)}] step={step:,} ...", end=" ", flush=True)
            result = self._evaluate_one(path, val_env, device)
            if result is None:
                print("SKIP")
                continue
            results.append(result)
            print(
                f"score={result['score']:+.3f} SR={result['SR']:+.3f} "
                f"CR={result['CR']:+.1f}% MPB={result['MPB']:.1f}% Trades={result['Trades']}"
            )
        return results

    def select_top(self, results: List[Dict], k: int = 3, min_trades: int = 20, min_trade_rate: float = 0.10) -> List[Dict]:
        if not results:
            return []
        eligible = [
            r for r in results
            if int(r.get("Trades", 0)) >= int(min_trades)
            and float(r.get("TradeRate", 0.0)) >= float(min_trade_rate)
        ]
        if not eligible:
            eligible = results

        sorted_by_score = sorted(eligible, key=lambda r: r["score"], reverse=True)
        selected = {r["path"]: r for r in sorted_by_score[:k]}
        selected[max(eligible, key=lambda r: r["SR"])["path"]] = max(eligible, key=lambda r: r["SR"])
        selected[min(eligible, key=lambda r: r["MPB"])["path"]] = min(eligible, key=lambda r: r["MPB"])
        final = list(selected.values())
        skipped = len(results) - len(eligible)
        print(
            f"\n[CheckpointSelector] Выбрано {len(final)} моделей по validation score/diversity "
            f"(cash-like skipped={skipped}, min_trades={min_trades}, min_trade_rate={min_trade_rate}):"
        )
        for r in final:
            print(
                f"  step={r['step']:,} | score={r['score']:+.3f} | SR={r['SR']:+.3f} | "
                f"CR={r['CR']:+.1f}% | MPB={r['MPB']:.1f}% | Trades={r['Trades']}"
            )
        return final
