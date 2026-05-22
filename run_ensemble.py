"""
run_ensemble.py — Оценка кросс-архитектурного ансамбля на тест-периоде.

Каждая модель получает observation из своей среды (с нужным use_rvi/use_hmm),
затем действия агрегируются медианой. Это позволяет смешивать модели
с разными observation space (baseline 124, RVI 125, HMM 127).

Использование:
    python run_ensemble.py
    python run_ensemble.py --config my_ensemble.json
    python run_ensemble.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from sb3_contrib import RecurrentPPO

from data import DataConfig, prepare_datasets
from env import MOEXTradingEnvV2
from train import (
    TrainConfig,
    compute_baseline_metrics,
    compute_metrics,
    make_env_cfg,
    _base_env_kwargs_from_cfg,
)

EPS = 1e-8


# ══════════════════════════════════════════════════════════════════════════════
# Конфигурация ансамблей
# Каждая запись: (path, use_rvi, use_hmm)
# ══════════════════════════════════════════════════════════════════════════════

def get_default_ensembles() -> Dict[str, List[Tuple[str, bool, bool]]]:
    """
    7 вариаций ансамбля с разными принципами отбора компонентов.
    Каждая запись: (path_to_checkpoint, use_rvi, use_hmm).

    Отредактируй пути под свою файловую структуру.
    Найди нужные чекпоинты в val_test_report.json → ensemble_paths.
    """
    rvi42   = "./final1_rvi_seed42"
    rvi748  = "./final1_rvi_seed748"
    rvi1777 = "./final1_rvi_seed1777"
    hmm42   = "./final1_hmm_seed42"
    hmm748  = "./final1_hmm_seed748"
    b42     = "./final1_seed42"
    b748    = "./final1_seed748"
    b1777   = "./final1_seed1777"

    return {

        # ── Вариация 1: Три лучших по SR агента (RVI) ────────────────────────
        # LSTM s42 SR=1.756, xLSTM Attn s42 SR=1.368, xLSTM Large s748 SR=1.255
        # Гипотеза: топ-3 по SR дают лучший результат
        "v1_top3_sr": [
            (f"{rvi42}/lstm/checkpoints/lstm_stage2_175000_steps.zip",                True,  False),
            (f"{rvi42}/xlstm_attn/checkpoints/xlstm_attn_stage2_195000_steps.zip",    True,  False),
            (f"{rvi748}/xlstm_large/checkpoints/xlstm_large_stage2_75000_steps.zip",  True,  False),
        ],

        # ── Вариация 2: Самые стабильные архитектуры (baseline) ──────────────
        # xLSTM Base, xLSTM Attn, xLSTM Large — baseline лучший для mLSTM
        # Гипотеза: матричная память без сигналов наиболее надёжна
        "v2_stable_baseline": [
            (f"{b42}/xlstm_base/checkpoints/xlstm_base_stage2_30000_steps.zip",        False, False),
            (f"{b42}/xlstm_attn/checkpoints/xlstm_attn_stage2_195000_steps.zip",       False, False),
            (f"{b748}/xlstm_large/checkpoints/xlstm_large_stage2_75000_steps.zip",     False, False),
        ],

        # ── Вариация 3: Смешанный RVI + HMM (разные obs space) ───────────────
        # xLSTM Attn RVI + LSTM RVI + TFT HMM
        # Гипотеза: разные сигналы дают диверсификацию информации
        "v3_mixed_signals": [
            (f"{rvi42}/xlstm_attn/checkpoints/xlstm_attn_stage2_195000_steps.zip",    True,  False),
            (f"{rvi42}/lstm/checkpoints/lstm_stage2_175000_steps.zip",                 True,  False),
            (f"{hmm42}/tft_like/checkpoints/tft_like_stage2_175000_steps.zip",         False, True),
        ],

        # ── Вариация 4: Одна архитектура × три seeds (xLSTM Attn RVI) ────────
        # Seeds 42, 748, 1777 для xLSTM Attn с RVI
        # Гипотеза: ансамбль seeds одной архитектуры усредняет случайность
        "v4_seeds_xlstm_attn": [
            (f"{rvi42}/xlstm_attn/checkpoints/xlstm_attn_stage2_195000_steps.zip",    True,  False),
            (f"{rvi748}/xlstm_attn/checkpoints/xlstm_attn_stage2_120000_steps.zip",   True,  False),
            (f"{rvi1777}/xlstm_attn/checkpoints/xlstm_attn_stage2_65000_steps.zip",   True,  False),
        ],

        # ── Вариация 5: Максимальная архитектурная диверсификация (5 моделей) ─
        # MLP + LSTM + Transformer + xLSTM Attn + TFT — разные классы
        # Гипотеза: разнородные архитектуры имеют наименее коррелированные ошибки
        "v5_max_diversity": [
            (f"{rvi42}/xlstm_attn/checkpoints/xlstm_attn_stage2_195000_steps.zip",    True,  False),
            (f"{rvi42}/lstm/checkpoints/lstm_stage2_175000_steps.zip",                 True,  False),
            (f"{hmm42}/tft_like/checkpoints/tft_like_stage2_175000_steps.zip",         False, True),
            (f"{hmm748}/transformer_causal/checkpoints/transformer_causal_stage2_185000_steps.zip", False, True),
            (f"{b42}/xlstm_base/checkpoints/xlstm_base_stage2_30000_steps.zip",        False, False),
        ],

        # ── Вариация 6: Лучший по стабильности (min std) среди RVI ───────────
        # xLSTM Attn RVI (std=0.434) + MLP RVI (std=0.411) — наименее вариабельные
        # Гипотеза: стабильные компоненты дают стабильный ансамбль
        "v6_min_std_rvi": [
            (f"{rvi42}/xlstm_attn/checkpoints/xlstm_attn_stage2_195000_steps.zip",    True,  False),
            (f"{rvi748}/xlstm_attn/checkpoints/xlstm_attn_stage2_120000_steps.zip",   True,  False),
            (f"{rvi42}/mlp/checkpoints/mlp_stage2_35000_steps.zip",                   True,  False),
            (f"{rvi748}/mlp/checkpoints/mlp_stage2_165000_steps.zip",                 True,  False),
        ],

        # ── Вариация 7: Полный кросс-ансамбль (6 моделей) ───────────────────
        # Три лучших RVI + два лучших HMM + лучший baseline
        # Гипотеза: больше компонентов = меньше variance
        "v7_full_cross": [
            (f"{rvi42}/lstm/checkpoints/lstm_stage2_175000_steps.zip",                 True,  False),
            (f"{rvi42}/xlstm_attn/checkpoints/xlstm_attn_stage2_195000_steps.zip",    True,  False),
            (f"{rvi748}/xlstm_large/checkpoints/xlstm_large_stage2_75000_steps.zip",  True,  False),
            (f"{hmm42}/tft_like/checkpoints/tft_like_stage2_175000_steps.zip",         False, True),
            (f"{hmm748}/transformer_causal/checkpoints/transformer_causal_stage2_185000_steps.zip", False, True),
            (f"{b42}/xlstm_base/checkpoints/xlstm_base_stage2_30000_steps.zip",        False, False),
        ],

    }


# ══════════════════════════════════════════════════════════════════════════════
# Загрузка данных для всех нужных конфигураций
# ══════════════════════════════════════════════════════════════════════════════

def load_all_bundles(
    ensembles: Dict[str, List[Tuple[str, bool, bool]]],
    base_cfg: DataConfig,
) -> Dict[Tuple[bool, bool], object]:
    needed = set()
    for configs in ensembles.values():
        for _, use_rvi, use_hmm in configs:
            needed.add((use_rvi, use_hmm))
    # Всегда нужна база без сигналов для portfolio_history ref-env
    needed.add((False, False))

    bundle_map = {}
    for use_rvi, use_hmm in needed:
        print(f"\n{'─'*55}")
        print(f"  Данные: use_rvi={use_rvi}  use_hmm={use_hmm}")
        print(f"{'─'*55}")
        cfg = DataConfig(
            tickers              = base_cfg.tickers,
            train_start          = base_cfg.train_start,
            train_end            = base_cfg.train_end,
            val_start            = base_cfg.val_start,
            val_end              = base_cfg.val_end,
            test_start           = base_cfg.test_start,
            test_end             = base_cfg.test_end,
            data_file            = base_cfg.data_file,
            turbulence_lookback  = base_cfg.turbulence_lookback,
            turbulence_percentile= base_cfg.turbulence_percentile,
            use_rvi              = use_rvi,
            use_hmm              = use_hmm,
            rvi_file             = base_cfg.rvi_file,
            n_regimes            = base_cfg.n_regimes,
        )
        bundle_map[(use_rvi, use_hmm)] = prepare_datasets(cfg)
    return bundle_map


def build_test_env(bundle_map, env_cfg, use_rvi: bool, use_hmm: bool):
    bundle = bundle_map[(use_rvi, use_hmm)]
    kwargs = _base_env_kwargs_from_cfg(env_cfg, bundle.threshold)
    return MOEXTradingEnvV2(
        close_prices = bundle.close_test,
        features     = bundle.features_test,
        turbulence   = bundle.turb_test,
        atr_values   = getattr(bundle, "atr_test",    None),
        regime       = getattr(bundle, "regime_test", None),
        rvi          = getattr(bundle, "rvi_test",    None),
        **kwargs,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Ансамбль с per-model environment
# ══════════════════════════════════════════════════════════════════════════════

class MultiEnvEnsemble:
    """
    Каждая модель работает со своей средой (нужный obs shape).
    Действия агрегируются медианой/взвешенным средним.
    Все среды продвигаются одним агрегированным действием для синхронности.
    """

    def __init__(
        self,
        model_paths: List[str],
        envs: List[MOEXTradingEnvV2],
        device: str = "cpu",
        weights: Optional[np.ndarray] = None,
    ):
        assert len(model_paths) == len(envs)
        print(f"[MultiEnvEnsemble] Загружаем {len(model_paths)} моделей...")
        self.models = []
        for path in model_paths:
            print(f"  {path}")
            self.models.append(RecurrentPPO.load(path, device=device))
        self.envs        = envs
        self.weights     = weights
        self.lstm_states = [None] * len(self.models)

    def evaluate(self, ref_env: MOEXTradingEnvV2) -> list:
        """
        ref_env — базовая среда (без сигналов) для накопления portfolio_history.
        Все среды запускаются синхронно, действие одно — агрегированное.
        """
        obs_list = []
        for env in self.envs:
            obs, _ = env.reset()
            obs_list.append(obs)
        ref_env.reset()
        self.lstm_states = [None] * len(self.models)

        is_first = True
        done = False

        while not done:
            ep_start = np.array([is_first], dtype=bool)
            actions  = []

            for i, (model, obs) in enumerate(zip(self.models, obs_list)):
                a_batch, state = model.predict(
                    obs[np.newaxis, ...],
                    state         = self.lstm_states[i],
                    episode_start = ep_start,
                    deterministic = True,
                )
                actions.append(a_batch[0])
                self.lstm_states[i] = state

            stacked = np.stack(actions, axis=0)
            if self.weights is None:
                agg = np.median(stacked, axis=0).astype(np.float32)
            else:
                w   = self.weights / (self.weights.sum() + EPS)
                agg = np.average(stacked, axis=0, weights=w).astype(np.float32)

            new_obs = []
            for env in self.envs:
                o, _, t, tr, _ = env.step(agg)
                new_obs.append(o)

            _, _, ref_t, ref_tr, _ = ref_env.step(agg)
            done     = bool(ref_t or ref_tr)
            obs_list = new_obs
            is_first = False

        return list(ref_env.portfolio_history)


# ══════════════════════════════════════════════════════════════════════════════
# Основная функция
# ══════════════════════════════════════════════════════════════════════════════

def run_ensemble_eval(
    ensembles:   Dict[str, List[Tuple[str, bool, bool]]],
    data_cfg:    Optional[DataConfig] = None,
    train_cfg:   Optional[TrainConfig] = None,
    device:      str = "cpu",
    output_json: str = "results/ensemble_results.json",
    aggregation: str = "median",
) -> Dict:
    data_cfg  = data_cfg  or DataConfig(
        tickers   = ["SBER", "GAZP", "LKOH", "NVTK", "GMKN"],
        data_file = "moex_data_v2.csv",
    )
    train_cfg = train_cfg or TrainConfig()
    env_cfg   = make_env_cfg(train_cfg)

    bundle_map = load_all_bundles(ensembles, data_cfg)

    # Бенчмарки
    base = bundle_map[(False, False)]
    print("\nВычисляем бенчмарки...")
    baselines = compute_baseline_metrics(
        base.close_test, base.turb_test, base.threshold,
        train_cfg.initial_balance,
        window_size    = train_cfg.window_size,
        commission     = train_cfg.commission,
        risk_free_rate = train_cfg.risk_free_rate,
    )
    for bname in ["buy_and_hold", "equal_weight"]:
        bm = baselines[bname]
        print(f"  {bname:15s}: SR={bm['SR']:+.3f}  CR={bm['CR']:+.2f}%")

    results = {"baselines": baselines, "ensembles": {}}

    for ens_name, model_configs in ensembles.items():
        print(f"\n{'='*60}")
        print(f"  Ансамбль: {ens_name}  ({len(model_configs)} моделей)")
        print(f"{'='*60}")

        paths    = [mc[0] for mc in model_configs]
        use_rvis = [mc[1] for mc in model_configs]
        use_hmms = [mc[2] for mc in model_configs]

        missing = [p for p in paths
                   if not os.path.exists(p) and not os.path.exists(p + ".zip")]
        if missing:
            print(f"  [SKIP] Файлы не найдены: {missing}")
            results["ensembles"][ens_name] = {"error": "files not found", "missing": missing}
            continue

        try:
            per_model_envs = [
                build_test_env(bundle_map, env_cfg, r, h)
                for r, h in zip(use_rvis, use_hmms)
            ]
            ref_env = build_test_env(bundle_map, env_cfg, False, False)

            weights = np.ones(len(paths), dtype=np.float32) if aggregation == "weighted" else None
            ens     = MultiEnvEnsemble(paths, per_model_envs, device=device, weights=weights)

            print("  Запускаем...")
            curve    = ens.evaluate(ref_env)
            metrics  = compute_metrics(
                curve, train_cfg.initial_balance,
                getattr(ref_env, "trade_count", 0),
                risk_free_rate=train_cfg.risk_free_rate,
            )

            print(f"  SR={metrics['SR']:+.4f}  Sortino={metrics['Sortino']:+.4f}  "
                  f"Calmar={metrics['Calmar']:+.4f}  CR={metrics['CR']:+.3f}%  "
                  f"MPB={metrics['MPB']:.3f}%  PF={metrics['ProfitFactor']:.3f}  "
                  f"Trades={metrics['Trades']}  FV={metrics['FinalValue']:,.0f} ₽")
            print(f"  vs Buy&Hold: Δ SR = {metrics['SR'] - baselines['buy_and_hold']['SR']:+.3f}")

            results["ensembles"][ens_name] = {
                "metrics":           metrics,
                "portfolio_history": curve,
                "n_models":          len(paths),
                "paths":             paths,
                "signals":           [{"use_rvi": r, "use_hmm": h}
                                      for r, h in zip(use_rvis, use_hmms)],
            }

        except Exception as exc:
            import traceback
            print(f"  [ERROR] {ens_name}: {exc}")
            traceback.print_exc()
            results["ensembles"][ens_name] = {"error": str(exc)}

    # Итоговая таблица
    print("\n" + "="*65)
    print("  ИТОГОВАЯ ТАБЛИЦА")
    print("="*65)
    print(f"{'Конфигурация':22s} {'SR':>8} {'CR%':>7} {'MPB%':>7} {'Calmar':>8} {'Trades':>7} {'FV ₽':>12}")
    print("-"*65)
    for bname in ["buy_and_hold","equal_weight","momentum_trader","cash"]:
        bm = baselines.get(bname, {})
        if bm:
            print(f"{bname:22s} {bm.get('SR',0):>+8.3f} {bm.get('CR',0):>+7.2f} "
                  f"{bm.get('MPB',0):>7.3f} {bm.get('Calmar',0):>+8.3f} "
                  f"{bm.get('Trades',0):>7} {bm.get('FinalValue',0):>12,.0f}")
    print("-"*65)
    for ens_name, er in results["ensembles"].items():
        if "error" in er:
            print(f"{ens_name:22s}  ERROR: {er['error']}")
            continue
        m = er["metrics"]
        print(f"{ens_name:22s} {m['SR']:>+8.3f} {m['CR']:>+7.2f} "
              f"{m['MPB']:>7.3f} {m['Calmar']:>+8.3f} "
              f"{m['Trades']:>7} {m['FinalValue']:>12,.0f}")

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Сохранено: {output_json}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default=None)
    parser.add_argument("--device",      default="cpu")
    parser.add_argument("--output",      default="results/ensemble_results.json")
    parser.add_argument("--data-file",   default="moex_data_v2.csv")
    parser.add_argument("--aggregation", default="median", choices=["median","weighted"])
    args = parser.parse_args()

    if args.config and os.path.exists(args.config):
        with open(args.config, encoding="utf-8") as f:
            raw = json.load(f)
        ensembles = {k: [tuple(x) for x in v] for k, v in raw.items()}
    else:
        ensembles = get_default_ensembles()
        print("Используются пути по умолчанию. "
              "Отредактируй get_default_ensembles() в run_ensemble.py.")

    data_cfg = DataConfig(
        tickers   = ["SBER", "GAZP", "LKOH", "NVTK", "GMKN"],
        data_file = args.data_file,
    )
    run_ensemble_eval(
        ensembles   = ensembles,
        data_cfg    = data_cfg,
        device      = args.device,
        output_json = args.output,
        aggregation = args.aggregation,
    )


if __name__ == "__main__":
    cli()
