"""
diagnose.py — Фаза 1: диагностика без переобучения

Запуск:
    python diagnose.py --model-dir ./models/xlstm_large

Что проверяет:
    1. Action collapse — распределение действий агента на тестовых данных
    2. YNDX data quality — NaN/константы в close_prices на тестовом периоде
    3. Reward audit — сравнение clipped vs log-return reward на сохранённой модели
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

from data_v2 import DEFAULT_CONFIG, DataConfig, prepare_datasets, set_global_seed
from env_6 import EnvFactoryConfig, build_env, MOEXTradingEnvShort

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. Action collapse
# ---------------------------------------------------------------------------

def diagnose_action_collapse(model: RecurrentPPO, bundle, env_cfg: EnvFactoryConfig, output_dir: str):
    """
    Прогоняет модель на тестовых данных и собирает все actions.
    Здоровый агент должен использовать весь диапазон [-1, 1] по каждому активу.
    Признак коллапса: std близко к 0 или distribution сосредоточена в одной точке.
    """
    print("\n=== 1. ACTION COLLAPSE ===")

    env = build_env(
        close_prices=bundle.close_test,
        features=bundle.features_test,
        turbulence=bundle.turb_test,
        turbulence_threshold=bundle.threshold,
        short=True,
        cfg=env_cfg,
    )
    vec_env = DummyVecEnv([lambda: env])

    obs = vec_env.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    actions_log = []
    done = False

    while not done:
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=True,
        )
        actions_log.append(action[0].copy())
        obs, _, dones, _ = vec_env.step(action)
        episode_starts = dones
        done = bool(dones[0])

    actions_arr = np.array(actions_log)  # shape: (T, n_assets)
    tickers = bundle.tickers

    print(f"Шагов в тесте: {len(actions_arr)}")
    print(f"Активов: {len(tickers)}")
    print()

    stats = {}
    collapse_detected = False
    for i, ticker in enumerate(tickers):
        a = actions_arr[:, i]
        s = {
            "ticker": ticker,
            "mean":   float(np.mean(a)),
            "std":    float(np.std(a)),
            "min":    float(np.min(a)),
            "max":    float(np.max(a)),
            "pct_long":  float(np.mean(a > 0.1)),
            "pct_short": float(np.mean(a < -0.1)),
            "pct_flat":  float(np.mean(np.abs(a) <= 0.1)),
        }
        stats[ticker] = s
        flag = ""
        if s["std"] < 0.05:
            flag = "  <<< КОЛЛАПС: std слишком мала!"
            collapse_detected = True
        elif s["pct_flat"] > 0.9:
            flag = "  <<< КОЛЛАПС: агент почти всегда в flat!"
            collapse_detected = True
        elif s["pct_long"] > 0.95 or s["pct_short"] > 0.95:
            flag = "  <<< КОЛЛАПС: агент всегда в одну сторону!"
            collapse_detected = True

        print(
            f"  {ticker}: mean={s['mean']:+.3f}  std={s['std']:.3f}  "
            f"long={s['pct_long']:.1%}  short={s['pct_short']:.1%}  flat={s['pct_flat']:.1%}{flag}"
        )

    print()
    if collapse_detected:
        print("ВЫВОД: action collapse обнаружен. Рекомендуется повысить ent_coef (фаза 2).")
    else:
        print("ВЫВОД: action collapse не обнаружен. Агент использует весь диапазон действий.")

    # График
    fig, axes = plt.subplots(1, len(tickers), figsize=(3 * len(tickers), 3), sharey=True)
    if len(tickers) == 1:
        axes = [axes]
    for i, (ticker, ax) in enumerate(zip(tickers, axes)):
        ax.hist(actions_arr[:, i], bins=40, range=(-1, 1), color="#378ADD", edgecolor="none", alpha=0.8)
        ax.set_title(ticker, fontsize=11)
        ax.set_xlabel("action", fontsize=9)
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("count", fontsize=9)
    fig.suptitle("Action distribution per asset (test period)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(output_dir, "action_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"График сохранён: {path}")

    with open(os.path.join(output_dir, "action_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return actions_arr, stats, collapse_detected


# ---------------------------------------------------------------------------
# 2. Data quality (YNDX / NaN)
# ---------------------------------------------------------------------------

def diagnose_data_quality(bundle, output_dir: str):
    """
    Проверяет close_prices на тестовом периоде:
    - NaN после ffill
    - константные серии (признак делистинга / заморозки торгов)
    - большие пропуски
    """
    print("\n=== 2. DATA QUALITY (YNDX и другие) ===")

    tickers = bundle.tickers
    close_test = pd.DataFrame(
        bundle.close_test,
        columns=tickers,
        index=bundle.dates_all[bundle.test_mask],
    )

    issues = {}
    for ticker in tickers:
        s = close_test[ticker]
        n_nan = int(s.isna().sum())
        n_zero = int((s == 0).sum())
        n_total = len(s)

        # Константные серии: std очень мала относительно mean
        rolling_std = s.rolling(20).std()
        n_flat_windows = int((rolling_std < 1e-4).sum())

        # Процент дней, когда цена не меняется вообще
        pct_no_change = float((s.diff() == 0).mean())

        # Максимальный прирост и просадка — для проверки адекватности
        returns = s.pct_change().dropna()
        max_return = float(returns.max())
        min_return = float(returns.min())

        flag = ""
        if n_nan > 0:
            flag += f" NaN={n_nan}"
        if n_zero > 0:
            flag += f" ZERO={n_zero}"
        if pct_no_change > 0.5:
            flag += f" ЗАМОРОЖЕН({pct_no_change:.0%} дней без движения)"
        if n_flat_windows > n_total * 0.3:
            flag += " КОНСТАНТА"

        issues[ticker] = {
            "n_total": n_total,
            "n_nan": n_nan,
            "n_zero": n_zero,
            "pct_no_change": pct_no_change,
            "n_flat_windows_20d": n_flat_windows,
            "max_daily_return": max_return,
            "min_daily_return": min_return,
            "flag": flag.strip(),
        }

        status = "OK" if not flag else f"ПРОБЛЕМА:{flag}"
        print(f"  {ticker}: {status}")
        print(
            f"    NaN={n_nan}, zero={n_zero}, no_change={pct_no_change:.1%}, "
            f"max_ret={max_return:+.2%}, min_ret={min_return:+.2%}"
        )

    # Визуализация
    fig, ax = plt.subplots(figsize=(10, 4))
    for ticker in tickers:
        s = close_test[ticker]
        s_norm = s / s.iloc[0]
        ax.plot(close_test.index, s_norm, label=ticker, linewidth=1.2)
    ax.set_title("Close prices normalized (test 2022–2024), base=1.0", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized price")
    ax.legend()
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle="--")
    plt.tight_layout()
    path = os.path.join(output_dir, "close_prices_test.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"График сохранён: {path}")

    with open(os.path.join(output_dir, "data_quality.json"), "w", encoding="utf-8") as f:
        json.dump(issues, f, ensure_ascii=False, indent=2)

    bad_tickers = [t for t, v in issues.items() if v["flag"]]
    if bad_tickers:
        print(f"\nВЫВОД: проблемные тикеры — {bad_tickers}. Рассмотреть замену в DataConfig.")
    else:
        print("\nВЫВОД: данные чистые, явных проблем нет.")

    return issues, bad_tickers


# ---------------------------------------------------------------------------
# 3. Reward audit
# ---------------------------------------------------------------------------

def _log_return_reward(new_value: float, prev_value: float) -> float:
    """Log-return reward без clip. Более стабильный сигнал при сильных движениях."""
    if prev_value <= 0:
        return 0.0
    return float(np.log(max(new_value, 1e-8) / max(prev_value, 1e-8)))


def _sortino_reward(new_value: float, prev_value: float, downside_penalty: float = 2.0) -> float:
    """
    Простой Sortino-style reward:
    - положительные returns — без изменений
    - отрицательные returns — штрафуются с коэффициентом downside_penalty
    """
    r = (new_value - prev_value) / max(prev_value, 1e-8)
    if r < 0:
        r *= downside_penalty
    return float(np.clip(r, -1.0, 1.0))


def diagnose_reward(model: RecurrentPPO, bundle, env_cfg: EnvFactoryConfig, output_dir: str):
    """
    Прогоняет модель трижды с одинаковыми весами, но разными reward функциями.
    Не переобучает — только собирает статистику reward сигнала.
    Показывает: была ли проблема в clip или в самом агенте.
    """
    print("\n=== 3. REWARD AUDIT ===")

    reward_fns = {
        "clipped (текущий)": lambda nv, pv: float(np.clip(
            (nv - pv) / max(pv, 1e-8), -1.0, 1.0
        )),
        "log_return":        _log_return_reward,
        "sortino_style":     _sortino_reward,
    }

    results = {}

    for reward_name, reward_fn in reward_fns.items():
        env = build_env(
            close_prices=bundle.close_test,
            features=bundle.features_test,
            turbulence=bundle.turb_test,
            turbulence_threshold=bundle.threshold,
            short=True,
            cfg=env_cfg,
        )
        vec_env = DummyVecEnv([lambda: env])
        obs = vec_env.reset()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

        rewards_collected = []
        portfolio_history = [env_cfg.initial_balance]
        prev_value = env_cfg.initial_balance
        done = False

        while not done:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )
            obs, _, dones, infos = vec_env.step(action)
            episode_starts = dones

            # Берём portfolio_value из info и считаем reward по нашей формуле
            info = infos[0] if infos else {}
            new_value = info.get("portfolio_value", prev_value)
            r = reward_fn(new_value, prev_value)
            rewards_collected.append(r)
            portfolio_history.append(new_value)
            prev_value = new_value
            done = bool(dones[0])

        rewards_arr = np.array(rewards_collected)
        final_value = portfolio_history[-1]
        cr = (final_value / env_cfg.initial_balance - 1) * 100

        stats = {
            "reward_mean":   float(np.mean(rewards_arr)),
            "reward_std":    float(np.std(rewards_arr)),
            "reward_min":    float(np.min(rewards_arr)),
            "reward_max":    float(np.max(rewards_arr)),
            "pct_clipped":   float(np.mean(np.abs(rewards_arr) >= 0.999)) if "clipped" in reward_name else None,
            "final_value":   final_value,
            "cumulative_return_pct": cr,
        }
        results[reward_name] = stats

        clip_info = ""
        if stats["pct_clipped"] is not None:
            clip_info = f"  clip_rate={stats['pct_clipped']:.1%}"
        print(
            f"  [{reward_name}]: mean={stats['reward_mean']:+.5f}  "
            f"std={stats['reward_std']:.5f}  "
            f"CR={cr:+.2f}%{clip_info}"
        )

    # Вывод: если clip_rate высокий, это причина проблемы
    clipped_stats = results.get("clipped (текущий)", {})
    clip_rate = clipped_stats.get("pct_clipped", 0) or 0
    print()
    if clip_rate > 0.1:
        print(
            f"ВЫВОД: clip срабатывает в {clip_rate:.1%} шагов — "
            "сигнал обрезается. Переход на log_return улучшит обучение (фаза 2)."
        )
    else:
        print(
            f"ВЫВОД: clip срабатывает редко ({clip_rate:.1%}). "
            "Проблема не в reward clipping — смотреть на action collapse и данные."
        )

    # График
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = ["#888780", "#378ADD", "#1D9E75"]
    for ax, (name, stats), color in zip(axes, results.items(), colors):
        r_data = []  # нам нужен массив — пересчитаем быстро
        # просто рисуем распределение из stats — повторный прогон дорог,
        # поэтому только summary bars
        ax.bar(
            ["mean", "std", "|min|", "max"],
            [
                stats["reward_mean"],
                stats["reward_std"],
                abs(stats["reward_min"]),
                stats["reward_max"],
            ],
            color=color,
            alpha=0.8,
        )
        ax.set_title(name, fontsize=10)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_ylabel("value")
    plt.suptitle("Reward function comparison (same model weights)", fontsize=11)
    plt.tight_layout()
    path = os.path.join(output_dir, "reward_audit.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"График сохранён: {path}")

    with open(os.path.join(output_dir, "reward_audit.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1 diagnostics")
    parser.add_argument(
        "--model-dir",
        default="./models/xlstm_large",
        help="Папка с сохранённой моделью (должна содержать xlstm_large.zip)",
    )
    parser.add_argument("--model-name", default="xlstm_large")
    parser.add_argument("--data-file", default="moex_data.csv")
    parser.add_argument("--output-dir", default="./diagnostics")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_global_seed(args.seed)

    # Данные
    print("Загружаем данные...")
    data_cfg = DataConfig(
        tickers=DEFAULT_CONFIG.tickers,
        data_file=args.data_file,
    )
    bundle = prepare_datasets(data_cfg)

    env_cfg = EnvFactoryConfig()

    # Модель
    model_path = os.path.join(args.model_dir, f"{args.model_name}.zip")
    if not os.path.exists(model_path):
        # попробовать без .zip
        model_path = os.path.join(args.model_dir, args.model_name)
    print(f"Загружаем модель: {model_path}")

    dummy_env = build_env(
        close_prices=bundle.close_test,
        features=bundle.features_test,
        turbulence=bundle.turb_test,
        turbulence_threshold=bundle.threshold,
        short=True,
        cfg=env_cfg,
    )
    vec_env = DummyVecEnv([lambda: dummy_env])
    model = RecurrentPPO.load(model_path, env=vec_env)
    print("Модель загружена.")

    # Запускаем три диагностики
    actions_arr, action_stats, collapse = diagnose_action_collapse(model, bundle, env_cfg, args.output_dir)
    data_issues, bad_tickers = diagnose_data_quality(bundle, args.output_dir)
    reward_results = diagnose_reward(model, bundle, env_cfg, args.output_dir)

    # Итоговый отчёт
    report = {
        "action_collapse_detected": collapse,
        "bad_tickers": bad_tickers,
        "reward_clip_rate": reward_results.get("clipped (текущий)", {}).get("pct_clipped"),
        "recommendations": [],
    }
    if collapse:
        report["recommendations"].append(
            "Повысить ent_coef (0.02 -> 0.05) в TrainConfig — фаза 2"
        )
    if bad_tickers:
        report["recommendations"].append(
            f"Заменить тикеры {bad_tickers} на ликвидные альтернативы — фаза 2"
        )
    clip_rate = report["reward_clip_rate"] or 0
    if clip_rate > 0.1:
        report["recommendations"].append(
            "Заменить clipped reward на log_return в env.py — фаза 2"
        )

    report_path = os.path.join(args.output_dir, "phase1_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("ИТОГОВЫЙ ОТЧЁТ ФАЗЫ 1")
    print("=" * 60)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nВсе файлы сохранены в: {args.output_dir}")


if __name__ == "__main__":
    main()
