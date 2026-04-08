from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

import matplotlib.pyplot as plt
import pandas as pd

from data import DEFAULT_CONFIG, DataConfig
from train import TrainConfig, train_model


EXPERIMENT_ORDER = [
    "lstm",
    "xlstm_base",
    "xlstm_attn",
    "xlstm_large",
]


def run_experiments(data_cfg: DataConfig, train_cfg: TrainConfig, model_names=None, output_dir: str = "./artifacts"):
    os.makedirs(output_dir, exist_ok=True)
    model_names = model_names or EXPERIMENT_ORDER

    results = []
    portfolios = {}
    for model_name in model_names:
        print("=" * 72)
        result = train_model(model_name, data_cfg=data_cfg, train_cfg=train_cfg, run_test=True)
        row = {"model": model_name, **(result["metrics"] or {})}
        results.append(row)
        portfolios[model_name] = result["portfolio_history"]

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(["SR", "CR"], ascending=[False, False]).reset_index(drop=True)

    csv_path = os.path.join(output_dir, "comparison_metrics.csv")
    json_path = os.path.join(output_dir, "comparison_metrics.json")
    plot_path = os.path.join(output_dir, "comparison_plot.png")
    portfolio_plot_path = os.path.join(output_dir, "portfolio_curves.png")

    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    if not df.empty:
        plt.figure(figsize=(10, 5))
        plt.bar(df["model"], df["SR"])
        plt.title("Sharpe Ratio by Model")
        plt.xlabel("Model")
        plt.ylabel("Sharpe Ratio")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

        plt.figure(figsize=(10, 5))
        for model_name, hist in portfolios.items():
            if hist is not None:
                plt.plot(hist, label=model_name)
        plt.title("Portfolio Curves on Test")
        plt.xlabel("Step")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(portfolio_plot_path, dpi=150)
        plt.close()

    print("\nИтоговая таблица:")
    print(df.to_string(index=False))
    print(f"\nСохранено:\n- {csv_path}\n- {json_path}\n- {plot_path}\n- {portfolio_plot_path}")
    return df, {
        "csv": csv_path,
        "json": json_path,
        "plot": plot_path,
        "portfolio_plot": portfolio_plot_path,
    }


def cli():
    parser = argparse.ArgumentParser(description="Run full MOEX model comparison")
    parser.add_argument("--models", nargs="*", default=EXPERIMENT_ORDER)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage1-steps", type=int, default=300_000)
    parser.add_argument("--stage2-steps", type=int, default=700_000)
    parser.add_argument("--save-dir", default="./models")
    parser.add_argument("--tensorboard-log", default="./tb_logs")
    parser.add_argument("--output-dir", default="./artifacts")
    parser.add_argument("--data-file", default="moex_data.csv")
    args = parser.parse_args()

    data_cfg = DataConfig(**{**asdict(DEFAULT_CONFIG), "data_file": args.data_file})
    train_cfg = TrainConfig(
        seed=args.seed,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        save_dir=args.save_dir,
        tensorboard_log=args.tensorboard_log,
    )
    run_experiments(data_cfg=data_cfg, train_cfg=train_cfg, model_names=args.models, output_dir=args.output_dir)


if __name__ == "__main__":
    cli()
