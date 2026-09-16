#!/usr/bin/env python3
"""Print the four result sections from evaluation outputs.

1. All frames — Gemma and Qwen
2. ImageHash ablation — all configurations — Gemma and Qwen
3. Reduced frames — GPT-4o, Gemma, and Qwen
4. All frames vs reduced — Gemma and Qwen

Usage:
    python summarize_results.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
HASH = ROOT / "ImageHash_tvalues" / "outputs"


def fmt(df, cols=("precision", "recall", "f1")):
    show = df.copy()
    for col in cols:
        if col in show.columns:
            show[col] = show[col].map(lambda x: f"{x:.4f}")
    return show


def section(title, df):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
    if df.empty:
        print("(no data)")
        return
    print(fmt(df).to_string(index=False))


def main():
    det = pd.read_csv(OUT / "detection_metrics.csv")

    section(
        "1. All frames — Gemma and Qwen (logo and screenshot separately)",
        det[det["model"].isin(["gemma_all", "qwen_all"])][
            ["model", "channel", "task", "n", "precision", "recall", "f1"]
        ].sort_values(["model", "channel", "task"]),
    )

    hash_path = HASH / "hash_ablation_only.csv"
    if hash_path.exists():
        hab = pd.read_csv(hash_path)
        section(
            "2. ImageHash ablation — all configurations — Gemma and Qwen",
            hab[
                ["channel", "model", "task", "threshold", "strategy",
                 "retention_pct", "precision", "recall", "f1"]
            ].sort_values(["channel", "model", "task", "threshold", "strategy"]),
        )
    else:
        print("\n(ImageHash results not found — run ImageHash_tvalues/run_hash_eval.py)")

    section(
        "3. Reduced frames — GPT-4o, Gemma, and Qwen",
        det[det["model"].isin(["gpt4o", "gemma_4_31b", "qwen38_27b"])][
            ["model", "channel", "task", "n", "precision", "recall", "f1"]
        ].sort_values(["model", "channel", "task"]),
    )

    cmp_path = OUT / "all_vs_reduced_comparison.csv"
    if cmp_path.exists():
        section(
            "4. All frames vs reduced — Gemma and Qwen",
            pd.read_csv(cmp_path)[
                ["model_family", "channel", "task", "f1_all", "f1_reduced", "f1_delta",
                 "recall_all", "recall_reduced", "recall_delta"]
            ],
        )


if __name__ == "__main__":
    main()
