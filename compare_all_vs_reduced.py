#!/usr/bin/env python3
"""Compare all-frames vs reduced-frame detection for Gemma and Qwen.

Reads outputs/detection_metrics.csv produced by run.py and writes
outputs/all_vs_reduced_comparison.csv with per-task F1 deltas.

Usage:
    python compare_all_vs_reduced.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DET = OUT / "detection_metrics.csv"

PAIRS = [
    ("gemma_all", "gemma_4_31b", "Gemma"),
    ("qwen_all", "qwen38_27b", "Qwen"),
]


def main():
    det = pd.read_csv(DET)
    rows = []
    for all_model, reduced_model, label in PAIRS:
        all_df = det[det["model"] == all_model].set_index(["channel", "task"])
        red_df = det[det["model"] == reduced_model].set_index(["channel", "task"])
        for (channel, task) in all_df.index.intersection(red_df.index):
            a = all_df.loc[(channel, task)]
            r = red_df.loc[(channel, task)]
            rows.append({
                "model_family": label,
                "all_frames_model": all_model,
                "reduced_model": reduced_model,
                "channel": channel,
                "task": task,
                "n_all": int(a["n"]),
                "n_reduced": int(r["n"]),
                "precision_all": a["precision"],
                "precision_reduced": r["precision"],
                "recall_all": a["recall"],
                "recall_reduced": r["recall"],
                "f1_all": a["f1"],
                "f1_reduced": r["f1"],
                "f1_delta": round(a["f1"] - r["f1"], 4),
                "precision_delta": round(a["precision"] - r["precision"], 4),
                "recall_delta": round(a["recall"] - r["recall"], 4),
            })

    out = pd.DataFrame(rows)
    out = out.sort_values(["model_family", "channel", "task"])
    out.to_csv(OUT / "all_vs_reduced_comparison.csv", index=False)

    print("All frames vs reduced (detection F1 by task)\n")
    show = out.copy()
    for col in ("f1_all", "f1_reduced", "f1_delta", "precision_all", "precision_reduced",
                "recall_all", "recall_reduced"):
        show[col] = show[col].map(lambda x: f"{x:.4f}")
    print(show.to_string(index=False))
    print(f"\nWrote {OUT / 'all_vs_reduced_comparison.csv'}")


if __name__ == "__main__":
    main()
