#!/usr/bin/env python3
"""ImageHash ablation vs gold with explicit negatives.

Scores each threshold x strategy x model x channel for logo, screenshot,
and combined (logo ∨ screenshot).

Dedup-preservation metric (hash study only; not the main benchmark F1):
  Precision = among retained frames with VLM=Yes, fraction gold-positive.
  Recall    = among gold-positive frames the all-frames VLM also labels Yes,
              fraction whose group representative remains VLM=Yes.
              Gold positives the full VLM misses are excluded from recall.

Also writes an all-frames baseline row per channel/model/task.

Usage:
    python run_hash_eval.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
GOLD_DIR = ROOT / "gold"
OUT_DIR = ROOT / "outputs"

CHANNELS = {
    "CNN": {"gold": "cnn.csv", "dir": "CNN", "meta": "CNN"},
    "FOXNEWS": {"gold": "foxnews.csv", "dir": "FOXNEWS", "meta": "FOXNEWS"},
    "MSNBC": {"gold": "msnbc.csv", "dir": "MSNBC", "meta": "MSNBC"},
}

MODELS = {
    "gemma_4_31b": "*gemma_4_31b.csv",
    "qwen38_27b": "*qwen_27b.csv",
}

TASKS = ("logo", "screenshot", "combined")


def is_yes(series):
    return series.astype(str).str.strip().str.lower().eq("yes")


def rates(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def parse_gold(path):
    gold = pd.read_csv(path)
    base = gold["filename"].astype(str).str.strip().replace("\\", "/").str.lstrip("/")
    gold = gold.copy()
    gold["show"] = base.str.replace(r"-(\d+)(-\d+)?\.jpg$", "", regex=True)
    gold["frame_id"] = base.str.extract(r"-(\d+)(?:-\d+)?\.jpg$")[0].astype(int)
    return gold


def load_vlm(channel_dir, pattern):
    files = sorted(channel_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No VLM CSVs matching {pattern} in {channel_dir}")
    vlm = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    parts = vlm["filename"].astype(str).str.replace("\\", "/").str.strip("/").str.split("/")
    vlm = vlm.copy()
    vlm["show"] = parts.str[-2].str.replace(r"\.frames1fps$", "", regex=True)
    vlm["frame_id"] = parts.str[-1].str.replace(r"\.jpg$", "", regex=True).astype(int)
    return vlm


def task_pred_series(vlm, task):
    logo = is_yes(vlm["Social Media Logo"])
    shot = is_yes(vlm["Social Media Post Screenshot"])
    if task == "logo":
        return logo
    if task == "screenshot":
        return shot
    return logo | shot


def task_gold_pos(gold, task):
    logo = is_yes(gold["Social Media Logo"])
    shot = is_yes(gold["Social Media Screenshot"])
    if task == "logo":
        yes = logo
    elif task == "screenshot":
        yes = shot
    else:
        yes = logo | shot
    return {(show, int(frame)) for show, frame, hit in zip(gold["show"], gold["frame_id"], yes) if hit}


def task_pred_map(vlm, task):
    yes = task_pred_series(vlm, task)
    return {
        (show, int(frame)): bool(hit)
        for show, frame, hit in zip(vlm["show"], vlm["frame_id"], yes)
    }


def score_setting(meta, pred, gold_pos, oracle, thr, strat):
    sub = meta[(meta.threshold == thr) & (meta.strategy == strat)]
    n_retained = len(sub)
    n_raw = int(sub["group_size"].sum())
    retention_pct = 100.0 * n_retained / n_raw if n_raw else 0.0

    frame_pred = {}
    retained = []
    for show, gs, ge, rf in zip(
        sub["show"], sub["group_start"], sub["group_end"], sub["representative_frame"]
    ):
        gs, ge, rf = int(gs), int(ge), int(rf)
        rp = pred.get((show, rf), False)
        retained.append((show, rf, rp))
        for frame in range(gs, ge + 1):
            frame_pred[(show, frame)] = rp

    ret_tp = sum(1 for show, rf, rp in retained if rp and (show, rf) in gold_pos)
    ret_fp = sum(1 for show, rf, rp in retained if rp and (show, rf) not in gold_pos)
    precision, _, _ = rates(ret_tp, ret_fp, 0)

    preserved = sum(1 for key in oracle if frame_pred.get(key, False))
    recall = preserved / len(oracle) if oracle else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "threshold": thr,
        "strategy": strat,
        "n_raw": n_raw,
        "n_retained": n_retained,
        "retention_pct": round(retention_pct, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "retained_tp": ret_tp,
        "retained_fp": ret_fp,
        "oracle_preserved": preserved,
        "oracle_size": len(oracle),
        "n_gold_pos": len(gold_pos),
    }


def all_frames_baseline(vlm, gold, task, shows):
    pred = task_pred_map(vlm, task)
    gold_pos = {key for key in task_gold_pos(gold, task) if key[0] in shows}
    yes = task_pred_series(vlm, task)

    tp = fp = fn = tn = 0
    for show, frame, hit in zip(vlm["show"], vlm["frame_id"], yes):
        if show not in shows:
            continue
        key = (show, int(frame))
        gold_yes = key in gold_pos
        if hit and gold_yes:
            tp += 1
        elif hit and not gold_yes:
            fp += 1
        elif not hit and gold_yes:
            fn += 1
        else:
            tn += 1

    precision, recall, f1 = rates(tp, fp, fn)
    return {
        "threshold": "all",
        "strategy": "all",
        "n_raw": tp + fp + fn + tn,
        "n_retained": tp + fp + fn + tn,
        "retention_pct": 100.0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "retained_tp": tp,
        "retained_fp": fp,
        "oracle_preserved": tp,
        "oracle_size": tp + fn,
        "n_gold_pos": len(gold_pos),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def evaluate_channel_model(channel_name, cfg, model_name, pattern):
    gold = parse_gold(GOLD_DIR / cfg["gold"])
    vlm = load_vlm(ROOT / cfg["dir"], pattern)
    shows = sorted(set(vlm["show"]) & set(gold["show"]))

    meta = pd.read_csv(ROOT / "group_metadata.csv")
    meta = meta[meta["channel"] == cfg["meta"]].copy()
    meta["show"] = meta["recording"].str.replace(r"\.frames1fps\.tar$", "", regex=True)
    meta = meta[meta["show"].isin(shows)].copy()

    rows = []
    for task in TASKS:
        pred = task_pred_map(vlm, task)
        gold_pos = {key for key in task_gold_pos(gold, task) if key[0] in shows}
        oracle = {key for key in gold_pos if pred.get(key)}

        rows.append({
            "channel": channel_name,
            "model": model_name,
            "task": task,
            **all_frames_baseline(vlm, gold, task, shows),
        })

        for thr in (3, 4, 5):
            for strat in ("first", "middle", "last"):
                row = score_setting(meta, pred, gold_pos, oracle, thr, strat)
                rows.append({
                    "channel": channel_name,
                    "model": model_name,
                    "task": task,
                    **row,
                })

    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for channel_name, cfg in CHANNELS.items():
        for model_name, pattern in MODELS.items():
            print(f"\n=== {channel_name} / {model_name} ===")
            rows = evaluate_channel_model(channel_name, cfg, model_name, pattern)
            all_rows.extend(rows)
            ablation = [r for r in rows if r["threshold"] != "all"]
            for task in TASKS:
                best = max(
                    (r for r in ablation if r["task"] == task),
                    key=lambda r: (r["f1"], -r["retention_pct"]),
                )
                print(
                    f"  {task}: best t={best['threshold']} {best['strategy']} "
                    f"P={best['precision']:.4f} R={best['recall']:.4f} F1={best['f1']:.4f}"
                )

    out = pd.DataFrame(all_rows)
    out.to_csv(OUT_DIR / "hash_ablation_results.csv", index=False)

    ablation = out[out["threshold"] != "all"].copy()
    ablation.to_csv(OUT_DIR / "hash_ablation_only.csv", index=False)

    baseline = out[out["threshold"] == "all"].copy()
    baseline.to_csv(OUT_DIR / "hash_all_frames_baseline.csv", index=False)

    for channel in CHANNELS:
        sub = ablation[ablation["channel"] == channel]
        sub.to_csv(OUT_DIR / f"hash_ablation_{channel.lower()}.csv", index=False)

    print(f"\nWrote {OUT_DIR / 'hash_ablation_results.csv'} ({len(out)} rows)")


if __name__ == "__main__":
    main()
