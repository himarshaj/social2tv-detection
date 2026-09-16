"""Score logo and screenshot predictions against the fixed gold labels.

Detection uses the Yes/No column. Platform identification uses the type
column. A Yes with no platform name is a failed identification: it counts
as a false positive, not as a non-prediction.

Layout
------
inputs/gold/{cnn,foxnews,msnbc}.csv
inputs/predictions/<model>/{cnn,foxnews,msnbc}.csv
outputs/<model>/detection.csv
outputs/<model>/platform_identification.csv

The gpt4o files already in inputs/predictions/gpt4o are the Run 5
all-episode results used for the reported tables:
  cnn     chatgpt40_CNN_t3_v4_cleaned.csv
          against gold_standard_images_cnn_t3_v4_.csv
  foxnews chatgpt40_FOXNEWS_t3_v3_cleaned.csv
          against gold_standard_images_foxnews_t3_v3_.csv
  msnbc   chatgpt40_MSNBC_t3_v3_cleaned.csv
          against gold_standard_images_msnbc_t3_v3_.csv

To score another model, add a folder with those three filenames and the
columns below, then run:

    python evaluate.py newmodel

With no arguments, every folder under inputs/predictions is scored.

Prediction columns
------------------
filename
Social Media Logo                 Yes or No
Social Media Logo Type            comma-separated platforms, blank if none
Social Media Post Screenshot      Yes or No
Social Media Screenshot Type      comma-separated platforms, blank if none

Gold columns
------------
filename
Social Media Logo
Social Media Logo Type
Social Media Screenshot
Social Media Screenshot Type

Rows are joined on the filename basename. A prediction with no gold row is
a negative. Gold rows that do not match any prediction are reported and
excluded, because a missing prediction cannot be scored.

Internet Archive repeats one minute at each program boundary. The frames at
or after the next program's start are that program's opening, already stored
as its first minute. When both recordings are in the prediction file, the
earlier copy is dropped. A gold label that exists only on the dropped copy is
moved onto the next program so the appearance is still scored once.
"""

import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
GOLD_DIR = ROOT / "inputs" / "gold"
PRED_DIR = ROOT / "inputs" / "predictions"
OUT_DIR = ROOT / "outputs"

CHANNELS = ("cnn", "foxnews", "msnbc")
CHANNEL_NAMES = {"cnn": "CNN", "foxnews": "FOX News", "msnbc": "MSNBC"}

TASKS = (
    {
        "task": "logo",
        "pred_yes": "logo_yes_pred",
        "gold_yes": "logo_yes_gold",
        "pred_type": "logo_type_pred",
        "gold_type": "logo_type_gold",
    },
    {
        "task": "screenshot",
        "pred_yes": "screenshot_yes_pred",
        "gold_yes": "screenshot_yes_gold",
        "pred_type": "screenshot_type_pred",
        "gold_type": "screenshot_type_gold",
    },
)

EMPTY_LABELS = {"n/a", "na", "none", "-", "nan"}
UNIDENTIFIED = "unidentified"


def basename(value):
    return str(value).strip().replace("\\", "/").split("/")[-1]


def is_yes(value):
    return isinstance(value, str) and value.strip().lower() == "yes"


def parse_labels(value):
    if not isinstance(value, str):
        return []
    labels = []
    for part in value.split(","):
        label = part.strip().lower()
        if label and label not in EMPTY_LABELS:
            labels.append(label)
    return list(dict.fromkeys(labels))


def rates(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def require_columns(df, columns, path):
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing columns: {', '.join(missing)}")


SPAN_NAME = re.compile(r"^(.*?)-(\d+)(?:-(\d+))?\.jpg$")
EPISODE_ID = re.compile(r"^(?P<stem>.+?)_(?P<date>\d{8})_(?P<tod>\d{6})_(?P<show>.+)$")


def parse_span(name):
    match = SPAN_NAME.match(str(name))
    if not match:
        return None
    start = int(match.group(2))
    end = int(match.group(3) or start)
    return match.group(1), start, end


def episodes_in(bases):
    found = []
    seen = set()
    for base in bases:
        span = parse_span(base)
        if span is None or span[0] in seen:
            continue
        seen.add(span[0])
        found.append(span[0])
    return found


def next_program(episodes):
    """Map each episode to (seconds until the next, next episode) on its station.

    The next item in the file is the following program only when it is the
    next recording on that station. A skipped hour has a gap larger than any
    frame offset, so nothing is dropped for it.
    """
    parsed = []
    for episode in episodes:
        match = EPISODE_ID.match(episode)
        if not match:
            continue
        started = datetime.strptime(match.group("date") + match.group("tod"), "%Y%m%d%H%M%S")
        parsed.append((match.group("stem"), started, episode))
    parsed.sort()
    nxt = {}
    for (stem, started, episode), (stem2, started2, episode2) in zip(parsed, parsed[1:]):
        if stem != stem2:
            continue
        gap = int((started2 - started).total_seconds())
        if gap > 0:
            nxt[episode] = (gap, episode2)
    return nxt


def rewrite_base(base, episode, start, end):
    match = SPAN_NAME.match(str(base))
    width = len(match.group(2))
    start_txt = f"{start:0{width}d}"
    if match.group(3) is None:
        return f"{episode}-{start_txt}.jpg"
    return f"{episode}-{start_txt}-{end:0{width}d}.jpg"


def drop_overlap_tails(pred):
    """Drop spans that begin at or after the next program's start.

    Those frames are the next program's opening. The next program's own
    first minute is the copy that is scored.
    """
    nxt = next_program(episodes_in(pred["base"]))
    if not nxt:
        return pred, 0
    keep = []
    for base in pred["base"]:
        span = parse_span(base)
        if span is None or span[0] not in nxt:
            keep.append(True)
            continue
        gap, _ = nxt[span[0]]
        keep.append(span[1] < gap)
    keep = pd.Series(keep, index=pred.index)
    return pred.loc[keep].copy(), int((~keep).sum())


def dedupe_gold_overlap(gold, pred):
    """Count a boundary gold label once, on the next program.

    The prediction file decides which copy is kept. If the next program
    already labels that time, the earlier label is dropped. If only the
    earlier recording is labeled, the label is moved onto the next program
    so it is still scored against the kept prediction.
    """
    nxt = next_program(episodes_in(pred["base"]))
    if not nxt or gold.empty:
        return gold, 0, 0

    spans = gold["base"].map(parse_span)
    by_episode = {}
    for idx, span in spans.items():
        if span is None:
            continue
        by_episode.setdefault(span[0], []).append((span[1], span[2]))

    drop = []
    moves = []
    for idx, span in spans.items():
        if span is None or span[0] not in nxt:
            continue
        episode, start, end = span
        gap, following = nxt[episode]
        if start < gap:
            continue
        shifted_start = start - gap
        shifted_end = end - gap
        already = any(
            frame_start <= shifted_end and shifted_start <= frame_end
            for frame_start, frame_end in by_episode.get(following, [])
        )
        if already:
            drop.append(idx)
            continue
        new_base = rewrite_base(gold.at[idx, "base"], following, shifted_start, shifted_end)
        moves.append((idx, new_base))
        by_episode.setdefault(following, []).append((shifted_start, shifted_end))

    out = gold.drop(index=drop).copy()
    for idx, new_base in moves:
        out.at[idx, "base"] = new_base
        if "filename" in out.columns:
            out.at[idx, "filename"] = new_base
    return out, len(drop), len(moves)


def union_types(values):
    seen = []
    for value in values:
        if not isinstance(value, str):
            continue
        for part in value.split(","):
            label = part.strip()
            if label and label.lower() not in EMPTY_LABELS and label not in seen:
                seen.append(label)
    return ", ".join(seen)


def aggregate_gold_into_spans(pred, gold):
    """Attach gold labels to prediction rows that cover or sit inside them.

    Reduced predictions are often wider than a gold frame: a span is Yes if
    any contained gold frame is Yes. All-frame predictions are one second
    wide: a gold span such as 136-141 labels every second inside it. Spans
    with no gold stay unlabeled and count as negatives.
    """
    gold = gold[gold["base"].notna() & gold["base"].ne("") & gold["base"].ne("nan")].copy()
    pred_spans = pred["base"].map(parse_span)
    gold_spans = gold["base"].map(parse_span)
    if pred_spans.isna().any() or gold_spans.isna().all():
        return None

    by_episode = {}
    for idx, span in gold_spans.items():
        if span is None:
            continue
        episode, start, end = span
        by_episode.setdefault(episode, []).append((start, end, idx))

    rows = []
    covered_gold = set()
    for base, span in zip(pred["base"], pred_spans):
        if span is None:
            continue
        episode, start, end = span
        indexes = []
        for frame_start, frame_end, idx in by_episode.get(episode, []):
            gold_in_pred = start <= frame_start and frame_end <= end
            pred_in_gold = frame_start <= start and end <= frame_end
            if gold_in_pred or pred_in_gold:
                indexes.append(idx)
        if not indexes:
            continue
        covered_gold.update(indexes)
        frame_df = gold.loc[indexes]
        logo_yes = frame_df["Social Media Logo"].fillna("").astype(str).str.strip().str.lower().eq("yes")
        shot_yes = frame_df["Social Media Screenshot"].fillna("").astype(str).str.strip().str.lower().eq("yes")
        rows.append({
            "base": base,
            "logo_yes_gold": "Yes" if logo_yes.any() else "No",
            "logo_type_gold": union_types(frame_df.loc[logo_yes, "Social Media Logo Type"]),
            "screenshot_yes_gold": "Yes" if shot_yes.any() else "No",
            "screenshot_type_gold": union_types(frame_df.loc[shot_yes, "Social Media Screenshot Type"]),
        })
    if not covered_gold:
        return None
    return pd.DataFrame(rows), len(covered_gold)

def load_channel(model, channel):
    gold_path = GOLD_DIR / f"{channel}.csv"
    pred_path = PRED_DIR / model / f"{channel}.csv"
    if not gold_path.exists():
        raise SystemExit(f"Missing gold file: {gold_path}")
    if not pred_path.exists():
        raise SystemExit(f"Missing prediction file: {pred_path}")

    gold = pd.read_csv(gold_path)
    pred = pd.read_csv(pred_path)
    require_columns(
        gold,
        ["filename", "Social Media Logo", "Social Media Logo Type",
         "Social Media Screenshot", "Social Media Screenshot Type"],
        gold_path,
    )
    require_columns(
        pred,
        ["filename", "Social Media Logo", "Social Media Logo Type",
         "Social Media Post Screenshot", "Social Media Screenshot Type"],
        pred_path,
    )

    gold = gold.copy()
    pred = pred.copy()
    gold["base"] = gold["filename"].map(basename)
    pred["base"] = pred["filename"].map(basename)
    gold = gold[gold["base"].notna() & gold["base"].ne("") & gold["base"].ne("nan")]
    if pred["base"].duplicated().any():
        raise SystemExit(f"{channel}: duplicate prediction filenames after taking the basename")

    pred, dropped_pred = drop_overlap_tails(pred)
    gold, dropped_gold, moved_gold = dedupe_gold_overlap(gold, pred)
    print(f"  {CHANNEL_NAMES[channel]}: shared minute counted on the next program "
          f"({dropped_pred} prediction spans dropped, "
          f"{dropped_gold} gold frames already there, {moved_gold} gold frames moved)")

    matched = int(gold["base"].isin(set(pred["base"])).sum())
    if gold["base"].duplicated().any():
        matched = 0
    span_gold = None
    if matched < len(gold):
        span_gold = aggregate_gold_into_spans(pred, gold)
    if span_gold is not None:
        span_df, covered = span_gold
        print(f"  {CHANNEL_NAMES[channel]}: {len(pred)} predictions, "
              f"{covered}/{len(gold)} gold frames mapped into {len(span_df)} spans")
    elif matched == 0:
        raise SystemExit(f"{channel}: no gold filenames matched the predictions")
    else:
        print(f"  {CHANNEL_NAMES[channel]}: {len(pred)} predictions, "
              f"{matched}/{len(gold)} gold rows matched")

    pred = pred.rename(columns={
        "Social Media Logo": "logo_yes_pred",
        "Social Media Logo Type": "logo_type_pred",
        "Social Media Post Screenshot": "screenshot_yes_pred",
        "Social Media Screenshot Type": "screenshot_type_pred",
    })
    gold = gold.rename(columns={
        "Social Media Logo": "logo_yes_gold",
        "Social Media Logo Type": "logo_type_gold",
        "Social Media Screenshot": "screenshot_yes_gold",
        "Social Media Screenshot Type": "screenshot_type_gold",
    })
    keep_pred = ["base", "logo_yes_pred", "logo_type_pred",
                 "screenshot_yes_pred", "screenshot_type_pred"]
    keep_gold = ["base", "logo_yes_gold", "logo_type_gold",
                 "screenshot_yes_gold", "screenshot_type_gold"]
    if span_gold is not None:
        gold_aligned = span_gold[0]
    else:
        gold_aligned = gold[keep_gold]
    merged = pred[keep_pred].merge(gold_aligned, on="base", how="left")
    return merged


def detection_row(model, channel, task, y_true, y_pred):
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    precision, recall, f1 = rates(tp, fp, fn)
    return {
        "model": model,
        "channel": CHANNEL_NAMES[channel],
        "task": task,
        "n": len(y_true),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def label_sets(frame, yes_col, type_col, penalize_blank_yes):
    sets = []
    for yes_value, type_value in zip(frame[yes_col], frame[type_col]):
        labels = parse_labels(type_value)
        if penalize_blank_yes and is_yes(yes_value) and not labels:
            labels = [UNIDENTIFIED]
        sets.append(set(labels))
    return sets


def platform_rows(model, channel, task, gold_sets, pred_sets):
    labels = sorted((set().union(*gold_sets) | set().union(*pred_sets)) - {UNIDENTIFIED})
    rows = []
    micro_tp = micro_fp = micro_fn = 0

    def add(label, tp, fp, fn):
        precision, recall, f1 = rates(tp, fp, fn)
        rows.append({
            "model": model,
            "channel": CHANNEL_NAMES[channel],
            "task": task,
            "label": label,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        })

    for label in labels:
        tp = sum(label in gold and label in pred for gold, pred in zip(gold_sets, pred_sets))
        fp = sum(label not in gold and label in pred for gold, pred in zip(gold_sets, pred_sets))
        fn = sum(label in gold and label not in pred for gold, pred in zip(gold_sets, pred_sets))
        add(label, tp, fp, fn)
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

    unid_fp = sum(UNIDENTIFIED in pred for pred in pred_sets)
    add(UNIDENTIFIED, 0, unid_fp, 0)
    micro_fp += unid_fp
    add("micro", micro_tp, micro_fp, micro_fn)
    return rows


def evaluate_model(model, channels=None):
    print(f"\n{model}")
    detection = []
    platform = []
    for channel in (CHANNELS if channels is None else channels):
        merged = load_channel(model, channel)
        for spec in TASKS:
            y_true = merged[spec["gold_yes"]].map(is_yes)
            y_pred = merged[spec["pred_yes"]].map(is_yes)
            detection.append(detection_row(model, channel, spec["task"], y_true, y_pred))

            gold_sets = label_sets(merged, spec["gold_yes"], spec["gold_type"], False)
            pred_sets = label_sets(merged, spec["pred_yes"], spec["pred_type"], True)
            platform.extend(platform_rows(model, channel, spec["task"], gold_sets, pred_sets))

    detection_df = pd.DataFrame(detection)
    platform_df = pd.DataFrame(platform)
    out = OUT_DIR / model
    out.mkdir(parents=True, exist_ok=True)
    detection_df.to_csv(out / "detection.csv", index=False)
    platform_df.to_csv(out / "platform_identification.csv", index=False)
    return detection_df, platform_df


def print_detection(df):
    show = df.copy()
    for col in ("accuracy", "precision", "recall", "f1"):
        show[col] = show[col].map(lambda x: f"{x:.4f}")
    print("\nDetection (Yes/No)")
    print(show.to_string(index=False))


def print_platform_micro(df):
    micro = df[df["label"] == "micro"][
        ["model", "channel", "task", "tp", "fp", "fn", "precision", "recall", "f1", "support"]
    ].copy()
    for col in ("precision", "recall", "f1"):
        micro[col] = micro[col].map(lambda x: f"{x:.4f}")
    print("\nPlatform identification (micro, blank Yes counted as a failure)")
    print(micro.to_string(index=False))


def models_to_run(argv):
    if argv:
        return argv
    if not PRED_DIR.exists():
        raise SystemExit(f"No predictions folder: {PRED_DIR}")
    found = sorted(path.name for path in PRED_DIR.iterdir() if path.is_dir())
    if not found:
        raise SystemExit(f"Put a model folder in {PRED_DIR}")
    return found


def main(argv, channels=None):
    frames = [evaluate_model(model, channels) for model in models_to_run(argv)]
    detection = pd.concat([item[0] for item in frames], ignore_index=True)
    platform = pd.concat([item[1] for item in frames], ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detection.to_csv(OUT_DIR / "detection_metrics.csv", index=False)
    platform.to_csv(OUT_DIR / "platform_metrics.csv", index=False)
    print_detection(detection)
    print_platform_micro(platform)
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])
