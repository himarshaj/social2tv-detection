"""Clean Qwen predictions.

Same rules as clean_gemma.py. Qwen stores the episode as a folder
(EpisodeName/000000-000001.jpg). Gold expects EpisodeName-000000-000001.jpg.
Do not use clean_gpt4o.py.

    python cleaning/clean_qwen.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_gpt4o import (
    REQUIRED,
    TYPE_COLUMNS,
    YES_NO_COLUMNS,
    check_otherplatform_instances,
    check_twitter_instances,
    clean_binary_column,
)

ROOT = Path(__file__).resolve().parent
MODEL = "qwen38_27b"
CHANNELS = ("cnn", "foxnews", "msnbc")


def episode_filename(value):
    parts = [part for part in str(value).strip().replace("\\", "/").split("/") if part]
    if len(parts) >= 2:
        episode, frame = parts[-2], parts[-1]
        if episode.endswith(".frames1fps"):
            episode = episode[: -len(".frames1fps")]
        prefix = episode + "-"
        if frame.startswith(prefix):
            return frame
        return prefix + frame
    name = parts[-1] if parts else ""
    return name.replace(".frames1fps-", "-")


def clean_qwen(df):
    missing = [col for col in REQUIRED if col not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {', '.join(missing)}")
    df = df.copy()
    df["filename"] = df["filename"].map(episode_filename)
    for column in YES_NO_COLUMNS:
        df = clean_binary_column(df, column)
    for column in TYPE_COLUMNS:
        df[column] = df[column].apply(check_twitter_instances)
        df[column] = df[column].apply(check_otherplatform_instances)
    return df


def run(channels=None, model=None):
    model = model or MODEL
    selected = list(CHANNELS if channels is None else channels)
    src_dir = ROOT / "inputs" / "predictions" / model
    out_dir = ROOT / "outputs" / model
    out_dir.mkdir(parents=True, exist_ok=True)
    wrote = []
    for channel in selected:
        src = src_dir / f"{channel}.csv"
        if not src.exists():
            print(f"{model}: no raw file for {channel}, skipping", file=sys.stderr)
            continue
        df = pd.read_csv(src, keep_default_na=False)
        cleaned = clean_qwen(df)
        dest = out_dir / f"{channel}.csv"
        cleaned.to_csv(dest, index=False)
        print(f"Wrote {dest} ({len(cleaned)} rows)")
        print(f"  filename sample: {cleaned['filename'].iloc[0]}")
        wrote.append(channel)
    if not wrote:
        raise SystemExit(f"No Qwen CSVs cleaned from {src_dir}")
    return wrote


def main():
    argv = sys.argv[1:]
    model = MODEL
    if argv and argv[0] in ("qwen38_27b", "qwen_all"):
        model = argv[0]
        argv = argv[1:]
    run(argv or None, model=model)


if __name__ == "__main__":
    main()
