"""Clean gpt4o predictions only.

Previous filename: cleaning_chatgpt_output.py, then clean_predictions.py.
Used on the Run 5 files, for example chatgpt40_CNN_t3_v4.csv. The old script
stripped one machine-specific path prefix. This version keeps the filename
basename, which is already EpisodeName-000000-000001.jpg for gpt4o.

Do not use this on Gemma or Qwen. Those store the episode as a folder.

    python cleaning/clean_gpt4o.py

Required columns:
  filename
  Social Media Logo
  Social Media Logo Type
  Social Media Post Screenshot
  Social Media Screenshot Type

Yes/No phrasing such as "Answer: Yes" or "N/A" is mapped to Yes or No.
Platform names are collapsed to the gold spelling: "Twitter (bird logo)" to
"Twitter", standalone "X" to "Twitter (X)", "Facebook (stylized)" to
"Facebook". Anything else in a Yes/No column is left unchanged and printed.
"""

import re
import sys
from pathlib import Path

import pandas as pd

REQUIRED = [
    "filename",
    "Social Media Logo",
    "Social Media Logo Type",
    "Social Media Post Screenshot",
    "Social Media Screenshot Type",
]
YES_NO_COLUMNS = ["Social Media Logo", "Social Media Post Screenshot"]
TYPE_COLUMNS = ["Social Media Logo Type", "Social Media Screenshot Type"]


def clean_binary_column(df, column_name):
    df[column_name] = df[column_name].replace(
        ["N/A", "- N/A", "- Answer: No", "Answer: N/A", "- Answer: N/A"],
        "No",
    )
    df[column_name] = df[column_name].replace(
        ["- Answer: Yes", "Answer: Yes"],
        "Yes",
    )
    values = df[column_name].astype(str).str.strip().str.lower()
    unexpected = sorted(set(values[~values.isin(["yes", "no"])]))
    if unexpected:
        print(f"Unexpected values still in {column_name}: {unexpected}", file=sys.stderr)
    return df


def replace_exact_x(value):
    if "Twitter (X)" in value:
        return value
    return re.sub(r"\bX\b", "Twitter (X)", value)


def check_twitter_instances(value):
    if pd.isna(value):
        return value
    original = str(value).strip()
    lower = original.lower()
    has_twitter = "twitter (bird logo)" in lower
    has_x = (
        "x (x logo)" in lower
        or "twitter (x logo)" in lower
        or re.search(r"\bX\b", original)
    )
    if has_twitter:
        original = original.replace("Twitter (bird logo)", "Twitter")
    if has_x:
        original = original.replace("X (X logo)", "Twitter (X)")
        original = original.replace("Twitter (X logo)", "Twitter (X)")
        original = replace_exact_x(original)
    return original


def check_otherplatform_instances(value):
    if pd.isna(value):
        return value
    original = str(value).strip()
    original = original.replace("- N/A", "N/A")
    original = original.replace("Facebook (stylized)", "Facebook")
    return original


def clean(df):
    missing = [col for col in REQUIRED if col not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {', '.join(missing)}")
    df = df.copy()
    df["filename"] = df["filename"].astype(str).str.strip().str.replace("\\", "/", regex=False)
    df["filename"] = df["filename"].str.split("/").str[-1]
    for column in YES_NO_COLUMNS:
        df = clean_binary_column(df, column)
    for column in TYPE_COLUMNS:
        df[column] = df[column].apply(check_twitter_instances)
        df[column] = df[column].apply(check_otherplatform_instances)
    return df


ROOT = Path(__file__).resolve().parent
MODEL = "gpt4o"
PRED_DIR = ROOT / "inputs" / "predictions" / MODEL
OUT_DIR = ROOT / "outputs" / MODEL
CHANNELS = ("cnn", "foxnews", "msnbc")


def clean_file(src, dest):
    df = pd.read_csv(src, keep_default_na=False)
    dest.parent.mkdir(parents=True, exist_ok=True)
    clean(df).to_csv(dest, index=False)
    print(f"Wrote {dest}")


def run(channels=None):
    selected = list(CHANNELS if channels is None else channels)
    if not PRED_DIR.is_dir():
        raise SystemExit(f"No raw folder: {PRED_DIR}")
    wrote = []
    for name in selected:
        src = PRED_DIR / f"{name}.csv"
        if not src.exists():
            print(f"{MODEL}: no raw file for {name}, skipping", file=sys.stderr)
            continue
        dest = OUT_DIR / f"{name}.csv"
        clean_file(src, dest)
        wrote.append(name)
    if not wrote:
        raise SystemExit(f"{MODEL}: nothing cleaned")
    return wrote


def main():
    run(sys.argv[1:] or None)


if __name__ == "__main__":
    main()
