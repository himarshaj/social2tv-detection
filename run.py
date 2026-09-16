"""Clean, copy, and score. Comment out a model or channel to skip it.

    python run.py

Gold labels are read from inputs/gold/. Replace those files first if the
gold standard changed. This script does not edit them.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "cleaning"))

import clean_gemma
import clean_gpt4o
import clean_qwen
from evaluate import (
    OUT_DIR,
    evaluate_model,
    print_detection,
    print_platform_micro,
)

import pandas as pd

ROOT = Path(__file__).resolve().parent
CLEAN_OUT = ROOT / "cleaning" / "outputs"
PRED_IN = ROOT / "inputs" / "predictions"

# Comment out a line to skip that model or channel.
MODELS = [
    "gpt4o",
    "gemma_4_31b",
    "gemma_all",
    "qwen38_27b",
    "qwen_all",
]
CHANNELS = [
    "cnn",
    "foxnews",
    "msnbc",
]

CLEANERS = {
    "gpt4o": clean_gpt4o.run,
    "gemma_4_31b": clean_gemma.run,
    "gemma_all": lambda channels=None: clean_gemma.run(channels, model="gemma_all"),
    "qwen38_27b": clean_qwen.run,
    "qwen_all": lambda channels=None: clean_qwen.run(channels, model="qwen_all"),
}


def copy_cleaned(model, channels):
    copied = []
    dest_dir = PRED_IN / model
    dest_dir.mkdir(parents=True, exist_ok=True)
    for channel in channels:
        src = CLEAN_OUT / model / f"{channel}.csv"
        if not src.exists():
            print(f"{model}: no cleaned {channel}.csv, skipping", file=sys.stderr)
            continue
        dest = dest_dir / f"{channel}.csv"
        shutil.copy2(src, dest)
        print(f"Copied {dest.relative_to(ROOT)}")
        copied.append(channel)
    return copied


def main():
    unknown = [model for model in MODELS if model not in CLEANERS]
    if unknown:
        raise SystemExit(f"No cleaner for: {', '.join(unknown)}")

    ready = {}
    for model in MODELS:
        print(f"\n=== clean {model} ===")
        CLEANERS[model](CHANNELS)
        copied = copy_cleaned(model, CHANNELS)
        if copied:
            ready[model] = copied
        else:
            print(f"{model}: nothing to score", file=sys.stderr)

    if not ready:
        raise SystemExit("Nothing to evaluate")

    frames = [evaluate_model(model, channels) for model, channels in ready.items()]
    detection = pd.concat([item[0] for item in frames], ignore_index=True)
    platform = pd.concat([item[1] for item in frames], ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detection.to_csv(OUT_DIR / "detection_metrics.csv", index=False)
    platform.to_csv(OUT_DIR / "platform_metrics.csv", index=False)
    print_detection(detection)
    print_platform_micro(platform)
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
