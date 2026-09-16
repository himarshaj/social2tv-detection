# ICWSM evaluation (release)

## Layout

```text
.
├── evaluate.py                 Score predictions vs gold
├── compare_all_vs_reduced.py   All-frames vs reduced comparison table
├── summarize_results.py        Print summary tables from outputs/
├── run.py                      Full clean+score pipeline (uses cleaning/)
├── inputs/
│   ├── gold/                   cnn.csv, foxnews.csv, foxnews_static.csv, msnbc.csv, readme.txt
│   └── predictions/<model>/    Per-channel prediction CSVs
├── outputs/
│   ├── detection_metrics.csv
│   ├── platform_metrics.csv
│   ├── all_vs_reduced_comparison.csv
│   └── <model>/{detection,platform_identification}.csv
├── cleaning/
│   ├── clean_gemma.py
│   ├── clean_gpt4o.py
│   └── clean_qwen.py
└── ImageHash_tvalues/
    ├── run_hash_eval.py
    ├── reduce_image_frames.py
    ├── group_metadata.csv
    ├── gold/
    ├── {CNN,FOXNEWS,MSNBC}/    Episode prediction CSVs
    └── outputs/hash_*.csv
```

Models: `gpt4o`, `gemma_4_31b`, `qwen38_27b`, `gemma_all`, `qwen_all`.

`cleaning/*.py` is for the `run.py` pipeline; `evaluate.py` scores `inputs/predictions` directly.

## Run

From the repo root (requires `pandas`):

```bash
# Score all models under inputs/predictions/ (or pass model folder names)
python evaluate.py
python evaluate.py gpt4o gemma_4_31b

# All-frames vs reduced detection comparison
python compare_all_vs_reduced.py

# Optional: print tables from existing outputs/
python summarize_results.py

# ImageHash ablation (from repo root or from ImageHash_tvalues/)
python ImageHash_tvalues/run_hash_eval.py
```

Outputs are written under `outputs/` and `ImageHash_tvalues/outputs/`.
