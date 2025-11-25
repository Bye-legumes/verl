#!/usr/bin/env python
"""Utility to convert the Pangu JSONL coding dataset into VERL RL parquet files.

Each row in the output parquet contains:
- prompt: a single-turn conversation (list of {role, content}) so VERL can apply chat templates.
- data_source: the original dataset name (e.g., "taco").
- ability: hard-coded "code" for filtering/metrics.
- reward_model: stores the ground-truth reference code (not used by SEGym reward but kept for completeness).
- extra_info: metadata required by the SEGym reward manager (dataset/index/md5 hashes/etc.).

Usage:
    python convert_pangu_jsonl_to_parquet.py \
        --input /path/to/pangu.jsonl \
        --train-output /path/to/train.parquet \
        --val-output /path/to/val.parquet \
        --val-ratio 0.05 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import pandas as pd


def _build_record(obj: dict[str, Any]) -> dict[str, Any]:
    dataset = obj.get("dataset", "unknown")
    dataset_index = int(obj.get("dataset_index", obj.get("index", 0)))
    prompt = obj.get("prompt", "")
    prompt_ground_truth = obj.get("prompt_ground_truth", "")

    record = {
        "prompt": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "data_source": dataset,
        "ability": "code",
        "reward_model": {
            "style": "reference_code",
            "ground_truth": prompt_ground_truth,
        },
        "extra_info": {
            "dataset": dataset,
            "dataset_index": dataset_index,
            "prompt_md5hash": obj.get("prompt_md5hash"),
            "dataset_problem_md5hash": obj.get("dataset_problem_md5hash"),
            "pangu_pass_count": obj.get("pangu_pass_count"),
            "prompt_type": obj.get("prompt_type"),
            "dataset_problem": obj.get("dataset_problem"),
            "language": obj.get("language", "python"),
            "timeout": obj.get("timeout"),
        },
    }
    return record


def convert_jsonl(
    input_path: Path,
    train_output: Path,
    val_output: Path | None,
    val_ratio: float,
    seed: int,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSON on line {line_no}: {exc}") from exc
            records.append(_build_record(obj))

    if not records:
        raise RuntimeError(f"No samples were loaded from {input_path}")

    df = pd.DataFrame(records)
    indices = list(range(len(df)))
    rnd = random.Random(seed)
    rnd.shuffle(indices)

    split_point = int(len(indices) * (1 - val_ratio)) if val_output else len(indices)
    train_idx = indices[:split_point]
    val_idx = indices[split_point:]

    os.makedirs(train_output.parent, exist_ok=True)
    df.iloc[train_idx].to_parquet(train_output, index=False)
    print(f"Wrote {len(train_idx)} samples to {train_output}")

    if val_output:
        os.makedirs(val_output.parent, exist_ok=True)
        if val_idx:
            df.iloc[val_idx].to_parquet(val_output, index=False)
            print(f"Wrote {len(val_idx)} samples to {val_output}")
        else:
            print("Validation split requested but val_ratio yielded 0 samples; skipping val output.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Pangu JSONL to VERL RL parquet dataset")
    parser.add_argument("--input", required=True, type=Path, help="Path to the source JSONL file")
    parser.add_argument(
        "--train-output",
        required=True,
        type=Path,
        help="Output path for the training parquet",
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        default=None,
        help="Output path for the validation parquet (optional)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="Fraction of data reserved for validation (only used if --val-output is provided)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_jsonl(args.input, args.train_output, args.val_output, max(args.val_ratio, 0.0), args.seed)
