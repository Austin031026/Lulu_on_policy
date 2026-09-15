"""Prepare deduplicated, disjoint Lulu train/dev questions from DAPO or local data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lulu.data import DEFAULT_CACHE_DIR, DEFAULT_DATASET, prepare_dataset
from lulu.paths import DEFAULT_OUTPUT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HF dataset ID, JSON/JSONL/Parquet/CSV/Arrow, or saved Dataset directory")
    parser.add_argument("--dataset-config")
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "data/lulu_dapo"))
    parser.add_argument("--question-column", help="Explicit question/messages field (nested dotted paths supported)")
    parser.add_argument("--answer-column", help="Explicit gold field, e.g. reward_model.ground_truth")
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--dev-size", type=int, default=256)
    size.add_argument("--dev-fraction", type=float)
    parser.add_argument("--gold-conflict-policy", choices=("drop", "error"), default="drop", help="Drop all copies of questions with contradictory labels, or abort")
    parser.add_argument("--eval-data-source", default="math", help="Shared evaluator answer-parser alias for the held-out Parquet export")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-download", action="store_true", help="Permit dataset downloads (default: use local cache only)")
    args = parser.parse_args()
    manifest = prepare_dataset(
        args.output_dir, dataset=args.dataset, dataset_config=args.dataset_config,
        split=args.split, cache_dir=args.cache_dir, offline=not args.allow_download,
        question_column=args.question_column, answer_column=args.answer_column,
        dev_size=args.dev_fraction if args.dev_fraction is not None else args.dev_size,
        train_limit=args.train_limit, seed=args.seed, gold_conflict_policy=args.gold_conflict_policy,
        eval_data_source=args.eval_data_source,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
