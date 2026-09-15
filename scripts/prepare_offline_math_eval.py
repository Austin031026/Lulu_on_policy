#!/usr/bin/env python3
"""Convert the established offline math JSONL suite to Lulu evaluator Parquet."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lulu.data import export_eval_parquet


SOURCES = {
    "math500": "math500/problems.jsonl",
    "olympiadbench": "olympiadbench/problems.jsonl",
    "aime25": "aime2025/problems.jsonl",
}


def convert(source_root: Path, output_dir: Path):
    benchmarks = {}
    for name, relative in SOURCES.items():
        source = source_root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        rows = []
        with source.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                record = json.loads(line)
                answers = [str(value) for value in (record.get("gold_answers") or [])]
                if not answers:
                    raise ValueError(f"{source}: row {index} has no gold_answers")
                gold = answers[0] if len(answers) == 1 else json.dumps(answers, ensure_ascii=False)
                rows.append({
                    "id": str(record.get("problem_id", f"{name}-{index}")),
                    "messages": [{"role": "user", "content": str(record["question"])}],
                    "gold_answer": gold,
                })
        destination = output_dir / f"{name}.parquet"
        metadata = export_eval_parquet(rows, destination, data_source="math")
        benchmarks[name] = {
            "full": metadata["path"], "probe": metadata["path"],
            "full_examples": metadata["examples"], "probe_examples": metadata["examples"],
            "sha256": metadata["sha256"], "scorer": "math",
            "source_jsonl": str(source.resolve()),
        }
        print(f"[lulu-eval-data] {name}: {metadata['examples']} -> {metadata['path']}")
    manifest = {"schema_version": 1, "benchmarks": benchmarks}
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"[lulu-eval-data] manifest -> {manifest_path.resolve()}")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True,
                        help="directory containing math500/, olympiadbench/, and aime2025/")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    convert(Path(args.source_root).expanduser().resolve(),
            Path(args.output_dir).expanduser().resolve())


if __name__ == "__main__":
    main()
