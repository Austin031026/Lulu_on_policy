#!/usr/bin/env python3
"""Show per-benchmark and overall rollout completion for a Lulu vLLM evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


SOURCE_COUNTS = {"math500": 500, "olympiadbench": 580, "aime2025": 30}


def line_count(paths):
    total = 0
    for path in paths:
        with path.open("rb") as handle:
            total += sum(1 for line in handle if line.strip())
    return total


def snapshot(root):
    plan_path = root / "eval_plan.json"
    if not plan_path.is_file():
        return False, ["Waiting for eval_plan.json (adapter may still be merging)."]
    plan = json.loads(plan_path.read_text())
    n = int(plan["num_rollouts"])
    limit = int(plan["max_examples"])
    rows = []
    complete_total = expected_total = 0
    for name, source_count in SOURCE_COUNTS.items():
        problems = min(source_count, limit) if limit else source_count
        expected = problems * n
        complete = line_count((root / name / plan["checkpoint_name"] / "shards").glob("*.jsonl"))
        complete_total += complete
        expected_total += expected
        rows.append(f"{name}: {complete}/{expected} rollouts ({100 * complete / expected:.1f}%)")
    rows.append(f"overall: {complete_total}/{expected_total} rollouts "
                f"({100 * complete_total / expected_total:.1f}%)")
    done = (root / "summary.json").is_file()
    if done:
        rows.append("summary.json: ready")
    return done, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    while True:
        done, rows = snapshot(args.output_dir.expanduser().resolve())
        print("\033[2J\033[H" + "\n".join(rows), flush=True)
        if not args.watch or done:
            return
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    main()
