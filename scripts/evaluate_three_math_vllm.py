#!/usr/bin/env python3
"""Evaluate one Lulu checkpoint on three math benchmarks with vLLM pass@4."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


BENCHMARKS = {
    "math500": ("math500/problems.jsonl", 500),
    "olympiadbench": ("olympiadbench/problems.jsonl", 580),
    "aime2025": ("aime2025/problems.jsonl", 30),
}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def checkpoint_digest(checkpoint):
    digest = hashlib.sha256()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        path = checkpoint / name
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def resolve_model(checkpoint, merged_root, base_model=None):
    if (checkpoint / "config.json").is_file() and not (checkpoint / "adapter_config.json").is_file():
        return checkpoint
    if not (checkpoint / "adapter_config.json").is_file():
        raise ValueError(f"checkpoint is neither a full HF model nor a PEFT adapter: {checkpoint}")

    digest = checkpoint_digest(checkpoint)
    destination = merged_root / f"{checkpoint.name}_{digest[:12]}"
    marker = destination / "lulu_merge_manifest.json"
    if marker.is_file() and json.loads(marker.read_text()).get("checkpoint_sha256") == digest:
        print(f"[lulu-vllm] reuse merged model: {destination}", flush=True)
        return destination
    if destination.exists():
        raise ValueError(f"incomplete merged model directory already exists: {destination}")

    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = PeftConfig.from_pretrained(str(checkpoint))
    base = base_model or config.base_model_name_or_path
    print(f"[lulu-vllm] merging adapter={checkpoint} base={base}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    merged = PeftModel.from_pretrained(model, str(checkpoint), is_trainable=False).merge_and_unload()
    temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    temporary.mkdir(parents=True)
    merged.save_pretrained(temporary, safe_serialization=True)
    tokenizer_source = checkpoint if (checkpoint / "tokenizer_config.json").is_file() else base
    AutoTokenizer.from_pretrained(str(tokenizer_source)).save_pretrained(temporary)
    (temporary / "lulu_merge_manifest.json").write_text(json.dumps({
        "source_checkpoint": str(checkpoint), "checkpoint_sha256": digest,
        "base_model": str(base),
    }, indent=2) + "\n")
    temporary.rename(destination)
    print(f"[lulu-vllm] merged model -> {destination}", flush=True)
    return destination


def run_checked(command, **kwargs):
    print("[lulu-vllm]", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True, **kwargs)


def generate_benchmark(args, model, name, data):
    bench_root = args.output_dir / name / args.checkpoint_name
    shards = bench_root / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    processes = []
    logs = []
    world = len(args.gpus)
    for shard_id, gpu in enumerate(args.gpus):
        output = shards / f"shard_{shard_id:02d}_of_{world:02d}.jsonl"
        log_path = args.output_dir / "logs" / f"{name}_gpu{gpu}.log"
        log = log_path.open("w")
        logs.append(log)
        cache = args.output_dir / "cache" / name / f"gpu{gpu}"
        for child in ("vllm", "torchinductor", "triton"):
            (cache / child).mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), VLLM_CACHE_ROOT=str(cache / "vllm"),
                   TORCHINDUCTOR_CACHE_DIR=str(cache / "torchinductor"),
                   TRITON_CACHE_DIR=str(cache / "triton"))
        command = [
            args.python, "-u", args.external_root / "11_generate_eval_rollouts.py",
            "--dataset_jsonl", data, "--output_jsonl", output, "--model", model,
            "--n", args.num_rollouts, "--problem_batch_size", args.batch_size,
            "--temperature", args.temperature, "--top_p", args.top_p,
            "--top_k", args.top_k, "--min_p", args.min_p,
            "--max_tokens", args.max_tokens, "--max_model_len", args.max_model_len,
            "--gpu_memory_utilization", args.gpu_memory_utilization,
            "--dtype", args.dtype, "--seed", args.seed,
            "--shard_id", shard_id, "--num_shards", world,
            "--enable_thinking", int(args.thinking),
        ]
        processes.append(subprocess.Popen(list(map(str, command)), env=env, stdout=log,
                                          stderr=subprocess.STDOUT))
        print(f"[lulu-vllm] benchmark={name} gpu={gpu} shard={shard_id}/{world} log={log_path}",
              flush=True)
    failures = []
    for shard_id, process in enumerate(processes):
        if process.wait() != 0:
            failures.append(shard_id)
    for log in logs:
        log.close()
    if failures:
        raise RuntimeError(f"{name} generation failed on shards {failures}; see {args.output_dir}/logs")

    merged = bench_root / "rollouts.jsonl"
    scored = bench_root / "rollouts_scored.jsonl"
    run_checked([args.python, args.external_root / "02_merge_jsonl.py",
                 "--input_glob", shards / f"shard_*_of_{world:02d}.jsonl",
                 "--expected_shards", world, "--dedup_keys", "problem_index,rollout_index",
                 "--output_jsonl", merged, "--assign_trajectory_id"])
    run_checked([args.python, args.external_root / "03_verify_math_rollouts.py",
                 "--input_jsonl", merged, "--output_jsonl", scored])
    return scored


def summarize(path, expected_n):
    groups = defaultdict(list)
    for row in read_jsonl(path):
        groups[int(row["problem_index"])].append(row)
    first, any_correct, all_rollouts = [], [], []
    for problem, rows in groups.items():
        rows.sort(key=lambda row: int(row["rollout_index"]))
        indices = [int(row["rollout_index"]) for row in rows]
        if indices != list(range(expected_n)):
            raise ValueError(f"problem {problem} has rollout indices {indices}")
        values = [row.get("is_correct") for row in rows]
        if any(value is None for value in values):
            raise ValueError(f"problem {problem} contains an unscored rollout")
        values = [bool(value) for value in values]
        first.append(values[0]); any_correct.append(any(values)); all_rollouts.extend(values)
    return {"problems": len(groups), "rollouts": len(all_rollouts),
            "pass@1": sum(first) / len(first),
            f"pass@{expected_n}": sum(any_correct) / len(any_correct),
            "rollout_accuracy": sum(all_rollouts) / len(all_rollouts)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--merged-model-root", type=Path, required=True)
    parser.add_argument("--base-model")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", default="0,1,2,4,5,6,7")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thinking", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.source_root = args.source_root.expanduser().resolve()
    args.external_root = args.external_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.merged_model_root = args.merged_model_root.expanduser().resolve()
    args.gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not args.gpus or len(set(args.gpus)) != len(args.gpus):
        parser.error("--gpus must contain unique GPU IDs")
    if min(args.batch_size, args.num_rollouts) <= 0 or args.max_examples < 0:
        parser.error("batch size and rollout count must be positive; max examples must be nonnegative")
    return args


def main():
    args = parse_args()
    required = ["11_generate_eval_rollouts.py", "02_merge_jsonl.py", "03_verify_math_rollouts.py"]
    for name in required:
        if not (args.external_root / name).is_file():
            raise FileNotFoundError(args.external_root / name)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"output directory must be empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs").mkdir()
    args.merged_model_root.mkdir(parents=True, exist_ok=True)
    model = resolve_model(args.checkpoint, args.merged_model_root, args.base_model)

    plan = vars(args).copy()
    plan["gpus"] = args.gpus
    for key, value in list(plan.items()):
        if isinstance(value, Path):
            plan[key] = str(value)
    plan["engine"] = "vllm"
    (args.output_dir / "eval_plan.json").write_text(json.dumps(plan, indent=2) + "\n")

    reports = {}
    input_root = args.output_dir / "inputs"
    for name, (relative, expected) in BENCHMARKS.items():
        rows = read_jsonl(args.source_root / relative)
        if len(rows) != expected:
            raise ValueError(f"{name}: expected {expected} source rows, found {len(rows)}")
        if args.max_examples:
            rows = rows[:args.max_examples]
        for index, row in enumerate(rows):
            row["problem_index"] = index
        data = input_root / f"{name}.jsonl"
        write_jsonl(data, rows)
        scored = generate_benchmark(args, model, name, data)
        reports[name] = summarize(scored, args.num_rollouts)
        print(json.dumps({name: reports[name]}, indent=2), flush=True)
    p1 = [value["pass@1"] for value in reports.values()]
    pk = [value[f"pass@{args.num_rollouts}"] for value in reports.values()]
    summary = {"checkpoint": str(args.checkpoint), "engine": "vllm",
               "num_rollouts": args.num_rollouts, "batch_size_questions_per_gpu": args.batch_size,
               "benchmarks": reports, "macro_pass@1": sum(p1) / len(p1),
               f"macro_pass@{args.num_rollouts}": sum(pk) / len(pk)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
