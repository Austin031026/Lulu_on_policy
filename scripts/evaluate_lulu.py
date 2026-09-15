#!/usr/bin/env python3
"""Batched ordinary-policy evaluation for Lulu / ReN-OPD student checkpoints.

Uses the existing PlainModelRunner, benchmark_parser, benchmark manifest and
LiveCodeBench export/scoring tools. Planning and summaries need only stdlib;
model dependencies are imported inside GPU workers.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from lulu.paths import DEFAULT_OUTPUT_ROOT, soraka_root

DEFAULT_BENCHMARKS = ("math500", "aime25", "olympiadbench", "mmlu_pro", "gpqa_diamond")


def named_value(value):
    name, sep, location = value.partition("=")
    if not sep or not location or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError(f"expected NAME=PATH with a simple unique NAME, got {value!r}")
    return name, location


def resolve_devices(value):
    if value == "auto":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None:
            value = visible
        else:
            try:
                output = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
                )
                value = ",".join(output.split())
            except (OSError, subprocess.CalledProcessError):
                value = ""
    devices = [x.strip() for x in value.split(",") if x.strip()]
    if not devices or "-1" in devices:
        raise ValueError("No visible GPUs; set --gpus 0,1,... (or --gpus cpu for a tiny CPU test)")
    if len(set(devices)) != len(devices) or ("cpu" in devices and len(devices) != 1):
        raise ValueError("GPU identifiers must be unique; cpu must be used alone")
    return devices


def resolve_model_reference(value):
    """Freeze local paths before recording a plan; leave Hub identifiers intact."""
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.exists() or value.startswith((".", "/", "~")):
        return str(path.resolve())
    return value


def build_plan(args):
    framework_root = soraka_root(args.soraka_root)
    if args.batch_size <= 0 or args.max_response_tokens <= 0 or args.max_prompt_tokens <= 0:
        raise ValueError("batch-size and token limits must be positive")
    if args.max_examples < 0:
        raise ValueError("max-examples must be nonnegative")
    if args.lcb_processes <= 0:
        raise ValueError("lcb-processes must be positive")
    models = []
    checkpoint_groups = (
        ("full", args.full_checkpoint),
        ("lora", args.lora_checkpoint),
        ("auto", args.checkpoint),
    )
    has_checkpoint = any(values for _, values in checkpoint_groups)
    if not has_checkpoint or args.include_base:
        models.append({"name": "base", "model": resolve_model_reference(args.model),
                       "checkpoint_type": "full"})
    for checkpoint_type, values in checkpoint_groups:
        for name, raw_path in map(named_value, values):
            model_path = resolve_model_reference(raw_path)
            if checkpoint_type != "auto":
                if not Path(model_path).is_dir():
                    raise FileNotFoundError(f"{checkpoint_type} checkpoint directory: {model_path}")
                resolve_checkpoint_type(model_path, checkpoint_type)
            models.append({"name": name, "model": model_path,
                           "checkpoint_type": checkpoint_type})
    if len({x["name"] for x in models}) != len(models):
        raise ValueError("checkpoint names must be unique; base is reserved when evaluating the base")
    source = {}
    manifest_root = Path.cwd()
    if args.data_manifest:
        manifest_path = Path(args.data_manifest).expanduser().resolve()
        source = json.loads(manifest_path.read_text())["benchmarks"]
        manifest_root = manifest_path.parent
    direct = dict(map(named_value, args.benchmark))
    if len(direct) != len(args.benchmark):
        raise ValueError("benchmark names must be unique")
    if not source and not direct:
        raise ValueError("provide --data-manifest or --benchmark NAME=PARQUET")
    if args.benchmarks == "all":
        names = list(dict.fromkeys([*source, *direct]))
    elif args.benchmarks:
        names = [x.strip() for x in args.benchmarks.split(",") if x.strip()]
    elif direct:
        names = list(direct)
    else:
        names = list(DEFAULT_BENCHMARKS)
    if not names or len(set(names)) != len(names):
        raise ValueError("select at least one benchmark; benchmark names must be unique")
    benchmarks = []
    for name in names:
        named_value(f"{name}=unused")  # Names become output path components.
        metadata = dict(source.get(name, {}))
        if name in direct:
            path = Path(direct[name]).expanduser().resolve()
            metadata.pop(f"{args.split}_examples", None)
        else:
            if name not in source:
                raise ValueError(f"benchmark {name!r} missing from manifest; set --benchmarks explicitly")
            path = Path(metadata[args.split]).expanduser()
            if not path.is_absolute():
                path = manifest_root / path
            path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"benchmark {name}: {path}")
        default_scorer = ("livecodebench" if name == "livecodebench" else
                          "choice" if name in {"mmlu_pro", "gpqa_diamond"} else "math")
        scorer = metadata.get("scorer", default_scorer)
        if scorer == "livecodebench":
            if not args.lcb_repo or not Path(args.lcb_repo).expanduser().is_dir():
                raise ValueError("LiveCodeBench requires --lcb-repo pointing to the official evaluator checkout")
            if args.max_examples:
                raise ValueError("LiveCodeBench official scorer requires the complete split; omit --max-examples")
            if not metadata.get("livecodebench", {}).get("release_version"):
                raise ValueError("LiveCodeBench requires release_version metadata in --data-manifest")
        expected = metadata.get(f"{args.split}_examples")
        if expected is not None and args.max_examples:
            expected = min(int(expected), args.max_examples)
        benchmarks.append({"name": name, "path": str(path), "scorer": scorer,
                           "expected_examples": expected, "livecodebench": metadata.get("livecodebench")})
    parser_path = (Path(args.parser_path).expanduser().resolve() if args.parser_path else
                   framework_root / "scripts" / "benchmark_parser.py")
    if not parser_path.is_file():
        raise FileNotFoundError(parser_path)
    return {
        "schema_version": 1, "policy": "ordinary_student_greedy", "models": models,
        "soraka_root": str(framework_root),
        "benchmarks": benchmarks, "split": args.split, "devices": resolve_devices(args.gpus),
        "batch_size": args.batch_size, "max_response_tokens": args.max_response_tokens,
        "max_prompt_tokens": args.max_prompt_tokens, "max_examples": args.max_examples,
        "thinking": args.thinking, "dtype": args.dtype, "store_text": args.store_text,
        "trust_remote_code": args.trust_remote_code,
        "adapter_base_model": resolve_model_reference(args.adapter_base_model),
        "parser_path": str(parser_path), "output_dir": str(Path(args.output_dir).expanduser().resolve()),
        "lcb_repo": str(Path(args.lcb_repo).expanduser().resolve()) if args.lcb_repo else None,
        "lcb_python": args.lcb_python, "lcb_processes": args.lcb_processes,
    }


def resolve_checkpoint_type(model_id, checkpoint_type="auto"):
    """Resolve and validate a local checkpoint without importing model libraries."""
    if checkpoint_type not in {"auto", "full", "lora"}:
        raise ValueError(f"unknown checkpoint type: {checkpoint_type}")
    path = Path(model_id).expanduser()
    is_local = path.exists()
    has_adapter = (path / "adapter_config.json").is_file()
    has_adapter_weights = any((path / name).is_file() for name in
                              ("adapter_model.safetensors", "adapter_model.bin"))
    has_full_config = (path / "config.json").is_file()
    if checkpoint_type == "auto":
        checkpoint_type = "lora" if has_adapter else "full"
    if checkpoint_type == "lora":
        if not has_adapter or not has_adapter_weights:
            raise ValueError(f"declared LoRA checkpoint is incomplete: {path}")
    elif is_local:
        if has_adapter:
            raise ValueError(f"declared full-model checkpoint contains adapter_config.json: {path}")
        if not has_full_config:
            raise ValueError(f"declared full-model checkpoint has no config.json: {path}")
    return checkpoint_type


def load_model_assets(model_id, *, dtype, device, thinking, trust_remote_code=False,
                      adapter_base_model=None, checkpoint_type="auto"):
    """Load one explicitly typed full HF model or PEFT LoRA adapter."""
    checkpoint_type = resolve_checkpoint_type(model_id, checkpoint_type)
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    path = Path(model_id).expanduser()
    kwargs = dict(torch_dtype=dtype, trust_remote_code=trust_remote_code, low_cpu_mem_usage=True)
    if checkpoint_type == "lora":
        from peft import PeftConfig, PeftModel
        config = PeftConfig.from_pretrained(str(path))
        base_id = adapter_base_model or config.base_model_name_or_path
        tokenizer_id = str(path) if (path / "tokenizer_config.json").is_file() else base_id
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=trust_remote_code)
        base = AutoModelForCausalLM.from_pretrained(base_id, **kwargs)
        model = PeftModel.from_pretrained(base, str(path), is_trainable=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must define a pad or EOS token")
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    model = model.to(device).eval()
    model.config.use_cache = True
    # Compare checkpoints under identical strict greedy decoding, independent of
    # generation_config.json sampling defaults or repetition penalties.
    model.generation_config = GenerationConfig(
        do_sample=False, num_beams=1, repetition_penalty=1.0,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    class ChatTokenizer:
        def __getattr__(self, key):
            return getattr(tokenizer, key)

        def __call__(self, *args, **kwargs):
            return tokenizer(*args, **kwargs)

        def apply_chat_template(self, *args, **kwargs):
            kwargs["enable_thinking"] = thinking
            return tokenizer.apply_chat_template(*args, **kwargs)

    return model, ChatTokenizer()


def add_shared_framework_paths(plan):
    """Resolve both sibling import roots explicitly in parent and fresh workers."""
    framework_root = soraka_root(plan.get("soraka_root"))
    for path in (framework_root, framework_root / "scripts"):
        location = str(path)
        if location not in sys.path:
            sys.path.insert(0, location)
    return framework_root


def make_runner(plan, device):
    add_shared_framework_paths(plan)
    from evaluate_plain_model import PlainModelRunner

    checkpoint_types = {}
    for item in plan["models"]:
        previous = checkpoint_types.setdefault(item["model"], item.get("checkpoint_type", "auto"))
        if previous != item.get("checkpoint_type", "auto"):
            raise ValueError(f"one model path has conflicting checkpoint types: {item['model']}")

    class LuluRunner(PlainModelRunner):
        def load_model(self, model_id, **kwargs):
            if self.model_id == model_id and self.model is not None:
                return
            self.close()
            self.model, self.tokenizer = load_model_assets(
                model_id, dtype=self.dtype, device=device, thinking=plan["thinking"],
                trust_remote_code=plan["trust_remote_code"],
                adapter_base_model=plan["adapter_base_model"],
                checkpoint_type=checkpoint_types.get(model_id, "auto"),
            )
            self.model_id = model_id
            print(f"[lulu-eval][READY] model={model_id} device={device}", flush=True)

    return LuluRunner(parser_path=plan["parser_path"], batch_size=plan["batch_size"],
                      max_response_tokens=plan["max_response_tokens"],
                      max_prompt_tokens=plan["max_prompt_tokens"], dtype=plan["dtype"])


def run_worker(plan, shard_id):
    device = "cpu" if plan["devices"] == ["cpu"] else "cuda:0"
    runner = make_runner(plan, device)
    try:
        for model in plan["models"]:
            for benchmark in plan["benchmarks"]:
                output = Path(plan["output_dir"]) / model["name"] / benchmark["name"]
                runner.evaluate_shard(
                    model_id=model["model"], input_parquet=benchmark["path"],
                    output=str(output / f"shard-{shard_id:03d}.jsonl"),
                    shard_id=shard_id, num_shards=len(plan["devices"]),
                    max_examples=plan["max_examples"], store_text=plan["store_text"],
                    progress_every=8, scorer=benchmark["scorer"],
                )
    finally:
        runner.close()


def load_complete_rows(directory, num_shards, expected_examples):
    """Only accept exactly the current shard set, with unique complete row coverage."""
    directory = Path(directory)
    expected_files = {f"shard-{i:03d}.jsonl" for i in range(num_shards)}
    actual_files = {p.name for p in directory.glob("shard-*.jsonl")}
    if actual_files != expected_files:
        raise ValueError(f"incomplete or stale shards in {directory}: {actual_files ^ expected_files}")
    indexed = {}
    for sid in range(num_shards):
        for line in (directory / f"shard-{sid:03d}.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            index = int(row["prompt_index"])
            if index in indexed or index % num_shards != sid:
                raise ValueError(f"duplicate or incorrectly assigned prompt_index={index} in {directory}")
            indexed[index] = row
    if set(indexed) != set(range(expected_examples)):
        raise ValueError(f"generation coverage {len(indexed)} does not match {expected_examples} in {directory}")
    return [indexed[i] for i in range(expected_examples)]


def summarize_rows(rows):
    rewards = [float(x["reward"]) for x in rows if x.get("reward") is not None]
    n = len(rows)
    return {
        "schema_version": 1, "examples": n, "scored_examples": len(rewards),
        "accuracy": sum(rewards) / len(rewards) if rewards else None,
        "mean_response_tokens": sum(x["response_tokens"] for x in rows) / n if n else None,
        "mean_prompt_tokens": sum(x["prompt_tokens"] for x in rows) / n if n else None,
        "hit_cap_fraction": sum(bool(x["hit_cap"]) for x in rows) / n if n else None,
    }


def paired_comparison(base, candidate):
    if [x["prompt_index"] for x in base] != [x["prompt_index"] for x in candidate]:
        raise ValueError("base and checkpoint row indices must match")
    pairs = [(a["reward"], b["reward"]) for a, b in zip(base, candidate)
             if a.get("reward") is not None and b.get("reward") is not None]
    return {"paired_examples": len(pairs),
            "accuracy_delta": sum(float(b) - float(a) for a, b in pairs) / len(pairs) if pairs else None,
            "rescues": sum(a == 0 and b == 1 for a, b in pairs),
            "degradations": sum(a == 1 and b == 0 for a, b in pairs)}


def score_livecodebench(plan, benchmark, raw_dir):
    scripts = soraka_root(plan.get("soraka_root")) / "scripts"
    custom = raw_dir / "livecodebench_custom.json"
    subprocess.run([sys.executable, str(scripts / "export_livecodebench_custom.py"),
                    "--data", benchmark["path"], "--generation-dir", str(raw_dir),
                    "--output", str(custom)], check=True)
    subprocess.run([plan["lcb_python"], "-m", "lcb_runner.runner.custom_evaluator",
                    "--custom_output_file", str(custom), "--scenario", "codegeneration",
                    "--release_version", benchmark["livecodebench"]["release_version"],
                    "--n", "1", "--temperature", "0.0",
                    "--num_process_evaluate", str(plan["lcb_processes"])],
                   cwd=plan["lcb_repo"], check=True)
    scored = raw_dir / "scored"
    subprocess.run([sys.executable, str(scripts / "inject_livecodebench_rewards.py"),
                    "--data", benchmark["path"], "--generation-dir", str(raw_dir),
                    "--lcb-eval-all", str(custom.with_name(custom.stem + "_codegeneration_output_eval_all.json")),
                    "--output-dir", str(scored)], check=True)
    return scored


def finish_suite(plan):
    root = Path(plan["output_dir"])
    result = {"schema_version": 1, "decoding": "greedy", "thinking": plan["thinking"],
              "split": plan["split"], "models": {}}
    base_rows = {}
    for model in plan["models"]:
        summaries = {}
        for benchmark in plan["benchmarks"]:
            directory = root / model["name"] / benchmark["name"]
            rows = load_complete_rows(directory, len(plan["devices"]), benchmark["expected_examples"])
            if benchmark["scorer"] == "livecodebench":
                scored = score_livecodebench(plan, benchmark, directory)
                rows = load_complete_rows(scored, len(plan["devices"]), benchmark["expected_examples"])
            if any(x.get("reward") is None for x in rows):
                raise ValueError(f"unscored rows remain in {directory}")
            summary = summarize_rows(rows)
            if model["name"] == "base":
                base_rows[benchmark["name"]] = rows
            elif benchmark["name"] in base_rows:
                summary["vs_base"] = paired_comparison(base_rows[benchmark["name"]], rows)
            summaries[benchmark["name"]] = summary
            (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        accuracies = [s["accuracy"] for s in summaries.values() if s["accuracy"] is not None]
        result["models"][model["name"]] = {"model": model["model"], "benchmarks": summaries,
            "macro_accuracy": sum(accuracies) / len(accuracies) if accuracies else None}
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def run_suite(plan):
    import pyarrow.parquet as pq

    # Preflight dataset coverage and the existing scorer before allocating GPUs.
    add_shared_framework_paths(plan)
    from evaluate_plain_model import load_parser
    parser = load_parser(plan["parser_path"])
    if any(b["scorer"] == "math" for b in plan["benchmarks"]) and hasattr(parser, "_math_parser"):
        parser._math_parser()
    for benchmark in plan["benchmarks"]:
        total = pq.ParquetFile(benchmark["path"]).metadata.num_rows
        actual = min(total, plan["max_examples"]) if plan["max_examples"] else total
        if actual <= 0:
            raise ValueError(f"empty benchmark: {benchmark['name']}")
        if benchmark["expected_examples"] is not None and benchmark["expected_examples"] != actual:
            raise ValueError(f"manifest example count disagrees with parquet for {benchmark['name']}")
        benchmark["expected_examples"] = actual
    root = Path(plan["output_dir"])
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"output directory must be empty to prevent mixing runs: {root}")
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "eval_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    workers = []
    logs = []
    try:
        for sid, gpu in enumerate(plan["devices"]):
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = "" if gpu == "cpu" else gpu
            env.setdefault("TOKENIZERS_PARALLELISM", "false")
            env.setdefault("OMP_NUM_THREADS", "4")
            log_path = root / f"worker-{sid:03d}.log"
            log = log_path.open("w")
            logs.append(log)
            workers.append(subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve()), "--worker-plan", str(plan_path),
                 "--shard-id", str(sid)], env=env, stdout=log, stderr=subprocess.STDOUT))
            print(f"[lulu-eval] gpu={gpu} shard={sid}/{len(plan['devices'])} log={log_path}", flush=True)
        failures = [i for i, worker in enumerate(workers) if worker.wait() != 0]
        if failures:
            raise RuntimeError(f"evaluation workers failed: {failures}; see {root}/worker-*.log")
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
        for worker in workers:
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
        for log in logs:
            log.close()
    finish_suite(plan)


def argument_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B", help="base student identifier")
    p.add_argument("--full-checkpoint", action="append", default=[], metavar="NAME=PATH",
                   help="repeat for complete Hugging Face model checkpoints")
    p.add_argument("--lora-checkpoint", action="append", default=[], metavar="NAME=PATH",
                   help="repeat for PEFT LoRA adapter checkpoints")
    p.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH",
                   help="legacy auto-detected HF/PEFT checkpoint; prefer an explicit typed option")
    p.add_argument("--include-base", action="store_true", help="also evaluate base and compute paired deltas")
    p.add_argument("--adapter-base-model", help="override the base path stored in LoRA adapters")
    p.add_argument("--data-manifest", help="existing prepare_crossbench_data.py manifest.json")
    p.add_argument("--benchmark", action="append", default=[], metavar="NAME=PARQUET")
    p.add_argument("--benchmarks", help="comma-separated manifest names; default five math/general benchmarks; 'all' includes coding")
    p.add_argument("--split", choices=["probe", "full"], default="full")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "evaluation"))
    p.add_argument("--soraka-root", help="evaluation framework root (default: this Lulu checkout; LULU_SORAKA_ROOT overrides)")
    p.add_argument("--gpus", default="auto", help="physical GPU IDs/UUIDs; auto respects CUDA_VISIBLE_DEVICES")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-response-tokens", type=int, default=8192)
    p.add_argument("--max-prompt-tokens", type=int, default=4096)
    p.add_argument("--max-examples", type=int, default=0)
    p.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--store-text", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--parser-path", help="benchmark parser (default: <soraka-root>/scripts/benchmark_parser.py)")
    p.add_argument("--lcb-repo")
    p.add_argument("--lcb-python", default=sys.executable)
    p.add_argument("--lcb-processes", type=int, default=16)
    p.add_argument("--dry-run", action="store_true", help="print resolved plan without importing model dependencies or writing outputs")
    p.add_argument("--worker-plan", help=argparse.SUPPRESS)
    p.add_argument("--shard-id", type=int, default=0, help=argparse.SUPPRESS)
    return p


def main():
    args = argument_parser().parse_args()
    if args.worker_plan:
        run_worker(json.loads(Path(args.worker_plan).read_text()), args.shard_id)
        return
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
    else:
        run_suite(plan)


if __name__ == "__main__":
    main()
