#!/usr/bin/env python3
"""Measure safe LuLu inference batches with the production model code.

Each candidate runs in a fresh process so a CUDA OOM cannot poison the next
measurement. Synthetic records use the configured maximum prompt/response
lengths and select every response token, making scoring deliberately strict.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
MARKER = "LULU_PROFILE_RESULT="
ALL_ROLES = ("student-rollout", "student-score", "hindsight-score", "teacher-score")


def csv_ints(value):
    result = [int(item) for item in value.split(",") if item.strip()]
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return sorted(set(result))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--roles", default=",".join(ALL_ROLES))
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--teacher-model", default="Qwen/Qwen3-32B")
    p.add_argument("--student-checkpoint", default="")
    p.add_argument("--student-gpu", default="0")
    p.add_argument("--hindsight-gpu", default="5")
    p.add_argument("--teacher-gpus", default="6,7")
    p.add_argument("--rollout-batches", type=csv_ints,
                   default=csv_ints("1,2,4,8,12,16"))
    p.add_argument("--score-batches", type=csv_ints,
                   default=csv_ints("1,2,4,6,8,12,16"))
    p.add_argument("--prompt-tokens", type=int, default=4096)
    p.add_argument("--response-tokens", type=int, default=8192)
    p.add_argument("--max-sequence-tokens", type=int, default=16384)
    p.add_argument("--top-k", type=int, default=32)
    p.add_argument("--logit-chunk-size", type=int, default=32)
    p.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-target-modules",
                   default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--global-batch-prompts", type=int, default=64)
    p.add_argument("--rollouts-per-prompt", type=int, default=1)
    p.add_argument("--student-world-size", type=int, default=4)
    p.add_argument("--gpu-memory-gib", type=float, default=80.0)
    p.add_argument("--safe-memory-fraction", type=float, default=0.90)
    p.add_argument("--timeout", type=float, default=7200)
    p.add_argument("--output", default="lulu_batch_profile.json")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--worker-role", choices=ALL_ROLES, help=argparse.SUPPRESS)
    p.add_argument("--batch-size", type=int, help=argparse.SUPPRESS)
    return p


def validate(a):
    roles = tuple(item.strip() for item in a.roles.split(",") if item.strip())
    unknown = set(roles) - set(ALL_ROLES)
    if unknown:
        raise ValueError(f"unknown roles: {sorted(unknown)}")
    if a.prompt_tokens < 1 or a.response_tokens < 1:
        raise ValueError("prompt and response lengths must be positive")
    if min(a.global_batch_prompts, a.rollouts_per_prompt, a.student_world_size) < 1:
        raise ValueError("global batch, rollouts per prompt and Student world size must be positive")
    if a.prompt_tokens + a.response_tokens > a.max_sequence_tokens:
        raise ValueError("prompt_tokens + response_tokens exceeds max_sequence_tokens")
    if not 0 < a.safe_memory_fraction <= 1:
        raise ValueError("safe_memory_fraction must be in (0, 1]")
    teacher = [item for item in a.teacher_gpus.split(",") if item.strip()]
    if len(teacher) != 2:
        raise ValueError("Qwen3-32B production profiling requires exactly two Teacher GPUs")
    used = [a.student_gpu, a.hindsight_gpu, *teacher]
    if len(used) != len(set(used)):
        raise ValueError("Student, Hindsight and Teacher GPU assignments must be disjoint")
    if "3" in used:
        raise ValueError("GPU 3 is marked unhealthy and cannot be used by this profiler")
    return roles


def common_args(a):
    values = [
        "--model", a.model, "--teacher-model", a.teacher_model,
        "--prompt-tokens", str(a.prompt_tokens),
        "--response-tokens", str(a.response_tokens),
        "--max-sequence-tokens", str(a.max_sequence_tokens),
        "--top-k", str(a.top_k), "--logit-chunk-size", str(a.logit_chunk_size),
        "--dtype", a.dtype, "--lora-rank", str(a.lora_rank),
        "--lora-alpha", str(a.lora_alpha),
        "--lora-target-modules", a.lora_target_modules,
        "--timeout", str(a.timeout),
    ]
    if a.student_checkpoint:
        values += ["--student-checkpoint", a.student_checkpoint]
    return values


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def command_for(a, role, batch):
    worker = [str(Path(__file__).resolve()), "--worker-role", role,
              "--batch-size", str(batch), *common_args(a)]
    if role == "teacher-score":
        port = 29501 if a.dry_run else free_port()
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nnodes", "1",
               "--node_rank", "0", "--master_addr", "127.0.0.1",
               "--master_port", str(port), "--nproc_per_node", "2", *worker]
        visible = a.teacher_gpus
    else:
        cmd = [sys.executable, *worker]
        visible = a.student_gpu if role.startswith("student-") else a.hindsight_gpu
    return cmd, visible


def parse_result(output):
    for line in reversed(output.splitlines()):
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER):])
    return None


def run_candidate(a, role, batch):
    cmd, visible = command_for(a, role, batch)
    if a.dry_run:
        return {"role": role, "batch_size": batch, "status": "dry-run",
                "cuda_visible_devices": visible, "command": cmd}
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=visible, TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS=env.get("OMP_NUM_THREADS", "1"),
               PYTHONPATH=str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""))
    started = time.monotonic()
    try:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              timeout=a.timeout)
        result = parse_result(proc.stdout)
        if result is None:
            status = "oom" if "out of memory" in proc.stdout.lower() else "error"
            result = {"role": role, "batch_size": batch, "status": status,
                      "error_tail": proc.stdout[-4000:]}
        result["exit_code"] = proc.returncode
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        result = {"role": role, "batch_size": batch, "status": "timeout",
                  "error_tail": output[-4000:] if isinstance(output, str) else str(output)[-4000:]}
    result["wall_seconds"] = time.monotonic() - started
    return result


def summarize(a, measurements):
    limit = a.gpu_memory_gib * a.safe_memory_fraction
    maxima = {}
    for role in ALL_ROLES:
        safe = [row for row in measurements if row["role"] == role and row["status"] == "ok"
                and max(row["peak_reserved_gib_per_rank"]) <= limit]
        maxima[role] = max((row["batch_size"] for row in safe), default=None)
    score_values = [maxima[name] for name in ("student-score", "hindsight-score", "teacher-score")
                    if maxima[name] is not None]
    exhausted = {}
    for role in ALL_ROLES:
        rows = [row for row in measurements if row["role"] == role]
        exhausted[role] = bool(rows and rows[-1]["status"] == "ok")
    local_jobs = math.ceil(a.global_batch_prompts * a.rollouts_per_prompt / a.student_world_size)
    rollout = min(maxima["student-rollout"], local_jobs) if maxima["student-rollout"] else None
    score = min(score_values) if len(score_values) == 3 else None
    # Services receive one Student rollout chunk at a time, so a score batch
    # larger than rollout_batch_size cannot improve production throughput.
    if score is not None and rollout is not None:
        score = min(score, rollout)
    return {"safe_reserved_limit_gib": limit, "largest_tested_safe_batch_by_role": maxima,
            "local_trajectories_per_student_rank": local_jobs,
            "recommended_rollout_batch_size": rollout,
            "recommended_score_batch_size": score,
            "largest_candidate_passed_by_role": exhausted,
            "recommendation_note": "score_batch_size is shared, so all three score roles must pass"}


def controller(a):
    roles = validate(a)
    measurements = []
    for role in roles:
        candidates = a.rollout_batches if role == "student-rollout" else a.score_batches
        for batch in candidates:
            row = run_candidate(a, role, batch)
            measurements.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if row["status"] not in ("ok", "dry-run"):
                break
    report = {
        "configuration": {"roles": roles, "student_gpu": a.student_gpu,
            "hindsight_gpu": a.hindsight_gpu, "teacher_gpus": a.teacher_gpus,
            "prompt_tokens": a.prompt_tokens, "response_tokens": a.response_tokens,
            "top_k": a.top_k, "logit_chunk_size": a.logit_chunk_size,
            "global_batch_prompts": a.global_batch_prompts,
            "rollouts_per_prompt": a.rollouts_per_prompt,
            "student_world_size": a.student_world_size,
            "gpu_memory_gib": a.gpu_memory_gib,
            "safe_memory_fraction": a.safe_memory_fraction},
        "measurements": measurements,
        "summary": summarize(a, measurements) if not a.dry_run else {},
    }
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {output.resolve()}")


def profile_args(a):
    return SimpleNamespace(
        model=a.model, teacher_model=a.teacher_model, dtype=a.dtype, cpu=False,
        lora_rank=a.lora_rank, lora_alpha=a.lora_alpha,
        lora_target_modules=a.lora_target_modules, gradient_checkpointing=True,
        max_sequence_tokens=a.max_sequence_tokens, score_batch_size=a.batch_size,
        logit_chunk_size=a.logit_chunk_size, top_k=a.top_k, method="ren_opd",
        worker_timeout=a.timeout,
    )


def load_student_for_profile(a, *, include_optimizer_state):
    import torch
    from lulu import training as tr
    pa = profile_args(a)
    checkpoint = Path(a.student_checkpoint) if a.student_checkpoint else None
    model = tr.load_student(pa, checkpoint, trainable=True)
    if checkpoint is None and a.lora_rank:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=a.lora_rank, lora_alpha=a.lora_alpha, lora_dropout=0.0,
            target_modules=a.lora_target_modules.split(","), bias="none", task_type="CAUSAL_LM"))
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    # Student rollout/scoring happens after earlier optimizer updates. Keep two
    # Adam moment buffers per trainable parameter resident during measurement.
    optimizer_state = []
    if include_optimizer_state:
        optimizer_state = [(torch.zeros_like(p), torch.zeros_like(p))
                           for p in model.parameters() if p.requires_grad]
    model.requires_grad_(False).eval()
    return model, tr.load_tokenizer(a.model), pa, optimizer_state


def valid_token(tok):
    ids = tok.encode("0", add_special_tokens=False)
    if not ids:
        raise ValueError("tokenizer produced no ordinary token for the synthetic input")
    return ids[0]


def records_for(a, tok, *, hindsight=False):
    import torch
    token = valid_token(tok)
    prompt = [token] * a.prompt_tokens
    response = [token] * a.response_tokens
    positions = list(range(a.response_tokens))
    correction = torch.arange(a.top_k, dtype=torch.long).repeat(a.response_tokens, 1)
    result = []
    for _ in range(a.batch_size):
        record = {"causal_prompt_ids": prompt, "response_ids": response,
                  "positions": positions, "correction_ids": correction}
        if hindsight:
            record["hindsight_prompt_ids"] = prompt
            record["causal_topk_ids"] = correction
        result.append(record)
    return result


def memory_result(a, started):
    import torch
    import torch.distributed as dist
    torch.cuda.synchronize()
    local = {"allocated": torch.cuda.max_memory_allocated() / 2**30,
             "reserved": torch.cuda.max_memory_reserved() / 2**30}
    ranks = [None] * dist.get_world_size() if dist.is_initialized() else [local]
    if dist.is_initialized():
        dist.all_gather_object(ranks, local)
    elapsed = time.monotonic() - started
    return {"role": a.worker_role, "batch_size": a.batch_size, "status": "ok",
            "peak_allocated_gib_per_rank": [item["allocated"] for item in ranks],
            "peak_reserved_gib_per_rank": [item["reserved"] for item in ranks],
            "operation_seconds": elapsed,
            "trajectories_per_second": a.batch_size / elapsed}


def worker(a):
    import torch
    from lulu import training as tr
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if a.worker_role == "teacher-score":
        import torch.distributed as dist
        from datetime import timedelta
        from lulu.teacher_service import _load_teacher, score_records
        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=a.timeout))
        pa = profile_args(a)
        model, tok = _load_teacher(pa, world, device)
        records = records_for(a, tok)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        score_records(model, tok, records, pa)
        result = memory_result(a, started)
        if rank == 0:
            print(MARKER + json.dumps(result), flush=True)
        dist.destroy_process_group()
        return

    model, tok, pa, optimizer_state = load_student_for_profile(
        a, include_optimizer_state=a.worker_role.startswith("student-"))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    if a.worker_role == "student-rollout":
        token = valid_token(tok)
        inputs = torch.full((a.batch_size, a.prompt_tokens), token,
                            dtype=torch.long, device="cuda")
        attention = torch.ones_like(inputs)
        eos = model.generation_config.eos_token_id or tok.eos_token_id
        with torch.inference_mode():
            model.generate(input_ids=inputs, attention_mask=attention,
                max_new_tokens=a.response_tokens, min_new_tokens=a.response_tokens,
                do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=eos,
                use_cache=True)
    elif a.worker_role == "student-score":
        records = records_for(a, tok)
        head = tr.base_model(model).get_output_embeddings()
        with torch.inference_mode():
            hidden = tr.selected_hidden(model, tok, records, "causal_prompt_ids", pa)
            for h in hidden:
                h.detach().to("cpu", copy=True)
                for start in range(0, len(h), a.logit_chunk_size):
                    head(h[start:start + a.logit_chunk_size]).float().topk(a.top_k, -1)
    else:
        from lulu.hindsight_service import score_hindsight
        records = records_for(a, tok, hindsight=True)
        with torch.inference_mode():
            score_hindsight(model, tok, records, pa)
    print(MARKER + json.dumps(memory_result(a, started)), flush=True)


def main():
    a = parser().parse_args()
    if a.worker_role:
        import torch
        try:
            worker(a)
        except torch.cuda.OutOfMemoryError as error:
            print(MARKER + json.dumps({"role": a.worker_role, "batch_size": a.batch_size,
                  "status": "oom", "error": str(error)}), flush=True)
            raise SystemExit(42)
    else:
        controller(a)


if __name__ == "__main__":
    main()
