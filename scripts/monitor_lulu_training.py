#!/usr/bin/env python3
"""Show live LuLu training progress and round-time estimates."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def metric_rows(run_dir):
    rows = []
    for path in sorted((run_dir / "metrics").glob("round_*.json")):
        payload = read_json(path)
        if isinstance(payload, dict):
            payload = [payload]
        if isinstance(payload, list):
            rows.extend(row for row in payload if isinstance(row, dict) and "completed_updates" in row)
    by_step = {int(row["completed_updates"]): row for row in rows}
    return [by_step[step] for step in sorted(by_step)]


def process_running(pid_file):
    if not pid_file:
        return None, None
    try:
        pid = int(Path(pid_file).read_text().strip())
    except (FileNotFoundError, ValueError, OSError):
        return None, None
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return pid, False
    return pid, True


def duration(seconds):
    if seconds is None:
        return "n/a"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def snapshot(run_dir, pid_file=None, recent_window=10, bar_width=50):
    config = read_json(run_dir / "run_config.json") or {}
    latest = read_json(run_dir / "latest.json") or {}
    rows = metric_rows(run_dir)
    total = int(config.get("rounds", 0) or 0)
    completed = int(latest.get("completed_updates", rows[-1]["completed_updates"] if rows else 0) or 0)
    if total <= 0:
        total = max(completed, 1)
    fraction = min(max(completed / total, 0.0), 1.0)
    filled = min(bar_width, int(fraction * bar_width))
    bar = "█" * filled + "░" * (bar_width - filled)

    times = [float(row["round_seconds"]) for row in rows
             if isinstance(row.get("round_seconds"), (int, float)) and row["round_seconds"] > 0]
    overall = sum(times) / len(times) if times else None
    recent_values = times[-recent_window:]
    recent = sum(recent_values) / len(recent_values) if recent_values else None
    estimate_rate = recent if recent is not None else overall
    eta = (total - completed) * estimate_rate if estimate_rate is not None else None
    last = rows[-1] if rows else {}
    pid, running = process_running(pid_file)
    if running is True:
        process = f"RUNNING (PID={pid})"
    elif running is False:
        process = f"NOT RUNNING (PID={pid})"
    else:
        process = "UNKNOWN (no readable PID file)"

    lines = [
        "# LuLu On-policy 训练实时进度",
        "",
        f"[{bar}] {fraction * 100:6.2f}%",
        f"已完成 optimizer steps：{completed} / {total}",
        f"训练模式：{config.get('training_mode', 'lora' if config.get('lora_rank') else 'full_parameter')}",
        f"KL：{config.get('kl_direction', 'forward')}，pointwise clip={config.get('pointwise_kl_clip', 'n/a')}",
        f"KL diagnostics：{config.get('kl_diagnostics', False)}",
        f"训练主进程：{process}",
        f"最近一轮耗时：{duration(last.get('round_seconds'))}",
        f"全部已完成轮平均：{duration(overall)}",
        f"最近 {min(recent_window, len(recent_values))} 轮平均：{duration(recent)}",
        f"按最近平均速度预计剩余：{duration(eta)}",
    ]
    if latest.get("checkpoint"):
        lines.append(f"当前 checkpoint：{latest['checkpoint']}")
    for key in ("optimization_kl", "forward_kl", "grad_norm", "reasoning_tokens",
                "supervised_trajectories", "response_tokens"):
        if key in last:
            lines.append(f"{key}：{last[key]}")
    lines.append(f"刷新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return "\n".join(lines), completed >= total and total > 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--recent-window", type=int, default=10)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    if args.recent_window <= 0 or args.interval <= 0:
        parser.error("--recent-window and --interval must be positive")
    args.run_dir = args.run_dir.expanduser().resolve()
    if args.pid_file:
        args.pid_file = args.pid_file.expanduser().resolve()
    return args


def main():
    args = parse_args()
    while True:
        report, complete = snapshot(args.run_dir, args.pid_file, args.recent_window)
        if args.watch:
            print("\033[2J\033[H", end="")
        print(report, flush=True)
        if not args.watch or complete:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
