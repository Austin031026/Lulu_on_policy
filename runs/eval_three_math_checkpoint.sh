#!/usr/bin/env bash
# Run Math500, OlympiadBench, and AIME2025 for one HF/PEFT checkpoint.
# Usage: bash runs/eval_three_math_checkpoint.sh /absolute/path/to/checkpoint

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this file. Run: bash runs/eval_three_math_checkpoint.sh CHECKPOINT"
  return 2
fi

set -uo pipefail

main() {
  local run_dir root_dir workspace_root checkpoint requested_checkpoint checkpoint_name checkpoint_type
  local python_bin source_root data_dir manifest output_root stamp output_dir log_file info_file
  local full_checkpoint lora_checkpoint

  run_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  root_dir="$(cd "$run_dir/.." && pwd)"
  workspace_root="$(cd "$root_dir/.." && pwd)"
  checkpoint="${1:-${CHECKPOINT:-}}"
  python_bin="${PYTHON_BIN:-${PYTHON:-$(command -v python)}}"
  source_root="${EVAL_JSON_ROOT:-/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Rona_Lulu/Lulu_outputs/ren_distill/qwen17b_final_5k/eval_data/math}"
  data_dir="${EVAL_PARQUET_ROOT:-$workspace_root/data/lulu_offline_math_eval}"
  output_root="${LULU_OUTPUT_ROOT:-$workspace_root/Lulu_outputs}"
  checkpoint_type="${CHECKPOINT_TYPE:-full}"

  if [[ -z "$checkpoint" ]]; then
    echo "Missing checkpoint path."
    echo "Usage: bash runs/eval_three_math_checkpoint.sh /absolute/path/to/checkpoint"
    return 2
  fi
  requested_checkpoint="$checkpoint"
  checkpoint="$(cd "$requested_checkpoint" 2>/dev/null && pwd)" || {
    echo "Checkpoint directory does not exist: $requested_checkpoint"
    return 2
  }
  full_checkpoint=""
  lora_checkpoint=""
  case "$checkpoint_type" in
    full)
      if [[ ! -f "$checkpoint/config.json" || -f "$checkpoint/adapter_config.json" ]]; then
        echo "Expected a full-model checkpoint with config.json: $checkpoint"
        return 2
      fi
      full_checkpoint="$checkpoint"
      ;;
    lora)
      if [[ ! -f "$checkpoint/adapter_config.json" || ( ! -f "$checkpoint/adapter_model.safetensors" && ! -f "$checkpoint/adapter_model.bin" ) ]]; then
        echo "Expected a complete LoRA adapter checkpoint: $checkpoint"
        return 2
      fi
      lora_checkpoint="$checkpoint"
      ;;
    *) echo "CHECKPOINT_TYPE must be full or lora: $checkpoint_type"; return 2 ;;
  esac
  if [[ ! -x "$python_bin" ]]; then
    echo "Python executable is invalid: $python_bin"
    return 2
  fi
  for relative in math500/problems.jsonl olympiadbench/problems.jsonl aime2025/problems.jsonl; do
    if [[ ! -f "$source_root/$relative" ]]; then
      echo "Missing benchmark JSONL: $source_root/$relative"
      return 2
    fi
  done

  mkdir -p "$data_dir" "$output_root/evaluation" "$output_root/logs"
  manifest="$data_dir/manifest.json"
  if [[ ! -f "$manifest" ]]; then
    echo "[lulu-eval] preparing local benchmark manifest"
    if ! "$python_bin" -u "$root_dir/scripts/prepare_offline_math_eval.py" \
        --source-root "$source_root" --output-dir "$data_dir"; then
      echo "Benchmark conversion failed; evaluation was not started."
      return 3
    fi
  fi

  if ! "$python_bin" - "$manifest" <<'PY'
import json, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
expected = {"math500": 500, "olympiadbench": 580, "aime25": 30}
actual = {name: int(manifest["benchmarks"][name]["full_examples"]) for name in expected}
if actual != expected:
    raise SystemExit(f"benchmark counts disagree: expected={expected}, actual={actual}")
for name in expected:
    path = Path(manifest["benchmarks"][name]["full"])
    if not path.is_file():
        raise SystemExit(f"missing parquet for {name}: {path}")
print(f"[lulu-eval] benchmark counts verified: {actual}")
PY
  then
    echo "Benchmark validation failed; evaluation was not started."
    return 3
  fi

  checkpoint_name="${CHECKPOINT_NAME:-$(basename "$checkpoint")}"
  stamp="$(date +%Y%m%d_%H%M%S)"
  output_dir="${OUTPUT_DIR:-$output_root/evaluation/${checkpoint_name}_three_math_b${BATCH_SIZE:-8}_$stamp}"
  log_file="${EVAL_LOG:-$output_root/logs/${checkpoint_name}_three_math_b${BATCH_SIZE:-8}_$stamp.log}"
  info_file="${EVAL_INFO_FILE:-$output_root/logs/latest_three_math_eval.info}"

  echo "[lulu-eval] checkpoint=$checkpoint"
  echo "[lulu-eval] checkpoint_type=$checkpoint_type"
  echo "[lulu-eval] benchmarks=math500,olympiadbench,aime25"
  echo "[lulu-eval] gpus=${GPUS:-0,1,2,4,5,6,7} batch_size=${BATCH_SIZE:-8}"
  echo "[lulu-eval] output=$output_dir"
  echo "[lulu-eval] log=$log_file"

  nohup env \
    PYTHON_BIN="$python_bin" \
    PYTHONPATH="$root_dir${PYTHONPATH:+:$PYTHONPATH}" \
    LULU_SORAKA_ROOT="$root_dir" \
    HF_HOME="${HF_HOME:-/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Rona_Lulu/Lulu_outputs/.cache/huggingface}" \
    HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$workspace_root/.cache/huggingface/datasets}" \
    TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
    MODEL="${MODEL:-Qwen/Qwen3-1.7B}" \
    CHECKPOINT="" \
    FULL_CHECKPOINT="$full_checkpoint" \
    LORA_CHECKPOINT="$lora_checkpoint" \
    CHECKPOINT_NAME="$checkpoint_name" \
    INCLUDE_BASE="${INCLUDE_BASE:-0}" \
    DATA_MANIFEST="$manifest" \
    BENCHMARKS=math500,olympiadbench,aime25 \
    GPUS="${GPUS:-0,1,2,4,5,6,7}" \
    BATCH_SIZE="${BATCH_SIZE:-8}" \
    MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-4096}" \
    MAX_RESPONSE_TOKENS="${MAX_RESPONSE_TOKENS:-8192}" \
    MAX_EXAMPLES="${MAX_EXAMPLES:-0}" \
    THINKING="${THINKING:-1}" \
    STORE_TEXT="${STORE_TEXT:-1}" \
    OUTPUT_DIR="$output_dir" \
    bash "$root_dir/runs/eval_lulu.sh" >"$log_file" 2>&1 </dev/null &

  local eval_pid=$!
  printf 'EVAL_PID=%s\nEVAL_LOG=%q\nOUTPUT_DIR=%q\nCHECKPOINT=%q\n' \
    "$eval_pid" "$log_file" "$output_dir" "$checkpoint" >"$info_file"

  sleep 3
  if kill -0 "$eval_pid" 2>/dev/null; then
    echo "[lulu-eval] started pid=$eval_pid"
    echo "[lulu-eval] state=$info_file"
    echo "[lulu-eval] monitor: tail -f \"$log_file\""
    echo "[lulu-eval] worker progress: watch -n 5 'tail -n 3 \"$output_dir\"/worker-*.log'"
  else
    echo "[lulu-eval] process exited during startup; last log lines:"
    tail -n 80 "$log_file" 2>/dev/null || true
    return 4
  fi
}

main "$@"
