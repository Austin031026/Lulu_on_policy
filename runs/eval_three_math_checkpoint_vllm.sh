#!/usr/bin/env bash
# vLLM pass@4 evaluation for one Lulu checkpoint on three fixed math benchmarks.
# Usage: bash runs/eval_three_math_checkpoint_vllm.sh /absolute/path/to/checkpoint

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this file. Run it with bash and a checkpoint path."
  return 2
fi

set -uo pipefail

main() {
  local run_dir root_dir workspace_root checkpoint requested python_bin
  local source_root external_root output_root label stamp output_dir log_file info_file
  run_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  root_dir="$(cd "$run_dir/.." && pwd)"
  workspace_root="$(cd "$root_dir/.." && pwd)"
  requested="${1:-${CHECKPOINT:-}}"
  python_bin="${PYTHON_BIN:-${PYTHON:-$(command -v python)}}"
  source_root="${EVAL_JSON_ROOT:-/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Rona_Lulu/Lulu_outputs/ren_distill/qwen17b_final_5k/eval_data/math}"
  external_root="${LULU_OFFLINE_EVAL_ROOT:-/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Rona_Lulu/Lulu}"
  output_root="${LULU_OUTPUT_ROOT:-$workspace_root/Lulu_outputs}"

  if [[ -z "$requested" ]]; then
    echo "Usage: bash runs/eval_three_math_checkpoint_vllm.sh /absolute/path/to/checkpoint"
    return 2
  fi
  checkpoint="$(cd "$requested" 2>/dev/null && pwd)" || {
    echo "Checkpoint directory does not exist: $requested"
    return 2
  }
  if [[ ! -f "$checkpoint/adapter_config.json" && ! -f "$checkpoint/config.json" ]]; then
    echo "Checkpoint must contain adapter_config.json or config.json: $checkpoint"
    return 2
  fi
  for file in 11_generate_eval_rollouts.py 02_merge_jsonl.py 03_verify_math_rollouts.py; do
    if [[ ! -f "$external_root/$file" ]]; then
      echo "Missing reusable offline evaluator file: $external_root/$file"
      return 2
    fi
  done
  if ! "$python_bin" -c 'import torch, transformers, peft, vllm, math_verify' >/dev/null; then
    echo "Python environment is missing torch/transformers/peft/vllm/math_verify: $python_bin"
    return 2
  fi

  label="${CHECKPOINT_NAME:-$(basename "$checkpoint")}"
  stamp="$(date +%Y%m%d_%H%M%S)"
  output_dir="${OUTPUT_DIR:-$output_root/evaluation/${label}_math_pass4_b${BATCH_SIZE:-8}_$stamp}"
  log_file="${EVAL_LOG:-$output_root/logs/${label}_math_pass4_b${BATCH_SIZE:-8}_$stamp.log}"
  info_file="${EVAL_INFO_FILE:-$output_root/logs/latest_three_math_vllm_eval.info}"
  mkdir -p "$output_root/evaluation" "$output_root/evaluation_models" "$output_root/logs"

  echo "[lulu-vllm] checkpoint=$checkpoint"
  echo "[lulu-vllm] engine=vllm n=${NUM_ROLLOUTS:-4} questions_per_gpu_batch=${BATCH_SIZE:-8}"
  echo "[lulu-vllm] gpus=${GPUS:-0,1,2,4,5,6,7} max_examples_per_benchmark=${MAX_EXAMPLES:-0}"
  echo "[lulu-vllm] output=$output_dir"

  nohup env \
    PYTHONPATH="$root_dir${PYTHONPATH:+:$PYTHONPATH}" \
    HF_HOME="${HF_HOME:-/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Rona_Lulu/Lulu_outputs/.cache/huggingface}" \
    HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$workspace_root/.cache/huggingface/datasets}" \
    TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
    "$python_bin" -u "$root_dir/scripts/evaluate_three_math_vllm.py" \
      --checkpoint "$checkpoint" --checkpoint-name "$label" \
      --source-root "$source_root" --external-root "$external_root" \
      --output-dir "$output_dir" --merged-model-root "$output_root/evaluation_models" \
      --python "$python_bin" --gpus "${GPUS:-0,1,2,4,5,6,7}" \
      --batch-size "${BATCH_SIZE:-8}" --num-rollouts "${NUM_ROLLOUTS:-4}" \
      --max-examples "${MAX_EXAMPLES:-0}" --temperature "${TEMPERATURE:-0.6}" \
      --top-p "${TOP_P:-0.95}" --top-k "${TOP_K:-20}" --min-p "${MIN_P:-0.0}" \
      --max-tokens "${MAX_RESPONSE_TOKENS:-16384}" --max-model-len "${MAX_MODEL_LEN:-32768}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
      --dtype "${DTYPE:-bfloat16}" --seed "${SEED:-42}" --thinking "${THINKING:-1}" \
      --base-model "${ADAPTER_BASE_MODEL:-}" \
      >"$log_file" 2>&1 </dev/null &
  local eval_pid=$!
  printf 'EVAL_PID=%s\nEVAL_LOG=%q\nOUTPUT_DIR=%q\nCHECKPOINT=%q\n' \
    "$eval_pid" "$log_file" "$output_dir" "$checkpoint" >"$info_file"
  sleep 3
  if kill -0 "$eval_pid" 2>/dev/null; then
    echo "[lulu-vllm] started pid=$eval_pid"
    echo "[lulu-vllm] state=$info_file"
    echo "[lulu-vllm] monitor: tail -f \"$log_file\""
    echo "[lulu-vllm] progress: \"$python_bin\" \"$root_dir/scripts/monitor_three_math_vllm.py\" --output-dir \"$output_dir\" --watch"
  else
    echo "[lulu-vllm] startup failed; last log lines:"
    tail -n 100 "$log_file" 2>/dev/null || true
    return 4
  fi
}

main "$@"
