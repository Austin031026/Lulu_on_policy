#!/usr/bin/env bash
# Example: DATA_MANIFEST=/path/manifest.json CHECKPOINT=/path/student bash runs/eval_lulu.sh
set -euo pipefail
RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$RUN_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
SORAKA_ROOT="${LULU_SORAKA_ROOT:-$ROOT_DIR}"
export MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
# Preserve the legacy parser override for compatible external evaluation roots.
[[ -z "${S2T_PARSER:-}" ]] || export S2T_MATH_PARSER="${S2T_MATH_PARSER:-$S2T_PARSER}"
args=(--model "$MODEL" --soraka-root "$SORAKA_ROOT"
      --output-dir "${OUTPUT_DIR:-${LULU_OUTPUT_ROOT:-$WORKSPACE_ROOT/LuLu_outputs}/evaluation}"
      --gpus "${GPUS:-0,1,2,4,5,6,7}" --batch-size "${BATCH_SIZE:-8}"
      --max-response-tokens "${MAX_RESPONSE_TOKENS:-8192}"
      --max-prompt-tokens "${MAX_PROMPT_TOKENS:-4096}"
      --max-examples "${MAX_EXAMPLES:-0}" --split "${SPLIT:-full}")
[[ -z "${DATA_MANIFEST:-}" ]] || args+=(--data-manifest "$DATA_MANIFEST")
[[ -z "${EVAL_DATA:-}" ]] || args+=(--benchmark "${BENCHMARK_NAME:-custom}=$EVAL_DATA")
[[ -z "${BENCHMARKS:-}" ]] || args+=(--benchmarks "$BENCHMARKS")
[[ -z "${CHECKPOINT:-}" ]] || args+=(--checkpoint "${CHECKPOINT_NAME:-lulu}=$CHECKPOINT")
[[ "${INCLUDE_BASE:-1}" != 1 ]] || args+=(--include-base)
[[ "${THINKING:-1}" != 0 ]] || args+=(--no-thinking)
[[ "${STORE_TEXT:-0}" != 1 ]] || args+=(--store-text)
[[ -z "${LCB_REPO:-}" ]] || args+=(--lcb-repo "$LCB_REPO")
[[ "${DRY_RUN:-0}" != 1 ]] || args+=(--dry-run)
exec "${PYTHON_BIN:-${PYTHON:-python}}" -u "$ROOT_DIR/scripts/evaluate_lulu.py" "${args[@]}" "$@"
