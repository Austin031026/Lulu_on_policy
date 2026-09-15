#!/usr/bin/env bash
# Evaluate the base Student and a retained sequence of LuLu checkpoints.
# Required: TRAIN_OUTPUT_DIR and DATA_MANIFEST.
# Example:
#   TRAIN_OUTPUT_DIR=/path/to/run DATA_MANIFEST=/path/to/manifest.json \
#   GPUS=0,1,2,4,5,6,7 bash runs/eval_lulu_checkpoints.sh
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$RUN_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
SORAKA_ROOT="${LULU_SORAKA_ROOT:-$ROOT_DIR}"

: "${TRAIN_OUTPUT_DIR:?Set TRAIN_OUTPUT_DIR to the completed LuLu training run}"
: "${DATA_MANIFEST:?Set DATA_MANIFEST to the benchmark manifest.json}"

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
STEPS="${STEPS:-20,40,60,80,100}"
BENCHMARKS="${BENCHMARKS:-aime25,olympiadbench}"
CHECKPOINT_TYPE="${CHECKPOINT_TYPE:-full}"
OUTPUT_DIR="${OUTPUT_DIR:-$TRAIN_OUTPUT_DIR/evaluation/checkpoints_${STEPS//,/_}}"
if [[ "$CHECKPOINT_TYPE" != full && "$CHECKPOINT_TYPE" != lora ]]; then
  echo "CHECKPOINT_TYPE must be full or lora: $CHECKPOINT_TYPE" >&2
  exit 2
fi
checkpoint_option="--${CHECKPOINT_TYPE}-checkpoint"

args=(--model "$MODEL" --include-base
      --soraka-root "$SORAKA_ROOT"
      --data-manifest "$DATA_MANIFEST"
      --benchmarks "$BENCHMARKS"
      --output-dir "$OUTPUT_DIR"
      --gpus "${GPUS:-0,1,2,4,5,6,7}"
      --batch-size "${BATCH_SIZE:-8}"
      --max-response-tokens "${MAX_RESPONSE_TOKENS:-8192}"
      --max-prompt-tokens "${MAX_PROMPT_TOKENS:-4096}"
      --max-examples "${MAX_EXAMPLES:-0}"
      --split "${SPLIT:-full}")

IFS=',' read -r -a step_values <<< "$STEPS"
for raw_step in "${step_values[@]}"; do
  step="${raw_step//[[:space:]]/}"
  if [[ ! "$step" =~ ^[0-9]+$ ]]; then
    echo "Invalid checkpoint step in STEPS: $raw_step" >&2
    exit 2
  fi
  printf -v padded '%06d' "$step"
  checkpoint="$TRAIN_OUTPUT_DIR/checkpoints/step_$padded"
  if [[ ! -d "$checkpoint" ]]; then
    echo "Missing retained checkpoint: $checkpoint" >&2
    exit 2
  fi
  if [[ "$CHECKPOINT_TYPE" == full ]]; then
    [[ -f "$checkpoint/config.json" && ! -f "$checkpoint/adapter_config.json" ]] || {
      echo "Expected a full-model checkpoint with config.json: $checkpoint" >&2; exit 2;
    }
  else
    [[ -f "$checkpoint/adapter_config.json" && ( -f "$checkpoint/adapter_model.safetensors" || -f "$checkpoint/adapter_model.bin" ) ]] || {
      echo "Expected a complete LoRA adapter checkpoint: $checkpoint" >&2; exit 2;
    }
  fi
  args+=("$checkpoint_option" "step_$step=$checkpoint")
done

[[ "${THINKING:-1}" != 0 ]] || args+=(--no-thinking)
[[ "${STORE_TEXT:-0}" != 1 ]] || args+=(--store-text)
[[ "${DRY_RUN:-0}" != 1 ]] || args+=(--dry-run)

exec "${PYTHON_BIN:-${PYTHON:-python}}" -u "$ROOT_DIR/scripts/evaluate_lulu.py" "${args[@]}" "$@"
