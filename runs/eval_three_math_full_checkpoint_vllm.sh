#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CHECKPOINT_TYPE=full
exec bash "$RUN_DIR/eval_three_math_checkpoint_vllm.sh" "$@"
