#!/usr/bin/env bash
# Full-parameter Forward-KL LuLu preset with OPSD pointwise clip diagnostics.
set -euo pipefail
RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$RUN_DIR/train_lulu_7gpu.sh" \
  --lora-rank 0 \
  --kl-direction forward \
  --pointwise-kl-clip 0.05 \
  --kl-diagnostics \
  --kl-diagnostic-thresholds 0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.1 \
  "$@"
