#!/usr/bin/env bash
set -euo pipefail

# Current machine layout: physical GPU 3 is unhealthy and must never be used.
# Four Student replicas run on 0,1,2,4; the synchronized privileged Student
# runs on 5; the 32B Teacher is one tensor-parallel group across 6,7.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
profile_args=()
if [[ -n "${LULU_BATCH_PROFILE:-}" ]]; then
  read -r rollout_batch score_batch < <(
    "$PYTHON_BIN" -c 'import json,sys; s=json.load(open(sys.argv[1]))["summary"]; r=s["recommended_rollout_batch_size"]; q=s["recommended_score_batch_size"]; assert isinstance(r,int) and isinstance(q,int), "profile has no complete safe recommendation"; print(r,q)' "$LULU_BATCH_PROFILE"
  )
  profile_args=(--rollout-batch-size "$rollout_batch" --score-batch-size "$score_batch")
fi
exec "$ROOT_DIR/runs/train_lulu.sh" \
  --backend persistent \
  --gpus 0,1,2,4,5,6,7 \
  --student-gpus 0,1,2,4 \
  --hindsight-gpus 5 \
  --teacher-gpus 6,7 \
  "${profile_args[@]}" \
  "$@"
