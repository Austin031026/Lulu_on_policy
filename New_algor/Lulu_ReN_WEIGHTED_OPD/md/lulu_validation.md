# Lulu implementation validation — 2026-09-13

## Resident backend and checkpoint retention

The current default is `--backend persistent`. With eight A100-80GB GPUs, the default ReN allocation is five full Student replicas for data-parallel rollout / DDP updates, one synchronized privileged Student replica, and one external Teacher instance using two-GPU native Transformers tensor parallelism. Explicit flags are `--student-gpus 0,1,2,3,4 --hindsight-gpus 5 --teacher-gpus 6,7`.

Models and optimizer remain resident. Microbatches stream through Student rollout/causal scoring, privileged scoring and answer-blind Teacher scoring inside a single frozen-policy round. All targets must finish before the one DDP update; the privileged Student receives current trainable weights before every new round. Hidden-state caches remain in Student RAM. `--update-passes 1` is required; `--backend staged` preserves the historical implementation for debugging and snapshot reuse.

The checkpoint manager passed **14 focused CPU tests** in 3.12 seconds. Tests cover periodic retention, latest model and optimizer recovery, interrupted model writes, interrupted symlink publication, a stale JSON manifest after successful symlink publication, retained final checkpoints, changing the retention cadence, foreign-directory preservation, and rejection of broken/external pointers or duplicate update writes. At `--save-every 20`, it retains initialization, multiples of 20, the explicit final update, and current latest. It still writes latest after every optimizer update; this policy reduces retained disk usage, not write frequency. Readers resume from the atomic `checkpoints/latest` pointer and checkpoint metadata.

The resident backend records complete `round_seconds` including snapshot synchronization, rollout/scoring, DDP update and checkpoint save. One-time model startup is separate in `runtime_plan.json`. Per-service times overlap and must not be added to estimate wall time.

Current production defaults are `--max-new-tokens 8192 --rollout-batch-size 4 --score-batch-size 1 --train-micro-batch-size 1`, with global batch 64, prompt cap 4096 and full sequence cap 16384. Both causal and hindsight prompt lengths are checked before generation; a prompt at its cap plus a response at its cap needs 12288 tokens. Smaller scoring batches limit padded long-context activations. The full 8192-token configuration has not been GPU-validated, and Qwen3-32B is not present in the checked cache. No additional timing or long-context GPU run is being requested or launched for documentation; no exact ETA or 32B speedup is claimed.

## Final CPU checks after the 8192-token update

The final affected training/layout/pipeline/lifecycle/context-budget suite passed **58 tests**. It includes exact 8192-response boundary indexing (4096 prompt + 8192 response), rejection of insufficient context budgets, real two-rank tiny-Qwen DDP with empty/dummy trajectories, independent byte-based tensor IPC and bounded stalled transfers. Privileged service checks passed 18 tests; checkpoint retention/recovery checks passed 14 tests.

Teacher service checks passed 33 CPU tests, with the GPU test skipped after the request to minimize resource use. One additional real two-rank Gloo test applies the actual synchronous native Transformers RowwiseParallel hooks to an 8192-token projection. Its [1,8192,1024] all-reduced output matches the dense reference with maximum absolute error below 2e-7. This uses only small CPU linear layers and does not load a language model.

The larger GPU attempt below exposed a collective timeout with different outstanding NCCL sequence numbers on the two ranks. The current implementation retains native TP matrix sharding but explicitly completes rowwise reductions with `async_op=False`. The CPU tests verify the applied workaround's math and synchronization semantics. Its large-model GPU behavior has **not** been revalidated, and no claim of a proven GPU timeout resolution or full 8192-token training validation is made.

Automatic approval review rejected restarting the isolated real 8B large-batch test because it conflicted with the user's request to limit resource use. That test was not restarted or bypassed. The final checks were CPU-only; no LuLu GPU job remains running.

## Completed resident short smoke

Student **Qwen/Qwen3-1.7B**, explicitly selected Teacher **Qwen/Qwen3-8B**, LoRA rank 16, eight A100-80GB GPUs allocated 5+1+2, three rounds of 10 DAPO prompts with a 64-token response cap. This completed before the default response cap was raised to 8192. Models stayed resident across the three rounds, Teacher used two-GPU TP, and checkpoint saving and privileged snapshot synchronization completed every round.

One-time startup took **44.8967 seconds**. The completed round records are:

| Round | Complete round seconds | Latest checkpoint save seconds | Hindsight weight sync seconds |
| --- | ---: | ---: | ---: |
| 0 | 9.488 | 0.468 | 0.416 |
| 1 | 8.775 | 0.352 | 0.366 |
| 2 | 8.734 | 0.459 | 0.324 |

These short-run measurements establish execution of the resident pipeline under that specific setup. They are not evidence of useful learning, a comparison against the differently sized historical staged run, 32B Teacher throughput, or full 8192-token GPU validation. A larger 64-prompt/256-token stress attempt exposed a native-TP timeout; its partial outputs are not performance results. No further GPU run is being used to estimate training time.

Artifacts:

- [Resident runtime plan](../../LuLu_outputs/validation/persistent_qwen8b_tp2/runtime_plan.json)
- [Resident run configuration](../../LuLu_outputs/validation/persistent_qwen8b_tp2/run_config.json)
- [First round](../../LuLu_outputs/validation/persistent_qwen8b_tp2/metrics/round_0000.json), [second round](../../LuLu_outputs/validation/persistent_qwen8b_tp2/metrics/round_0001.json), [third round](../../LuLu_outputs/validation/persistent_qwen8b_tp2/metrics/round_0002.json)
- [Final checkpoint state](../../LuLu_outputs/validation/persistent_qwen8b_tp2/checkpoints/step_000003/lulu_state.json)

## Historical staged implementation

The following measurements were collected before the code moved from `Soraka/Global_reasoning` to the sibling `LuLu` project. Historical data and experiment artifacts remain at their original paths; the links below refer to those preserved results, not a new training run. Current commands and project layout are documented in the [LuLu README](../README.md).

The focused suite passed: **70 tests and 6 subtests**, 30.78 seconds, using the existing `envs/trl/bin/python` environment. The only warning was the host's deprecated `TRANSFORMERS_CACHE` variable; runtime commands explicitly unset it and point to the existing HF cache. Both launch scripts passed `bash -n`.

Coverage includes exact dense-versus-sparse targets, forward KL and stopped source gradients, all five objective modes, conservative reasoning masks, gold isolation, DAPO duplicates/conflicts, tiny real Qwen3/PEFT backward passes, bounded vocabulary activations, and a real two-rank CPU DDP update with an empty/dummy rank. DDP optimizer moments agree with the serial reference within floating-point tolerance. Functional attention dropout and module dropout are disabled for snapshot-consistent training.

### Historical Qwen smoke (`--backend staged`)

Student **Qwen/Qwen3-1.7B**, Teacher **Qwen/Qwen3-8B**, 2 GPUs, 2 rounds, 2 DAPO prompts per round, 24 generated tokens per trajectory. The 8B Teacher was selected explicitly because it is cached; the implementation default remains Qwen3-32B. The run performed independent Student rollout/scoring workers, external Teacher scoring, DDP updates, checkpoint reload, and optimizer restoration across rounds.

| Round | Reasoning tokens | Corrected positions | Positive corrections | Forward KL | Gradient norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 46 | 15 | 23 | 2.5885e-8 | 9.6616e-7 |
| 1 | 46 | 10 | 16 | 8.4264e-8 | 1.4968e-6 |

The adapter file hashes differ across initialization, round1 and round2. Both updated checkpoints contain optimizer state. The signal in these first 24 tokens is small; this is an execution check, not evidence of useful learning or a tuned training recipe.

The container's unresolved hostname prevented `torchrun --standalone`; the launcher now uses the repository's existing localhost rendezvous pattern and loopback interfaces. The resumed real two-GPU run completed successfully.

Artifacts:

- [Validation summary](../../Soraka_rlrl/experiments/lulu_ren_opd_smoke_20260913/validation_summary.json)
- [Run configuration](../../Soraka_rlrl/experiments/lulu_ren_opd_smoke_20260913/run_config.json)
- [Updated checkpoint](../../Soraka_rlrl/experiments/lulu_ren_opd_smoke_20260913/checkpoints/round_0002/lulu_state.json)

### Historical evaluator smoke

The saved Student adapter and base Qwen3 model ran through the shared evaluator on **MMLU-Pro and GPQA Diamond**, two examples per benchmark/model, across two GPUs. The updated Student also ran through the shared **math parser on two held-out DAPO examples**. Checkpoint loading, batching, sharding, scoring, completeness checks and report aggregation all completed.

All responses hit the deliberately short 24-token cap. These smoke accuracies are **not benchmark quality results**. Full evaluation defaults to 8192 response tokens and should be run separately on the selected final checkpoints.

- [General reasoning smoke report](../../Soraka_rlrl/experiments/lulu_ren_opd_smoke_20260913/evaluation_smoke/summary.json)
- [DAPO math smoke report](../../Soraka_rlrl/experiments/lulu_ren_opd_smoke_20260913/dapo_eval_smoke/summary.json)

The eight-GPU evaluation dry-run against the existing `crossbench_v631` manifest resolved all five default tasks: MATH500 (500), AIME25 (30), OlympiadBench (512), MMLU-Pro (1400), GPQA Diamond (198), total2640 examples per model. These are the existing manifest's fixed subsets. No full benchmark or Qwen3-32B training job was launched. Production eight-GPU throughput and a 32B Teacher run were not measured in this historical validation.

### Prepared DAPO

The cached source has1,791,700 rows. Normalized question deduplication yields17,188 question groups;12 groups have contradictory gold labels and are excluded entirely. The remaining17,176 questions are split into **16,920 train / 256 dev** with disjoint question hashes. Both dev JSONL and the evaluator-compatible dev Parquet are available. Source cache fingerprint and output SHA256 are recorded in the [data manifest](../../Soraka_rlrl/data/lulu_dapo/manifest.json).


## Standalone sibling project migration

All LuLu-owned source, launchers, tests and documentation now live under the sibling `LuLu/` project. Soraka retains only a documentation link and its existing shared evaluation implementation. Historical datasets, checkpoints, benchmark manifests and measurements remain at their original paths.

Post-migration verification passed **78 tests + 6 subtests**:40 objective/data,23 training/layout and15 evaluation. New cases exercise real CPU child-process initialization from a foreign cwd, two-rank DDP, the actual shared evaluator with tiny HF/PEFT checkpoints, explicit alternate framework roots, independent data output, and data/config hash checks when resuming with different path spellings. The package built as `lulu-ren-opd`; its unpacked wheel can invoke `python -m lulu.training --dry-run` without the source checkout or Soraka.

The historical Qwen smoke's original temporary input had expired. Its two source IDs were recovered from frozen rollout caches and matched to the prepared DAPO data; reconstructed input bytes match the original manifest's SHA256 exactly. The relocated launcher accepted the new data/output paths, preserved the old manifest, and skipped the already completed checkpoints successfully. See [resume verification](../../LuLu_outputs/validation/relocation/resume_check.json). No new GPU training was required for this migration.

## Historical evaluator-only throughput

Before the resident backend, base Qwen3-1.7B was timed through the actual evaluator on a single A100-80GB using 16 DAPO dev questions. These warm-generation measurements do not include LoRA, Teacher, backward or checkpoint work.

| Per-GPU batch | Response cap | Generated tokens | Generation seconds | Tokens/s | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 | 256 | 4096 | 14.30 | 286.52 | 3.69 |
| 16 | 256 | 4096 | 7.14 | 573.32 | 4.15 |
| 8 | 1024 | 16384 | 57.03 | 287.31 | 4.38 |

All responses reached the cap. The historical staged smoke also recorded about 76.33 seconds between its second-round checkpoints, versus 0.78 seconds for the update loop alone. Neither measurement predicts the current 8192-token training configuration. See [historical evaluator throughput](../../audit/lulu_batch_timing_check.json) and [profiling script](../../audit/lulu_profile_runtime.py).
