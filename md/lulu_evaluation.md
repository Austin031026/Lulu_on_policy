# LuLu / ReN-OPD evaluation

评测直接加载训练产出的 Student，使用仓库内置的 `scripts/evaluate_plain_model.py` 和 `scripts/benchmark_parser.py`。所有模型使用相同的 greedy decoding；默认启用 Qwen3 thinking。部署时只需本 Lulu checkout、Student checkpoint 和 benchmark 数据。

## 已有数据与直接运行

从 `Rona_Soraka/LuLu` 目录执行：

```bash
export PYTHON_BIN="$(command -v python)"
export DATA_MANIFEST="$(realpath ../Soraka_rlrl/experiments/v6_5-success-q-scale-pool4096-phase11024-seed42/crossbench_v631/data/manifest.json)"

FULL_CHECKPOINT=/path/to/lulu-run/checkpoints/step_000100 \
OUTPUT_DIR=/path/to/new-evaluation-dir \
GPUS=0,1,2,3,4,5,6,7 BATCH_SIZE=8 \
bash runs/eval_lulu.sh
```

普通数学与选择题评测默认以当前 Lulu checkout 为框架根目录，直接加载本仓库的 runner 和 parser，不需要设置 `LULU_SORAKA_ROOT`。`--soraka-root`/`LULU_SORAKA_ROOT` 仅保留为显式兼容覆盖。launcher 不改变当前目录，所以相对 `CHECKPOINT`、`DATA_MANIFEST` 和 `OUTPUT_DIR` 均相对调用时的工作目录。

Persistent 后端中，`step_000000` 是初始化 checkpoint，`step_000001` 是第一次 optimizer update 后的 Student；旧 staged 后端使用 `round_XXXX` 命名。新训练默认产生包含 `config.json` 和完整权重的 Hugging Face 模型目录，使用 `FULL_CHECKPOINT` 或 `--full-checkpoint` 加载。历史 LoRA run 使用 `LORA_CHECKPOINT` 或 `--lora-checkpoint` 加载，目录必须包含 adapter 配置与权重。旧 `CHECKPOINT`/`--checkpoint` 只作为自动识别兼容入口。`MODEL` 默认是 `Qwen/Qwen3-1.7B`，用于 base 对照；迁移 LoRA adapter 后若内部 base 路径失效，可追加 `--adapter-base-model /new/base/path`。完整命令见 [全参数训练与分类型 checkpoint 测评](lulu_full_parameter_training_and_evaluation.md)。

默认同时评测 base 和指定 checkpoint。上述现有 manifest 的 `full` 包含：

| Benchmark | 样本数 |
| --- | ---: |
| MATH-500 | 500 |
| AIME 2025 | 30 |
| OlympiadBench | 512 |
| MMLU-Pro | 1400 |
| GPQA Diamond | 198 |

其中 OlympiadBench 和 MMLU-Pro 是现有框架准备的固定子集；`full` 指该 manifest 的 full split。总计每模型 2640 个样本，包含数学与 general reasoning。`selector_crossbench_v6_52/data/manifest.json` 使用另一种 schema，不适用于此入口；这里使用 `crossbench_v631/data/manifest.json`。

数学题使用仓库内置 parser 调用 Math-Verify；MCQ 使用仓库内置答案字母提取与比较逻辑。逐题输出保留解析结果、命中的 gold answer 和评分错误。

## 并行、预算和快速检查

`GPUS=auto` 默认读取 `CUDA_VISIBLE_DEVICES`，否则使用 `nvidia-smi` 可见 GPU。每 GPU 一个常驻 worker，按索引分片，每次批量生成 `BATCH_SIZE` 个问题。worker 在同一模型下依次处理 benchmarks，避免每个分片重新加载权重。

```bash
# 只打印解析后的数据、模型与 GPU 计划，不加载模型、不写评测输出。
DRY_RUN=1 GPUS=0,1,2,3 bash runs/eval_lulu.sh

# 快速检查：只用 general reasoning 的 probe split。
FULL_CHECKPOINT=/path/to/lulu-run/checkpoints/step_000001 \
OUTPUT_DIR=/path/to/new-probe-dir \
BENCHMARKS=mmlu_pro,gpqa_diamond SPLIT=probe \
MAX_EXAMPLES=16 MAX_RESPONSE_TOKENS=1024 \
GPUS=0,1 BATCH_SIZE=8 bash runs/eval_lulu.sh
```

默认 `MAX_RESPONSE_TOKENS=8192`、`MAX_PROMPT_TOKENS=4096`。显存允许时可提高 `BATCH_SIZE`；长回答的 KV cache 会增加内存占用。超长 prompt 会明确报错，避免截断 benchmark 输入。比较各方法时保持 split、样本数、thinking 和 token budget 一致；报告中记录 `hit_cap_fraction`，用于观察回答预算是否限制结果。

`THINKING=0` 关闭 thinking；`INCLUDE_BASE=0` 只评测 checkpoint；`STORE_TEXT=1` 保留生成文本。每次使用新的空 `OUTPUT_DIR`，防止混合旧分片。

## 多方法与自定义数据

同一训练 run 的多个保留节点可以一次评测。下面默认加载 base、step 20/40/60/80/100，并评测 AIME25 和 OlympiadBench：

```bash
TRAIN_OUTPUT_DIR=/path/to/lulu-run \
DATA_MANIFEST=/path/to/crossbench/manifest.json \
OUTPUT_DIR=/path/to/new-checkpoint-evaluation \
GPUS=0,1,2,3,4,5,6,7 \
bash runs/eval_lulu_checkpoints.sh
```

用 `STEPS=20,40,60` 或 `BENCHMARKS=aime25,olympiadbench,math500` 调整节点和数据集。脚本在启动 GPU worker 前检查所有指定的 `step_XXXXXX` 目录；缺失节点会直接报错。评测默认保持 Thinking Mode。

```bash
"$PYTHON" scripts/evaluate_lulu.py \
  --model Qwen/Qwen3-1.7B --include-base \
  --full-checkpoint ren=/path/ren/checkpoints/step_000100 \
  --full-checkpoint opd=/path/vanilla-opd/checkpoints/step_000100 \
  --lora-checkpoint opsd=/path/opsd/checkpoints/step_000100 \
  --data-manifest "$DATA_MANIFEST" \
  --output-dir /path/to/new-comparison-dir \
  --gpus 0,1,2,3,4,5,6,7 --batch-size 8
```

可用 `--benchmarks math500,mmlu_pro,gpqa_diamond` 选择 manifest 中的任务。更换评测数据时，可重复传 `--benchmark NAME=/path/data.parquet`；或者通过 launcher 设置 `EVAL_DATA` 与 `BENCHMARK_NAME`。Parquet 保持现有字段：`prompt`（chat messages）、`data_source`、`reward_model.ground_truth`；MCQ ground truth 保持 `__CHOICE__A` 形式。新的评分任务可传 `--parser-path`，实现现有 `extract_answer` / `math_equal` 接口。

LiveCodeBench 可通过 `BENCHMARKS=all LCB_REPO=/path/to/official/LiveCodeBench` 加入。它复用现有代码导出、官方 custom evaluator 和 reward 回填脚本；必须保留 manifest 的完整 release split，不支持 `MAX_EXAMPLES` 截取。可追加 `--lcb-python /path/python --lcb-processes 16` 指定官方 evaluator 环境。

## 输出

- `eval_plan.json`：数据路径、模型、thinking、预算和设备配置。
- `worker-NNN.log`：每个 GPU 的进度与错误。
- `MODEL/BENCHMARK/shard-NNN.jsonl`：与现有框架兼容的逐题结果。
- `MODEL/BENCHMARK/summary.json`：准确率、平均回答 / prompt 长度、达到 token cap 的比例；有 base 时附配对 accuracy delta、rescues、degradations。
- `summary.json`：所有方法与 benchmarks 的汇总，包括不按数据集大小加权的 macro accuracy。

任何分片失败、重复样本、缺失样本或未完成评分都会阻止生成成功汇总。
