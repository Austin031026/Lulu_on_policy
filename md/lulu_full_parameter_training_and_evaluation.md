# 全参数训练与分类型 checkpoint 测评

## 默认训练模式

LuLu 现在默认对 Student 的全部参数进行优化：

```text
lora_rank = 0
training_mode = full_parameter
checkpoint_format = huggingface_full_model
```

训练命令不需要再写 `--lora-rank 0`。为了让日志和 checkpoint 可审计，`runtime_plan.json` 会记录 `training_mode`、`checkpoint_format` 和 `lora_rank`；每个 `lulu_state.json` 也会记录训练模式与 checkpoint 格式。

全参数模式下，checkpoint 目录包含完整 Hugging Face 模型的 `config.json` 和模型权重，以及 `optimizer.pt`、tokenizer 和 `lulu_state.json`。每次 optimizer update 仍会发布可恢复的 `checkpoints/latest`，`--save-every` 只控制历史节点的保留。完整模型和全参数 AdamW optimizer 的保存量、通信量与反向传播显存明显高于 LoRA；以前用 LoRA rank 16 跑过的压力测试不能作为全参数训练不 OOM 的证据，正式长跑前应使用相同上下文长度和 batch 做一轮完整 canary。

LoRA 改为显式选择。例如：

```bash
bash runs/train_lulu_7gpu.sh \
  --train-data /absolute/path/train.jsonl \
  --output-dir /absolute/path/lora_run \
  --lora-rank 16
```

旧 run 恢复时，`run_config.json` 中原有的 `lora_rank` 仍参与严格配置核对，因此不会把已有 LoRA run 当成全参数 run 接着训练。全参数实验应使用新的空输出目录。

## 普通 Hugging Face 测评

完整模型和 LoRA adapter 使用不同的显式参数。完整模型：

```bash
FULL_CHECKPOINT=/absolute/path/checkpoints/step_000020 \
CHECKPOINT_NAME=step_20 \
DATA_MANIFEST=/absolute/path/manifest.json \
INCLUDE_BASE=0 \
bash runs/eval_lulu.sh
```

LoRA adapter：

```bash
LORA_CHECKPOINT=/absolute/path/checkpoints/step_000020 \
CHECKPOINT_NAME=step_20_lora \
DATA_MANIFEST=/absolute/path/manifest.json \
INCLUDE_BASE=0 \
bash runs/eval_lulu.sh
```

Python 入口分别是：

```bash
python scripts/evaluate_lulu.py --full-checkpoint step20=/path/full_checkpoint ...
python scripts/evaluate_lulu.py --lora-checkpoint step20=/path/lora_checkpoint ...
```

`--full-checkpoint` 要求本地目录包含 `config.json`，并拒绝含 `adapter_config.json` 的目录。`--lora-checkpoint` 要求同时存在 adapter 配置和 adapter 权重。旧的 `--checkpoint` 继续保留为自动识别兼容入口，新实验应使用显式类型。

一次测多个保留 step 时，默认按完整模型读取：

```bash
TRAIN_OUTPUT_DIR=/absolute/path/full_parameter_run \
DATA_MANIFEST=/absolute/path/manifest.json \
STEPS=20,40,60,80,100 \
bash runs/eval_lulu_checkpoints.sh
```

测历史 LoRA run 时显式设置：

```bash
CHECKPOINT_TYPE=lora \
TRAIN_OUTPUT_DIR=/absolute/path/lora_run \
DATA_MANIFEST=/absolute/path/manifest.json \
STEPS=20,40,60 \
bash runs/eval_lulu_checkpoints.sh
```

## 三个数学 benchmark 的 vLLM 测评

完整模型 checkpoint：

```bash
export GPUS=0,1,2,4,5,6,7
export BATCH_SIZE=8
export NUM_ROLLOUTS=4
bash runs/eval_three_math_full_checkpoint_vllm.sh \
  /absolute/path/checkpoints/step_000020
```

LoRA checkpoint：

```bash
export GPUS=0,1,2,4,5,6,7
export BATCH_SIZE=8
export NUM_ROLLOUTS=4
bash runs/eval_three_math_lora_checkpoint_vllm.sh \
  /absolute/path/checkpoints/step_000020
```

全参数 checkpoint 直接交给 vLLM。LoRA 入口先加载其 base model、合并 adapter，并缓存合并后的模型，再交给 vLLM。两条入口都会验证 checkpoint 类型；把 LoRA 目录传给完整模型入口会在加载 GPU 前失败。

普通 HF generate 版本也分开：

```bash
bash runs/eval_three_math_full_checkpoint.sh /path/full_checkpoint
bash runs/eval_three_math_lora_checkpoint.sh /path/lora_checkpoint
```

底层通用脚本 `eval_three_math_checkpoint[_vllm].sh` 默认 `CHECKPOINT_TYPE=full`，仅用于兼容现有调用。推荐直接使用上面的类型化入口。
