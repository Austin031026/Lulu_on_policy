# LuLu ReN On-Policy 正式实验运行记录

本文是当前 `Qwen3-1.7B Student + Qwen3-32B Teacher` 的 LuLu ReN on-policy 蒸馏实验记录。本文中的参数、路径和命令构成这次实验的固定配置。集群命令统一从代码根目录执行，避免混用旧 checkout、旧 5K 数据或不可写的 Hugging Face cache。

记录日期：2026-09-14。

## 1. 实验状态

完整的一轮端到端压力测试已经通过，覆盖了以下阶段：

```text
Student rollout
→ causal Student score
→ privileged/hindsight Student score
→ Qwen3-32B Teacher score
→ ReN target reconstruction
→ pointwise-clipped forward KL
→ DDP backward
→ AdamW optimizer update
→ checkpoint save
```

压力测试参数为：

```text
rollout_batch_size=6
score_batch_size=6
train_micro_batch_size=2
global_batch_prompts=64
rounds=1
```

压力测试退出码为 `0`，64 条轨迹全部参与监督，反向传播、optimizer step 和 checkpoint 保存均成功。因此正式 100-round 实验固定使用 `6/6/2`。

## 2. 代码、环境与模型路径

| 项目 | 固定值 |
| --- | --- |
| 集群代码根目录 | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_on_policy` |
| Git 仓库 | `https://github.com/Austin031026/Lulu_on_policy` |
| Conda 环境 | `lulu_probe_qwen3` |
| Python | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/conda/envs/lulu_probe_qwen3/bin/python` |
| 7-GPU launcher | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_on_policy/runs/train_lulu_7gpu.sh` |
| Student | `Qwen/Qwen3-1.7B` |
| Teacher | `Qwen/Qwen3-32B` |
| Hugging Face 根 cache | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Rona_Lulu/Lulu_outputs/.cache/huggingface` |
| Hugging Face Hub cache | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Rona_Lulu/Lulu_outputs/.cache/huggingface/hub` |
| Teacher 本地 snapshot | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Rona_Lulu/Lulu_outputs/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137` |

必须清除可能继承的旧 `TRANSFORMERS_CACHE`，并显式使用以上可读 cache。不要从 `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_OPSD_diff` 启动本实验。

## 3. 训练数据

原始本地 DAPO-Math-17k：

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Rona_Lulu/Lulu_outputs/local_datasets/DAPO-Math-17k
```

准备后的数据目录：

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/data/lulu_on_policy_full_dapo/
├── train.jsonl
├── dev.jsonl
├── dev_eval.parquet
└── manifest.json
```

正式训练输入：

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/data/lulu_on_policy_full_dapo/train.jsonl
```

冻结的数据统计：

| 指标 | 数值 |
| --- | ---: |
| 原始 rows | 17,391 |
| 唯一题目组 | 17,181 |
| 删除的重复 rows | 210 |
| 删除的 gold-conflict 题目 | 5 |
| 可用训练题目 | 17,176 |
| dev 题目 | 0 |
| 数据 seed | 42 |
| train SHA256 | `2504195b702a00f6ad11b0906e62c8808b1500899b9a1ae8c63045733e0ea252` |

当前实验不复用旧 5K 实验的筛选结果。训练 scheduler 使用 seed 42 的确定性 shuffle。正式实验共使用：

```text
100 rounds × 64 prompts × 1 rollout = 6,400 trajectories
```

因为 6,400 小于 17,176，100 rounds 内不会重复选择题目。断点续训根据已完成的 round 继续读取确定性数据流，不会重新运行前面的题目。

## 4. GPU 角色

| 物理 GPU | 角色 | 是否反向传播 |
| --- | --- | --- |
| 0、1、2、4 | 四个 Qwen3-1.7B Student worker；rollout、causal score、DDP update | 是 |
| 3 | 故障卡，完全排除 | 否 |
| 5 | 每轮同步的 privileged/hindsight Student | 否 |
| 6、7 | 一个 Qwen3-32B Teacher，Transformers 原生 tensor parallel=2 | 否 |

Teacher 和 Hindsight Student 只做 inference。Student 的 GPU 0、1、2、4 先做 rollout，评分全部完成后仍由这四张卡完成 backward 和 AdamW update。

## 5. 每轮训练策略

每个 round 严格执行一次新的 on-policy 数据采集和一次参数更新：

1. Controller 从确定性数据流选择 64 道当前 round 的题目。
2. GPU 0、1、2、4 使用当前 round 开始时的 Student 参数生成 64 条 Thinking rollout；每张卡负责 16 条，rollout chunk 为 `6+6+4`。
3. Student rollout 使用 Qwen3 chat template，并显式设置 `enable_thinking=True`。
4. Sampling 参数为 `temperature=1.0`、`top_p=1.0`、`top_k=0`，最大生成长度为 8192 tokens。
5. Causal Student 在每个 reasoning position 得到当前完整词表背景分布，并提供 causal top-32 token。
6. GPU 5 的 Hindsight Student 在包含 gold answer 的 privileged prompt 下得到 hindsight top-32，并选择不在 causal top-32 中的候选 token，即 ReN 的新候选集合。
7. GPU 6、7 的 Qwen3-32B Teacher 只接收原始 causal prompt、Student response prefix、位置和候选 token ID。Teacher 不接收 gold answer、hindsight prompt 或 Student hidden states。
8. Teacher 对候选 token 计算完整词表归一化概率 `q_T(token | causal prompt, Student prefix)`。
9. 以 round 开始时冻结的 Student 完整词表分布为背景。仅当候选 token 的 Teacher 概率高于 causal Student 概率时进行替换，随后重新归一化，构造 ReN target。
10. Live Student 在相同 prompt 和 rollout prefix 上重新 forward，计算 Student 到 ReN target 的 full-vocabulary forward KL。
11. 对每个 response position、每个词表项的 KL contribution 使用 `0.05` pointwise clipping，然后再对词表和位置汇总。
12. 每张 Student 卡用 `train_micro_batch_size=2` 处理本卡的 16 条轨迹，即每张卡 8 个 micro-batch；四卡 DDP 汇总梯度。
13. 使用 `max_grad_norm=1.0` 做 gradient clipping，然后执行一次 AdamW optimizer step。
14. 成功完成 optimizer step 后发布新的原子 checkpoint。下一轮 rollout 必须等待本轮 update 完成，并使用更新后的 Student。

因此：

```text
1 round = 64 prompts = 64 trajectories = 1 AdamW optimizer update
```

## 6. 正式训练参数

```text
backend=persistent
method=ren_opd
rounds=100
global_batch_prompts=64
rollouts_per_prompt=1
rollout_batch_size=6
score_batch_size=6
train_micro_batch_size=2
update_passes=1
temperature=1.0
top_p=1.0
top_k_sampling=0
teacher_candidate_top_k=32
max_prompt_tokens=4096
max_new_tokens=8192
max_sequence_tokens=16384
logit_chunk_size=32
lora_rank=16
lora_alpha=32
lora_targets=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
learning_rate=1e-5
weight_decay=0
max_grad_norm=1.0
pointwise_kl_clip=0.05
gradient_checkpointing=true
save_every=20
seed=42
worker_timeout=7200
dtype=bfloat16
```

Student optimizer 是 AdamW，只更新 `requires_grad=True` 的 LoRA adapter 参数。Base Qwen3-1.7B 权重冻结，不为完整基础模型创建 AdamW moments。

`score_batch_size` 不继续提高到 16，因为评分服务一次最多接收一个大小为 6 的 rollout chunk，设置为大于 6 不会增加当前 pipeline 的有效 batch。Backward 在压力测试中只占整轮约 3.3%，因此继续提高 train micro-batch 的总体加速收益很小。

## 7. 已完成的端到端压力测试

压力测试输出：

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/lulu_stress_b6_s6_m2_20260914_060454
```

压力测试日志：

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/logs/lulu_stress_b6_s6_m2_20260914_060454/
├── train.log
├── gpu_memory.csv
├── gpu_processes.log
└── gpu_peak_summary.txt
```

完整一轮结果：

| 指标 | 结果 |
| --- | ---: |
| exit code | 0 |
| completed updates | 1 |
| trajectories | 64 |
| supervised trajectories | 64 |
| response tokens | 431,407 |
| reasoning tokens | 324,091 |
| 平均 response tokens/trajectory | 6,740.73 |
| response token budget 使用率 | 82.28% |
| forward KL | 0.0001811262 |
| grad norm | 0.000527963 |
| round seconds | 1,375.27 秒 |
| pipeline seconds | 1,329.57 秒 |
| update seconds | 44.94 秒 |
| Teacher score seconds | 177.04 秒 |
| Hindsight score seconds | 42.20 秒 |
| checkpoint seconds | 0.33 秒 |

GPU 峰值：

| GPU | 峰值 | 剩余显存 |
| --- | ---: | ---: |
| 0 | 23,925 MiB | 57,995 MiB |
| 1 | 24,123 MiB | 57,797 MiB |
| 2 | 26,705 MiB | 55,215 MiB |
| 4 | 23,703 MiB | 58,217 MiB |
| 5 | 12,647 MiB | 69,273 MiB |
| 6 | 46,763 MiB | 35,157 MiB |
| 7 | 46,763 MiB | 35,157 MiB |

这次是高长度的真实训练压力测试，但 `max_new_tokens=8192` 是上限，不强制所有轨迹都生成满 8192。此前合成极限测试中，Student rollout batch 8 达到约 78.7 GiB reserved，因此正式实验不采用 batch 8。已经完整验证的 `6/6/2` 是正式配置。

按照压力测试速度，100 rounds 预计约需 38.2 小时。实际耗时会随每轮生成长度变化。

## 8. 正式 100-round 输出

正式 run 目录：

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/lulu_ren_opd_dapo17k_b6_s6_m2_seed42_100r
```

正式日志：

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/logs/lulu_ren_opd_dapo17k_b6_s6_m2_seed42_100r.log
```

PID 文件：

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/logs/lulu_ren_opd_dapo17k_b6_s6_m2_seed42_100r.log.pid
```

## 9. 正式启动命令

以下命令从 base Student 开始新的 100-round 实验。正式 `OUTPUT_DIR` 不应预先包含其他文件。

```bash
cd /pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_on_policy

export PYTHON_BIN=/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/conda/envs/lulu_probe_qwen3/bin/python
export HF_HOME=/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Rona_Lulu/Lulu_outputs/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

unset TRANSFORMERS_CACHE
unset LULU_BATCH_PROFILE

mkdir -p /pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/logs

export OUTPUT_DIR=/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/lulu_ren_opd_dapo17k_b6_s6_m2_seed42_100r
export TRAIN_LOG=/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/logs/lulu_ren_opd_dapo17k_b6_s6_m2_seed42_100r.log

if [ -e "$OUTPUT_DIR" ]; then
    echo "停止启动：OUTPUT_DIR 已存在：$OUTPUT_DIR"
else
    nohup bash runs/train_lulu_7gpu.sh \
        --model Qwen/Qwen3-1.7B \
        --teacher-model /pfss/mlde/workspaces/mlde_wsp_Model_Distil/Rona_Lulu/Lulu_outputs/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 \
        --train-data /pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/data/lulu_on_policy_full_dapo/train.jsonl \
        --output-dir "$OUTPUT_DIR" \
        --method ren_opd \
        --rounds 100 \
        --global-batch-prompts 64 \
        --rollouts-per-prompt 1 \
        --rollout-batch-size 6 \
        --score-batch-size 6 \
        --train-micro-batch-size 2 \
        --update-passes 1 \
        --top-k 32 \
        --max-new-tokens 8192 \
        --max-prompt-tokens 4096 \
        --max-sequence-tokens 16384 \
        --logit-chunk-size 32 \
        --lora-rank 16 \
        --lora-alpha 32 \
        --learning-rate 1e-5 \
        --weight-decay 0 \
        --max-grad-norm 1.0 \
        --pointwise-kl-clip 0.05 \
        --gradient-checkpointing \
        --save-every 20 \
        --seed 42 \
        --worker-timeout 7200 \
        > "$TRAIN_LOG" 2>&1 < /dev/null &

    TRAIN_PID=$!
    echo "$TRAIN_PID" > "${TRAIN_LOG}.pid"
    echo "训练已启动：PID=$TRAIN_PID"
    echo "OUTPUT_DIR=$OUTPUT_DIR"
    echo "TRAIN_LOG=$TRAIN_LOG"
fi
```

## 10. 运行监控

查看训练日志：

```bash
tail -f /pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/logs/lulu_ren_opd_dapo17k_b6_s6_m2_seed42_100r.log
```

查看 GPU：

```bash
watch -n 2 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader'
```

查看主进程：

```bash
ps -fp "$(cat /pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/logs/lulu_ren_opd_dapo17k_b6_s6_m2_seed42_100r.log.pid)"
```

每轮完成后会写：

```text
<OUTPUT_DIR>/metrics/round_0000.json
<OUTPUT_DIR>/metrics/round_0001.json
...
<OUTPUT_DIR>/latest.json
```

日志中每轮的重要字段包括 `forward_kl`、`grad_norm`、`reasoning_tokens`、`supervised_trajectories`、`response_tokens`、`update_seconds`、`round_seconds` 和 `checkpoint`。

`pynvml package is deprecated` 和 `torch_dtype is deprecated` 是依赖库警告，不代表训练失败。真正的失败标记包括 `Traceback`、`CUDA out of memory`、`worker failed` 或主进程退出码非零。

## 11. Checkpoint 保留规则

当前 persistent backend 每完成一次 optimizer update，都会先写一个可恢复 checkpoint，并原子更新：

```text
<OUTPUT_DIR>/checkpoints/latest
<OUTPUT_DIR>/latest.json
```

`save_every=20` 表示长期保留以下节点：

```text
step_000000
step_000020
step_000040
step_000060
step_000080
step_000100
```

同时，当前最新的非周期 step 也会保留，用于断点恢复。例如训练完成 step 37 后，`latest` 指向 `step_000037`；step 38 完成后，旧的非周期 step 37 被新的 step 38 替代。step 20、40 等周期节点不会被删除。

每个 LoRA checkpoint 包含：

- LoRA adapter 权重和 adapter 配置；
- tokenizer；
- `optimizer.pt`，即 AdamW optimizer 状态；
- `lulu_state.json`，包含 completed updates、方法和 checkpoint 元数据。

## 12. 在 step 20 中断与恢复

中断前先确认 step 20 已经原子发布：

```bash
readlink -f /pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/lulu_ren_opd_dapo17k_b6_s6_m2_seed42_100r/checkpoints/latest
cat /pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/lulu_ren_opd_dapo17k_b6_s6_m2_seed42_100r/latest.json
```

如果 `latest` 指向 `step_000020`，恢复时使用与正式实验相同的所有科学参数，并增加 `--resume`。程序加载 adapter 和 `optimizer.pt`，从第 21 个 round 开始，不会重新运行前 20 rounds。

```bash
cd /pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_on_policy

export PYTHON_BIN=/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/conda/envs/lulu_probe_qwen3/bin/python
export HF_HOME=/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Rona_Lulu/Lulu_outputs/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
unset TRANSFORMERS_CACHE
unset LULU_BATCH_PROFILE

nohup bash runs/train_lulu_7gpu.sh \
    --resume \
    --model Qwen/Qwen3-1.7B \
    --teacher-model /pfss/mlde/workspaces/mlde_wsp_Model_Distil/Rona_Lulu/Lulu_outputs/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 \
    --train-data /pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/data/lulu_on_policy_full_dapo/train.jsonl \
    --output-dir /pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/lulu_ren_opd_dapo17k_b6_s6_m2_seed42_100r \
    --method ren_opd \
    --rounds 100 \
    --global-batch-prompts 64 \
    --rollouts-per-prompt 1 \
    --rollout-batch-size 6 \
    --score-batch-size 6 \
    --train-micro-batch-size 2 \
    --update-passes 1 \
    --top-k 32 \
    --max-new-tokens 8192 \
    --max-prompt-tokens 4096 \
    --max-sequence-tokens 16384 \
    --logit-chunk-size 32 \
    --lora-rank 16 \
    --lora-alpha 32 \
    --learning-rate 1e-5 \
    --weight-decay 0 \
    --max-grad-norm 1.0 \
    --pointwise-kl-clip 0.05 \
    --gradient-checkpointing \
    --save-every 20 \
    --seed 42 \
    --worker-timeout 7200 \
    >> /pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/logs/lulu_ren_opd_dapo17k_b6_s6_m2_seed42_100r.log 2>&1 < /dev/null &

echo $! > /pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/logs/lulu_ren_opd_dapo17k_b6_s6_m2_seed42_100r.log.pid
```

如果在 step 21 尚未完成时中断，未完成的 step 21 不会覆盖 step 20；恢复后重新执行 step 21。Checkpoint 目录先写入临时目录，完整写入后才原子发布 `latest`。

恢复前必须确认旧训练进程已经退出，不能让两个进程同时写同一个 `OUTPUT_DIR`。

## 13. Checkpoint 评测入口

保留的 step 20、40、60、80、100 可以由以下 launcher 一次性评测：

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_on_policy/runs/eval_lulu_checkpoints.sh
```

该脚本默认加载 base Student 作为对照，并依次加载 `step_000020`、`step_000040`、`step_000060`、`step_000080` 和 `step_000100` 的 LoRA adapter。正式运行前需要提供已准备好的 benchmark `DATA_MANIFEST`；runner 与 parser 已包含在 Lulu 仓库内，不再要求外部 `LULU_SORAKA_ROOT`。评测默认保持 Thinking Mode。

模板：

```bash
cd /pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_on_policy

export PYTHON_BIN=/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/conda/envs/lulu_probe_qwen3/bin/python
export TRAIN_OUTPUT_DIR=/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/lulu_ren_opd_dapo17k_b6_s6_m2_seed42_100r
export DATA_MANIFEST=/absolute/path/to/benchmark/manifest.json
export STEPS=20,40,60,80,100
export BENCHMARKS=aime25,olympiadbench
export GPUS=0,1,2,4,5,6,7

bash runs/eval_lulu_checkpoints.sh
```

不要把压力测试的 `step_000001` 当作正式训练 checkpoint。正式评测使用正式 run 目录中的保留节点。
