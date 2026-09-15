# Lulu On-Policy Distillation：通用设计

本文只记录两个算法共享的训练框架。算法目标的差异分别见：

- [原算法：ReN OPD](lulu_ren_opd.md)
- [新算法：ReN Weighted OPD](lulu_weighted_opd.md)

## 共同符号

对同一条 Student rollout 的每个 reasoning 位置：

```text
pC = 本轮冻结 Student 在 causal prompt + sampled prefix 上的分布
pH = 同一冻结 Student 在 hindsight prompt + 相同 sampled prefix 上的分布
qT = external Teacher 在 causal prompt + 相同 sampled prefix 上的分布
pθ = 当前可训练 Student 分布
```

Gold answer 只出现在 hindsight prompt。Student rollout、Teacher 输入和可训练 Student 输入都看不到 gold answer。Teacher 始终是 answer-blind。

## 共同训练流程

1. 在一轮开始时冻结 Student 快照 `S_r`。
2. `S_r` 根据 causal prompt 生成 rollout。
3. 只选择 `<think>...</think>` 内的 reasoning tokens；排除 prompt、标签、special/EOS 以及最终答案部分。
4. 对相同 sampled prefix 计算 frozen causal Student、hindsight Student 和 Teacher 信号。
5. 根据 `--method` 构造该算法的逐位置 loss。
6. 每条有效轨迹先对 reasoning positions 求平均，再对全局有效轨迹求平均。
7. 完成一次 optimizer update，得到 `S_{r+1}`，同步 hindsight Student 后开始下一轮 rollout。

训练严格使用本轮 Student 自己生成的轨迹。下一轮必须等待当前更新完成，不提前用旧参数采样。

## 共同系统设计

- 默认使用 LoRA + AdamW；Student update 使用 DDP。
- Student、hindsight Student 和 Teacher 分配在独立 GPU 角色上。
- Teacher 只接收 causal prompt、sampled response IDs 和 reasoning positions。
- sampled token IDs 直接复用，不对 rollout 重新 tokenize。
- full-vocabulary 分布按 `--logit-chunk-size` 分块重建，避免保存整条轨迹的 dense logits。
- 默认每 20 个 optimizer steps 保留历史 checkpoint；`latest` 每次更新后保存，可使用 `--resume` 恢复。

## Clip 开关

`--pointwise-kl-clip` 是共同参数：

- `0`：关闭 clip。
- 正数：在 vocabulary 求和之前，对每个 vocabulary contribution 设置上限。

它不是整个位点 KL clip，也不是新旧 Student 之间的 trust-region 约束。两个算法具体在哪里应用 clip，见各自子文档。

## 选择算法

```bash
# 原算法
--method ren_opd

# 新的加权算法
--method ren_weighted_opd
```

两种算法应使用不同的 `--output-dir`，不能在同一个 run 中途切换 method 后继续 `--resume`。
