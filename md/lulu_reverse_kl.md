# Reverse KL 与 pointwise clip 诊断

## 目标

现有 forward 目标保持默认：

```text
KL(q_target || p_student)
```

新增实验方向：

```text
KL(p_student || q_target)
```

通过 `--kl-direction forward|reverse` 选择。Reverse 只改变 KL 方向，不改变 ReN target、Teacher prompt、reasoning token mask、DAPO 数据、rollout 或优化器超参数。为了得到公平对照，reverse 应从同一个 base model 新建输出目录，不应从已有 forward checkpoint 切换目标续训。

Reverse target 在 KL 数学精度下将因 FP32 softmax 下溢产生的零值 floor 到最小正数并重新归一化。加入的总概率质量可忽略，但避免数值下溢人为制造无穷 reverse KL。

## Pointwise clipping

两个方向都先计算每个 response position、每个词表项的 contribution：

```text
forward: q[v] * (log q[v] - log p[v])
reverse: p[v] * (log p[v] - log q[v])
```

随后对每个 contribution 执行上限裁剪，再对词表求和：

```text
contribution[v] = min(contribution[v], pointwise_kl_clip)
```

负 contribution 不做下限裁剪。这不是位置 KL clipping，也不是更新前后 Student KL 约束。

## 在线统计

Reverse 实验自动开启 KL 诊断；forward 实验可显式传 `--kl-diagnostics`。每个 optimizer step 的 JSON 同时记录：

- `optimization_kl` 与 `kl_direction`；
- 同一批 token 上、相同 configured clip 下的 `forward_kl` 和 `reverse_kl`；
- clipping 前与 configured clipping 后的 position mean；
- 正、负 contribution 数量与质量；
- 最大单项 contribution；
- 每个候选阈值的超限数量；
- 至少有一个词表项被该阈值裁剪的 token 数量与比例；
- 超限项占所有 vocabulary entries 和正 contribution 的比例；
- 阈值会移除的正 contribution 质量及其比例；
- 该阈值隐含的 clipped position mean。

默认候选阈值：

```text
0.001,0.002,0.005,0.01,0.02,0.05,0.1
```

这些统计在现有 `logit_chunk_size` 块内计算并跨 Student DDP ranks 汇总，不保存逐 token 全词表张量。`vocabulary_entry_count` 等计数是精确值；候选阈值给出 pointwise contribution 分布的精确 complementary CDF 采样点。

## Reverse 对照实验参数

在原 forward 实验命令上增加：

```text
--kl-direction reverse
--kl-diagnostics
--kl-diagnostic-thresholds 0.001,0.002,0.005,0.01,0.02,0.05,0.1
```

并使用新的输出目录。学习率、seed、batch、Teacher、pointwise clip 和所有 rollout 参数应与 forward 基线一致。例如当前基线保持：

```text
--learning-rate 1e-5
--pointwise-kl-clip 0.05
```

首次运行先使用 `--rounds 1` 的独立 canary 输出目录验证显存和耗时，再启动完整100步实验。诊断会增加两种 KL 的无梯度逐词表计算，因此吞吐会比只记录训练 loss 略低。

## 汇总

```bash
python scripts/analyze_kl_diagnostics.py \
  --run-dir /absolute/path/to/reverse_run
```

输出：

```text
analysis/kl_diagnostics/kl_by_step.csv
analysis/kl_diagnostics/pointwise_clip_by_step.csv
analysis/kl_diagnostics/pointwise_clip_aggregate.csv
analysis/kl_diagnostics/summary.json
analysis/kl_diagnostics/kl_curve.png
analysis/kl_diagnostics/pointwise_clip_distribution.png
```

默认经验筛选规则是在已测候选值中选最小阈值，同时满足：

```text
被裁 vocabulary entries 比例 <= 1e-4
被移除正 contribution 质量比例 <= 5%
```

这只是为 clip 消融提供起点，最终阈值仍需联合 benchmark accuracy、训练稳定性和 gradient norm 判断。
