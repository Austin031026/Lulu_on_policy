# 新算法：ReN Weighted OPD

启动参数：

```bash
--method ren_weighted_opd
```

## 每个位置的目标

Hindsight Student 只提供 recognition set：

```text
H = TopK(pH)
alpha = sum_{a ∈ H} qT(a)
```

`alpha` 是 Teacher 在 hindsight Top-K 上的完整词表概率质量。训练目标为：

```text
loss_t = alpha * KL(stop_gradient(qT) || pθ)
       + (1-alpha) * KL(stop_gradient(pC) || pθ)
```

- `alpha` 较大：该位置更多学习完整 Teacher 分布。
- `alpha` 较小：该位置更多保持本轮冻结 Student。
- `pH` 的概率值不作为 target，只用其 Top-K token IDs。

该算法不再构造 `H \\ C` 的稀疏 graft target。Teacher 返回 selected hidden states，Student worker 用 Teacher output head 分块重建 full-vocabulary `qT`，再计算 `alpha` 和 loss。`recognition_ids` 不发送给 Teacher。

## Clip

第一次加权实验先使用：

```bash
--pointwise-kl-clip 0
```

即保留 clip 功能，但关闭它以单独观察 weight 的效果。

以后若设置正数，先计算每个词表项的加权 contribution：

```text
weighted[v] = alpha * teacher_KL_contribution[v]
            + (1-alpha) * preserve_KL_contribution[v]
```

再执行 `min(weighted[v], clip)`，最后对 vocabulary 求和。例如：

```bash
--pointwise-kl-clip 0.05
```
