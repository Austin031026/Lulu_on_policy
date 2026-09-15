# 原算法：ReN OPD

启动参数：

```bash
--method ren_opd
```

## 每个位置的目标

先计算：

```text
C = TopK(pC)
H = TopK(pH)
R = H \\ C
```

只保留 `R` 中满足 `qT(a) > pC(a)` 的 token。把这些 token 的 Student 概率替换为 Teacher 概率，其余词表继续使用 frozen causal Student：

```text
q_hat[a] = qT[a]，当 a ∈ R 且 qT[a] > pC[a]
q_hat[a] = pC[a]，其他情况
q_target = q_hat / sum(q_hat)
loss_t = KL(stop_gradient(q_target) || pθ)
```

因此该算法执行的是稀疏 positive Teacher graft：hindsight 决定候选 token，Teacher 只修正少量满足条件的 token，完整 `pC` 保留为背景分布。

没有 positive correction 时，`q_target=pC`。在本轮更新起点该位置梯度为零；Student 被其他位置更新后，它会约束 Student 回到本轮冻结策略。

## Clip

默认配置为：

```bash
--pointwise-kl-clip 0.05
```

clip 作用于 `KL(q_target||pθ)` 的每个 vocabulary contribution，然后才对词表求和。传 `0` 可关闭。
