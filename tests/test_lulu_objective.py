from __future__ import annotations

import pytest
import torch

from lulu.objective import (build_cached_target, build_target, directional_kl,
                            forward_kl, pointwise_kl_statistics,
                            probability_mass_at_ids, recognition_weighted_forward_kl,
                            reduce_position_losses, reverse_kl)


def _probabilities():
    # C top-2 = {0,1}; H top-2 = {2,3}; only token 2 has qT > pC.
    c = torch.tensor([[.40, .30, .15, .10, .05]], dtype=torch.float64)
    h = torch.tensor([[.10, .10, .40, .30, .10]], dtype=torch.float64)
    t = torch.tensor([[.20, .10, .50, .05, .15]], dtype=torch.float64)
    return c, h, t


def test_ren_exact_positive_teacher_graft_and_full_vocab_normalization():
    c, h, t = _probabilities()
    actual, stats = build_target(c.log(), h.log(), t.log(), 2, return_diagnostics=True)
    unnormalized = c.clone()
    unnormalized[0, 2] = t[0, 2]
    expected = unnormalized / unnormalized.sum(-1, keepdim=True)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.sum(-1), torch.ones(1, dtype=c.dtype))
    assert stats["novelty_count"].item() == 2
    assert stats["correction_count"].item() == 1
    assert stats["added_mass"].item() == pytest.approx(.35)
    # Outside the grafted region, relative student preferences remain intact.
    assert (actual[0, 0] / actual[0, 4]).item() == pytest.approx(8.)


def test_hindsight_defines_only_frontier_not_target_values():
    c, h, t = _probabilities()
    changed_h = torch.tensor([[.01, .01, .55, .42, .01]], dtype=h.dtype)
    first = build_target(c.log(), h.log(), t.log(), 2)
    second = build_target(c.log(), changed_h.log(), t.log(), 2)
    torch.testing.assert_close(first, second)


def test_weighted_target_uses_teacher_mass_on_hindsight_topk():
    c, h, t = _probabilities()
    actual, stats = build_target(
        c.log(), h.log(), t.log(), 2, method="ren_weighted_opd", return_diagnostics=True)
    alpha = t[0, [2, 3]].sum()
    expected = alpha * t + (1.0 - alpha) * c
    torch.testing.assert_close(actual, expected)
    assert stats["recognition_mass"].item() == pytest.approx(alpha.item())


def test_weighted_objective_and_optional_clip_are_exact():
    c, h, t = _probabilities()
    ids = h.topk(2, -1).indices
    alpha = probability_mass_at_ids(t.log(), ids)
    student = c.log().clone().requires_grad_()
    actual, parts = recognition_weighted_forward_kl(
        student, c.log(), t.log(), alpha, return_components=True)
    expected = alpha.squeeze() * forward_kl(student, t) + (1.0 - alpha.squeeze()) * forward_kl(student, c)
    torch.testing.assert_close(actual, expected)
    assert parts["recognition_mass"].item() == pytest.approx(alpha.item())
    actual.backward()
    assert student.grad.abs().sum().item() > 0

    clipped_student = c.log().clone().requires_grad_()
    logp = clipped_student.log_softmax(-1)
    teacher_terms = torch.xlogy(t, t) - t * logp
    causal_terms = torch.xlogy(c, c) - c * logp
    weighted_terms = alpha.unsqueeze(-1) * teacher_terms + (1.0 - alpha).unsqueeze(-1) * causal_terms
    clipped = recognition_weighted_forward_kl(
        clipped_student, c.log(), t.log(), alpha, pointwise_clip=0.05)
    torch.testing.assert_close(clipped, weighted_terms.clamp(max=0.05).sum(-1).mean())


def test_weighted_alpha_endpoints_preserve_or_match_teacher():
    c, _, t = _probabilities()
    preserve_student = c.log().clone().requires_grad_()
    preserve = recognition_weighted_forward_kl(
        preserve_student, c.log(), t.log(), torch.zeros(1, dtype=c.dtype))
    preserve.backward()
    torch.testing.assert_close(preserve, torch.zeros_like(preserve), atol=1e-15, rtol=0)
    torch.testing.assert_close(preserve_student.grad, torch.zeros_like(preserve_student.grad), atol=1e-15, rtol=0)

    teacher_student = c.log().clone().requires_grad_()
    weighted = recognition_weighted_forward_kl(
        teacher_student, c.log(), t.log(), torch.ones(1, dtype=c.dtype))
    vanilla = forward_kl(teacher_student, t)
    torch.testing.assert_close(weighted, vanilla)


@pytest.mark.parametrize("empty_reason", ["same_frontier", "no_positive_correction", "whole_vocabulary"])
def test_no_correction_keeps_snapshot_and_zero_initial_gradient(empty_reason):
    c, h, t = _probabilities()
    k = 2
    if empty_reason == "same_frontier":
        h = c
    elif empty_reason == "no_positive_correction":
        t = c
    else:
        k = 100
    train_logits = c.log().requires_grad_()
    target = build_target(c.log(), h.log(), t.log(), k)
    loss = forward_kl(train_logits, target)
    loss.backward()
    torch.testing.assert_close(target, c)
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-15, rtol=0)
    torch.testing.assert_close(train_logits.grad, torch.zeros_like(train_logits), atol=1e-15, rtol=0)


def test_no_correction_positions_regularize_student_drift_within_round():
    c, _, t = _probabilities()
    target = build_target(c.log(), c.log(), t.log(), 2)
    changed_logits = (c.log() + torch.tensor([[0., 0., 1., 0., 0.]])).requires_grad_()
    loss = forward_kl(changed_logits, target)
    loss.backward()
    assert loss.item() > 0
    assert changed_logits.grad.abs().sum().item() > 0


def test_target_and_kl_detach_every_supervision_source():
    sources = [v.log().requires_grad_() for v in _probabilities()]
    target, stats = build_target(*sources, 2, return_diagnostics=True)
    assert not target.requires_grad
    assert not any(value.requires_grad for value in stats.values())
    # forward_kl independently enforces stopping even for caller-built targets.
    supplied_target = target.requires_grad_()
    student = sources[0].detach().clone().requires_grad_()
    forward_kl(student, supplied_target).backward()
    assert student.grad is not None and student.grad.abs().sum() > 0
    assert supplied_target.grad is None
    assert all(source.grad is None for source in sources)


@pytest.mark.parametrize("method", ["vanilla_opd", "opsd", "causal_topk", "union_topk"])
def test_baselines_have_explicit_distinct_targets(method):
    c, h, t = _probabilities()
    expected = h.clone() if method == "opsd" else t.clone()
    if method == "causal_topk":
        expected[:, 2:] = 0
    elif method == "union_topk":
        expected[:, 4:] = 0
    expected /= expected.sum(-1, keepdim=True)
    actual = build_target(c.log(), h.log(), t.log(), 2, method)
    torch.testing.assert_close(actual, expected)
    student = c.log().requires_grad_()
    loss = forward_kl(student, actual)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(student.grad).all()


def test_unused_baseline_sources_are_optional():
    c, h, t = _probabilities()
    torch.testing.assert_close(build_target(c.log(), None, t.log(), method="vanilla_opd"), t)
    torch.testing.assert_close(build_target(c.log(), h.log(), None, method="opsd"), h)
    with pytest.raises(ValueError, match="hindsight_logits is required"):
        build_target(c.log(), None, t.log())


def test_sequence_mean_masking_does_not_reward_longer_trajectories():
    losses = torch.tensor([[1., 3., 100.], [9., 100., 100.], [100., 100., 100.]], requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, False, False], [False, False, False]])
    result = reduce_position_losses(losses, mask)
    assert result.item() == pytest.approx(5.5)
    assert reduce_position_losses(losses, mask, reduction="token_mean").item() == pytest.approx(13/3)
    result.backward()
    torch.testing.assert_close(losses.grad, torch.tensor([[.25, .25, 0.], [.5, 0., 0.], [0., 0., 0.]]))


def test_flat_sequence_ids_match_padded_sequence_reduction():
    losses = torch.tensor([1., 3., 9., 1e5], requires_grad=True)
    mask = torch.tensor([True, True, True, False])
    sequence_ids = torch.tensor([7, 7, 2000, 2000])
    actual = reduce_position_losses(losses, mask, sequence_ids=sequence_ids)
    assert actual.item() == pytest.approx(5.5)
    actual.backward()
    torch.testing.assert_close(losses.grad, torch.tensor([.25, .25, .5, 0.]))


def test_all_masked_batch_has_connected_zero_loss_and_zero_gradient():
    logits = torch.randn(2, 3, 7, requires_grad=True)
    target = torch.randn_like(logits).softmax(-1)
    loss = forward_kl(logits, target, torch.zeros(2, 3, dtype=torch.bool))
    loss.backward()
    assert loss.item() == 0
    assert torch.count_nonzero(logits.grad).item() == 0


@pytest.mark.parametrize("chunk_size", [1, 4, 100])
def test_chunked_targets_and_position_kl_match_dense_values_and_gradients(chunk_size):
    torch.manual_seed(62)
    c, h, t = [torch.randn(11, 23, dtype=torch.float64) for _ in range(3)]
    dense_student = torch.randn_like(c, requires_grad=True)
    dense_target = build_target(c, h, t, 5)
    dense_losses = forward_kl(dense_student, dense_target, reduction="none")
    dense_losses.mean().backward()
    chunk_student = dense_student.detach().clone().requires_grad_()
    chunks = []
    for start in range(0, len(c), chunk_size):
        sl = slice(start, start + chunk_size)
        target = build_target(c[sl], h[sl], t[sl], 5)
        chunks.append(forward_kl(chunk_student[sl], target, reduction="none"))
    actual = torch.cat(chunks)
    actual.mean().backward()
    torch.testing.assert_close(actual, dense_losses)
    torch.testing.assert_close(chunk_student.grad, dense_student.grad)


def test_large_logits_are_stable_and_half_inputs_use_float32_math():
    c = torch.tensor([[1000., 999., 998., -1000.]], dtype=torch.float16)
    h = torch.tensor([[999., 998., 1000., -1000.]], dtype=torch.float16)
    t = torch.tensor([[999., 998., 1000., -1000.]], dtype=torch.float16)
    target = build_target(c, h, t, 1)
    assert target.dtype == torch.float32
    assert torch.isfinite(target).all()
    loss = forward_kl(c.requires_grad_(), target)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(c.grad).all()


def test_pointwise_kl_clip_caps_vocab_contributions_before_sum():
    target = torch.tensor([[0.8, 0.2]], dtype=torch.float64)
    student = torch.tensor([[0.01, 0.99]], dtype=torch.float64).log().requires_grad_()
    unclipped_terms = target * (target.log() - student.log_softmax(-1))
    expected = unclipped_terms.clamp(max=0.05).sum(-1).mean()
    actual = forward_kl(student, target, pointwise_clip=0.05)
    torch.testing.assert_close(actual, expected)
    assert actual < unclipped_terms.sum()


def test_reverse_kl_and_pointwise_clip_are_exact_and_stop_target_gradient():
    target = torch.tensor([[0.8, 0.2]], dtype=torch.float64, requires_grad=True)
    student_probs = torch.tensor([[0.01, 0.99]], dtype=torch.float64)
    student = student_probs.log().requires_grad_()
    terms = student_probs * (student_probs.log() - target.detach().log())

    actual = reverse_kl(student, target)
    clipped = reverse_kl(student, target, pointwise_clip=0.05)
    torch.testing.assert_close(actual, terms.sum(-1).mean())
    torch.testing.assert_close(clipped, terms.clamp(max=0.05).sum(-1).mean())
    torch.testing.assert_close(
        directional_kl(student, target, direction="reverse"), actual)

    clipped.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert target.grad is None


def test_reverse_kl_floors_underflowed_dense_target_without_nonfinite_loss():
    target = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    student = torch.tensor([[0.0, 0.0]], requires_grad=True)
    loss = reverse_kl(student, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(student.grad).all()


def test_pointwise_statistics_report_exact_threshold_tail_for_both_directions():
    target = torch.tensor([[0.8, 0.2]], dtype=torch.float64)
    student_probs = torch.tensor([[0.01, 0.99]], dtype=torch.float64)
    logits = student_probs.log()
    thresholds = [0.01, 0.05]
    sums, maxima = pointwise_kl_statistics(
        logits, target, thresholds, pointwise_clip=0.05)

    expected = [
        target * (target.log() - student_probs.log()),
        student_probs * (student_probs.log() - target.log()),
    ]
    assert sums.shape == (2, 14)
    for index, contributions in enumerate(expected):
        assert sums[index, 0].item() == 1
        assert sums[index, 1].item() == 2
        torch.testing.assert_close(sums[index, 2], contributions.sum())
        torch.testing.assert_close(sums[index, 3], contributions.clamp(max=0.05).sum())
        torch.testing.assert_close(sums[index, 4], contributions.clamp_min(0).sum())
        torch.testing.assert_close(sums[index, 5], contributions.clamp_max(0).sum())
        torch.testing.assert_close(maxima[index], contributions.max())
        for threshold_index, threshold in enumerate(thresholds):
            offset = 8 + 3 * threshold_index
            assert sums[index, offset].item() == (contributions > threshold).sum().item()
            assert sums[index, offset + 1].item() == (contributions > threshold).any(-1).sum().item()
            torch.testing.assert_close(
                sums[index, offset + 2], (contributions - threshold).clamp_min(0).sum())


@pytest.mark.parametrize("value", [0, -0.1, True])
def test_invalid_pointwise_kl_clip_is_rejected(value):
    target = torch.tensor([[0.5, 0.5]])
    with pytest.raises(ValueError, match="pointwise_clip"):
        forward_kl(target.log(), target, pointwise_clip=value)


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_topk_fails_clearly(k):
    c, h, t = _probabilities()
    with pytest.raises(ValueError, match="positive integer"):
        build_target(c.log(), h.log(), t.log(), k)


@pytest.mark.parametrize("method", ["ren_opd", "causal_topk", "union_topk"])
def test_sparse_cached_target_matches_dense_teacher_and_detaches_sources(method):
    c, h, t = _probabilities()
    support = {"ren_opd": [2, 3], "causal_topk": [0, 1], "union_topk": [0, 1, 2, 3]}[method]
    ids = torch.tensor([support + [-1, -1]])
    teacher_probs = torch.tensor([[*t[0, support].tolist(), 999., 999.]], dtype=t.dtype, requires_grad=True)
    actual, stats = build_cached_target(c.log().requires_grad_(), ids, teacher_probs, method=method, return_diagnostics=True)
    expected = build_target(c.log(), h.log(), t.log(), 2, method)
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad
    assert not any(value.requires_grad for value in stats.values())


def test_sparse_padding_does_not_overwrite_token_zero_correction():
    causal = torch.tensor([[.05, .15, .8]], dtype=torch.float64)
    actual = build_cached_target(causal.log(), torch.tensor([[0, -1, -1]]), torch.tensor([[.5, 0., 0.]], dtype=causal.dtype))
    expected = torch.tensor([[.5, .15, .8]], dtype=causal.dtype)
    expected /= expected.sum(-1, keepdim=True)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("width", [0, 3])
def test_empty_sparse_corrections_preserve_full_causal_background(width):
    c, _, _ = _probabilities()
    actual = build_cached_target(c.log(), torch.full((1, width), -1, dtype=torch.long), torch.zeros(1, width))
    torch.testing.assert_close(actual, c)
