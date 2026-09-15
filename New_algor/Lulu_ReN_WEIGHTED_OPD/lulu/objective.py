"""Objectives for LuLu / ReN on-policy distillation.

The current ``ren_opd`` objective deliberately does *not* construct supervision
from ``p_H - p_C`` and does not splice selected Teacher probabilities into the
Student distribution.  Instead the three frozen views have separate roles:

* ``p_C`` (causal Student snapshot) is the policy-preservation anchor;
* ``p_H`` (same Student + gold outcome) defines recognition only;
* ``q_T`` (answer-blind external Teacher) is the knowledge source.

For each reasoning position we compute the Teacher probability mass on the
hindsight Student's Top-K actions,

    alpha_t = q_T(TopK(p_H)).

The trainable Student then minimizes

    alpha_t KL(q_T || p_theta) + (1-alpha_t) KL(p_C || p_theta).

Thus hindsight never becomes a target distribution.  It only controls how much
we trust the external Teacher; low-recognition positions stay close to the
on-policy Student snapshot.  ``ren_graft`` retains the previous H\\C positive-
correction target as an explicit legacy/ablation mode.

All functions accept arbitrary leading position dimensions.  Dense vocabulary
math should be called in small position chunks to bound peak memory.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


METHODS = ("ren_opd", "ren_graft", "vanilla_opd", "opsd", "causal_topk", "union_topk")


def _math_dtype(*values: torch.Tensor | None) -> torch.dtype:
    return torch.float64 if any(v is not None and v.dtype == torch.float64 for v in values) else torch.float32


def _check_source(source: torch.Tensor | None, causal: torch.Tensor, name: str) -> torch.Tensor:
    if source is None:
        raise ValueError(f"{name} is required for this objective")
    if source.shape != causal.shape or source.device != causal.device:
        raise ValueError(f"{name} must have the same shape and device as causal_logits")
    if not source.is_floating_point():
        raise ValueError(f"{name} must contain floating-point logits")
    return source.detach()


def _novel_indices(hindsight_ids: torch.Tensor, causal_ids: torch.Tensor) -> torch.Tensor:
    """Membership in H \\ C using O(positions * K) memory, not a K² mask."""
    ordered = causal_ids.sort(dim=-1).values.contiguous()
    where = torch.searchsorted(ordered, hindsight_ids.contiguous())
    found = ordered.gather(-1, where.clamp_max(ordered.shape[-1] - 1))
    return (where == ordered.shape[-1]) | (found != hindsight_ids)


@torch.no_grad()
def probability_mass_at_ids(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Exact full-vocabulary probability mass at ``ids``.

    ``logits`` has shape ``[..., V]`` and ``ids`` shape ``[..., K]`` with the
    same leading dimensions.  ``-1`` may pad IDs and contributes zero mass.
    The returned tensor has shape ``[...]`` and is detached FP32/FP64.
    """
    if logits.ndim < 2 or not logits.is_floating_point():
        raise ValueError("logits must be floating-point [..., vocabulary]")
    if ids.dtype != torch.long or ids.shape[:-1] != logits.shape[:-1] or ids.device != logits.device:
        raise ValueError("ids must be torch.long [..., K] on the same device with matching leading dimensions")
    vocab = logits.shape[-1]
    valid = ids >= 0
    if bool((ids < -1).any()) or bool((ids[valid] >= vocab).any()):
        raise ValueError("ids must be -1 padding or valid vocabulary IDs")
    dtype = _math_dtype(logits)
    math_logits = logits.detach().to(dtype)
    safe = ids.clamp_min(0)
    selected_logp = math_logits.gather(-1, safe) - math_logits.logsumexp(-1, keepdim=True)
    mass = selected_logp.exp().masked_fill(~valid, 0.0).sum(-1)
    # Repeated IDs would double-count probability mass and invalidate alpha.
    if ids.shape[-1] > 1:
        sorted_ids = ids.masked_fill(~valid, vocab).sort(dim=-1).values
        dup = (sorted_ids[..., 1:] == sorted_ids[..., :-1]) & (sorted_ids[..., 1:] != vocab)
        if bool(dup.any()):
            raise ValueError("ids must be unique within each position")
    return mass.clamp(0.0, 1.0)


@torch.no_grad()
def build_target(
    causal_logits: torch.Tensor,
    hindsight_logits: torch.Tensor | None,
    teacher_logits: torch.Tensor | None,
    top_k: int = 32,
    method: str = "ren_opd",
    *,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Construct detached dense targets for reference/tests and baselines.

    ``ren_opd`` returns the convex target

        m_t = alpha_t q_T + (1-alpha_t) p_C,
        alpha_t = q_T(TopK(p_H)).

    Optimizing ``KL(m_t || p_theta)`` has the same gradient w.r.t. Student
    logits as the production weighted objective

        alpha KL(q_T||p_theta) + (1-alpha) KL(p_C||p_theta),

    but production uses the latter so logged losses preserve the two-term
    interpretation.  Hindsight values never enter the target except through
    the Top-K support used to compute alpha.

    ``ren_graft`` is the previous ReN target: positive Teacher corrections on
    TopK(p_H)\\TopK(p_C) grafted into p_C and renormalized.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method={method!r}; expected one of {METHODS}")
    if causal_logits.ndim < 2 or causal_logits.shape[-1] < 1 or not causal_logits.is_floating_point():
        raise ValueError("causal_logits must be floating-point [..., vocabulary] with at least one position dimension")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    causal = causal_logits.detach()
    hindsight = _check_source(hindsight_logits, causal, "hindsight_logits") if method in {"ren_opd", "ren_graft", "opsd", "union_topk"} else None
    teacher = _check_source(teacher_logits, causal, "teacher_logits") if method != "opsd" else None
    dtype = _math_dtype(causal, hindsight, teacher)
    k = min(top_k, causal.shape[-1])
    stats: dict[str, torch.Tensor] = {}

    if method == "vanilla_opd":
        target_logp = F.log_softmax(teacher.to(dtype), dim=-1)
        target = target_logp.exp()
    elif method == "opsd":
        target_logp = F.log_softmax(hindsight.to(dtype), dim=-1)
        target = target_logp.exp()
    elif method == "ren_opd":
        causal_probs = F.softmax(causal.to(dtype), dim=-1)
        teacher_probs = F.softmax(teacher.to(dtype), dim=-1)
        hindsight_ids = hindsight.topk(k, dim=-1, sorted=False).indices
        alpha = teacher_probs.gather(-1, hindsight_ids).sum(-1).clamp(0.0, 1.0)
        target = alpha.unsqueeze(-1) * teacher_probs + (1.0 - alpha).unsqueeze(-1) * causal_probs
        target_logp = target.log()
        if return_diagnostics:
            teacher_kl_causal = (teacher_probs * (F.log_softmax(teacher.to(dtype), -1) - F.log_softmax(causal.to(dtype), -1))).sum(-1)
            stats.update(recognition_mass=alpha, teacher_causal_kl=teacher_kl_causal)
    else:
        causal_ids = causal.topk(k, dim=-1, sorted=False).indices
        if method == "causal_topk":
            target_logits = torch.full_like(causal, -torch.inf, dtype=dtype)
            target_logits.scatter_(-1, causal_ids, teacher.gather(-1, causal_ids).to(dtype))
            target_logp = F.log_softmax(target_logits, dim=-1)
            target = target_logp.exp()
        else:
            hindsight_ids = hindsight.topk(k, dim=-1, sorted=False).indices
            novelty = _novel_indices(hindsight_ids, causal_ids)
            if method == "union_topk":
                target_logits = torch.full_like(causal, -torch.inf, dtype=dtype)
                target_logits.scatter_(-1, causal_ids, teacher.gather(-1, causal_ids).to(dtype))
                target_logits.scatter_(-1, hindsight_ids, teacher.gather(-1, hindsight_ids).to(dtype))
                target_logp = F.log_softmax(target_logits, dim=-1)
                target = target_logp.exp()
            else:  # legacy ren_graft
                causal_logp = F.log_softmax(causal.to(dtype), dim=-1)
                teacher_math = teacher.to(dtype)
                teacher_h_logp = teacher_math.gather(-1, hindsight_ids) - teacher_math.logsumexp(-1, keepdim=True)
                causal_h_logp = causal_logp.gather(-1, hindsight_ids)
                correction = novelty & (teacher_h_logp > causal_h_logp)
                replacement = torch.where(correction, teacher_h_logp, causal_h_logp)
                target_logits = causal_logp.clone()
                target_logits.scatter_(-1, hindsight_ids, replacement)
                target_logp = F.log_softmax(target_logits, dim=-1)
                target = target_logp.exp()
                if return_diagnostics:
                    teacher_h = teacher_h_logp.exp()
                    causal_h = causal_h_logp.exp()
                    stats.update(
                        novelty_count=novelty.sum(-1),
                        correction_count=correction.sum(-1),
                        added_mass=torch.where(correction, teacher_h - causal_h, 0.0).sum(-1),
                        teacher_mass_in_region=torch.where(novelty, teacher_h, 0.0).sum(-1),
                        causal_mass_in_region=torch.where(novelty, causal_h, 0.0).sum(-1),
                    )

    if return_diagnostics:
        safe_log = target_logp.masked_fill(target == 0, 0.0)
        stats["target_entropy"] = -(target * safe_log).sum(-1)
        return target, stats
    return target


@torch.no_grad()
def build_cached_target(
    causal_logits: torch.Tensor,
    correction_ids: torch.Tensor,
    teacher_probs: torch.Tensor,
    *,
    method: str = "ren_graft",
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reconstruct legacy sparse targets from cached exact Teacher probabilities.

    New ``ren_opd`` intentionally does not use this sparse target path because
    its objective needs the full Teacher distribution plus a scalar recognition
    weight.  This function remains for ``ren_graft`` and Top-K projection
    controls.
    """
    if method not in {"ren_graft", "causal_topk", "union_topk"}:
        raise ValueError("cached targets support ren_graft, causal_topk, and union_topk")
    if causal_logits.ndim < 2 or not causal_logits.is_floating_point():
        raise ValueError("causal_logits must be floating-point [..., vocabulary]")
    if correction_ids.shape != teacher_probs.shape or correction_ids.shape[:-1] != causal_logits.shape[:-1]:
        raise ValueError("cached IDs/probabilities must have matching [..., K] shapes and causal leading dimensions")
    if correction_ids.dtype != torch.long or not teacher_probs.is_floating_point():
        raise ValueError("correction_ids must be torch.long and teacher_probs floating-point")
    if correction_ids.device != causal_logits.device or teacher_probs.device != causal_logits.device:
        raise ValueError("cached targets and causal logits must be on the same device")
    dtype = _math_dtype(causal_logits, teacher_probs)
    probabilities = teacher_probs.detach().to(dtype)
    vocab_size = causal_logits.shape[-1]
    ids = torch.where(correction_ids == -1, vocab_size, correction_ids)
    valid = correction_ids != -1
    shape = (*causal_logits.shape[:-1], vocab_size + 1)
    scratch = torch.zeros(shape, dtype=dtype, device=causal_logits.device)
    stats: dict[str, torch.Tensor] = {}
    if method == "ren_graft":
        scratch[..., :vocab_size] = F.softmax(causal_logits.detach().to(dtype), dim=-1)
        causal_at_ids = scratch.gather(-1, ids)
        positive = valid & (probabilities > causal_at_ids)
        replacements = torch.where(positive, probabilities, causal_at_ids)
        if return_diagnostics:
            stats = {
                "novelty_count": valid.sum(-1),
                "correction_count": positive.sum(-1),
                "added_mass": torch.where(positive, probabilities - causal_at_ids, 0.0).sum(-1),
            }
    else:
        replacements = torch.where(valid, probabilities, 0.0)
    scratch.scatter_(-1, ids, replacements)
    target = scratch[..., :vocab_size]
    target = target / target.sum(-1, keepdim=True)
    if return_diagnostics:
        stats["target_entropy"] = -torch.xlogy(target, target).sum(-1)
        return target, stats
    return target


def reduce_position_losses(
    losses: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    reduction: str = "sequence_mean",
    sequence_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reduce token losses, giving each nonempty sequence equal weight."""
    if reduction not in {"none", "sum", "token_mean", "sequence_mean"}:
        raise ValueError(f"unknown reduction={reduction!r}")
    if losses.ndim < 1:
        raise ValueError("losses must have at least one position dimension")
    if mask is None:
        valid = torch.ones_like(losses, dtype=torch.bool)
    else:
        if mask.shape != losses.shape or mask.dtype != torch.bool or mask.device != losses.device:
            raise ValueError("mask must be boolean and match the loss shape and device")
        valid = mask
    selected = torch.where(valid, losses, 0.0)
    if reduction == "none":
        return selected
    if reduction == "sum":
        return selected.sum()
    if reduction == "token_mean":
        return selected.sum() / valid.sum().clamp_min(1)
    if sequence_ids is not None:
        if sequence_ids.shape != losses.shape or sequence_ids.device != losses.device or sequence_ids.dtype != torch.long:
            raise ValueError("sequence_ids must be torch.long and match the loss shape and device")
        if sequence_ids.numel() and bool((sequence_ids < 0).any()):
            raise ValueError("sequence_ids must be nonnegative")
        unique_ids, inverse = torch.unique(sequence_ids.reshape(-1), return_inverse=True)
        sums = losses.new_zeros(unique_ids.numel()).scatter_add(0, inverse, selected.reshape(-1))
        counts = losses.new_zeros(unique_ids.numel()).scatter_add(0, inverse, valid.reshape(-1).to(losses.dtype))
    elif losses.ndim == 1:
        return selected.sum() / valid.sum().clamp_min(1)
    else:
        sums = selected.flatten(1).sum(-1)
        counts = valid.flatten(1).sum(-1)
    nonempty = counts > 0
    return (sums / counts.clamp_min(1)).sum() / nonempty.sum().clamp_min(1)


def _position_forward_kl(student_logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    dtype = _math_dtype(student_logits, target_probs)
    target = target_probs.detach().to(dtype)
    log_student = F.log_softmax(student_logits.to(dtype), dim=-1)
    return (torch.xlogy(target, target) - target * log_student).sum(-1)


def forward_kl(
    student_logits: torch.Tensor,
    target_probs: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    reduction: str = "sequence_mean",
    sequence_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute KL(stopgrad(target) || student), differentiating only Student."""
    if student_logits.ndim < 2 or student_logits.shape != target_probs.shape:
        raise ValueError("student_logits and target_probs must have identical [..., vocabulary] shapes")
    if student_logits.device != target_probs.device:
        raise ValueError("student_logits and target_probs must be on the same device")
    if not student_logits.is_floating_point() or not target_probs.is_floating_point():
        raise ValueError("student_logits and target_probs must be floating-point tensors")
    position_kl = _position_forward_kl(student_logits, target_probs)
    return reduce_position_losses(position_kl, mask, reduction=reduction, sequence_ids=sequence_ids)


def recognition_weighted_forward_kl(
    student_logits: torch.Tensor,
    causal_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    recognition_mass: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    reduction: str = "sequence_mean",
    sequence_ids: torch.Tensor | None = None,
    return_components: bool = False,
):
    """ReN objective: trust Teacher by recognition, otherwise preserve Student.

    Per position:

        alpha KL(q_T || p_theta) + (1-alpha) KL(p_C || p_theta),

    where ``alpha`` is the *frozen* Teacher probability mass assigned to the
    hindsight Student's Top-K actions.  ``p_C`` and ``q_T`` are detached; the
    hindsight Student affects only ``alpha`` and never supplies target values.
    """
    if student_logits.shape != causal_logits.shape or student_logits.shape != teacher_logits.shape:
        raise ValueError("student, causal and teacher logits must have identical [..., vocabulary] shapes")
    if student_logits.ndim < 2 or not all(x.is_floating_point() for x in (student_logits, causal_logits, teacher_logits)):
        raise ValueError("student, causal and teacher logits must be floating-point [..., vocabulary]")
    if recognition_mass.shape != student_logits.shape[:-1] or recognition_mass.device != student_logits.device:
        raise ValueError("recognition_mass must match the logits leading dimensions and device")
    if not recognition_mass.is_floating_point():
        raise ValueError("recognition_mass must be floating-point")
    alpha = recognition_mass.detach().to(_math_dtype(recognition_mass, student_logits))
    tolerance = 1e-6
    if bool((alpha < -tolerance).any()) or bool((alpha > 1 + tolerance).any()):
        raise ValueError("recognition_mass must lie in [0, 1]")
    alpha = alpha.clamp(0.0, 1.0)
    dtype = _math_dtype(student_logits, causal_logits, teacher_logits, alpha)
    causal_probs = F.softmax(causal_logits.detach().to(dtype), dim=-1)
    teacher_probs = F.softmax(teacher_logits.detach().to(dtype), dim=-1)
    teacher_kl = _position_forward_kl(student_logits, teacher_probs)
    preserve_kl = _position_forward_kl(student_logits, causal_probs)
    position_loss = alpha * teacher_kl + (1.0 - alpha) * preserve_kl
    loss = reduce_position_losses(position_loss, mask, reduction=reduction, sequence_ids=sequence_ids)
    if not return_components:
        return loss
    return loss, {
        "teacher_kl": reduce_position_losses(teacher_kl, mask, reduction=reduction, sequence_ids=sequence_ids),
        "preserve_kl": reduce_position_losses(preserve_kl, mask, reduction=reduction, sequence_ids=sequence_ids),
        "recognition_mass": reduce_position_losses(alpha, mask, reduction=reduction, sequence_ids=sequence_ids),
    }
