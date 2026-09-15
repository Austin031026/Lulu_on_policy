"""Stopped full-vocabulary targets for Lulu / ReN on-policy distillation.

``ren_opd`` grafts *positive* external-teacher corrections in H \\ C onto
the frozen causal student distribution, then normalizes over the whole
vocabulary. Hindsight probabilities never serve as target values in this mode.
No score weights the loss, and states without a correction retain their frozen
causal target (which can still regularize drift later in a training round).

``ren_weighted_opd`` leaves that target construction untouched and adds a
separate recognition-weighted objective. At each response position, hindsight
Top-K IDs define a recognition set, the answer-blind Teacher probability mass
on that set supplies ``alpha``, and the loss is
``alpha * KL(qT || pθ) + (1-alpha) * KL(pC || pθ)``.

The functions accept arbitrary leading position dimensions. Call them on small
position chunks after vocabulary projection to bound peak memory; no function
below stores a trajectory-sized vocabulary tensor internally or constructs a
dense frontier mask. The caller is responsible for supplying distributions from
the same frozen snapshot and masking reasoning positions, excluding answers.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


METHODS = ("ren_opd", "ren_weighted_opd", "vanilla_opd", "opsd", "causal_topk", "union_topk")


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
    """Membership in H \\ C using O(positions * K) memory, not a K² comparison."""
    ordered = causal_ids.sort(dim=-1).values.contiguous()
    where = torch.searchsorted(ordered, hindsight_ids.contiguous())
    found = ordered.gather(-1, where.clamp_max(ordered.shape[-1] - 1))
    return (where == ordered.shape[-1]) | (found != hindsight_ids)


@torch.no_grad()
def probability_mass_at_ids(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Return exact full-vocabulary probability mass assigned to unique IDs.

    ``logits`` has shape ``[..., V]`` and ``ids`` has shape ``[..., K]`` with
    identical leading dimensions. ``-1`` is accepted as padding and contributes
    no mass. The result is detached and has shape ``[...]``.
    """
    if logits.ndim < 2 or not logits.is_floating_point():
        raise ValueError("logits must be floating-point [..., vocabulary]")
    if ids.dtype != torch.long or ids.shape[:-1] != logits.shape[:-1] or ids.device != logits.device:
        raise ValueError("ids must be torch.long [..., K] on the same device with matching leading dimensions")
    vocabulary = logits.shape[-1]
    valid = ids >= 0
    if bool((ids < -1).any()) or bool((ids[valid] >= vocabulary).any()):
        raise ValueError("ids must be -1 padding or valid vocabulary IDs")
    if ids.shape[-1] > 1:
        ordered = ids.masked_fill(~valid, vocabulary).sort(dim=-1).values
        duplicate = (ordered[..., 1:] == ordered[..., :-1]) & (ordered[..., 1:] != vocabulary)
        if bool(duplicate.any()):
            raise ValueError("ids must be unique within each position")
    math_logits = logits.detach().to(_math_dtype(logits))
    selected_logp = math_logits.gather(-1, ids.clamp_min(0)) - math_logits.logsumexp(-1, keepdim=True)
    return selected_logp.exp().masked_fill(~valid, 0.0).sum(-1).clamp(0.0, 1.0)


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
    """Construct a detached probability target of shape ``[..., vocabulary]``.

    Methods:
      * ``ren_opd``: qhat[a] = qT[a] iff a ∈ TopK(pH) \\ TopK(pC) and
        qT[a] > pC[a], otherwise qhat[a] = pC[a]; return qhat / sum(qhat).
      * ``ren_weighted_opd``: the convex target with the same Student gradient
        as the recognition-weighted two-KL objective used by training.
      * ``vanilla_opd``: the external teacher's full probability distribution.
      * ``opsd``: the frozen hindsight student's full probability distribution.
      * ``causal_topk``: qT projected onto TopK(pC), then renormalized.
      * ``union_topk``: qT projected onto TopK(pC) ∪ TopK(pH), renormalized.

    The last two are support-projection controls, distinct from ReN grafting.
    ``top_k`` must be positive and is capped at vocabulary size. Rank selection
    uses the supplied logits directly; all normalization uses FP32 (FP64 if an
    input is FP64). Inputs must be finite model logits. None is accepted for
    hindsight/teacher only when the selected baseline does not use that source.

    Optional diagnostics have shape ``[...]``; no diagnostic changes the loss.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method={method!r}; expected one of {METHODS}")
    if causal_logits.ndim < 2 or causal_logits.shape[-1] < 1 or not causal_logits.is_floating_point():
        raise ValueError("causal_logits must be floating-point [..., vocabulary] with at least one position dimension")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    causal = causal_logits.detach()
    hindsight = _check_source(hindsight_logits, causal, "hindsight_logits") if method in {"ren_opd", "ren_weighted_opd", "opsd", "union_topk"} else None
    teacher = _check_source(teacher_logits, causal, "teacher_logits") if method != "opsd" else None
    dtype = _math_dtype(causal, hindsight, teacher)
    k = min(top_k, causal.shape[-1])
    stats: dict[str, torch.Tensor] = {}

    if method == "vanilla_opd":
        target_logp = F.log_softmax(teacher.to(dtype), dim=-1)
    elif method == "opsd":
        target_logp = F.log_softmax(hindsight.to(dtype), dim=-1)
    elif method == "ren_weighted_opd":
        causal_probs = F.softmax(causal.to(dtype), dim=-1)
        teacher_probs = F.softmax(teacher.to(dtype), dim=-1)
        recognition_ids = hindsight.topk(k, dim=-1, sorted=False).indices
        recognition_mass = teacher_probs.gather(-1, recognition_ids).sum(-1).clamp(0.0, 1.0)
        target = (recognition_mass.unsqueeze(-1) * teacher_probs
                  + (1.0 - recognition_mass).unsqueeze(-1) * causal_probs)
        target_logp = target.log()
        if return_diagnostics:
            stats["recognition_mass"] = recognition_mass
    else:
        causal_ids = causal.topk(k, dim=-1, sorted=False).indices
        if method == "causal_topk":
            # Global teacher log-normalization cancels under the projection.
            target_logits = torch.full_like(causal, -torch.inf, dtype=dtype)
            target_logits.scatter_(-1, causal_ids, teacher.gather(-1, causal_ids).to(dtype))
            target_logp = F.log_softmax(target_logits, dim=-1)
        else:
            hindsight_ids = hindsight.topk(k, dim=-1, sorted=False).indices
            novelty = _novel_indices(hindsight_ids, causal_ids)
            if method == "union_topk":
                target_logits = torch.full_like(causal, -torch.inf, dtype=dtype)
                target_logits.scatter_(-1, causal_ids, teacher.gather(-1, causal_ids).to(dtype))
                target_logits.scatter_(-1, hindsight_ids, teacher.gather(-1, hindsight_ids).to(dtype))
                target_logp = F.log_softmax(target_logits, dim=-1)
            else:
                causal_logp = F.log_softmax(causal.to(dtype), dim=-1)
                # Only K teacher probabilities are needed, but their normalizer
                # must cover the full vocabulary to compare qT[a] with pC[a].
                teacher_math = teacher.to(dtype)
                teacher_h_logp = teacher_math.gather(-1, hindsight_ids) - teacher_math.logsumexp(-1, keepdim=True)
                causal_h_logp = causal_logp.gather(-1, hindsight_ids)
                correction = novelty & (teacher_h_logp > causal_h_logp)
                replacement = torch.where(correction, teacher_h_logp, causal_h_logp)
                target_logits = causal_logp.clone()
                target_logits.scatter_(-1, hindsight_ids, replacement)
                target_logp = F.log_softmax(target_logits, dim=-1)
                if return_diagnostics:
                    teacher_h = teacher_h_logp.exp()
                    causal_h = causal_h_logp.exp()
                    stats = {
                        "novelty_count": novelty.sum(-1),
                        "correction_count": correction.sum(-1),
                        "added_mass": torch.where(correction, teacher_h - causal_h, 0.0).sum(-1),
                        "teacher_mass_in_region": torch.where(novelty, teacher_h, 0.0).sum(-1),
                        "causal_mass_in_region": torch.where(novelty, causal_h, 0.0).sum(-1),
                    }

    target = target_logp.exp()
    if return_diagnostics:
        # Replace -inf at zero-probability projection entries before multiplying.
        stats["target_entropy"] = -(target * target_logp.masked_fill(target == 0, 0.0)).sum(-1)
        return target, stats
    return target



@torch.no_grad()
def build_cached_target(
    causal_logits: torch.Tensor,
    correction_ids: torch.Tensor,
    teacher_probs: torch.Tensor,
    *,
    method: str = "ren_opd",
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reconstruct an exact target from sparse, frozen teacher scoring results.

    ``causal_logits`` are the frozen round-start student's full-vocabulary
    logits. ``correction_ids`` and ``teacher_probs`` have shape ``[..., K]``
    with the same leading dimensions; -1 IDs pad shorter supports, with their
    probabilities ignored. Nonpadding IDs must be unique within each position.
    For ReN, IDs must have already been restricted to H \\ C. Probabilities
    are qT at those IDs, normalized by the teacher's *whole* vocabulary, never
    by the sparse candidate set. Positive corrections are checked again here.

    For ``causal_topk`` and ``union_topk``, IDs instead contain the projected
    support and probabilities are renormalized over that support. At least one
    nonzero support entry is required per position in those projection modes.
    This representation avoids retaining or transferring dense teacher logits
    during the optimizer passes, while preserving the exact ReN background.
    """
    if method not in {"ren_opd", "causal_topk", "union_topk"}:
        raise ValueError("cached targets support ren_opd, causal_topk, and union_topk")
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
    # A private padding column prevents padded -1 entries from overwriting a
    # legitimate token-0 correction during scatter. Invalid other IDs naturally
    # raise an indexing error instead of being silently clamped.
    ids = torch.where(correction_ids == -1, vocab_size, correction_ids)
    valid = correction_ids != -1
    shape = (*causal_logits.shape[:-1], vocab_size + 1)
    scratch = torch.zeros(shape, dtype=dtype, device=causal_logits.device)
    stats: dict[str, torch.Tensor] = {}
    if method == "ren_opd":
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
    """Reduce token losses, giving each nonempty sequence equal weight.

    ``none`` returns masked position losses, ``sum`` their sum, ``token_mean``
    their mean over valid positions, and ``sequence_mean`` the mean of each
    sequence's valid-position mean. A [B,T] tensor uses its first dimension as
    sequence index; a [T] tensor denotes one sequence unless ``sequence_ids``
    supplies nonnegative integer sequence indices of the same shape. Empty
    sequences are excluded, and an entirely masked batch returns a connected
    zero with zero gradients. Mask must be boolean; it never acts as a weight.
    """
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
        # Unique indices allow sparse/global sequence IDs without huge buffers.
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


def forward_kl(
    student_logits: torch.Tensor,
    target_probs: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    reduction: str = "sequence_mean",
    sequence_ids: torch.Tensor | None = None,
    pointwise_clip: float | None = None,
) -> torch.Tensor:
    """Compute KL(stopgrad(target) || student), differentiating only the student.

    ``target_probs`` must already be a normalized probability distribution.
    When ``pointwise_clip`` is set, each vocabulary-level KL contribution is
    capped before summing over the vocabulary, matching OPSD's pointwise
    clipping. Because individual KL contributions can be negative, the clipped
    vocabulary sum is not guaranteed to remain nonnegative.
    The reduction follows the per-sequence reasoning-position mean in ReN's
    objective. Use ``reduction='none'`` when accumulating projected chunks with
    sequence weights outside this function. Zero target entries are supported.
    """
    if student_logits.ndim < 2 or student_logits.shape != target_probs.shape:
        raise ValueError("student_logits and target_probs must have identical [..., vocabulary] shapes")
    if student_logits.device != target_probs.device:
        raise ValueError("student_logits and target_probs must be on the same device")
    if not student_logits.is_floating_point() or not target_probs.is_floating_point():
        raise ValueError("student_logits and target_probs must be floating-point tensors")
    if pointwise_clip is not None and (isinstance(pointwise_clip, bool) or pointwise_clip <= 0):
        raise ValueError("pointwise_clip must be a positive number or None")
    dtype = _math_dtype(student_logits, target_probs)
    target = target_probs.detach().to(dtype)
    log_student = F.log_softmax(student_logits.to(dtype), dim=-1)
    # xlogy defines the 0*log(0) limit without introducing epsilon mass.
    contributions = torch.xlogy(target, target) - target * log_student
    if pointwise_clip is not None:
        contributions = contributions.clamp(max=pointwise_clip)
    position_kl = contributions.sum(-1)
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
    pointwise_clip: float | None = None,
    return_components: bool = False,
):
    """Recognition-weighted full-vocabulary Teacher distillation.

    At each response position the un-clipped objective is

    ``alpha * KL(q_teacher || p_student) + (1-alpha) * KL(p_causal || p_student)``.

    ``alpha`` is the frozen Teacher probability mass on the hindsight
    Student's Top-K IDs. When clipping is enabled, it is applied to each
    vocabulary contribution after the two KL contributions are weighted and
    before the vocabulary sum. All three target-side inputs are detached.
    """
    if student_logits.shape != causal_logits.shape or student_logits.shape != teacher_logits.shape:
        raise ValueError("student, causal and teacher logits must have identical [..., vocabulary] shapes")
    if student_logits.ndim < 2 or not all(
            value.is_floating_point() for value in (student_logits, causal_logits, teacher_logits)):
        raise ValueError("student, causal and teacher logits must be floating-point [..., vocabulary]")
    if not (student_logits.device == causal_logits.device == teacher_logits.device):
        raise ValueError("student, causal and teacher logits must be on the same device")
    if (recognition_mass.shape != student_logits.shape[:-1]
            or recognition_mass.device != student_logits.device
            or not recognition_mass.is_floating_point()):
        raise ValueError("recognition_mass must be floating-point, on-device and match logits leading dimensions")
    if pointwise_clip is not None and (isinstance(pointwise_clip, bool) or pointwise_clip <= 0):
        raise ValueError("pointwise_clip must be a positive number or None")

    dtype = _math_dtype(student_logits, causal_logits, teacher_logits, recognition_mass)
    alpha = recognition_mass.detach().to(dtype)
    tolerance = 1e-6
    if bool((alpha < -tolerance).any()) or bool((alpha > 1.0 + tolerance).any()):
        raise ValueError("recognition_mass must lie in [0, 1]")
    alpha = alpha.clamp(0.0, 1.0)
    causal = F.softmax(causal_logits.detach().to(dtype), dim=-1)
    teacher = F.softmax(teacher_logits.detach().to(dtype), dim=-1)
    log_student = F.log_softmax(student_logits.to(dtype), dim=-1)
    teacher_contributions = torch.xlogy(teacher, teacher) - teacher * log_student
    preserve_contributions = torch.xlogy(causal, causal) - causal * log_student
    weighted = (alpha.unsqueeze(-1) * teacher_contributions
                + (1.0 - alpha).unsqueeze(-1) * preserve_contributions)
    if pointwise_clip is not None:
        weighted = weighted.clamp(max=pointwise_clip)
    position_loss = weighted.sum(-1)
    loss = reduce_position_losses(position_loss, mask, reduction=reduction, sequence_ids=sequence_ids)
    if not return_components:
        return loss
    teacher_kl = teacher_contributions.sum(-1)
    preserve_kl = preserve_contributions.sum(-1)
    return loss, {
        "teacher_kl": reduce_position_losses(
            teacher_kl, mask, reduction=reduction, sequence_ids=sequence_ids),
        "preserve_kl": reduce_position_losses(
            preserve_kl, mask, reduction=reduction, sequence_ids=sequence_ids),
        "recognition_mass": reduce_position_losses(
            alpha, mask, reduction=reduction, sequence_ids=sequence_ids),
    }
