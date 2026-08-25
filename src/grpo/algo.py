"""Core GRPO math: group-relative advantages, KL estimation, clipped policy loss.

Everything in this module operates on plain tensors -- no model, tokenizer, or
dataset objects -- which is deliberate: it makes the parts of GRPO that are
easy to get subtly wrong (advantage normalization, the KL estimator, loss
aggregation, masking) unit-testable on CPU with no GPU and no downloads.
See ``docs/algorithm.md`` for the derivations behind each function.

Layout convention used throughout the package: a rollout batch of ``N``
sequences is ``num_prompts * group_size`` rows ordered so that sample ``j`` of
prompt ``i`` lives at row ``i * group_size + j``. Every function that reshapes
by group relies on that contiguity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

AdvantageMode = Literal["group_norm", "group_mean", "none"]
Aggregation = Literal["token_mean", "seq_mean_token_mean", "seq_mean_token_sum"]

__all__ = [
    "AdvantageMode",
    "Aggregation",
    "GroupStats",
    "compute_group_advantages",
    "group_reward_stats",
    "kl_divergence_k3",
    "aggregate_per_token_loss",
    "grpo_policy_loss",
]


# --------------------------------------------------------------------------
# Advantages
# --------------------------------------------------------------------------
def compute_group_advantages(
    rewards: torch.Tensor,
    group_size: int,
    mode: AdvantageMode = "group_norm",
    eps: float = 1e-4,
) -> torch.Tensor:
    """Estimate an advantage for every rollout, relative to its own group.

    GRPO's defining move: it drops the learned value network that PPO uses as a
    baseline and instead uses the *other samples drawn for the same prompt* as
    the baseline. With ``G`` samples per prompt the group mean is an unbiased,
    zero-parameter estimate of that prompt's expected return.

    Args:
        rewards: ``[N]`` scalar reward per rollout, grouped contiguously.
        group_size: ``G``; must divide ``N``.
        mode:
            ``"group_norm"``  -- ``(r - mean) / (std + eps)``, as published in
                the DeepSeekMath GRPO paper. Puts every prompt on a comparable
                advantage scale.
            ``"group_mean"``  -- ``r - mean`` only. Removes the difficulty bias
                that std-normalization introduces (see the note below).
            ``"none"``        -- raw rewards, no baseline. Diagnostic only; this
                is REINFORCE with a very high-variance gradient.
        eps: floor added to the std to keep the division finite.

    Returns:
        ``[N]`` advantages, one per rollout, in the same order as ``rewards``.

    Note:
        Dividing by the group std is not free. A prompt where the model is
        nearly always right (or nearly always wrong) has a tiny std, so its few
        deviating samples get amplified into huge advantages -- the update is
        dominated by the prompts that carry the least learning signal. That is
        the "difficulty bias" identified by Dr. GRPO, and ``"group_mean"``
        exists to turn it off. Which one is better is empirical; both are one
        config line apart on purpose.
    """
    if rewards.ndim != 1:
        raise ValueError(f"rewards must be 1-D [N], got shape {tuple(rewards.shape)}")
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}")
    if rewards.numel() % group_size != 0:
        raise ValueError(
            f"rewards length {rewards.numel()} is not divisible by group_size {group_size}"
        )

    grouped = rewards.view(-1, group_size).float()

    if mode == "none":
        return grouped.reshape(-1).to(rewards.dtype)

    centered = grouped - grouped.mean(dim=1, keepdim=True)
    if mode == "group_mean":
        return centered.reshape(-1).to(rewards.dtype)
    if mode != "group_norm":
        raise ValueError(f"unknown advantage mode {mode!r}")

    # unbiased=False when G == 1: torch's Bessel-corrected std is NaN for a
    # single sample, and a group of one carries no relative signal anyway, so
    # the right answer is a zero advantage rather than a NaN that silently
    # poisons the whole update.
    std = grouped.std(dim=1, unbiased=group_size > 1, keepdim=True)
    return (centered / (std + eps)).reshape(-1).to(rewards.dtype)


@dataclass(frozen=True)
class GroupStats:
    """Diagnostics about the reward distribution inside each prompt group."""

    mean: float
    std: float
    group_mean_std: float
    frac_zero_variance: float

    def as_metrics(self, prefix: str = "reward") -> dict[str, float]:
        return {
            f"{prefix}/mean": self.mean,
            f"{prefix}/std": self.std,
            f"{prefix}/group_std": self.group_mean_std,
            f"{prefix}/frac_zero_variance_groups": self.frac_zero_variance,
        }


def group_reward_stats(rewards: torch.Tensor, group_size: int) -> GroupStats:
    """Summarize a reward batch, including the zero-variance group fraction.

    ``frac_zero_variance`` is the metric to watch during a GRPO run. A group
    whose samples all earn the same reward produces an all-zero advantage and
    therefore contributes *no gradient at all*. If it climbs toward 1.0 the
    training signal has collapsed -- either the task became too easy (all
    correct), or the model never gets it right (all wrong), or the reward
    function has saturated. Either way the compute spent on rollouts is being
    thrown away, and the fix is a curriculum/filtering change, not a smaller LR.
    """
    grouped = rewards.view(-1, group_size).float()
    spread = grouped.max(dim=1).values - grouped.min(dim=1).values
    return GroupStats(
        mean=grouped.mean().item(),
        std=grouped.std(unbiased=grouped.numel() > 1).item(),
        group_mean_std=grouped.std(dim=1, unbiased=group_size > 1).mean().item()
        if group_size > 1
        else 0.0,
        frac_zero_variance=(spread == 0).float().mean().item(),
    )


# --------------------------------------------------------------------------
# KL regularization
# --------------------------------------------------------------------------
def kl_divergence_k3(
    policy_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    max_log_ratio: float = 20.0,
) -> torch.Tensor:
    """Per-token estimate of ``KL(pi_theta || pi_ref)`` (Schulman's ``k3``).

    For a token sampled from the policy, with ``d = log pi_ref - log pi_theta``::

        k3 = exp(d) - d - 1

    This is the estimator GRPO uses, and it beats the obvious alternatives for
    a specific reason: the naive ``-d`` estimator is unbiased but can be
    *negative* for a single sample, so the penalty term occasionally pays the
    policy to move away from the reference. ``k3`` is unbiased **and**
    non-negative for every sample (``exp(d) - d - 1 >= 0`` for all real ``d``),
    which makes the penalty behave like a penalty at every step, not just in
    expectation.

    ``max_log_ratio`` clamps ``d`` before the exponential. Without it a single
    token the policy has driven to near-zero probability sends ``exp(d)`` to
    ``inf``, and one ``inf`` turns the whole batch's loss into ``NaN``. The
    clamp bounds the penalty's contribution instead of losing the step.
    """
    log_ratio = torch.clamp(ref_logps - policy_logps, min=-max_log_ratio, max=max_log_ratio)
    return torch.exp(log_ratio) - log_ratio - 1.0


# --------------------------------------------------------------------------
# Loss aggregation
# --------------------------------------------------------------------------
def aggregate_per_token_loss(
    per_token_loss: torch.Tensor,
    completion_mask: torch.Tensor,
    aggregation: Aggregation = "token_mean",
    max_completion_length: int | None = None,
    normalizer: float | None = None,
) -> torch.Tensor:
    """Reduce a ``[N, T]`` per-token loss to a scalar under a chosen convention.

    ``normalizer`` overrides the denominator with a caller-supplied constant.
    That is what makes gradient accumulation correct rather than approximate:
    summing ``K`` micro-batch means is not the mean over their union unless every
    micro-batch carries the same denominator, and under ``token_mean`` they never
    do -- completions have different lengths. The trainer counts the whole rollout
    batch's unmasked tokens once and hands that count to every micro-batch, so
    the accumulated parts reproduce the undivided whole exactly. Left as ``None``
    the function normalizes locally, which is right for a single full-batch call.

    This looks like a bookkeeping detail and is actually a length-bias knob:

    ``"seq_mean_token_mean"``
        Mean over tokens within a sequence, then mean over sequences -- the
        formulation printed in the GRPO paper. Every *sequence* gets equal
        weight, so a token inside a 20-token completion pulls 10x harder on the
        gradient than a token inside a 200-token one.

    ``"token_mean"``
        Sum over every unmasked token in the batch, divided by that token
        count (DAPO). Every *token* gets equal weight. Usually the right
        default for reasoning tasks, where long chains are the ones you care
        about most.

    ``"seq_mean_token_sum"``
        Sum over tokens divided by the *constant* ``max_completion_length``,
        then mean over sequences (Dr. GRPO). Removes the ``1/|o_i|`` factor
        that otherwise makes a short wrong answer cheaper to keep than a long
        one, at the cost of a loss scale that depends on the length budget.
    """
    mask = completion_mask.to(per_token_loss.dtype)
    masked = per_token_loss * mask

    if aggregation == "token_mean":
        denom = mask.sum().clamp(min=1.0) if normalizer is None else _denom(normalizer, mask)
        return masked.sum() / denom
    if aggregation == "seq_mean_token_mean":
        per_seq = masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    elif aggregation == "seq_mean_token_sum":
        if not max_completion_length:
            raise ValueError(
                "aggregation='seq_mean_token_sum' needs max_completion_length "
                "(the constant normalizer is the whole point of the mode)"
            )
        per_seq = masked.sum(dim=1) / float(max_completion_length)
    else:
        raise ValueError(f"unknown aggregation {aggregation!r}")

    denom = (
        _denom(per_seq.numel(), per_seq) if normalizer is None else _denom(normalizer, per_seq)
    )
    return per_seq.sum() / denom


def _denom(normalizer: float, like: torch.Tensor) -> torch.Tensor:
    return torch.tensor(float(normalizer), dtype=like.dtype, device=like.device).clamp(min=1.0)


# --------------------------------------------------------------------------
# Policy loss
# --------------------------------------------------------------------------
def grpo_policy_loss(
    per_token_logps: torch.Tensor,
    old_per_token_logps: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    ref_per_token_logps: torch.Tensor | None = None,
    beta: float = 0.0,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
    aggregation: Aggregation = "token_mean",
    max_completion_length: int | None = None,
    normalizer: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """KL-regularized clipped surrogate loss, per token, masked to completions.

    ::

        rho_it = exp(log pi_theta(o_it) - log pi_old(o_it))
        J      = E[ min(rho_it * A_i, clip(rho_it, 1-eps_lo, 1+eps_hi) * A_i)
                    - beta * KL_k3(pi_theta || pi_ref)_it ]
        loss   = -J

    Note that ``A_i`` carries no ``t`` index: every token of a rollout is
    credited with the same group-relative advantage, because the reward is only
    observed once the sequence terminates and GRPO does not learn a value
    function to distribute it across timesteps.

    Args:
        per_token_logps: ``[N, T]`` under the *current* policy; carries grad.
        old_per_token_logps: ``[N, T]`` under the sampling policy; detached.
        advantages: ``[N]`` from :func:`compute_group_advantages`.
        completion_mask: ``[N, T]`` 1 for real completion tokens, 0 for padding
            and for anything past the first EOS.
        ref_per_token_logps: ``[N, T]`` frozen reference; required if ``beta > 0``.
        beta: KL penalty coefficient. ``0`` disables it, which also lets the
            caller skip loading a reference model entirely.
        normalizer: global loss denominator, for gradient accumulation. See
            :func:`aggregate_per_token_loss`.
        epsilon_low / epsilon_high: PPO clip range. Asymmetric by default in
            some recipes (DAPO's "clip-higher" raises only ``epsilon_high`` to
            stop low-probability tokens from being squeezed out and killing
            exploration); symmetric 0.2/0.2 here.

    Returns:
        ``(loss, metrics)`` -- metrics are detached floats for logging.
    """
    if per_token_logps.shape != old_per_token_logps.shape:
        raise ValueError(
            f"logp shape mismatch: current {tuple(per_token_logps.shape)} vs "
            f"old {tuple(old_per_token_logps.shape)}"
        )
    if advantages.shape[0] != per_token_logps.shape[0]:
        raise ValueError(
            f"advantages has {advantages.shape[0]} rows but logps has "
            f"{per_token_logps.shape[0]}"
        )
    if beta != 0.0 and ref_per_token_logps is None:
        raise ValueError("beta != 0 requires ref_per_token_logps")

    mask = completion_mask.to(per_token_logps.dtype)
    adv = advantages.to(per_token_logps.dtype).unsqueeze(1)

    ratio = torch.exp(per_token_logps - old_per_token_logps)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - epsilon_low, 1.0 + epsilon_high) * adv
    per_token_objective = torch.min(unclipped, clipped)

    metrics: dict[str, float] = {}
    if beta != 0.0:
        assert ref_per_token_logps is not None
        kl = kl_divergence_k3(per_token_logps, ref_per_token_logps)
        per_token_objective = per_token_objective - beta * kl
        metrics["kl"] = _masked_mean(kl, mask).item()
    else:
        metrics["kl"] = 0.0

    loss = aggregate_per_token_loss(
        -per_token_objective, mask, aggregation, max_completion_length, normalizer
    )

    with torch.no_grad():
        # "clipped" here means the clip actually bound -- i.e. min() selected
        # the clipped branch and the token's gradient was zeroed. With a single
        # inner epoch and old == current, ratio is exactly 1 and this is 0.0 by
        # construction; a nonzero value is only meaningful once the policy has
        # drifted from the sampler.
        was_clipped = (per_token_objective < unclipped).to(mask.dtype)
        metrics.update(
            {
                "loss": loss.item(),
                "clip_frac": _masked_mean(was_clipped, mask).item(),
                "ratio_mean": _masked_mean(ratio, mask).item(),
                "advantage_mean": advantages.float().mean().item(),
                "advantage_abs_mean": advantages.float().abs().mean().item(),
            }
        )
    return loss, metrics


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp(min=1.0)
