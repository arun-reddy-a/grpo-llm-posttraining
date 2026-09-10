"""Per-token log probabilities of completion tokens under a causal LM.

Three things in here are easy to get wrong and expensive to debug, so each is
isolated behind its own function with a test:

1. **Off-by-one alignment.** Position ``t``'s logits predict token ``t+1``.
2. **Temperature.** The ratio in the GRPO loss is only a valid importance
   weight if both log probs come from the *sampling* distribution, which means
   dividing logits by the same temperature used to generate.
3. **Memory.** A ``[N, T, V]`` logit tensor at ``V = 150k`` dwarfs the model
   itself; upcasting it to fp32 in one shot is what OOMs a GRPO run long before
   the optimizer states do.
"""

from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Any

import torch
import torch.nn as nn

__all__ = ["selective_log_softmax", "compute_per_token_logprobs", "batched_logprobs"]


def selective_log_softmax(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """``log_softmax(logits).gather(-1, index)`` without materializing the softmax.

    Args:
        logits: ``[N, T, V]``.
        index: ``[N, T]`` token ids to select.

    Returns:
        ``[N, T]`` log probabilities of the selected tokens.

    The two branches trade memory against compute for a reason. In fp32/fp64 the
    identity ``log p_i = z_i - logsumexp(z)`` is exact, so we gather the chosen
    logit and subtract a ``[N, T]`` logsumexp -- no ``[N, T, V]`` intermediate.
    In bf16/fp16 that subtraction loses too much precision to trust for a ratio
    that gets exponentiated, so we upcast -- but row by row, capping the
    transient at one sequence's worth of fp32 logits instead of the batch's.
    """
    if logits.dim() != 3:
        raise ValueError(f"logits must be [N, T, V], got {tuple(logits.shape)}")
    if index.shape != logits.shape[:2]:
        raise ValueError(
            f"index {tuple(index.shape)} does not match logits batch/time "
            f"{tuple(logits.shape[:2])}"
        )

    if logits.dtype in (torch.float32, torch.float64):
        selected = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        return selected - torch.logsumexp(logits, dim=-1)

    per_row = []
    for row_logits, row_index in zip(logits, index, strict=True):
        row_logps = torch.log_softmax(row_logits.float(), dim=-1)
        per_row.append(torch.gather(row_logps, dim=-1, index=row_index.unsqueeze(-1)).squeeze(-1))
    return torch.stack(per_row)


@lru_cache(maxsize=8)
def _logits_to_keep_kwarg(forward_cls: type) -> str | None:
    """Find this transformers version's name for the 'trim the logits' kwarg.

    It was ``num_logits_to_keep`` when introduced and was renamed to
    ``logits_to_keep`` in transformers 4.49. Sniffing the signature keeps the
    trainer working across both instead of pinning a narrow version range; if
    neither exists we fall back to slicing a full logits tensor, which is
    correct but allocates the thing we were trying to avoid.
    """
    try:
        params = inspect.signature(forward_cls.forward).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic module types
        return None
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in params:
            return name
    return None


def compute_per_token_logprobs(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_completion_tokens: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Log probs of the last ``num_completion_tokens`` tokens of each sequence.

    Args:
        model: a causal LM returning ``.logits``.
        input_ids: ``[N, L]`` prompt tokens followed by completion tokens.
        attention_mask: ``[N, L]``.
        num_completion_tokens: ``C``; the completion is ``input_ids[:, -C:]``.
        temperature: must equal the sampling temperature used to generate.

    Returns:
        ``[N, C]`` log probabilities, differentiable w.r.t. ``model``.

    Alignment: we ask for the last ``C + 1`` logit positions, which are
    ``L-C-1 .. L-1``, then drop the final one (it predicts the token *after* the
    sequence, which does not exist). What is left, ``L-C-1 .. L-2``, are exactly
    the positions whose predictions are the ``C`` completion tokens.
    """
    if num_completion_tokens < 1:
        raise ValueError(f"num_completion_tokens must be >= 1, got {num_completion_tokens}")
    if num_completion_tokens >= input_ids.shape[1]:
        raise ValueError(
            f"num_completion_tokens ({num_completion_tokens}) must leave at least one "
            f"prompt token in a sequence of length {input_ids.shape[1]}"
        )
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    keep = num_completion_tokens + 1
    kwargs: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
    kwarg_name = _logits_to_keep_kwarg(type(model))
    if kwarg_name is not None:
        kwargs[kwarg_name] = keep

    logits = model(**kwargs).logits
    # If the model understood the trim kwarg it already returned only `keep`
    # positions; otherwise (or if it silently ignored the kwarg) it returned
    # logits for the full sequence and we slice the tail ourselves -- correct
    # either way, just more memory when the kwarg isn't honored.
    if kwarg_name is None or logits.shape[1] != keep:
        logits = logits[:, -keep:, :]
    logits = logits[:, :-1, :]  # drop the position predicting past the sequence end

    completion_ids = input_ids[:, -num_completion_tokens:]
    if temperature != 1.0:
        logits = logits / temperature
    return selective_log_softmax(logits, completion_ids)


@torch.no_grad()
def batched_logprobs(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_completion_tokens: int,
    temperature: float = 1.0,
    micro_batch_size: int = 4,
) -> torch.Tensor:
    """Chunked, no-grad :func:`compute_per_token_logprobs` over the batch dim.

    Used for the reference-model and old-policy passes, which are pure
    inference: there is no reason to hold a whole rollout batch's activations
    when the result is a ``[N, C]`` tensor of floats.
    """
    if micro_batch_size < 1:
        raise ValueError(f"micro_batch_size must be >= 1, got {micro_batch_size}")
    chunks = [
        compute_per_token_logprobs(
            model,
            input_ids[i : i + micro_batch_size],
            attention_mask[i : i + micro_batch_size],
            num_completion_tokens,
            temperature,
        )
        for i in range(0, input_ids.shape[0], micro_batch_size)
    ]
    return torch.cat(chunks, dim=0)
