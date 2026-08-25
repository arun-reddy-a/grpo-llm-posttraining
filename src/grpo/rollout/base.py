"""The rollout interface that decouples GRPO's sampler from its trainer.

GRPO alternates between two workloads with opposite performance profiles:
memory-bound autoregressive *generation* (many short forward passes, batch of
one token) and compute-bound *training* (one big forward/backward). vLLM is
built for the first and useless for the second; a plain HF module is fine at
the second and slow at the first. Rather than hard-code either, the trainer
talks to :class:`RolloutEngine` and the backend is a config line.

Both backends must return token ids, never strings. Detokenizing a completion
and re-tokenizing it for the training forward pass is not an identity round
trip -- byte-level BPE merges across a boundary can change the token count, and
a shifted-by-one completion silently trains on the wrong log probs.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import torch

__all__ = ["RolloutBatch", "RolloutEngine", "pad_and_stack", "completion_mask_from_ids"]


@dataclass
class RolloutBatch:
    """One GRPO step's worth of sampled sequences.

    Rows are ordered so that sample ``j`` of prompt ``i`` is at index
    ``i * group_size + j``; :func:`grpo.algo.compute_group_advantages` reshapes
    on that assumption.

    Attributes:
        prompt_ids: ``[N, P]``, **left**-padded so every real prompt ends at the
            same column and generation continues from a common offset.
        completion_ids: ``[N, C]``, **right**-padded.
        completion_mask: ``[N, C]``, 1 through the first EOS inclusive, 0 after.
        finished: ``[N]`` bool -- False means the sample hit the token budget
            without emitting EOS, i.e. it was truncated mid-thought.
    """

    prompt_ids: torch.Tensor
    prompt_mask: torch.Tensor
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    prompts: list[str]
    completions: list[str]
    finished: torch.Tensor
    group_size: int

    def __post_init__(self) -> None:
        n = self.prompt_ids.shape[0]
        if n % self.group_size != 0:
            raise ValueError(f"batch of {n} rows is not divisible by group_size {self.group_size}")
        for name, tensor in (
            ("prompt_mask", self.prompt_mask),
            ("completion_ids", self.completion_ids),
            ("completion_mask", self.completion_mask),
            ("finished", self.finished),
        ):
            if tensor.shape[0] != n:
                raise ValueError(f"{name} has {tensor.shape[0]} rows, expected {n}")
        if len(self.prompts) != n or len(self.completions) != n:
            raise ValueError(
                f"expected {n} prompt/completion strings, got "
                f"{len(self.prompts)}/{len(self.completions)}"
            )

    @property
    def num_sequences(self) -> int:
        return self.prompt_ids.shape[0]

    @property
    def num_prompts(self) -> int:
        return self.num_sequences // self.group_size

    @property
    def max_completion_tokens(self) -> int:
        return self.completion_ids.shape[1]

    def to(self, device: torch.device | str) -> RolloutBatch:
        return RolloutBatch(
            prompt_ids=self.prompt_ids.to(device),
            prompt_mask=self.prompt_mask.to(device),
            completion_ids=self.completion_ids.to(device),
            completion_mask=self.completion_mask.to(device),
            prompts=self.prompts,
            completions=self.completions,
            finished=self.finished.to(device),
            group_size=self.group_size,
        )

    def merged(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate prompt and completion into the training forward's inputs."""
        return (
            torch.cat([self.prompt_ids, self.completion_ids], dim=1),
            torch.cat([self.prompt_mask, self.completion_mask], dim=1),
        )

    def completion_lengths(self) -> torch.Tensor:
        return self.completion_mask.sum(dim=1)


class RolloutEngine(abc.ABC):
    """Generates ``group_size`` samples for each prompt and syncs policy weights."""

    @abc.abstractmethod
    def generate(self, prompts: list[str], group_size: int) -> RolloutBatch:
        """Sample ``group_size`` completions per prompt, grouped contiguously."""

    @abc.abstractmethod
    def sync_weights(self, model: torch.nn.Module) -> None:
        """Push the current policy weights into the sampler.

        Called after every optimizer step. Skipping it is the single most
        common way to turn a GRPO run into an expensive no-op: the sampler
        keeps drawing from the initial policy, so the ratio drifts further from
        1 every step, clipping saturates, and the reported reward never moves.
        """

    def close(self) -> None:  # noqa: B027  # pragma: no cover - backend specific
        """Release sampler resources (GPU memory, worker processes).

        Intentionally concrete and empty rather than abstract: the HF backend
        owns no resources of its own, so forcing every engine to implement a
        no-op would be noise.
        """


def pad_and_stack(
    sequences: list[list[int]],
    pad_value: int,
    side: str = "right",
    max_length: int | None = None,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad ragged id lists into ``([N, T] ids, [N, T] mask)``."""
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    if not sequences:
        empty = torch.zeros((0, 0), dtype=torch.long, device=device)
        return empty, empty.clone()

    width = max_length or max((len(s) for s in sequences), default=0)
    width = max(width, 1)
    ids = torch.full((len(sequences), width), pad_value, dtype=torch.long, device=device)
    mask = torch.zeros((len(sequences), width), dtype=torch.long, device=device)
    for row, seq in enumerate(sequences):
        trimmed = seq[-width:] if side == "left" else seq[:width]
        span = slice(width - len(trimmed), width) if side == "left" else slice(0, len(trimmed))
        ids[row, span] = torch.tensor(trimmed, dtype=torch.long, device=device)
        mask[row, span] = 1
    return ids, mask


def completion_mask_from_ids(
    completion_ids: torch.Tensor,
    pad_mask: torch.Tensor,
    eos_token_ids: set[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask through the first EOS inclusive; report which rows terminated.

    Keeping EOS *inside* the mask is deliberate. Choosing to stop is an action
    the policy took, and if it carries no gradient the model is never rewarded
    for ending a correct answer -- one of the ways a run drifts toward rambling
    completions that run out the token budget.

    Args:
        completion_ids: ``[N, C]``.
        pad_mask: ``[N, C]`` 1 where the generator produced a token at all.
        eos_token_ids: every id that terminates generation (models often have
            several -- e.g. a chat template's end-of-turn token alongside the
            tokenizer's nominal EOS).

    Returns:
        ``(mask [N, C], finished [N] bool)``.
    """
    if not eos_token_ids:
        return pad_mask.clone(), pad_mask.new_zeros(completion_ids.shape[0], dtype=torch.bool)

    eos_tensor = torch.tensor(sorted(eos_token_ids), device=completion_ids.device)
    is_eos = torch.isin(completion_ids, eos_tensor) & pad_mask.bool()

    finished = is_eos.any(dim=1)
    first_eos = torch.where(
        finished,
        is_eos.float().argmax(dim=1),
        torch.full_like(is_eos[:, 0].long(), completion_ids.shape[1] - 1),
    )
    positions = torch.arange(completion_ids.shape[1], device=completion_ids.device)
    mask = (positions.unsqueeze(0) <= first_eos.unsqueeze(1)).long() * pad_mask
    return mask, finished
