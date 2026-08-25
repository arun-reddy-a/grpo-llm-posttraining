"""Learning-rate schedules.

Implemented directly on ``LambdaLR`` rather than pulled from ``transformers`` so
the schedule is visible and the package does not depend on transformers for
something this small.

A note on the LR itself: GRPO runs at ~1e-6, two to three orders of magnitude
below a typical SFT run. The gradient is a policy-gradient estimate whose scale
is set by the advantages, and the model being updated is already competent --
the job is to shift a distribution, not to fit one. An SFT-scale LR here
collapses the policy's entropy within tens of steps, after which every sample in
a group is identical, variance is zero, and the run is over.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

__all__ = ["build_scheduler"]


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler: str,
    total_steps: int,
    warmup_ratio: float = 0.0,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Return a ``LambdaLR`` implementing warmup plus the named decay."""
    warmup_steps = max(0, int(round(total_steps * warmup_ratio)))

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        if scheduler == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        if scheduler == "linear":
            decayed = 1.0 - progress
        elif scheduler == "cosine":
            decayed = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            raise ValueError(f"unknown scheduler {scheduler!r}")
        return min_lr_ratio + (1.0 - min_lr_ratio) * decayed

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _typed(lr_lambda))


def _typed(fn: Callable[[int], float]) -> Callable[[int], float]:
    return fn
