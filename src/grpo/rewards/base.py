"""Reward function protocol, registry, and weighted ensemble.

GRPO does not need a learned reward model. Because the advantage is computed
*within* a group of samples for the same prompt, only the relative ordering of
rewards inside that group matters -- which makes cheap, deterministic,
rule-based rewards a first-class choice rather than a fallback. Every reward
here is a pure function of ``(prompt, completion, dataset columns)``, so it is
free to evaluate, impossible to reward-hack via the usual RM exploits, and
testable without a GPU.

A reward may return ``None`` for a sample to mean "not applicable" (e.g. a
math-correctness reward on a row with no gold answer). ``None`` contributes 0
to the weighted total and is excluded from that function's logged mean, which
keeps a partially-applicable reward from silently dragging every group's mean
toward zero.
"""

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "RewardFunction",
    "RewardBatch",
    "RewardEnsemble",
    "register_reward",
    "build_reward",
    "available_rewards",
]


class RewardFunction(abc.ABC):
    """Scores completions. Stateless and deterministic by contract."""

    name: str = "reward"

    @abc.abstractmethod
    def __call__(
        self,
        prompts: Sequence[str],
        completions: Sequence[str],
        **columns: Any,
    ) -> list[float | None]:
        """Score each completion.

        Args:
            prompts: ``N`` prompt strings (repeated within a group).
            completions: ``N`` decoded completions.
            **columns: passthrough dataset fields, each an ``N``-length sequence
                aligned to ``completions`` -- ``answer`` is what the built-in
                math reward reads. Extra kwargs are ignored by design so a new
                reward can read a new column without touching the trainer.

        Returns:
            ``N`` floats, or ``None`` where the reward does not apply.
        """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(name={self.name!r})"


_REGISTRY: dict[str, Callable[..., RewardFunction]] = {}


def register_reward(name: str) -> Callable[[type], type]:
    """Class decorator adding a reward to the config-visible registry."""

    def decorator(cls: type) -> type:
        if name in _REGISTRY:
            raise ValueError(f"reward {name!r} is already registered")
        cls.name = name  # type: ignore[attr-defined]
        _REGISTRY[name] = cls  # type: ignore[assignment]
        return cls

    return decorator


def available_rewards() -> list[str]:
    return sorted(_REGISTRY)


def build_reward(name: str, **kwargs: Any) -> RewardFunction:
    """Instantiate a registered reward, failing loudly on an unknown name."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown reward {name!r}; available: {', '.join(available_rewards())}")
    return _REGISTRY[name](**kwargs)


@dataclass
class RewardBatch:
    """Weighted totals plus the per-function breakdown that makes them legible."""

    total: list[float]
    per_function: dict[str, list[float | None]] = field(default_factory=dict)

    def as_metrics(self) -> dict[str, float]:
        """Mean of each component reward, skipping non-applicable samples."""
        metrics: dict[str, float] = {}
        for name, values in self.per_function.items():
            applicable = [v for v in values if v is not None]
            if applicable:
                metrics[f"reward/{name}"] = sum(applicable) / len(applicable)
        return metrics


class RewardEnsemble:
    """Weighted sum of reward functions, evaluated together over one batch.

    Keeping the components separate through to logging is what makes a GRPO run
    debuggable: a flat total reward is ambiguous between "the model is not
    learning" and "the model learned to satisfy the format reward and stopped
    improving on correctness", and those call for opposite responses.
    """

    def __init__(self, functions: Sequence[RewardFunction], weights: Sequence[float] | None = None):
        if not functions:
            raise ValueError("RewardEnsemble needs at least one reward function")
        if weights is None:
            weights = [1.0] * len(functions)
        if len(weights) != len(functions):
            raise ValueError(f"got {len(functions)} functions but {len(weights)} weights")
        names = [f.name for f in functions]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate reward names in ensemble: {names}")
        self.functions = list(functions)
        self.weights = [float(w) for w in weights]

    def __call__(
        self,
        prompts: Sequence[str],
        completions: Sequence[str],
        **columns: Any,
    ) -> RewardBatch:
        n = len(completions)
        totals = [0.0] * n
        per_function: dict[str, list[float | None]] = {}

        for fn, weight in zip(self.functions, self.weights, strict=True):
            values = fn(prompts, completions, **columns)
            if len(values) != n:
                raise ValueError(
                    f"reward {fn.name!r} returned {len(values)} values for {n} completions"
                )
            per_function[fn.name] = list(values)
            for i, value in enumerate(values):
                if value is not None:
                    totals[i] += weight * float(value)

        return RewardBatch(total=totals, per_function=per_function)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        pairs = zip(self.functions, self.weights, strict=True)
        parts = ", ".join(f"{w}*{f.name}" for f, w in pairs)
        return f"RewardEnsemble({parts})"
