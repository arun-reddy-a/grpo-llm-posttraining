"""Rewards for the ``<think>``/``<answer>`` output contract.

These carry no task signal at all -- they pay the model to be *parseable*.
That is not busywork: the correctness reward can only fire on completions whose
answer can be located, so at the start of a run, when the model has never seen
the tags, a pure-correctness reward is near-zero for every sample in every group
and the group-relative advantage is therefore zero everywhere. No gradient, no
learning. A dense shaping reward breaks that cold-start symmetry.

The flip side is that a format reward is the easiest thing in the recipe to
reward-hack, since emitting four tags is much cheaper than solving the problem.
Hence the deliberately small default weights in ``configs/`` and the
per-function reward logging that makes the hack visible when it happens.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from .base import RewardFunction, register_reward

__all__ = ["FormatReward", "TagCountReward"]

_STRICT = re.compile(
    r"^\s*<think>(?P<think>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>\s*$",
    re.DOTALL,
)
_SOFT = re.compile(r"<think>.*?</think>.*?<answer>.*?</answer>", re.DOTALL)
_TAGS = ("<think>", "</think>", "<answer>", "</answer>")


@register_reward("format")
class FormatReward(RewardFunction):
    """All-or-nothing reward for producing exactly the required tag structure.

    ``strict=True`` demands the completion be *nothing but* the two blocks in
    order; ``strict=False`` only demands they appear in order somewhere. Strict
    is the better training target (it also suppresses trailing chatter after
    the answer, which otherwise burns the token budget), but it is a sharper
    cliff early on -- which is what :class:`TagCountReward` is for.

    ``require_nonempty`` rejects the degenerate ``<think></think><answer></answer>``
    that a model will otherwise discover within a few hundred steps.
    """

    def __init__(self, reward: float = 1.0, strict: bool = True, require_nonempty: bool = True):
        self.reward = float(reward)
        self.strict = bool(strict)
        self.require_nonempty = bool(require_nonempty)

    def __call__(
        self,
        prompts: Sequence[str],
        completions: Sequence[str],
        **columns: Any,
    ) -> list[float | None]:
        out: list[float | None] = []
        for completion in completions:
            match = _STRICT.match(completion) if self.strict else _SOFT.search(completion)
            ok = match is not None
            if ok and self.strict and self.require_nonempty:
                assert match is not None
                ok = bool(match.group("think").strip()) and bool(match.group("answer").strip())
            # Exactly one of each tag: catches a model that pads out several
            # empty blocks to raise its odds of matching.
            if ok and any(completion.count(tag) != 1 for tag in _TAGS):
                ok = False
            out.append(self.reward if ok else 0.0)
        return out


@register_reward("tag_count")
class TagCountReward(RewardFunction):
    """Partial credit: ``reward / 4`` for each of the four tags appearing once.

    A dense stand-in for :class:`FormatReward` during the cold start. Because it
    is continuous in the number of tags emitted, groups differ from each other
    on the very first step -- which is precisely the nonzero within-group
    variance GRPO needs to produce any gradient at all.
    """

    def __init__(self, reward: float = 1.0):
        self.reward = float(reward)

    def __call__(
        self,
        prompts: Sequence[str],
        completions: Sequence[str],
        **columns: Any,
    ) -> list[float | None]:
        per_tag = self.reward / len(_TAGS)
        return [
            sum(per_tag for tag in _TAGS if completion.count(tag) == 1)
            for completion in completions
        ]
