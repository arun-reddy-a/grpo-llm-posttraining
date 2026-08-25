"""Answer-extraction and numeric-equivalence rewards for math word problems.

The whole quality of a rule-based math reward lives in its parser. A false
negative (right answer, parser missed it) is worse than it looks under GRPO: it
does not merely lose one sample, it inverts that sample's advantage *relative to
its group*, actively training the model away from a correct behaviour. So the
extraction below is deliberately layered, most-explicit-first, and the
comparison is numeric rather than string equality -- ``0.5``, ``1/2`` and
``.50`` are the same answer.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from fractions import Fraction
from typing import Any

from .base import RewardFunction, register_reward

__all__ = [
    "extract_answer",
    "extract_gold_answer",
    "normalize_number",
    "answers_match",
    "MathCorrectnessReward",
]

_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_BOXED = re.compile(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}")
_GSM8K_GOLD = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)
_NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d+)?")
_STRIP = str.maketrans({"$": None, ",": None, "%": None, " ": None, "\u00a0": None})


def extract_answer(text: str) -> str | None:
    """Pull the model's final answer out of a completion.

    Precedence is explicit-marker before heuristic, because a marker is a
    behaviour we are actively rewarding and a trailing number is a guess:

    1. the **last** ``<answer>...</answer>`` block (last, not first, so a model
       that reconsiders is scored on its final claim);
    2. the last ``\\boxed{...}``;
    3. the last number anywhere in the text.
    """
    for pattern in (_ANSWER_TAG, _BOXED):
        matches = pattern.findall(text)
        if matches:
            candidate = matches[-1].strip()
            inner = _BOXED.findall(candidate)
            if inner:
                candidate = inner[-1].strip()
            numbers = _NUMBER.findall(candidate)
            return (numbers[-1] if numbers else candidate).strip() or None

    numbers = _NUMBER.findall(text)
    return numbers[-1].strip() if numbers else None


def extract_gold_answer(answer: str) -> str:
    """Normalize a dataset's gold field (GSM8K stores ``...\\n#### 42``)."""
    match = _GSM8K_GOLD.findall(answer)
    return (match[-1] if match else answer).strip()


def normalize_number(text: str | None) -> float | None:
    """Parse ``'$1,234.50'``, ``'3/4'``, ``'42.'`` into a float; ``None`` if not numeric."""
    if text is None:
        return None
    cleaned = text.strip().rstrip(".").translate(_STRIP)
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        pass
    try:  # bare fractions such as "3/4"
        return float(Fraction(cleaned))
    except (ValueError, ZeroDivisionError):
        return None


def answers_match(predicted: str | None, gold: str | None, rel_tol: float = 1e-6) -> bool:
    """Numeric comparison with a relative tolerance, falling back to string equality."""
    if predicted is None or gold is None:
        return False
    p, g = normalize_number(predicted), normalize_number(gold)
    if p is None or g is None:
        return predicted.strip().lower() == gold.strip().lower()
    return abs(p - g) <= rel_tol * max(1.0, abs(g))


@register_reward("math_correctness")
class MathCorrectnessReward(RewardFunction):
    """``correct_reward`` if the extracted answer matches gold, else ``incorrect_reward``.

    This is the only reward in the default recipe that carries real task signal;
    the format rewards below it exist to make *this* one parseable. Weight it
    well above them, or the model will find that reliably emitting tags pays
    better than reliably doing arithmetic.
    """

    def __init__(
        self,
        correct_reward: float = 1.0,
        incorrect_reward: float = 0.0,
        answer_column: str = "answer",
        rel_tol: float = 1e-6,
    ):
        self.correct_reward = float(correct_reward)
        self.incorrect_reward = float(incorrect_reward)
        self.answer_column = answer_column
        self.rel_tol = float(rel_tol)

    def __call__(
        self,
        prompts: Sequence[str],
        completions: Sequence[str],
        **columns: Any,
    ) -> list[float | None]:
        gold_values = columns.get(self.answer_column)
        if gold_values is None:
            # No gold column -> not applicable, rather than "everything is wrong".
            return [None] * len(completions)

        out: list[float | None] = []
        # strict=True: a gold column shorter than the batch would silently grade
        # later completions against nothing, which is worse than a crash.
        for completion, gold in zip(completions, gold_values, strict=True):
            if gold is None:
                out.append(None)
                continue
            predicted = extract_answer(completion)
            matched = answers_match(predicted, extract_gold_answer(str(gold)), self.rel_tol)
            out.append(self.correct_reward if matched else self.incorrect_reward)
        return out
