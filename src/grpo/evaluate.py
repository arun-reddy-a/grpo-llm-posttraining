"""Held-out evaluation: greedy accuracy and pass@k.

Kept separate from the training reward on purpose. The training reward is a
*shaped* signal (correctness plus format bonuses, tuned to make learning
possible); reporting it as though it were accuracy would flatter the run,
because a model can raise it substantially by learning the tags alone. What
goes on a results table is exact-match accuracy on a split the policy never
sampled from, measured the same way before and after training.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .data import Sample
from .rewards.math_reward import answers_match, extract_answer, extract_gold_answer

__all__ = ["EvalResult", "evaluate_model"]


@dataclass
class EvalResult:
    accuracy: float
    pass_at_k: float
    num_samples: int
    k: int
    mean_length: float
    frac_truncated: float

    def as_metrics(self, prefix: str = "eval") -> dict[str, float]:
        return {
            f"{prefix}/accuracy": self.accuracy,
            f"{prefix}/pass@{self.k}": self.pass_at_k,
            f"{prefix}/mean_length": self.mean_length,
            f"{prefix}/frac_truncated": self.frac_truncated,
        }

    def __str__(self) -> str:
        return (
            f"n={self.num_samples}  accuracy={self.accuracy:.2%}  "
            f"pass@{self.k}={self.pass_at_k:.2%}  "
            f"mean_len={self.mean_length:.0f}  truncated={self.frac_truncated:.1%}"
        )


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    tokenizer: Any,
    samples: Sequence[Sample],
    max_new_tokens: int = 512,
    batch_size: int = 8,
    k: int = 1,
    temperature: float = 0.7,
    max_prompt_length: int = 512,
) -> EvalResult:
    """Score ``samples`` by exact numeric match on the extracted answer.

    ``k == 1`` decodes greedily -- the deterministic, comparable number. ``k > 1``
    samples ``k`` completions per prompt at ``temperature`` and reports pass@k
    alongside the mean single-sample accuracy. Both are worth having: GRPO tends
    to move accuracy up and pass@k much less, because it mostly sharpens the
    policy onto solutions the base model could already reach rather than
    teaching it new ones.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    correct_any = 0
    correct_first = 0.0
    total_length = 0
    truncated = 0
    generated_count = 0

    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        encoded = tokenizer(
            [s.prompt for s in chunk],
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=True,
            max_length=max_prompt_length,
            add_special_tokens=False,
        ).to(device)

        outputs = model.generate(
            **encoded,
            do_sample=k > 1,
            temperature=temperature if k > 1 else None,
            top_p=1.0 if k > 1 else None,
            num_return_sequences=k,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_id,
        )
        completions = tokenizer.batch_decode(
            outputs[:, encoded["input_ids"].shape[1] :], skip_special_tokens=True
        )

        for i, sample in enumerate(chunk):
            gold = extract_gold_answer(sample.answer)
            group = completions[i * k : (i + 1) * k]
            hits = [answers_match(extract_answer(c), gold) for c in group]
            correct_any += int(any(hits))
            correct_first += sum(hits) / len(hits)
            for out_row in outputs[i * k : (i + 1) * k]:
                length = int((out_row[encoded["input_ids"].shape[1] :] != pad_id).sum())
                total_length += length
                truncated += int(length >= max_new_tokens)
                generated_count += 1

    if was_training:
        model.train()

    n = max(1, len(samples))
    return EvalResult(
        accuracy=correct_first / n,
        pass_at_k=correct_any / n,
        num_samples=len(samples),
        k=k,
        mean_length=total_length / max(1, generated_count),
        frac_truncated=truncated / max(1, generated_count),
    )
