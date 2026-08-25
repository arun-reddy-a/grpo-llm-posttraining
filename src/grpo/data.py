"""Dataset loading, prompt construction, and epoch-aware prompt sampling.

GRPO trains on *prompts*, not on (prompt, response) pairs -- the responses are
generated fresh each step. So a "batch" here is a list of prompt strings plus
whatever columns the reward functions need to grade the samples they produce.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "Sample",
    "SYSTEM_PROMPT",
    "load_gsm8k",
    "load_jsonl",
    "build_prompt",
    "PromptSampler",
]

SYSTEM_PROMPT = (
    "You are a careful mathematical reasoner. Think through the problem step by "
    "step inside <think> </think> tags, then give only the final numeric answer "
    "inside <answer> </answer> tags.\n"
    "Respond in exactly this format:\n"
    "<think>your step-by-step reasoning</think>\n"
    "<answer>final numeric answer</answer>"
)


@dataclass(frozen=True)
class Sample:
    """One training prompt plus the columns rewards grade against."""

    prompt: str
    answer: str
    metadata: dict[str, Any] = field(default_factory=dict)


def load_gsm8k(
    split: str = "train",
    limit: int | None = None,
    system_prompt: str | None = SYSTEM_PROMPT,
    tokenizer: Any | None = None,
) -> list[Sample]:
    """Load GSM8K via ``datasets`` and format each row into a chat prompt.

    Imported lazily so the package -- and its test suite -- stay importable
    without ``datasets`` installed and without network access.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "load_gsm8k needs the 'datasets' package: pip install -e '.[train]'"
        ) from exc

    raw = load_dataset("openai/gsm8k", "main", split=split)
    if limit is not None:
        raw = raw.select(range(min(limit, len(raw))))

    from .rewards.math_reward import extract_gold_answer

    return [
        Sample(
            prompt=build_prompt(row["question"], system_prompt, tokenizer),
            answer=extract_gold_answer(row["answer"]),
            metadata={"question": row["question"], "solution": row["answer"]},
        )
        for row in raw
    ]


def load_jsonl(
    path: str | Path,
    prompt_key: str = "question",
    answer_key: str = "answer",
    limit: int | None = None,
    system_prompt: str | None = SYSTEM_PROMPT,
    tokenizer: Any | None = None,
) -> list[Sample]:
    """Load prompts from a local JSONL file -- the escape hatch for your own task.

    Nothing above the reward functions is GSM8K-specific; swap this in with a
    matching reward and the trainer is unchanged.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dataset file not found: {path}")

    samples: list[Sample] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} is not valid JSON") from exc
            if prompt_key not in row:
                raise KeyError(f"{path}:{lineno} has no {prompt_key!r} field")
            samples.append(
                Sample(
                    prompt=build_prompt(row[prompt_key], system_prompt, tokenizer),
                    answer=str(row.get(answer_key, "")),
                    metadata={k: v for k, v in row.items() if k not in (prompt_key, answer_key)},
                )
            )
            if limit is not None and len(samples) >= limit:
                break
    return samples


def build_prompt(question: str, system_prompt: str | None, tokenizer: Any | None) -> str:
    """Render a question into the model's chat format.

    When a tokenizer with a chat template is supplied we use it, because a
    post-trained instruct model's behaviour is conditioned on seeing its own
    special tokens in the exact arrangement it was tuned with; hand-rolling the
    string is a reliable way to lose several points of accuracy before training
    even starts. ``add_generation_prompt=True`` leaves the sequence ending where
    the assistant turn begins, which is where generation must continue from.
    """
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system_prompt}\n\n{question}\n" if system_prompt else f"{question}\n"


class PromptSampler:
    """Infinite, seeded, epoch-shuffled iterator over prompt batches.

    GRPO runs for a step count rather than an epoch count -- each step consumes
    ``batch_size`` prompts and turns them into ``batch_size * group_size``
    sequences -- so the sampler wraps around and reshuffles instead of raising
    ``StopIteration``. ``drop_last`` keeps every step's batch the same shape,
    which keeps the advantage reshape and the logged metrics comparable
    across steps.
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        batch_size: int,
        seed: int = 0,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        if not samples:
            raise ValueError("PromptSampler needs a non-empty dataset")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if drop_last and batch_size > len(samples):
            raise ValueError(
                f"batch_size {batch_size} exceeds dataset size {len(samples)} with drop_last=True"
            )
        self.samples = list(samples)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0
        self._rng = random.Random(seed)
        self._order: list[int] = []
        self._cursor = 0
        self._reshuffle()

    def _reshuffle(self) -> None:
        self._order = list(range(len(self.samples)))
        if self.shuffle:
            self._rng.shuffle(self._order)
        self._cursor = 0

    def next_batch(self) -> list[Sample]:
        batch: list[Sample] = []
        while len(batch) < self.batch_size:
            if self._cursor >= len(self._order):
                self.epoch += 1
                self._reshuffle()
                if not self.drop_last and batch:
                    break
            take = min(self.batch_size - len(batch), len(self._order) - self._cursor)
            batch.extend(self.samples[i] for i in self._order[self._cursor : self._cursor + take])
            self._cursor += take
        return batch

    def __iter__(self) -> Iterator[list[Sample]]:
        while True:
            yield self.next_batch()
