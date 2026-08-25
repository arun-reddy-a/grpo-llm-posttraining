"""Rollout backend built on ``transformers.generate``.

Slow -- an unbatched-KV, Python-loop sampler is typically 5-20x behind vLLM on
the same GPU -- but it has no extra dependency, no second copy of the weights
in memory, and no weight-sync step that can silently desynchronize. That makes
it the right backend for correctness work: smoke tests, unit-level debugging,
and confirming a change to the loss actually moved the reward before paying for
a real run.
"""

from __future__ import annotations

from typing import Any

import torch

from .base import RolloutBatch, RolloutEngine, completion_mask_from_ids, pad_and_stack

__all__ = ["HFRolloutEngine"]


class HFRolloutEngine(RolloutEngine):
    """Samples with ``model.generate`` directly from the training module."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        max_prompt_length: int = 512,
        max_completion_length: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        generation_batch_size: int = 8,
        seed: int = 0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.generation_batch_size = max(1, generation_batch_size)
        self.seed = seed
        self._eos_ids = _eos_token_ids(tokenizer, getattr(model, "generation_config", None))
        self._pad_id = _pad_token_id(tokenizer)

    @torch.no_grad()
    def generate(self, prompts: list[str], group_size: int) -> RolloutBatch:
        was_training = self.model.training
        self.model.eval()
        device = next(self.model.parameters()).device

        # Left-padding is mandatory here, not stylistic: with right-padding the
        # pad tokens sit between the prompt and the first generated token, so
        # every sequence in the batch continues from a different offset and the
        # short prompts generate a continuation of padding.
        encoded = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=True,
            max_length=self.max_prompt_length,
            add_special_tokens=False,
        )
        prompt_ids = encoded["input_ids"].to(device)
        prompt_mask = encoded["attention_mask"].to(device)

        generator = torch.Generator(device=device.type)
        generator.manual_seed(self.seed)

        completion_rows: list[list[int]] = []
        # Expand prompts to [N, P] first so row i*G+j is sample j of prompt i --
        # the contiguous-group layout the advantage computation assumes.
        expanded_ids = prompt_ids.repeat_interleave(group_size, dim=0)
        expanded_mask = prompt_mask.repeat_interleave(group_size, dim=0)

        for start in range(0, expanded_ids.shape[0], self.generation_batch_size):
            chunk_ids = expanded_ids[start : start + self.generation_batch_size]
            chunk_mask = expanded_mask[start : start + self.generation_batch_size]
            out = self.model.generate(
                input_ids=chunk_ids,
                attention_mask=chunk_mask,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k if self.top_k > 0 else 0,
                max_new_tokens=self.max_completion_length,
                pad_token_id=self._pad_id,
                return_dict_in_generate=False,
            )
            generated = out[:, chunk_ids.shape[1] :]
            for row in generated:
                completion_rows.append(_strip_trailing_pad(row.tolist(), self._pad_id, self._eos_ids))

        if was_training:
            self.model.train()

        completion_ids, pad_mask = pad_and_stack(
            completion_rows,
            pad_value=self._pad_id,
            side="right",
            max_length=self.max_completion_length,
            device=device,
        )
        completion_mask, finished = completion_mask_from_ids(completion_ids, pad_mask, self._eos_ids)
        completions = self.tokenizer.batch_decode(
            [
                ids[:n]
                for ids, n in zip(
                    completion_rows, completion_mask.sum(dim=1).tolist(), strict=True
                )
            ],
            skip_special_tokens=True,
        )

        return RolloutBatch(
            prompt_ids=expanded_ids,
            prompt_mask=expanded_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            prompts=[p for p in prompts for _ in range(group_size)],
            completions=completions,
            finished=finished,
            group_size=group_size,
        )

    def sync_weights(self, model: torch.nn.Module) -> None:
        """No-op: this backend generates from the training module itself.

        The absence of a sync step is the backend's main safety property -- the
        sampler cannot fall behind the policy because it *is* the policy.
        """


def _pad_token_id(tokenizer: Any) -> int:
    for candidate in (tokenizer.pad_token_id, tokenizer.eos_token_id):
        if candidate is not None:
            return int(candidate)
    raise ValueError("tokenizer defines neither pad_token_id nor eos_token_id")


def _eos_token_ids(tokenizer: Any, generation_config: Any = None) -> set[int]:
    """Collect every id that ends generation.

    Chat models routinely have more than one: Qwen2.5 ends assistant turns with
    ``<|im_end|>`` while ``eos_token_id`` points at ``<|endoftext|>``. Masking on
    only the nominal EOS leaves the end-of-turn token unmasked and treats every
    completion as truncated.
    """
    ids: set[int] = set()
    for source in (tokenizer, generation_config):
        value = getattr(source, "eos_token_id", None)
        if isinstance(value, int):
            ids.add(value)
        elif isinstance(value, (list, tuple, set)):
            ids.update(int(v) for v in value if v is not None)
    return ids


def _strip_trailing_pad(ids: list[int], pad_id: int, eos_ids: set[int]) -> list[int]:
    """Trim ``generate``'s right padding without trimming a real trailing EOS.

    ``generate`` pads finished rows with ``pad_token_id``, which for many chat
    models *is* the EOS id -- so a naive rstrip of pad tokens deletes the EOS
    the policy earned and makes the sequence look truncated.
    """
    end = len(ids)
    while end > 0 and ids[end - 1] == pad_id:
        end -= 1
    if end < len(ids) and pad_id in eos_ids:
        end += 1  # give back the single EOS that terminated the sequence
    return ids[:end]
