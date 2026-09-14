"""Rollout backend built on vLLM, with in-place weight synchronization.

Rollout generation dominates GRPO's wall-clock: every step samples
``prompts_per_step * group_size`` full completions before a single gradient is
computed, and at ``8 x 8 x 512`` tokens that is far more forward passes than the
update itself. vLLM's paged KV cache and continuous batching are worth roughly
an order of magnitude here, which moves the bottleneck back onto the optimizer
where it belongs.

The cost is a **second copy of the weights** that must be pushed forward after
every optimizer step. That sync is the fragile part of any vLLM-based RL loop and
the failure is silent: without it the sampler keeps drawing from the initial
policy, so the model is trained on data from a distribution it has already left,
the importance ratio drifts, clipping saturates, and reward flatlines with no
error anywhere. :meth:`VLLMRolloutEngine.sync_weights` is therefore loud on
failure and the trainer calls it unconditionally.

**Colocated design.** The engine shares the training process's GPU rather than
running as a separate server, so ``gpu_memory_utilization`` must leave room for
the policy, its gradients, and the Adam state -- 0.3-0.4 is a sane starting
point, not vLLM's 0.9 default, which will OOM the trainer.
"""

from __future__ import annotations

import gc
import os
from collections.abc import Iterable
from typing import Any

import torch

# Newer vLLM defaults to an out-of-process EngineCore (MPClient), so the
# driver-process attribute paths below no longer resolve -- the model isn't in
# this process's memory. This repo's design is colocated/same-process by
# construction, so force the in-process engine the sync below depends on.
# Must be set before `LLM(...)` is constructed; setdefault so an explicit
# override wins.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from .base import RolloutBatch, RolloutEngine, completion_mask_from_ids, pad_and_stack

__all__ = ["VLLMRolloutEngine"]

# vLLM has moved the executor internals more than once (V0 -> V1 engine), so we
# probe known layouts instead of pinning one version. Ordered newest-first.
_MODEL_PATHS: tuple[tuple[str, ...], ...] = (
    ("llm_engine", "model_executor", "driver_worker", "model_runner", "model"),
    ("llm_engine", "model_executor", "driver_worker", "worker", "model_runner", "model"),
    ("llm_engine", "engine_core", "engine_core", "model_executor", "driver_worker",
     "model_runner", "model"),
)


class VLLMRolloutEngine(RolloutEngine):
    """Samples with vLLM; keeps its weights in step with the training policy."""

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: Any,
        max_prompt_length: int = 512,
        max_completion_length: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        seed: int = 0,
        gpu_memory_utilization: float = 0.35,
        enforce_eager: bool = False,
        max_model_len: int | None = None,
        dtype: str = "auto",
        trust_remote_code: bool = False,
    ):
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                "rollout.backend='vllm' needs vLLM: pip install -e '.[vllm]'. "
                "Use backend='hf' for a dependency-free (slower) sampler."
            ) from exc

        self._SamplingParams = SamplingParams
        self.tokenizer = tokenizer
        self.max_completion_length = max_completion_length
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        self.llm = LLM(
            model=model_name_or_path,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            max_model_len=max_model_len or (max_prompt_length + max_completion_length),
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            seed=seed,
            # GRPO samples group_size completions from the *same* prompt every
            # step, so caching its KV prefix turns G-1 of every G prefills into
            # cache hits -- the main reason `n=group_size` sampling (below) is
            # cheap rather than G independent generate calls.
            enable_prefix_caching=True,
        )
        self._eos_ids = _eos_token_ids(tokenizer)
        self._pad_id = _pad_token_id(tokenizer)
        self._max_prompt_length = max_prompt_length

    # ------------------------------------------------------------- sampling
    def generate(self, prompts: list[str], group_size: int) -> RolloutBatch:
        # n=group_size lets vLLM share one prefill across the whole group; with
        # prefix caching on, the prompt is encoded once no matter how large G is.
        params = self._SamplingParams(
            n=group_size,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k if self.top_k > 0 else -1,
            max_tokens=self.max_completion_length,
            detokenize=True,
        )
        outputs = self.llm.generate(list(prompts), params, use_tqdm=False)

        prompt_rows: list[list[int]] = []
        completion_rows: list[list[int]] = []
        completions: list[str] = []
        repeated_prompts: list[str] = []

        for prompt, request in zip(prompts, outputs, strict=True):
            samples = list(request.outputs)
            if len(samples) != group_size:  # pragma: no cover - vLLM contract
                raise RuntimeError(
                    f"vLLM returned {len(samples)} samples for n={group_size}"
                )
            for sample in samples:
                prompt_rows.append(list(request.prompt_token_ids)[-self._max_prompt_length :])
                completion_rows.append(list(sample.token_ids))
                completions.append(sample.text)
                repeated_prompts.append(prompt)

        prompt_ids, prompt_mask = pad_and_stack(prompt_rows, self._pad_id, side="left")
        completion_ids, pad_mask = pad_and_stack(
            completion_rows, self._pad_id, side="right", max_length=self.max_completion_length
        )
        completion_mask, finished = completion_mask_from_ids(completion_ids, pad_mask, self._eos_ids)

        return RolloutBatch(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            prompts=repeated_prompts,
            completions=completions,
            finished=finished,
            group_size=group_size,
        )

    # -------------------------------------------------------- weight syncing
    def sync_weights(self, model: torch.nn.Module) -> None:
        """Load the trained policy's weights into the running vLLM engine.

        In-place ``load_weights`` rather than re-instantiating ``LLM``: rebuilding
        the engine every step would re-profile the KV cache and re-run CUDA graph
        capture, which costs far more than the update it follows.
        """
        merged = _merge_lora_if_present(model)
        try:
            state_dict = _clean_state_dict(merged)
            _resolve_vllm_model(self.llm).load_weights(state_dict.items())
        finally:
            _unmerge_lora_if_present(model)
        gc.collect()

    def close(self) -> None:  # pragma: no cover - teardown
        del self.llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- internals
def _resolve_vllm_model(llm: Any) -> Any:
    """Find the nn.Module inside the vLLM engine that exposes ``load_weights``."""
    tried: list[str] = []
    for path in _MODEL_PATHS:
        node: Any = llm
        for attr in path:
            node = getattr(node, attr, None)
            if node is None:
                break
        tried.append(".".join(path))
        if node is not None and hasattr(node, "load_weights"):
            return node
    raise RuntimeError(
        "Could not locate the vLLM model runner to sync weights into. This "
        "usually means the installed vLLM moved its executor internals.\n"
        "Paths tried:\n  - " + "\n  - ".join(tried) + "\n"
        "Add the correct attribute path to _MODEL_PATHS in "
        "src/grpo/rollout/vllm_engine.py. Continuing without a working sync "
        "would train against a stale sampler, so this is fatal by design."
    )


def _clean_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Strip wrapper prefixes so keys match vLLM's expected parameter names.

    ``torch.compile`` prefixes ``_orig_mod.``, DDP prefixes ``module.``, and PEFT
    prefixes ``base_model.model.``; vLLM knows none of them and would skip every
    unmatched key -- a sync that reports success while updating nothing.
    """
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        for prefix in ("_orig_mod.", "module.", "base_model.model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        if ".lora_" in key or key.endswith(".base_layer.weight"):
            key = key.replace(".base_layer", "")
            if ".lora_" in key:
                continue  # adapter tensors were folded in by the merge above
        cleaned[key] = value
    return cleaned


def _merge_lora_if_present(model: torch.nn.Module) -> torch.nn.Module:
    """Fold LoRA deltas into the base weights so vLLM receives dense tensors."""
    if hasattr(model, "merge_adapter"):
        model.merge_adapter()
    return model


def _unmerge_lora_if_present(model: torch.nn.Module) -> None:
    """Undo the merge -- training must continue against the unmerged adapters."""
    if hasattr(model, "unmerge_adapter"):
        model.unmerge_adapter()


def _pad_token_id(tokenizer: Any) -> int:
    for candidate in (tokenizer.pad_token_id, tokenizer.eos_token_id):
        if candidate is not None:
            return int(candidate)
    raise ValueError("tokenizer defines neither pad_token_id nor eos_token_id")


def _eos_token_ids(tokenizer: Any) -> set[int]:
    value = getattr(tokenizer, "eos_token_id", None)
    if isinstance(value, int):
        return {value}
    if isinstance(value, Iterable):
        return {int(v) for v in value if v is not None}
    return set()
