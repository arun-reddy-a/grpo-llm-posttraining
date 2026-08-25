"""Rollout backends. Import the concrete engines lazily -- vLLM is optional."""

from .base import RolloutBatch, RolloutEngine, completion_mask_from_ids, pad_and_stack
from .hf_engine import HFRolloutEngine

__all__ = [
    "RolloutBatch",
    "RolloutEngine",
    "completion_mask_from_ids",
    "pad_and_stack",
    "HFRolloutEngine",
    "build_engine",
]


def build_engine(config, model, tokenizer):
    """Construct the rollout engine named by ``config.rollout.backend``."""
    r = config.rollout
    if r.backend == "hf":
        return HFRolloutEngine(
            model=model,
            tokenizer=tokenizer,
            max_prompt_length=r.max_prompt_length,
            max_completion_length=r.max_completion_length,
            temperature=r.temperature,
            top_p=r.top_p,
            top_k=r.top_k,
            generation_batch_size=r.generation_batch_size,
            seed=r.seed,
        )
    if r.backend == "vllm":
        from .vllm_engine import VLLMRolloutEngine  # imported here: optional dep

        return VLLMRolloutEngine(
            model_name_or_path=config.model.name_or_path,
            tokenizer=tokenizer,
            max_prompt_length=r.max_prompt_length,
            max_completion_length=r.max_completion_length,
            temperature=r.temperature,
            top_p=r.top_p,
            top_k=r.top_k,
            seed=r.seed,
            gpu_memory_utilization=r.gpu_memory_utilization,
            enforce_eager=r.enforce_eager,
            max_model_len=r.max_model_len,
            dtype=r.dtype,
            trust_remote_code=config.model.trust_remote_code,
        )
    raise ValueError(f"unknown rollout backend {r.backend!r}")
