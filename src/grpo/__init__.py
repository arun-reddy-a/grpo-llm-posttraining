"""GRPO post-training for open-source LLMs.

A from-scratch implementation of Group Relative Policy Optimization: grouped
rollout generation, rule-based reward computation, group-relative advantage
estimation, and KL-regularized clipped policy updates.

The numerics live in :mod:`grpo.algo` and :mod:`grpo.logprobs` and depend only
on ``torch``, so they can be imported and tested without ``transformers``,
``datasets``, or a GPU. Everything that needs those is imported lazily.
"""

from .algo import (
    aggregate_per_token_loss,
    compute_group_advantages,
    group_reward_stats,
    grpo_policy_loss,
    kl_divergence_k3,
)
from .config import Config
from .logprobs import compute_per_token_logprobs, selective_log_softmax

__version__ = "0.1.0"

__all__ = [
    "Config",
    "aggregate_per_token_loss",
    "compute_group_advantages",
    "group_reward_stats",
    "grpo_policy_loss",
    "kl_divergence_k3",
    "compute_per_token_logprobs",
    "selective_log_softmax",
    "__version__",
]
