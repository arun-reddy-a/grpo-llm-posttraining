"""Composable, rule-based reward functions for GRPO.

Importing this package registers the built-ins, so ``build_reward("format")``
works from a config file without the config needing to know import paths.
"""

from .base import (
    RewardBatch,
    RewardEnsemble,
    RewardFunction,
    available_rewards,
    build_reward,
    register_reward,
)
from .format_reward import FormatReward, TagCountReward
from .math_reward import (
    MathCorrectnessReward,
    answers_match,
    extract_answer,
    extract_gold_answer,
    normalize_number,
)

__all__ = [
    "RewardBatch",
    "RewardEnsemble",
    "RewardFunction",
    "available_rewards",
    "build_reward",
    "register_reward",
    "FormatReward",
    "TagCountReward",
    "MathCorrectnessReward",
    "answers_match",
    "extract_answer",
    "extract_gold_answer",
    "normalize_number",
]


def build_ensemble(specs: list[dict]) -> RewardEnsemble:
    """Build a :class:`RewardEnsemble` from config dicts.

    Each spec is ``{"name": ..., "weight": 1.0, **kwargs}``; ``kwargs`` go to
    the reward's constructor, so a config typo raises a ``TypeError`` at startup
    instead of quietly training against a default.
    """
    functions, weights = [], []
    for spec in specs:
        params = dict(spec)
        name = params.pop("name", None)
        if not name:
            raise ValueError(f"reward spec missing 'name': {spec}")
        weights.append(float(params.pop("weight", 1.0)))
        functions.append(build_reward(name, **params))
    return RewardEnsemble(functions, weights)
