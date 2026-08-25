"""Typed, validated, YAML-backed configuration.

Two deliberate choices here.

**Unknown keys are errors, not warnings.** A GRPO run costs GPU-hours before it
produces a number you can interpret. Silently ignoring ``beta_kl:`` because the
field is spelled ``beta`` means discovering after the run that the KL penalty
was never applied. Every dataclass below rejects keys it does not recognize.

**Cross-field checks live in :meth:`Config.validate`.** Individual fields being
in range does not make a config coherent -- ``beta > 0`` needs a reference
model, ``seq_mean_token_sum`` needs a length budget, ``group_size = 1`` makes
group-relative advantages identically zero. These fail at startup with an
explanation rather than at step 400 with a NaN.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

__all__ = ["Config", "ModelConfig", "RolloutConfig", "AlgoConfig", "OptimConfig", "TrainConfig"]

T = TypeVar("T")


def _from_dict(cls: type[T], data: dict[str, Any], path: str) -> T:
    """Build a dataclass from a dict, rejecting unknown keys with context."""
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected a mapping, got {type(data).__name__}")
    known = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"{path}: unknown config key(s) {sorted(unknown)}. "
            f"Valid keys: {sorted(known)}"
        )
    return cls(**data)  # type: ignore[call-arg]


@dataclass
class ModelConfig:
    name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    torch_dtype: str = "bfloat16"
    attn_implementation: str | None = None
    gradient_checkpointing: bool = True
    trust_remote_code: bool = False
    # LoRA cuts the optimizer state and the reference model in one move: with
    # adapters, the frozen base weights *are* the reference, so disabling the
    # adapters reproduces pi_ref without a second copy of the model in memory.
    use_lora: bool = False
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )


@dataclass
class RolloutConfig:
    backend: str = "hf"  # "hf" | "vllm"
    group_size: int = 8
    prompts_per_step: int = 8
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    max_prompt_length: int = 512
    max_completion_length: int = 512
    seed: int = 0
    # vLLM-only
    gpu_memory_utilization: float = 0.35
    enforce_eager: bool = False
    max_model_len: int | None = None
    dtype: str = "auto"
    # HF-only
    generation_batch_size: int = 8


@dataclass
class AlgoConfig:
    beta: float = 0.02
    epsilon_low: float = 0.2
    epsilon_high: float = 0.2
    advantage_mode: str = "group_norm"
    aggregation: str = "token_mean"
    num_inner_epochs: int = 1
    # Zero out completions that hit the token budget without an EOS. They are
    # graded as wrong by any answer-extraction reward, so without this the model
    # is punished for the *truncation* rather than for the reasoning -- which
    # pushes it toward shorter, safer, worse answers.
    mask_truncated_completions: bool = True
    # Drop groups whose rewards are all identical. Their advantages are zero, so
    # they add nothing but a denominator; skipping them keeps token_mean
    # aggregation from being diluted by dead samples.
    filter_zero_variance_groups: bool = False
    kl_max_log_ratio: float = 20.0


@dataclass
class OptimConfig:
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    scheduler: str = "cosine"  # "cosine" | "linear" | "constant"
    warmup_ratio: float = 0.03
    min_lr_ratio: float = 0.1
    micro_batch_size: int = 2


@dataclass
class TrainConfig:
    steps: int = 500
    seed: int = 0
    output_dir: str = "outputs/run"
    dataset: str = "gsm8k"  # "gsm8k" | path to .jsonl
    dataset_split: str = "train"
    dataset_limit: int | None = None
    log_every: int = 1
    save_every: int = 100
    eval_every: int = 0
    eval_limit: int = 200
    log_completions: int = 2
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    resume_from: str | None = None


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    rewards: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"name": "math_correctness", "weight": 1.0},
            {"name": "format", "weight": 0.2, "strict": True},
            {"name": "tag_count", "weight": 0.1},
        ]
    )

    # ---------------------------------------------------------------- loading
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        data = dict(data or {})
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown top-level config section(s) {sorted(unknown)}")

        rewards = data.pop("rewards", None)
        if rewards is not None and not isinstance(rewards, list):
            raise TypeError(f"rewards: expected a list, got {type(rewards).__name__}")

        sections = {
            "model": ModelConfig,
            "rollout": RolloutConfig,
            "algo": AlgoConfig,
            "optim": OptimConfig,
            "train": TrainConfig,
        }
        kwargs: dict[str, Any] = {
            name: _from_dict(section_cls, data[name], name)
            for name, section_cls in sections.items()
            if name in data
        }
        if rewards is not None:
            kwargs["rewards"] = rewards
        config = cls(**kwargs)
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        import yaml

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"config not found: {path}")
        with path.open(encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle) or {})

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def apply_overrides(self, overrides: list[str]) -> Config:
        """Apply ``--set section.field=value`` CLI overrides onto this config.

        Values are parsed as YAML scalars so ``true``, ``3``, ``1e-6`` and
        ``null`` arrive as the right Python types rather than as strings that
        later compare unequal to every branch that checks them.
        """
        import yaml

        data = self.to_dict()
        for override in overrides:
            if "=" not in override:
                raise ValueError(f"--set expects section.field=value, got {override!r}")
            key, raw = override.split("=", 1)
            parts = key.strip().split(".")
            if len(parts) != 2:
                raise ValueError(f"--set key must be 'section.field', got {key!r}")
            section, name = parts
            if section not in data or not isinstance(data[section], dict):
                raise ValueError(f"--set: no config section {section!r}")
            if name not in data[section]:
                raise ValueError(f"--set: {section!r} has no field {name!r}")
            data[section][name] = yaml.safe_load(raw)
        return Config.from_dict(data)

    # ------------------------------------------------------------- validation
    def validate(self) -> Config:
        r, a, o, t, m = self.rollout, self.algo, self.optim, self.train, self.model

        if m.torch_dtype not in ("bfloat16", "float16", "float32", "auto"):
            raise ValueError(
                f"model.torch_dtype must be one of bfloat16/float16/float32/auto, "
                f"got {m.torch_dtype!r}"
            )
        if m.use_lora and m.lora_r < 1:
            raise ValueError(f"model.lora_r must be >= 1, got {m.lora_r}")

        if r.backend not in ("hf", "vllm"):
            raise ValueError(f"rollout.backend must be 'hf' or 'vllm', got {r.backend!r}")
        if r.group_size < 1:
            raise ValueError(f"rollout.group_size must be >= 1, got {r.group_size}")
        if r.group_size == 1 and a.advantage_mode != "none":
            raise ValueError(
                "rollout.group_size=1 gives every sample an advantage of exactly 0 under "
                f"advantage_mode={a.advantage_mode!r} -- a group of one has nothing to be "
                "relative to, so no gradient flows. Use group_size >= 2 (4-16 is typical), "
                "or set algo.advantage_mode='none' if you deliberately want plain REINFORCE."
            )
        if r.prompts_per_step < 1:
            raise ValueError(f"rollout.prompts_per_step must be >= 1, got {r.prompts_per_step}")
        if r.temperature <= 0:
            raise ValueError(
                f"rollout.temperature must be > 0, got {r.temperature}. Greedy decoding "
                "produces identical samples within a group, hence zero variance and no "
                "learning signal."
            )
        if not 0 < r.top_p <= 1:
            raise ValueError(f"rollout.top_p must be in (0, 1], got {r.top_p}")
        if r.max_completion_length < 1 or r.max_prompt_length < 1:
            raise ValueError("rollout.max_prompt_length and max_completion_length must be >= 1")

        if a.advantage_mode not in ("group_norm", "group_mean", "none"):
            raise ValueError(f"algo.advantage_mode invalid: {a.advantage_mode!r}")
        if a.aggregation not in ("token_mean", "seq_mean_token_mean", "seq_mean_token_sum"):
            raise ValueError(f"algo.aggregation invalid: {a.aggregation!r}")
        if a.beta < 0:
            raise ValueError(f"algo.beta must be >= 0, got {a.beta}")
        if a.num_inner_epochs < 1:
            raise ValueError(f"algo.num_inner_epochs must be >= 1, got {a.num_inner_epochs}")
        if a.epsilon_low <= 0 or a.epsilon_high <= 0:
            raise ValueError("algo.epsilon_low/epsilon_high must be > 0")
        total = r.prompts_per_step * r.group_size
        if o.micro_batch_size < 1:
            raise ValueError(f"optim.micro_batch_size must be >= 1, got {o.micro_batch_size}")
        if total % o.micro_batch_size != 0:
            raise ValueError(
                f"optim.micro_batch_size ({o.micro_batch_size}) must divide "
                f"prompts_per_step * group_size ({r.prompts_per_step} * {r.group_size} = {total}); "
                "otherwise the last gradient-accumulation micro-batch is a different size and "
                "silently reweights those samples."
            )
        if o.scheduler not in ("cosine", "linear", "constant"):
            raise ValueError(f"optim.scheduler invalid: {o.scheduler!r}")
        if not 0 <= o.warmup_ratio < 1:
            raise ValueError(f"optim.warmup_ratio must be in [0, 1), got {o.warmup_ratio}")
        if o.learning_rate <= 0:
            raise ValueError(f"optim.learning_rate must be > 0, got {o.learning_rate}")

        if t.steps < 1:
            raise ValueError(f"train.steps must be >= 1, got {t.steps}")
        if not self.rewards:
            raise ValueError("at least one reward function is required")
        for spec in self.rewards:
            if not isinstance(spec, dict) or "name" not in spec:
                raise ValueError(f"each reward spec needs a 'name' key, got {spec!r}")
        return self

    # --------------------------------------------------------------- derived
    @property
    def sequences_per_step(self) -> int:
        return self.rollout.prompts_per_step * self.rollout.group_size

    @property
    def grad_accum_steps(self) -> int:
        return self.sequences_per_step // self.optim.micro_batch_size

    @property
    def needs_reference_model(self) -> bool:
        """A separate frozen copy is only needed when KL is on and LoRA is off."""
        return self.algo.beta > 0 and not self.model.use_lora
