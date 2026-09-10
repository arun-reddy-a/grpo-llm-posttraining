"""The GRPO training loop.

One step, end to end::

    1. sample G completions for each of B prompts            (rollout engine)
    2. score every completion with rule-based rewards        (reward ensemble)
    3. center/normalize rewards within each prompt group     -> advantages
    4. forward the policy (and reference) over the batch     -> per-token logps
    5. clipped surrogate + beta * KL, accumulated over micro-batches
    6. optimizer step, then push new weights to the sampler

Steps 1 and 4 are the same sequences forwarded twice, which looks wasteful and
is not: step 1 runs under a sampler optimized for autoregressive decode and
returns only ids, while step 4 needs a differentiable teacher-forced pass over
the whole sequence at once. They are different computations on the same tokens.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from .algo import compute_group_advantages, group_reward_stats, grpo_policy_loss
from .config import Config
from .data import PromptSampler, Sample
from .logprobs import batched_logprobs, compute_per_token_logprobs
from .rewards import RewardEnsemble
from .rollout.base import RolloutBatch, RolloutEngine
from .utils.logging import MetricLogger
from .utils.schedule import build_scheduler
from .utils.seed import set_seed

__all__ = ["GRPOTrainer"]

# "auto" maps to None so callers can pass it straight through to `**kwargs`
# and let transformers/torch.autocast pick the dtype themselves, rather than
# this module hard-coding a default that would drift from theirs over time.
_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "auto": None,
}


class GRPOTrainer:
    """Owns the policy, the reference, the sampler, and the update loop."""

    def __init__(
        self,
        config: Config,
        model: torch.nn.Module,
        tokenizer: Any,
        engine: RolloutEngine,
        rewards: RewardEnsemble,
        samples: Sequence[Sample],
        ref_model: torch.nn.Module | None = None,
        logger: MetricLogger | None = None,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.engine = engine
        self.rewards = rewards
        self.ref_model = ref_model
        self.device = next(model.parameters()).device
        self._compute_dtype = _DTYPES.get(config.model.compute_dtype or "")

        if config.algo.beta > 0 and ref_model is None and not config.model.use_lora:
            raise ValueError(
                "algo.beta > 0 requires a reference model (or model.use_lora=True, "
                "where the frozen base weights serve as the reference)"
            )

        self.sampler = PromptSampler(
            samples,
            batch_size=config.rollout.prompts_per_step,
            seed=config.train.seed,
        )
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.optim.learning_rate,
            betas=(config.optim.adam_beta1, config.optim.adam_beta2),
            eps=config.optim.adam_epsilon,
            weight_decay=config.optim.weight_decay,
        )
        self.scheduler = build_scheduler(
            self.optimizer,
            config.optim.scheduler,
            total_steps=config.train.steps,
            warmup_ratio=config.optim.warmup_ratio,
            min_lr_ratio=config.optim.min_lr_ratio,
        )
        self.logger = logger or MetricLogger(
            config.train.output_dir,
            wandb_project=config.train.wandb_project,
            wandb_run_name=config.train.wandb_run_name,
            config=config.to_dict(),
            total_steps=config.train.steps,
        )
        self.step = 0

    def _autocast(self):
        """Run the forward/backward math in ``model.compute_dtype``.

        Master weights stay fp32 so a 1e-6 update survives rounding; autocast is
        what recovers bf16's speed and activation memory without paying for it
        in lost updates. A no-op on CPU, where autocast buys nothing and the
        tests run in fp32 anyway.
        """
        if self._compute_dtype is None or self.device.type == "cpu":
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self._compute_dtype)

    # ------------------------------------------------------------- main loop
    def train(self) -> None:
        set_seed(self.config.train.seed)
        self.model.train()
        total = self.config.train.steps

        for step in range(self.step + 1, total + 1):
            self.step = step
            started = time.time()
            metrics = self.train_step()
            metrics["time/step_s"] = time.time() - started

            if step % max(1, self.config.train.log_every) == 0:
                self.logger.log(metrics, step=step)
            if self.config.train.save_every and step % self.config.train.save_every == 0:
                self.save_checkpoint(Path(self.config.train.output_dir) / f"checkpoint-{step}")

        self.save_checkpoint(Path(self.config.train.output_dir) / "final")
        self.logger.close()

    def train_step(self) -> dict[str, float]:
        cfg = self.config
        batch = self.sampler.next_batch()
        prompts = [s.prompt for s in batch]
        columns = {"answer": [s.answer for s in batch]}

        # ---- 1. grouped rollouts ------------------------------------------
        rollout = self.engine.generate(prompts, cfg.rollout.group_size).to(self.device)
        group_size = rollout.group_size

        # ---- 2. rewards ----------------------------------------------------
        # `columns` is one value per prompt; the rollout is one row per sample,
        # group_size samples per prompt. Repeat each column value group_size
        # times so row i of `expanded` lines up with row i of the rollout batch.
        expanded = {k: [v for v in values for _ in range(group_size)] for k, values in columns.items()}
        reward_batch = self.rewards(rollout.prompts, rollout.completions, **expanded)
        rewards = torch.tensor(reward_batch.total, dtype=torch.float32, device=self.device)

        # ---- 3. group-relative advantages ---------------------------------
        advantages = compute_group_advantages(
            rewards, group_size, mode=cfg.algo.advantage_mode
        )
        stats = group_reward_stats(rewards, group_size)

        loss_mask = self._build_loss_mask(rollout, rewards)

        # ---- 4/5. forward + accumulated update -----------------------------
        metrics = self._optimize(rollout, advantages, loss_mask)

        # ---- 6. keep the sampler on the current policy ---------------------
        self.engine.sync_weights(self.model)

        lengths = rollout.completion_lengths().float()
        metrics.update(stats.as_metrics())
        metrics.update(reward_batch.as_metrics())
        metrics.update(
            {
                "reward/total": rewards.mean().item(),
                "completion/mean_length": lengths.mean().item(),
                "completion/max_length": lengths.max().item(),
                "completion/frac_truncated": (~rollout.finished).float().mean().item(),
                "completion/frac_masked_out": 1.0
                - (loss_mask.sum() / rollout.completion_mask.sum().clamp(min=1)).item(),
                "lr": self.scheduler.get_last_lr()[0],
                "epoch": float(self.sampler.epoch),
            }
        )
        self._log_samples(rollout, reward_batch, advantages)
        return metrics

    # --------------------------------------------------------------- masking
    def _build_loss_mask(self, rollout: RolloutBatch, rewards: torch.Tensor) -> torch.Tensor:
        """Completion mask with truncated / zero-variance samples optionally removed."""
        mask = rollout.completion_mask.clone()

        if self.config.algo.mask_truncated_completions:
            # A completion cut off at the token budget is graded wrong by any
            # answer-extraction reward, but it was not necessarily *reasoning*
            # wrong -- it ran out of room. Training on that signal teaches
            # brevity rather than correctness, so drop it from the loss. The
            # reward stays in the group baseline: removing it would change the
            # effective group size per prompt and break the fixed-G reshape.
            mask = mask * rollout.finished.long().unsqueeze(1)

        if self.config.algo.filter_zero_variance_groups:
            grouped = rewards.view(-1, rollout.group_size)
            has_signal = (grouped.max(dim=1).values > grouped.min(dim=1).values).long()
            mask = mask * has_signal.repeat_interleave(rollout.group_size).unsqueeze(1)

        return mask

    # -------------------------------------------------------------- updating
    def _optimize(
        self,
        rollout: RolloutBatch,
        advantages: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> dict[str, float]:
        cfg = self.config
        input_ids, attention_mask = rollout.merged()
        num_completion_tokens = rollout.max_completion_tokens
        temperature = cfg.rollout.temperature

        # The global denominator for gradient accumulation -- computed once over
        # the whole rollout batch so the accumulated micro-batch losses sum to
        # exactly the full-batch loss. See algo.aggregate_per_token_loss.
        normalizer = (
            float(loss_mask.sum().item())
            if cfg.algo.aggregation == "token_mean"
            else float(rollout.num_sequences)
        )

        ref_logps = self._reference_logprobs(input_ids, attention_mask, num_completion_tokens)

        # `old` log probs are only materialized when we actually reuse the batch.
        # With num_inner_epochs == 1 no optimizer step happens between sampling
        # and the loss, so the sampling policy IS the current policy: the ratio
        # is identically 1 and taking it from the detached current log probs is
        # exact, not an approximation. That saves a full forward pass over the
        # batch every step.
        old_logps = None
        if cfg.algo.num_inner_epochs > 1:
            with self._autocast():
                old_logps = batched_logprobs(
                    self.model,
                    input_ids,
                    attention_mask,
                    num_completion_tokens,
                    temperature,
                    micro_batch_size=cfg.optim.micro_batch_size,
                )

        micro = cfg.optim.micro_batch_size
        # Sequences are chunked into micro-batches purely to bound peak
        # activation memory; `normalizer` above is what keeps the accumulated
        # loss identical to a single full-batch call regardless of `micro`.
        num_micro = max(1, math.ceil(rollout.num_sequences / micro))
        num_updates = cfg.algo.num_inner_epochs
        accumulated: dict[str, float] = {}
        grad_norms: list[float] = []
        loss_total = 0.0

        for _ in range(num_updates):
            self.optimizer.zero_grad(set_to_none=True)
            for start in range(0, rollout.num_sequences, micro):
                sl = slice(start, start + micro)
                with self._autocast():
                    logps = compute_per_token_logprobs(
                        self.model,
                        input_ids[sl],
                        attention_mask[sl],
                        num_completion_tokens,
                        temperature,
                    )
                loss, part = grpo_policy_loss(
                    per_token_logps=logps,
                    old_per_token_logps=old_logps[sl] if old_logps is not None else logps.detach(),
                    advantages=advantages[sl],
                    completion_mask=loss_mask[sl],
                    ref_per_token_logps=ref_logps[sl] if ref_logps is not None else None,
                    beta=cfg.algo.beta,
                    epsilon_low=cfg.algo.epsilon_low,
                    epsilon_high=cfg.algo.epsilon_high,
                    aggregation=cfg.algo.aggregation,
                    max_completion_length=cfg.rollout.max_completion_length,
                    normalizer=normalizer,
                )
                loss.backward()
                # Micro-batch losses were normalized by the *global* denominator,
                # so they sum to the full-batch loss; every other metric is a
                # mean and is averaged over micro-batches and inner epochs.
                loss_total += part.pop("loss")
                for key, value in part.items():
                    accumulated[key] = accumulated.get(key, 0.0) + value

            grad_norms.append(
                float(
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        cfg.optim.max_grad_norm,
                    )
                )
            )
            self.optimizer.step()

        self.scheduler.step()
        metrics = {key: value / (num_micro * num_updates) for key, value in accumulated.items()}
        metrics["loss"] = loss_total / num_updates
        metrics["grad_norm"] = sum(grad_norms) / len(grad_norms)
        return metrics

    def _reference_logprobs(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, num_completion_tokens: int
    ) -> torch.Tensor | None:
        """Frozen-policy log probs, or ``None`` when the KL term is off."""
        if self.config.algo.beta == 0:
            return None

        temperature = self.config.rollout.temperature
        micro = self.config.optim.micro_batch_size

        if self.ref_model is not None:
            with self._autocast():
                return batched_logprobs(
                    self.ref_model, input_ids, attention_mask, num_completion_tokens,
                    temperature, micro_batch_size=micro,
                )

        # LoRA path: the reference policy is the model with its adapters off, so
        # no second set of weights is ever allocated. This is the single largest
        # memory saving available to a single-GPU GRPO run.
        disable = getattr(self.model, "disable_adapter", None)
        context = disable() if callable(disable) else nullcontext()
        with context, self._autocast():
            return batched_logprobs(
                self.model, input_ids, attention_mask, num_completion_tokens,
                temperature, micro_batch_size=micro,
            )

    # -------------------------------------------------------------- logging
    def _log_samples(self, rollout: RolloutBatch, reward_batch: Any, advantages: torch.Tensor) -> None:
        n = self.config.train.log_completions
        if n <= 0:
            return
        records = [
            {
                "prompt": rollout.prompts[i],
                "completion": rollout.completions[i],
                "reward": reward_batch.total[i],
                "advantage": float(advantages[i]),
                "finished": bool(rollout.finished[i]),
                "length": int(rollout.completion_mask[i].sum()),
                "components": {k: v[i] for k, v in reward_batch.per_function.items()},
            }
            for i in range(min(n, rollout.num_sequences))
        ]
        self.logger.log_completions(self.step, records)

    # ---------------------------------------------------------- checkpoints
    def save_checkpoint(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        torch.save(
            {
                "step": self.step,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            path / "trainer_state.pt",
        )
        (path / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        state = torch.load(Path(path) / "trainer_state.pt", map_location=self.device)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.step = int(state["step"])


# --------------------------------------------------------------- construction
def build_model_and_tokenizer(config: Config) -> tuple[torch.nn.Module, Any]:
    """Load the policy and tokenizer described by ``config.model``."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    m = config.model
    tokenizer = AutoTokenizer.from_pretrained(
        m.name_or_path, trust_remote_code=m.trust_remote_code, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        # Never add a *new* pad token here: resizing the embedding matrix changes
        # the model's vocab and would make the vLLM engine's weights the wrong
        # shape on the first sync. Reusing EOS costs nothing since pads are masked.
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {"trust_remote_code": m.trust_remote_code}
    dtype = _DTYPES[m.torch_dtype]
    if dtype is not None:
        kwargs["dtype"] = dtype
    if m.attn_implementation:
        kwargs["attn_implementation"] = m.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(m.name_or_path, **kwargs)

    if m.gradient_checkpointing:
        # use_reentrant=False is required, not preferred: the reentrant
        # implementation does not play well with inputs that require no grad and
        # silently drops gradients for some parameter groups.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    if m.use_lora:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                r=m.lora_r,
                lora_alpha=m.lora_alpha,
                lora_dropout=m.lora_dropout,
                target_modules=m.lora_target_modules,
                task_type="CAUSAL_LM",
            ),
        )
        # Upcast just the adapters to fp32. The frozen base never receives an
        # update, so 16-bit storage there is harmless and saves real memory --
        # but a bf16 adapter loses a 1e-5 update to rounding exactly the way a
        # bf16 full fine-tune loses a 1e-6 one. This is why model.use_lora is
        # exempt from the 16-bit check in Config.validate.
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.float()

    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer


def build_reference_model(config: Config) -> torch.nn.Module | None:
    """Load the frozen reference policy, if this configuration needs one."""
    if not config.needs_reference_model:
        return None
    from transformers import AutoModelForCausalLM

    m = config.model
    kwargs: dict[str, Any] = {"trust_remote_code": m.trust_remote_code}
    dtype = _DTYPES[m.torch_dtype]
    if dtype is not None:
        kwargs["dtype"] = dtype
    ref = AutoModelForCausalLM.from_pretrained(m.name_or_path, **kwargs)
    ref.eval()
    ref.requires_grad_(False)
    if torch.cuda.is_available():
        ref = ref.cuda()
    return ref
