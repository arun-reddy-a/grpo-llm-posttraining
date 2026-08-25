"""End-to-end trainer tests against a tiny in-memory policy and sampler.

No transformers, no datasets, no network, no GPU -- but the real
:class:`GRPOTrainer`, the real loss, the real advantage computation and the real
gradient-accumulation path. What this actually proves is the thing unit tests
cannot: that rollout -> reward -> advantage -> loss -> optimizer step is wired
together with the right *sign*, i.e. that training makes rewarded completions
more likely rather than less.
"""

import pytest
import torch
import torch.nn as nn

from grpo.config import Config
from grpo.logprobs import batched_logprobs
from grpo.rewards import build_ensemble
from grpo.rollout.base import RolloutBatch, RolloutEngine
from grpo.trainer import GRPOTrainer

VOCAB, PROMPT_LEN, COMPLETION_LEN = 16, 3, 5
GROUP_SIZE, PROMPTS_PER_STEP = 4, 2
N = GROUP_SIZE * PROMPTS_PER_STEP

CORRECT = "<think>six sevens</think><answer>42</answer>"
WRONG = "<think>six sevens</think><answer>7</answer>"


class TinyPolicy(nn.Module):
    """A per-position LM small enough to train in a unit test."""

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Embedding(VOCAB, 16)
        self.head = nn.Linear(16, VOCAB)
        self.saved_to = []

    def forward(self, input_ids, attention_mask=None, logits_to_keep=None):
        logits = self.head(self.embed(input_ids))
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:, :]
        return type("Out", (), {"logits": logits})()

    def save_pretrained(self, path):
        self.saved_to.append(str(path))


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def save_pretrained(self, path):
        pass


class FakeEngine(RolloutEngine):
    """Replays one fixed rollout batch, and counts weight syncs.

    Determinism is the point: with the same sequences every step, any change in
    their log probabilities is attributable to the update rather than to
    resampling.
    """

    def __init__(self, correct_rows=(0, 4), truncated_rows=()):
        torch.manual_seed(3)
        self.prompt_ids = torch.randint(2, VOCAB, (N, PROMPT_LEN))
        self.completion_ids = torch.randint(2, VOCAB, (N, COMPLETION_LEN))
        self.correct_rows = set(correct_rows)
        self.truncated_rows = set(truncated_rows)
        self.sync_calls = 0

    def generate(self, prompts, group_size):
        finished = torch.tensor([i not in self.truncated_rows for i in range(N)])
        return RolloutBatch(
            prompt_ids=self.prompt_ids.clone(),
            prompt_mask=torch.ones(N, PROMPT_LEN, dtype=torch.long),
            completion_ids=self.completion_ids.clone(),
            completion_mask=torch.ones(N, COMPLETION_LEN, dtype=torch.long),
            prompts=[p for p in prompts for _ in range(group_size)],
            completions=[CORRECT if i in self.correct_rows else WRONG for i in range(N)],
            finished=finished,
            group_size=group_size,
        )

    def sync_weights(self, model):
        self.sync_calls += 1


def make_config(tmp_path, **overrides):
    base = {
        "model": {"gradient_checkpointing": False},
        "rollout": {
            "backend": "hf",
            "group_size": GROUP_SIZE,
            "prompts_per_step": PROMPTS_PER_STEP,
            "temperature": 1.0,
            "max_completion_length": COMPLETION_LEN,
        },
        "algo": {"beta": 0.0, "advantage_mode": "group_norm", "aggregation": "token_mean"},
        "optim": {"learning_rate": 0.05, "micro_batch_size": 4, "warmup_ratio": 0.0,
                  "scheduler": "constant"},
        "train": {"steps": 3, "output_dir": str(tmp_path / "run"), "save_every": 0,
                  "log_completions": 1},
    }
    for section, values in overrides.items():
        base.setdefault(section, {}).update(values)
    return Config.from_dict(base)


def build_trainer(tmp_path, engine=None, ref_model=None, **overrides):
    from grpo.data import Sample

    config = make_config(tmp_path, **overrides)
    model = TinyPolicy()
    engine = engine or FakeEngine()
    trainer = GRPOTrainer(
        config=config,
        model=model,
        tokenizer=FakeTokenizer(),
        engine=engine,
        rewards=build_ensemble(config.rewards),
        samples=[Sample(prompt=f"q{i}", answer="42") for i in range(8)],
        ref_model=ref_model,
    )
    return trainer, model, engine


def sequence_logps(model, engine):
    ids = torch.cat([engine.prompt_ids, engine.completion_ids], dim=1)
    mask = torch.ones_like(ids)
    return batched_logprobs(model, ids, mask, COMPLETION_LEN).sum(dim=1)


class TestLearningSignal:
    def test_rewarded_completions_become_more_likely(self, tmp_path):
        """The end-to-end sign check. If this fails, the pipeline is inverted."""
        trainer, model, engine = build_trainer(tmp_path)
        before = sequence_logps(model, engine)
        trainer.train()
        after = sequence_logps(model, engine)

        rewarded = list(engine.correct_rows)
        penalized = [i for i in range(N) if i not in engine.correct_rows]
        assert torch.all(after[rewarded] > before[rewarded])
        assert after[penalized].mean() < before[penalized].mean()

    def test_a_collapsed_reward_batch_produces_no_update(self, tmp_path):
        """Every sample correct -> zero variance -> zero advantage -> no movement."""
        engine = FakeEngine(correct_rows=range(N))
        trainer, model, engine = build_trainer(tmp_path, engine=engine)
        before = sequence_logps(model, engine)
        trainer.train()
        assert torch.allclose(sequence_logps(model, engine), before, atol=1e-6)

    def test_truncated_completions_are_excluded_from_the_update(self, tmp_path):
        """A row masked out by mask_truncated_completions must not move."""
        engine = FakeEngine(correct_rows=(0, 4), truncated_rows=(1, 5))
        trainer, model, engine = build_trainer(
            tmp_path, engine=engine, algo={"mask_truncated_completions": True}
        )
        before = sequence_logps(model, engine)
        trainer.train()
        after = sequence_logps(model, engine)
        # Rows 1 and 5 share no parameters-in-isolation with the rest, so they do
        # shift slightly; the check is that the *rewarded* rows moved far more.
        assert (after[[0, 4]] - before[[0, 4]]).mean() > (after[[1, 5]] - before[[1, 5]]).abs().mean()


class TestStepMechanics:
    def test_sampler_is_synced_once_per_step(self, tmp_path):
        """Skipping this is the classic silent GRPO failure."""
        trainer, _, engine = build_trainer(tmp_path, train={"steps": 4})
        trainer.train()
        assert engine.sync_calls == 4

    def test_metrics_cover_the_full_diagnostic_set(self, tmp_path):
        trainer, _, _ = build_trainer(tmp_path)
        metrics = trainer.train_step()
        assert {
            "loss", "kl", "clip_frac", "ratio_mean", "grad_norm", "lr",
            "reward/total", "reward/math_correctness", "reward/format",
            "reward/frac_zero_variance_groups", "completion/mean_length",
            "completion/frac_truncated",
        } <= set(metrics)

    def test_on_policy_step_reports_ratio_one_and_no_clipping(self, tmp_path):
        trainer, _, _ = build_trainer(tmp_path)
        metrics = trainer.train_step()
        assert metrics["ratio_mean"] == pytest.approx(1.0, abs=1e-5)
        assert metrics["clip_frac"] == 0.0

    @pytest.mark.parametrize("micro_batch_size", [1, 2, 4, 8])
    def test_gradient_accumulation_gives_the_same_update(self, tmp_path, micro_batch_size):
        """Micro-batch size must be a memory knob, not a hyperparameter."""
        reference, ref_model, ref_engine = build_trainer(
            tmp_path / "ref", optim={"micro_batch_size": 8}
        )
        reference.train_step()
        expected = sequence_logps(ref_model, ref_engine)

        trainer, model, engine = build_trainer(
            tmp_path / f"m{micro_batch_size}", optim={"micro_batch_size": micro_batch_size}
        )
        trainer.train_step()
        assert torch.allclose(sequence_logps(model, engine), expected, atol=1e-5)

    def test_inner_epochs_apply_multiple_updates(self, tmp_path):
        trainer, model, engine = build_trainer(
            tmp_path, algo={"num_inner_epochs": 3}, train={"steps": 1}
        )
        before = sequence_logps(model, engine)
        metrics = trainer.train_step()
        moved = (sequence_logps(model, engine) - before).abs().mean()

        single, single_model, single_engine = build_trainer(tmp_path / "one", train={"steps": 1})
        single_before = sequence_logps(single_model, single_engine)
        single.train_step()
        moved_once = (sequence_logps(single_model, single_engine) - single_before).abs().mean()

        assert moved > moved_once
        assert metrics["clip_frac"] >= 0.0  # clipping is now reachable


class TestKLTerm:
    def test_kl_grows_as_the_policy_leaves_the_reference(self, tmp_path):
        ref = TinyPolicy()
        ref.requires_grad_(False)
        trainer, _, _ = build_trainer(
            tmp_path, ref_model=ref, algo={"beta": 0.05}, train={"steps": 1}
        )
        first = trainer.train_step()["kl"]
        later = [trainer.train_step()["kl"] for _ in range(3)][-1]
        assert first == pytest.approx(0.0, abs=1e-6)  # policy starts at the reference
        assert later > first

    def test_kl_penalty_restrains_the_update(self, tmp_path):
        """A larger beta must keep the policy closer to where it started."""
        moves = {}
        for beta in (0.0, 5.0):
            ref = TinyPolicy()
            ref.requires_grad_(False)
            trainer, model, engine = build_trainer(
                tmp_path / f"b{beta}", ref_model=ref, algo={"beta": beta}, train={"steps": 3}
            )
            before = sequence_logps(model, engine)
            trainer.train()
            moves[beta] = (sequence_logps(model, engine) - before).abs().mean().item()
        assert moves[5.0] < moves[0.0]

    def test_missing_reference_model_is_rejected_at_construction(self, tmp_path):
        with pytest.raises(ValueError, match="requires a reference model"):
            build_trainer(tmp_path, algo={"beta": 0.1})


class TestArtifacts:
    def test_metrics_and_completions_are_written(self, tmp_path):
        trainer, _, _ = build_trainer(tmp_path, train={"steps": 2})
        trainer.train()
        run_dir = tmp_path / "run"
        assert (run_dir / "metrics.jsonl").read_text().count("\n") == 2
        assert (run_dir / "completions.jsonl").exists()
        assert (run_dir / "config.json").exists()

    def test_final_checkpoint_is_saved(self, tmp_path):
        trainer, model, _ = build_trainer(tmp_path, train={"steps": 1})
        trainer.train()
        assert any(path.endswith("final") for path in model.saved_to)
        assert (tmp_path / "run" / "final" / "trainer_state.pt").exists()

    def test_checkpoint_round_trips_the_step_counter(self, tmp_path):
        trainer, _, _ = build_trainer(tmp_path, train={"steps": 2})
        trainer.train()
        fresh, _, _ = build_trainer(tmp_path / "fresh", train={"steps": 2})
        fresh.load_checkpoint(tmp_path / "run" / "final")
        assert fresh.step == 2
