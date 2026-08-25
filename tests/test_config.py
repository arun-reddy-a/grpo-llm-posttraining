"""Config parsing, override plumbing, and the cross-field coherence checks."""

from pathlib import Path

import pytest

from grpo.config import Config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


class TestLoading:
    def test_defaults_are_valid(self):
        assert Config().validate() is not None

    def test_from_dict_merges_onto_defaults(self):
        config = Config.from_dict({"algo": {"beta": 0.05}})
        assert config.algo.beta == 0.05
        assert config.optim.learning_rate == 1e-6  # untouched default

    @pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
    def test_every_shipped_config_loads_and_validates(self, path):
        """Regression guard: a config that no longer parses is a broken recipe."""
        assert Config.from_yaml(path).validate() is not None

    def test_missing_file_is_reported_clearly(self):
        with pytest.raises(FileNotFoundError, match="config not found"):
            Config.from_yaml("nope.yaml")

    def test_round_trips_through_to_dict(self):
        original = Config.from_yaml(CONFIG_DIR / "smoke.yaml")
        assert Config.from_dict(original.to_dict()).to_dict() == original.to_dict()


class TestUnknownKeys:
    def test_typo_in_a_field_is_an_error_not_a_silent_default(self):
        """The bug this guards: `beta_kl: 0.1` leaving the KL term off entirely."""
        with pytest.raises(ValueError, match="unknown config key"):
            Config.from_dict({"algo": {"beta_kl": 0.1}})

    def test_the_error_lists_the_valid_keys(self):
        with pytest.raises(ValueError, match="Valid keys"):
            Config.from_dict({"optim": {"lr": 1e-5}})

    def test_unknown_section_is_an_error(self):
        with pytest.raises(ValueError, match="unknown top-level config section"):
            Config.from_dict({"trainer": {}})

    def test_section_must_be_a_mapping(self):
        with pytest.raises(TypeError, match="expected a mapping"):
            Config.from_dict({"algo": [1, 2, 3]})


class TestValidation:
    def test_group_size_one_is_rejected_with_an_explanation(self):
        with pytest.raises(ValueError, match="nothing to be relative to"):
            Config.from_dict({"rollout": {"group_size": 1}})

    def test_group_size_one_is_allowed_when_advantages_are_off(self):
        config = Config.from_dict(
            {"rollout": {"group_size": 1}, "algo": {"advantage_mode": "none"},
             "optim": {"micro_batch_size": 1}}
        )
        assert config.rollout.group_size == 1

    def test_micro_batch_must_divide_the_rollout_batch(self):
        with pytest.raises(ValueError, match="must divide"):
            Config.from_dict(
                {"rollout": {"group_size": 4, "prompts_per_step": 4}, "optim": {"micro_batch_size": 5}}
            )

    def test_greedy_sampling_is_rejected(self):
        with pytest.raises(ValueError, match="identical samples within a group"):
            Config.from_dict({"rollout": {"temperature": 0.0}})

    @pytest.mark.parametrize(
        "patch,message",
        [
            ({"rollout": {"backend": "sglang"}}, "backend must be"),
            ({"algo": {"advantage_mode": "zscore"}}, "advantage_mode invalid"),
            ({"algo": {"aggregation": "mean"}}, "aggregation invalid"),
            ({"algo": {"beta": -1.0}}, "beta must be >= 0"),
            ({"algo": {"num_inner_epochs": 0}}, "num_inner_epochs must be >= 1"),
            ({"optim": {"learning_rate": 0.0}}, "learning_rate must be > 0"),
            ({"optim": {"scheduler": "exponential"}}, "scheduler invalid"),
            ({"optim": {"warmup_ratio": 1.5}}, "warmup_ratio must be in"),
            ({"train": {"steps": 0}}, "steps must be >= 1"),
            ({"model": {"torch_dtype": "int8"}}, "torch_dtype must be"),
            ({"rollout": {"top_p": 1.5}}, "top_p must be in"),
        ],
    )
    def test_out_of_range_values_are_rejected(self, patch, message):
        with pytest.raises(ValueError, match=message):
            Config.from_dict(patch)

    def test_rewards_must_be_present_and_named(self):
        with pytest.raises(ValueError, match="at least one reward"):
            Config.from_dict({"rewards": []})
        with pytest.raises(ValueError, match="needs a 'name' key"):
            Config.from_dict({"rewards": [{"weight": 1.0}]})


class TestOverrides:
    def test_values_are_parsed_as_yaml_scalars_not_strings(self):
        """`--set algo.beta=0` must give the float 0.0, not the string '0'."""
        config = Config().apply_overrides(["algo.beta=0"])
        assert config.algo.beta == 0 and not isinstance(config.algo.beta, str)

    def test_booleans_and_nulls_parse(self):
        config = Config().apply_overrides(
            ["model.gradient_checkpointing=false", "train.wandb_project=null"]
        )
        assert config.model.gradient_checkpointing is False
        assert config.train.wandb_project is None

    def test_multiple_overrides_apply_in_order(self):
        config = Config().apply_overrides(["train.steps=5", "train.steps=9"])
        assert config.train.steps == 9

    def test_overrides_are_revalidated(self):
        with pytest.raises(ValueError, match="nothing to be relative to"):
            Config().apply_overrides(["rollout.group_size=1"])

    @pytest.mark.parametrize(
        "bad,message",
        [
            ("algo.beta", "section.field=value"),
            ("beta=0.1", "section.field"),
            ("nope.beta=1", "no config section"),
            ("algo.nope=1", "has no field"),
        ],
    )
    def test_malformed_overrides_are_rejected(self, bad, message):
        with pytest.raises(ValueError, match=message):
            Config().apply_overrides([bad])


class TestDerived:
    def test_sequences_and_accumulation_steps(self):
        config = Config.from_dict(
            {"rollout": {"prompts_per_step": 8, "group_size": 8}, "optim": {"micro_batch_size": 4}}
        )
        assert config.sequences_per_step == 64
        assert config.grad_accum_steps == 16

    def test_reference_model_needed_only_with_kl_and_without_lora(self):
        assert Config.from_dict({"algo": {"beta": 0.02}}).needs_reference_model is True
        assert Config.from_dict({"algo": {"beta": 0.0}}).needs_reference_model is False
        assert (
            Config.from_dict(
                {"algo": {"beta": 0.02}, "model": {"use_lora": True}}
            ).needs_reference_model
            is False
        )


class TestWeightPrecision:
    """Guards the defect that 16-bit master weights silently discard GRPO updates.

    This is the failure the rest of the suite could not see: the toy policy in
    ``test_trainer.py`` is fp32 at a large learning rate, so it trains happily
    while a real bf16 recipe at 1e-6 would not move a single weight.
    """

    def test_a_policy_gradient_update_vanishes_in_bfloat16(self):
        """The numeric fact the validation rule is derived from.

        bf16 carries an 8-bit mantissa, so near a weight of magnitude 0.02 the
        smallest representable change is ~1.2e-4. An AdamW step is ~lr in size
        because Adam normalizes the gradient, so at lr=1e-6 the update is two
        orders of magnitude too small to land and rounds to exactly zero.
        """
        import torch

        moved = {}
        for dtype in (torch.bfloat16, torch.float32):
            param = torch.nn.Parameter(torch.full((256,), 0.02, dtype=dtype))
            optimizer = torch.optim.AdamW([param], lr=1e-6, betas=(0.9, 0.99))
            start = param.detach().clone().float()
            for _ in range(50):
                optimizer.zero_grad()
                param.grad = torch.full_like(param, 1e-3)
                optimizer.step()
            moved[dtype] = (param.detach().float() - start).abs().mean().item()

        assert moved[torch.bfloat16] == 0.0, "bf16 should lose the update entirely"
        assert moved[torch.float32] > 1e-6, "fp32 should accumulate it"

    def test_sixteen_bit_full_finetune_at_grpo_lr_is_rejected(self):
        for dtype in ("bfloat16", "float16"):
            with pytest.raises(ValueError, match="rounds to zero"):
                Config.from_dict({"model": {"torch_dtype": dtype}})

    def test_the_error_names_both_ways_out(self):
        with pytest.raises(ValueError, match="use_lora"):
            Config.from_dict({"model": {"torch_dtype": "bfloat16"}})

    def test_sixteen_bit_is_allowed_at_a_learning_rate_that_lands(self):
        config = Config.from_dict(
            {"model": {"torch_dtype": "bfloat16"}, "optim": {"learning_rate": 1e-3}}
        )
        assert config.model.torch_dtype == "bfloat16"

    def test_lora_is_exempt_because_its_adapters_are_upcast(self):
        config = Config.from_dict(
            {"model": {"torch_dtype": "bfloat16", "use_lora": True}}
        )
        assert config.model.torch_dtype == "bfloat16"

    def test_the_default_recipe_is_safe(self):
        """fp32 master weights, bf16 math -- the combination that actually trains."""
        assert Config().model.torch_dtype == "float32"
        assert Config().model.compute_dtype == "bfloat16"

    @pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
    def test_no_shipped_config_silently_discards_its_updates(self, path):
        config = Config.from_yaml(path)
        assert config.model.use_lora or config.model.torch_dtype not in ("bfloat16", "float16")

    def test_invalid_compute_dtype_is_rejected(self):
        with pytest.raises(ValueError, match="compute_dtype must be"):
            Config.from_dict({"model": {"compute_dtype": "int8"}})

    def test_compute_dtype_may_be_disabled(self):
        assert Config.from_dict({"model": {"compute_dtype": None}}).model.compute_dtype is None
