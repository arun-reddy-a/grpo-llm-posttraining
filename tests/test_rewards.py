"""Answer extraction, numeric matching, format rewards, and the ensemble."""

import pytest

from grpo.rewards import (
    FormatReward,
    MathCorrectnessReward,
    RewardEnsemble,
    TagCountReward,
    answers_match,
    available_rewards,
    build_ensemble,
    build_reward,
    extract_answer,
    extract_gold_answer,
    normalize_number,
)


class TestExtraction:
    def test_answer_tag_wins_over_a_trailing_number(self):
        text = "<think>7 times 6 is 42, and then I mention 99</think><answer>42</answer>"
        assert extract_answer(text) == "42"

    def test_boxed_is_used_when_no_answer_tag(self):
        assert extract_answer(r"so the total is \boxed{18} dollars") == "18"

    def test_last_number_is_the_fallback(self):
        assert extract_answer("first 3, then 5, so 8") == "8"

    def test_last_answer_tag_wins_when_the_model_reconsiders(self):
        text = "<answer>10</answer> wait, no. <answer>12</answer>"
        assert extract_answer(text) == "12"

    def test_number_is_pulled_out_of_a_wordy_answer_tag(self):
        assert extract_answer("<answer>The answer is 42 apples</answer>") == "42"

    def test_no_number_anywhere_returns_none(self):
        assert extract_answer("I have no idea how to solve this.") is None

    def test_gsm8k_gold_marker_is_stripped(self):
        assert extract_gold_answer("Janet sells 16 - 3 = 13 eggs.\n#### 18") == "18"

    def test_gold_without_a_marker_passes_through(self):
        assert extract_gold_answer("18") == "18"


class TestNormalization:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("42", 42.0),
            ("42.", 42.0),
            ("$1,234.50", 1234.5),
            ("  -7 ", -7.0),
            ("50%", 50.0),
            ("3/4", 0.75),
            ("0.5", 0.5),
        ],
    )
    def test_parses_common_answer_spellings(self, raw, expected):
        assert normalize_number(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", ["twelve", "", None, "1/0"])
    def test_non_numeric_returns_none(self, raw):
        assert normalize_number(raw) is None


class TestMatching:
    @pytest.mark.parametrize(
        "predicted,gold",
        [("42", "42"), ("42.0", "42"), ("1,234", "1234"), ("$18", "18"), ("0.5", "1/2")],
    )
    def test_numerically_equal_answers_match(self, predicted, gold):
        assert answers_match(predicted, gold)

    @pytest.mark.parametrize("predicted,gold", [("42", "43"), ("0", "1"), ("-5", "5")])
    def test_different_numbers_do_not_match(self, predicted, gold):
        assert not answers_match(predicted, gold)

    def test_non_numeric_falls_back_to_case_insensitive_string_equality(self):
        assert answers_match("Blue", "blue")
        assert not answers_match("blue", "red")

    def test_missing_values_never_match(self):
        assert not answers_match(None, "42")
        assert not answers_match("42", None)


class TestMathCorrectness:
    def test_scores_correct_and_incorrect(self):
        reward = MathCorrectnessReward()
        scores = reward(
            ["q", "q"],
            ["<answer>42</answer>", "<answer>7</answer>"],
            answer=["42", "42"],
        )
        assert scores == [1.0, 0.0]

    def test_custom_reward_values(self):
        reward = MathCorrectnessReward(correct_reward=2.0, incorrect_reward=-1.0)
        assert reward(["q"], ["<answer>5</answer>"], answer=["5"]) == [2.0]
        assert reward(["q"], ["<answer>6</answer>"], answer=["5"]) == [-1.0]

    def test_missing_gold_column_is_not_applicable_rather_than_wrong(self):
        """None must not be silently read as 'incorrect' -- that would train against nothing."""
        assert MathCorrectnessReward()(["q"], ["<answer>42</answer>"]) == [None]

    def test_a_single_missing_gold_value_is_isolated(self):
        scores = MathCorrectnessReward()(
            ["q", "q"], ["<answer>1</answer>", "<answer>2</answer>"], answer=[None, "2"]
        )
        assert scores == [None, 1.0]

    def test_gsm8k_style_gold_is_handled(self):
        scores = MathCorrectnessReward()(
            ["q"], ["<answer>18</answer>"], answer=["blah blah\n#### 18"]
        )
        assert scores == [1.0]


class TestFormatRewards:
    @pytest.fixture
    def good(self):
        return "<think>reasoning here</think><answer>42</answer>"

    def test_strict_accepts_the_exact_contract(self, good):
        assert FormatReward(strict=True)(["q"], [good]) == [1.0]

    def test_strict_rejects_trailing_chatter(self, good):
        assert FormatReward(strict=True)(["q"], [good + " hope that helps!"]) == [0.0]

    def test_soft_accepts_trailing_chatter(self, good):
        assert FormatReward(strict=False)(["q"], [good + " hope that helps!"]) == [1.0]

    def test_wrong_order_is_rejected(self):
        assert FormatReward(strict=False)(
            ["q"], ["<answer>42</answer><think>after the fact</think>"]
        ) == [0.0]

    def test_empty_blocks_are_rejected(self):
        assert FormatReward(strict=True)(["q"], ["<think></think><answer></answer>"]) == [0.0]

    def test_duplicate_tags_are_rejected(self):
        """Blocks the 'spray tags and hope one matches' degenerate strategy."""
        spam = "<think>a</think><answer>1</answer><think>b</think><answer>2</answer>"
        assert FormatReward(strict=False)(["q"], [spam]) == [0.0]

    def test_missing_tags_score_zero(self):
        assert FormatReward()(["q"], ["the answer is 42"]) == [0.0]


class TestTagCount:
    def test_full_credit_for_all_four_tags(self):
        assert TagCountReward()(["q"], ["<think>a</think><answer>1</answer>"]) == [1.0]

    def test_partial_credit_is_proportional(self):
        assert TagCountReward()(["q"], ["<think>a</think>"]) == [0.5]
        assert TagCountReward()(["q"], ["<think>"]) == [0.25]

    def test_no_credit_for_no_tags(self):
        assert TagCountReward()(["q"], ["plain text"]) == [0.0]

    def test_it_is_dense_where_the_strict_reward_is_not(self):
        """Why this reward exists: it separates samples the strict one cannot."""
        partial = ["<think>a</think>", "<think>a</think><answer>1", "plain"]
        strict = FormatReward(strict=True)(["q"] * 3, partial)
        dense = TagCountReward()(["q"] * 3, partial)
        assert len(set(strict)) == 1        # strict sees no difference at all
        assert len(set(dense)) == 3         # tag_count ranks them


class TestEnsemble:
    def test_weighted_sum(self):
        ensemble = RewardEnsemble(
            [MathCorrectnessReward(), TagCountReward()], weights=[1.0, 0.5]
        )
        batch = ensemble(
            ["q"], ["<think>a</think><answer>42</answer>"], answer=["42"]
        )
        assert batch.total == [pytest.approx(1.5)]

    def test_none_contributes_zero_but_is_kept_for_reporting(self):
        ensemble = RewardEnsemble([MathCorrectnessReward(), TagCountReward()])
        batch = ensemble(["q"], ["<think>a</think><answer>42</answer>"])  # no gold
        assert batch.total == [pytest.approx(1.0)]
        assert batch.per_function["math_correctness"] == [None]

    def test_metrics_skip_non_applicable_samples(self):
        ensemble = RewardEnsemble([MathCorrectnessReward()])
        batch = ensemble(["q", "q"], ["<answer>1</answer>"] * 2, answer=[None, "1"])
        assert batch.as_metrics()["reward/math_correctness"] == pytest.approx(1.0)

    def test_all_non_applicable_reports_no_metric(self):
        batch = RewardEnsemble([MathCorrectnessReward()])(["q"], ["<answer>1</answer>"])
        assert "reward/math_correctness" not in batch.as_metrics()

    def test_rejects_empty_and_mismatched_construction(self):
        with pytest.raises(ValueError, match="at least one"):
            RewardEnsemble([])
        with pytest.raises(ValueError, match="weights"):
            RewardEnsemble([TagCountReward()], weights=[1.0, 2.0])

    def test_rejects_duplicate_names(self):
        with pytest.raises(ValueError, match="duplicate reward names"):
            RewardEnsemble([TagCountReward(), TagCountReward()])


class TestRegistry:
    def test_builtins_are_registered(self):
        assert set(available_rewards()) >= {"math_correctness", "format", "tag_count"}

    def test_build_by_name_with_kwargs(self):
        reward = build_reward("format", strict=False)
        assert isinstance(reward, FormatReward) and reward.strict is False

    def test_unknown_name_lists_the_alternatives(self):
        with pytest.raises(KeyError, match="unknown reward"):
            build_reward("does_not_exist")

    def test_build_ensemble_from_config_specs(self):
        ensemble = build_ensemble(
            [
                {"name": "math_correctness", "weight": 1.0},
                {"name": "format", "weight": 0.2, "strict": False},
            ]
        )
        assert ensemble.weights == [1.0, 0.2]
        assert ensemble.functions[1].strict is False

    def test_spec_without_a_name_is_rejected(self):
        with pytest.raises(ValueError, match="missing 'name'"):
            build_ensemble([{"weight": 1.0}])

    def test_a_typo_in_a_reward_kwarg_fails_loudly(self):
        with pytest.raises(TypeError):
            build_ensemble([{"name": "format", "strictt": True}])
