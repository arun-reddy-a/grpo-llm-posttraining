"""Padding and EOS masking -- where a rollout batch becomes trainable tensors."""

import pytest
import torch

from grpo.rollout.base import RolloutBatch, completion_mask_from_ids, pad_and_stack


class TestPadAndStack:
    def test_right_padding_shapes_and_mask(self):
        ids, mask = pad_and_stack([[1, 2, 3], [4], [5, 6]], pad_value=0, side="right")
        assert ids.tolist() == [[1, 2, 3], [4, 0, 0], [5, 6, 0]]
        assert mask.tolist() == [[1, 1, 1], [1, 0, 0], [1, 1, 0]]

    def test_left_padding_aligns_the_ends(self):
        """Prompts must end at a common column so generation continues in step."""
        ids, mask = pad_and_stack([[1, 2, 3], [4]], pad_value=0, side="left")
        assert ids.tolist() == [[1, 2, 3], [0, 0, 4]]
        assert mask.tolist() == [[1, 1, 1], [0, 0, 1]]

    def test_right_truncation_keeps_the_start(self):
        ids, _ = pad_and_stack([[1, 2, 3, 4, 5]], pad_value=0, side="right", max_length=3)
        assert ids.tolist() == [[1, 2, 3]]

    def test_left_truncation_keeps_the_end(self):
        """An over-long prompt loses its head, not the question at its tail."""
        ids, _ = pad_and_stack([[1, 2, 3, 4, 5]], pad_value=0, side="left", max_length=3)
        assert ids.tolist() == [[3, 4, 5]]

    def test_pads_out_to_max_length(self):
        ids, mask = pad_and_stack([[1, 2]], pad_value=9, side="right", max_length=4)
        assert ids.tolist() == [[1, 2, 9, 9]]
        assert mask.sum().item() == 2

    def test_empty_input(self):
        ids, mask = pad_and_stack([], pad_value=0)
        assert ids.shape == (0, 0) and mask.shape == (0, 0)

    def test_invalid_side_rejected(self):
        with pytest.raises(ValueError, match="side must be"):
            pad_and_stack([[1]], pad_value=0, side="middle")


class TestCompletionMask:
    def test_masks_everything_after_the_first_eos(self):
        ids = torch.tensor([[5, 6, 2, 7, 8]])
        mask, finished = completion_mask_from_ids(ids, torch.ones_like(ids), {2})
        assert mask.tolist() == [[1, 1, 1, 0, 0]]
        assert finished.tolist() == [True]

    def test_eos_itself_stays_in_the_mask(self):
        """Choosing to stop is an action the policy took; it must carry gradient."""
        ids = torch.tensor([[5, 2, 9]])
        mask, _ = completion_mask_from_ids(ids, torch.ones_like(ids), {2})
        assert mask[0, 1].item() == 1

    def test_no_eos_means_truncated_and_fully_unmasked(self):
        ids = torch.tensor([[5, 6, 7]])
        mask, finished = completion_mask_from_ids(ids, torch.ones_like(ids), {2})
        assert mask.tolist() == [[1, 1, 1]]
        assert finished.tolist() == [False]

    def test_multiple_eos_ids_are_all_honoured(self):
        """Chat models end turns with a token that is not the nominal EOS."""
        ids = torch.tensor([[5, 151645, 7], [5, 151643, 7]])
        mask, finished = completion_mask_from_ids(
            ids, torch.ones_like(ids), {151643, 151645}
        )
        assert mask.tolist() == [[1, 1, 0], [1, 1, 0]]
        assert finished.tolist() == [True, True]

    def test_only_the_nominal_eos_would_miss_the_turn_token(self):
        """Counterpart to the test above: this is the bug it prevents."""
        ids = torch.tensor([[5, 151645, 7]])
        _, finished = completion_mask_from_ids(ids, torch.ones_like(ids), {151643})
        assert finished.tolist() == [False]

    def test_padding_is_never_unmasked_by_eos_logic(self):
        ids = torch.tensor([[5, 2, 0, 0]])
        pad_mask = torch.tensor([[1, 1, 0, 0]])
        mask, _ = completion_mask_from_ids(ids, pad_mask, {2})
        assert mask.tolist() == [[1, 1, 0, 0]]

    def test_eos_inside_the_padded_region_is_ignored(self):
        """A pad id that equals the EOS id must not be read as a real stop."""
        ids = torch.tensor([[5, 6, 2, 2]])
        pad_mask = torch.tensor([[1, 1, 0, 0]])
        _, finished = completion_mask_from_ids(ids, pad_mask, {2})
        assert finished.tolist() == [False]

    def test_rows_are_independent(self):
        ids = torch.tensor([[2, 5, 5], [5, 5, 2]])
        mask, finished = completion_mask_from_ids(ids, torch.ones_like(ids), {2})
        assert mask.tolist() == [[1, 0, 0], [1, 1, 1]]
        assert finished.tolist() == [True, True]

    def test_no_eos_ids_configured_leaves_the_mask_alone(self):
        ids = torch.tensor([[5, 6]])
        mask, finished = completion_mask_from_ids(ids, torch.ones_like(ids), set())
        assert mask.tolist() == [[1, 1]]
        assert finished.tolist() == [False]


def _batch(n=4, group_size=2, p=3, c=5):
    return RolloutBatch(
        prompt_ids=torch.ones(n, p, dtype=torch.long),
        prompt_mask=torch.ones(n, p, dtype=torch.long),
        completion_ids=torch.ones(n, c, dtype=torch.long),
        completion_mask=torch.ones(n, c, dtype=torch.long),
        prompts=["q"] * n,
        completions=["a"] * n,
        finished=torch.ones(n, dtype=torch.bool),
        group_size=group_size,
    )


class TestRolloutBatch:
    def test_derived_shapes(self):
        batch = _batch(n=6, group_size=3)
        assert (batch.num_sequences, batch.num_prompts, batch.max_completion_tokens) == (6, 2, 5)

    def test_merged_concatenates_prompt_then_completion(self):
        ids, mask = _batch().merged()
        assert ids.shape == (4, 8) and mask.shape == (4, 8)

    def test_completion_lengths_come_from_the_mask(self):
        batch = _batch()
        batch.completion_mask[0, 3:] = 0
        assert batch.completion_lengths().tolist() == [3, 5, 5, 5]

    def test_group_size_must_divide_the_batch(self):
        with pytest.raises(ValueError, match="not divisible by group_size"):
            _batch(n=5, group_size=2)

    def test_row_count_mismatch_is_caught(self):
        with pytest.raises(ValueError, match="expected 4 prompt/completion strings"):
            RolloutBatch(
                prompt_ids=torch.ones(4, 3, dtype=torch.long),
                prompt_mask=torch.ones(4, 3, dtype=torch.long),
                completion_ids=torch.ones(4, 5, dtype=torch.long),
                completion_mask=torch.ones(4, 5, dtype=torch.long),
                prompts=["q"],
                completions=["a"],
                finished=torch.ones(4, dtype=torch.bool),
                group_size=2,
            )

    def test_tensor_row_mismatch_is_caught(self):
        with pytest.raises(ValueError, match="completion_ids has 2 rows"):
            RolloutBatch(
                prompt_ids=torch.ones(4, 3, dtype=torch.long),
                prompt_mask=torch.ones(4, 3, dtype=torch.long),
                completion_ids=torch.ones(2, 5, dtype=torch.long),
                completion_mask=torch.ones(4, 5, dtype=torch.long),
                prompts=["q"] * 4,
                completions=["a"] * 4,
                finished=torch.ones(4, dtype=torch.bool),
                group_size=2,
            )
