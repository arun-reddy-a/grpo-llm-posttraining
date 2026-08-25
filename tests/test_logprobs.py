"""Log-prob extraction: alignment, temperature, dtype paths, chunking.

The models here are deliberately degenerate -- position ``t``'s logits depend
only on token ``t`` -- because that makes the *expected* answer computable in
closed form. A real transformer would make the alignment assertion circular.
"""

import pytest
import torch
import torch.nn as nn

from grpo.logprobs import batched_logprobs, compute_per_token_logprobs, selective_log_softmax


class TinyLM(nn.Module):
    """Per-position LM exposing the modern ``logits_to_keep`` kwarg."""

    def __init__(self, vocab=11, dim=8, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)

    def _logits(self, input_ids):
        return self.head(self.embed(input_ids))

    def forward(self, input_ids, attention_mask=None, logits_to_keep=None):
        logits = self._logits(input_ids)
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:, :]
        return type("Out", (), {"logits": logits})()


class LegacyTinyLM(TinyLM):
    """Same model behind the pre-4.49 ``num_logits_to_keep`` spelling."""

    def forward(self, input_ids, attention_mask=None, num_logits_to_keep=None):
        logits = self._logits(input_ids)
        if num_logits_to_keep:
            logits = logits[:, -num_logits_to_keep:, :]
        return type("Out", (), {"logits": logits})()


class NoKeepTinyLM(TinyLM):
    """A model with no trimming kwarg at all -- exercises the slicing fallback."""

    def forward(self, input_ids, attention_mask=None):
        return type("Out", (), {"logits": self._logits(input_ids)})()


@pytest.fixture
def batch():
    torch.manual_seed(7)
    ids = torch.randint(0, 11, (3, 9))
    return ids, torch.ones_like(ids)


class TestSelectiveLogSoftmax:
    def test_matches_the_reference_implementation(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 5, 17)
        index = torch.randint(0, 17, (4, 5))
        expected = torch.log_softmax(logits, dim=-1).gather(-1, index.unsqueeze(-1)).squeeze(-1)
        assert torch.allclose(selective_log_softmax(logits, index), expected, atol=1e-6)

    def test_bf16_row_path_agrees_with_fp32(self):
        """The low-precision branch takes a different code path; it must agree."""
        torch.manual_seed(0)
        logits = torch.randn(3, 4, 23)
        index = torch.randint(0, 23, (3, 4))
        fp32 = selective_log_softmax(logits, index)
        bf16 = selective_log_softmax(logits.bfloat16(), index)
        assert torch.allclose(bf16.float(), fp32, atol=2e-2)

    def test_result_is_a_valid_log_probability(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 3, 9)
        out = selective_log_softmax(logits, torch.randint(0, 9, (2, 3)))
        assert torch.all(out <= 0.0)

    def test_shape_validation(self):
        with pytest.raises(ValueError, match=r"must be \[N, T, V\]"):
            selective_log_softmax(torch.randn(4, 5), torch.zeros(4, 5, dtype=torch.long))
        with pytest.raises(ValueError, match="does not match"):
            selective_log_softmax(torch.randn(2, 3, 4), torch.zeros(2, 5, dtype=torch.long))


class TestAlignment:
    def test_selects_the_positions_that_predict_the_completion(self, batch):
        """The off-by-one, asserted against an independently computed answer.

        Token ``L-C+i`` is predicted by position ``L-C-1+i``. If the slice were
        shifted by one, every log prob would come from the wrong position -- the
        ratio would still look plausible and training would quietly optimize
        the wrong objective.
        """
        ids, mask = batch
        model, completion = TinyLM(), 4
        got = compute_per_token_logprobs(model, ids, mask, completion)

        full = torch.log_softmax(model._logits(ids), dim=-1)
        predicting = full[:, -completion - 1 : -1, :]
        expected = predicting.gather(-1, ids[:, -completion:].unsqueeze(-1)).squeeze(-1)

        assert got.shape == (3, completion)
        assert torch.allclose(got, expected, atol=1e-6)

    def test_a_one_position_shift_would_be_detected(self, batch):
        """Guards the guard: the assertion above is not vacuously true."""
        ids, mask = batch
        model, completion = TinyLM(), 4
        got = compute_per_token_logprobs(model, ids, mask, completion)

        full = torch.log_softmax(model._logits(ids), dim=-1)
        shifted = full[:, -completion:, :].gather(-1, ids[:, -completion:].unsqueeze(-1)).squeeze(-1)
        assert not torch.allclose(got, shifted, atol=1e-3)

    @pytest.mark.parametrize("cls", [TinyLM, LegacyTinyLM, NoKeepTinyLM])
    def test_all_transformers_kwarg_spellings_agree(self, batch, cls):
        """logits_to_keep / num_logits_to_keep / neither must give one answer."""
        ids, mask = batch
        reference = compute_per_token_logprobs(TinyLM(), ids, mask, 4)
        assert torch.allclose(compute_per_token_logprobs(cls(), ids, mask, 4), reference, atol=1e-6)


class TestTemperature:
    def test_scaling_is_applied_to_the_logits(self, batch):
        ids, mask = batch
        model = TinyLM()
        got = compute_per_token_logprobs(model, ids, mask, 4, temperature=0.5)

        scaled = torch.log_softmax(model._logits(ids) / 0.5, dim=-1)
        expected = (
            scaled[:, -5:-1, :].gather(-1, ids[:, -4:].unsqueeze(-1)).squeeze(-1)
        )
        assert torch.allclose(got, expected, atol=1e-6)

    def test_temperature_actually_changes_the_result(self, batch):
        """A no-op temperature would make the ratio silently wrong, not noisy."""
        ids, mask = batch
        model = TinyLM()
        assert not torch.allclose(
            compute_per_token_logprobs(model, ids, mask, 4, temperature=1.0),
            compute_per_token_logprobs(model, ids, mask, 4, temperature=0.5),
            atol=1e-3,
        )

    def test_low_temperature_sharpens_the_distribution(self):
        """Cold sampling concentrates mass, so a greedy sequence's logp rises.

        The sequence has to be greedy *under this model*, and because position
        t's logits depend on token t, that means chaining argmax forward rather
        than taking the argmax of some other sequence's logits.
        """
        model = TinyLM()
        vocab = model.head.out_features
        next_best = model._logits(torch.arange(vocab).unsqueeze(0))[0].argmax(dim=-1)

        row = [3]
        for _ in range(8):
            row.append(int(next_best[row[-1]]))
        ids = torch.tensor([row])
        mask = torch.ones_like(ids)

        hot = compute_per_token_logprobs(model, ids, mask, 4, temperature=2.0)
        cold = compute_per_token_logprobs(model, ids, mask, 4, temperature=0.25)
        assert cold.mean().item() > hot.mean().item()

    def test_invalid_temperature_rejected(self, batch):
        ids, mask = batch
        with pytest.raises(ValueError, match="temperature must be > 0"):
            compute_per_token_logprobs(TinyLM(), ids, mask, 4, temperature=0.0)


class TestBatching:
    @pytest.mark.parametrize("micro", [1, 2, 3, 8])
    def test_chunking_does_not_change_the_result(self, batch, micro):
        ids, mask = batch
        model = TinyLM()
        chunked = batched_logprobs(model, ids, mask, 4, micro_batch_size=micro)
        whole = compute_per_token_logprobs(model, ids, mask, 4)
        assert torch.allclose(chunked, whole.detach(), atol=1e-6)

    def test_batched_path_is_detached(self, batch):
        ids, mask = batch
        assert not batched_logprobs(TinyLM(), ids, mask, 4).requires_grad

    def test_micro_batch_size_validated(self, batch):
        ids, mask = batch
        with pytest.raises(ValueError, match="micro_batch_size must be >= 1"):
            batched_logprobs(TinyLM(), ids, mask, 4, micro_batch_size=0)


class TestValidation:
    def test_completion_must_leave_a_prompt(self, batch):
        ids, mask = batch
        with pytest.raises(ValueError, match="at least one prompt token"):
            compute_per_token_logprobs(TinyLM(), ids, mask, ids.shape[1])

    def test_completion_length_must_be_positive(self, batch):
        ids, mask = batch
        with pytest.raises(ValueError, match=">= 1"):
            compute_per_token_logprobs(TinyLM(), ids, mask, 0)
