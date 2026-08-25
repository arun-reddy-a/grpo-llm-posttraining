"""The clipped, KL-regularized surrogate loss and its aggregation modes."""

import pytest
import torch

from grpo.algo import aggregate_per_token_loss, grpo_policy_loss


def _logps(n=4, t=6, value=-1.0, grad=False):
    return torch.full((n, t), value, requires_grad=grad)


class TestOnPolicyBehaviour:
    def test_ratio_is_one_when_old_equals_current(self):
        logps = _logps()
        loss, metrics = grpo_policy_loss(
            logps, logps.clone(), torch.tensor([1.0, -1.0, 0.5, 0.0]), torch.ones(4, 6)
        )
        assert metrics["ratio_mean"] == pytest.approx(1.0, abs=1e-6)
        # With rho == 1 the objective is just the mean advantage.
        assert loss.item() == pytest.approx(-0.125, abs=1e-6)

    def test_nothing_is_clipped_on_policy(self):
        """Documented consequence of num_inner_epochs=1: clip_frac is 0 by construction."""
        logps = _logps()
        _, metrics = grpo_policy_loss(
            logps, logps.clone(), torch.randn(4), torch.ones(4, 6)
        )
        assert metrics["clip_frac"] == 0.0

    def test_positive_advantage_raises_logprob(self):
        logps = _logps(grad=True)
        loss, _ = grpo_policy_loss(
            logps, logps.detach().clone(), torch.tensor([1.0, 1.0, 1.0, 1.0]), torch.ones(4, 6)
        )
        loss.backward()
        # Descending the loss must increase log pi for a rewarded sample.
        assert torch.all(logps.grad < 0)

    def test_negative_advantage_lowers_logprob(self):
        logps = _logps(grad=True)
        loss, _ = grpo_policy_loss(
            logps, logps.detach().clone(), torch.full((4,), -1.0), torch.ones(4, 6)
        )
        loss.backward()
        assert torch.all(logps.grad > 0)

    def test_zero_advantage_gives_zero_gradient(self):
        logps = _logps(grad=True)
        loss, _ = grpo_policy_loss(
            logps, logps.detach().clone(), torch.zeros(4), torch.ones(4, 6)
        )
        loss.backward()
        assert torch.allclose(logps.grad, torch.zeros_like(logps.grad), atol=1e-8)


class TestClipping:
    def test_clip_binds_above_range_for_positive_advantage(self):
        """rho > 1+eps with A > 0: the token is already far more likely, so stop."""
        old = _logps(n=1, t=1, value=-2.0)
        cur = torch.tensor([[-1.0]], requires_grad=True)  # ratio = e ~ 2.72
        loss, metrics = grpo_policy_loss(
            cur, old, torch.tensor([1.0]), torch.ones(1, 1), epsilon_high=0.2
        )
        assert metrics["clip_frac"] == 1.0
        loss.backward()
        assert cur.grad.abs().item() == pytest.approx(0.0, abs=1e-8)
        assert loss.item() == pytest.approx(-1.2, abs=1e-6)  # -clip(rho)*A = -(1.2)(1)

    def test_clip_does_not_bind_below_range_for_positive_advantage(self):
        """rho < 1-eps with A > 0 stays unclipped: min() keeps the smaller term."""
        old = _logps(n=1, t=1, value=-1.0)
        cur = torch.tensor([[-3.0]], requires_grad=True)  # ratio ~ 0.135
        loss, metrics = grpo_policy_loss(
            cur, old, torch.tensor([1.0]), torch.ones(1, 1), epsilon_low=0.2
        )
        assert metrics["clip_frac"] == 0.0
        loss.backward()
        assert cur.grad.abs().item() > 0

    def test_clip_binds_below_range_for_negative_advantage(self):
        old = _logps(n=1, t=1, value=-1.0)
        cur = torch.tensor([[-3.0]], requires_grad=True)
        loss, metrics = grpo_policy_loss(
            cur, old, torch.tensor([-1.0]), torch.ones(1, 1), epsilon_low=0.2
        )
        assert metrics["clip_frac"] == 1.0
        loss.backward()
        assert cur.grad.abs().item() == pytest.approx(0.0, abs=1e-8)

    def test_asymmetric_clip_higher_widens_only_the_top(self):
        old = _logps(n=1, t=1, value=-2.0)
        cur = torch.tensor([[-1.0]])  # ratio ~ 2.72
        _, tight = grpo_policy_loss(cur, old, torch.tensor([1.0]), torch.ones(1, 1),
                                    epsilon_low=0.2, epsilon_high=0.2)
        _, wide = grpo_policy_loss(cur, old, torch.tensor([1.0]), torch.ones(1, 1),
                                   epsilon_low=0.2, epsilon_high=2.0)
        assert wide["loss"] < tight["loss"]  # more of the improvement is credited


class TestMasking:
    def test_masked_tokens_contribute_no_gradient(self):
        logps = _logps(n=2, t=4, grad=True)
        mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]])
        loss, _ = grpo_policy_loss(
            logps, logps.detach().clone(), torch.tensor([1.0, 1.0]), mask
        )
        loss.backward()
        assert torch.all(logps.grad[mask == 0] == 0)
        assert torch.all(logps.grad[mask == 1] != 0)

    def test_padding_values_cannot_change_the_loss(self):
        """Garbage in the padded region must be inert, not merely small."""
        torch.manual_seed(0)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
        adv = torch.tensor([1.0, -1.0])
        base = _logps(n=2, t=4)
        polluted = base.clone()
        polluted[mask == 0] = 50.0

        a, _ = grpo_policy_loss(base, base.clone(), adv, mask)
        b, _ = grpo_policy_loss(polluted, polluted.clone(), adv, mask)
        assert a.item() == pytest.approx(b.item(), abs=1e-6)

    def test_fully_masked_batch_is_finite(self):
        logps = _logps(n=2, t=4)
        loss, _ = grpo_policy_loss(logps, logps.clone(), torch.ones(2), torch.zeros(2, 4))
        assert torch.isfinite(loss) and loss.item() == 0.0


class TestKLTerm:
    def test_kl_penalty_increases_the_loss(self):
        cur = _logps(n=2, t=3, value=-1.0)
        ref = _logps(n=2, t=3, value=-2.0)
        adv = torch.tensor([1.0, 1.0])
        without, _ = grpo_policy_loss(cur, cur.clone(), adv, torch.ones(2, 3), beta=0.0)
        with_kl, m = grpo_policy_loss(
            cur, cur.clone(), adv, torch.ones(2, 3), ref_per_token_logps=ref, beta=0.1
        )
        assert with_kl.item() > without.item()
        assert m["kl"] > 0

    def test_beta_zero_reports_no_kl(self):
        cur = _logps(n=2, t=3)
        _, m = grpo_policy_loss(cur, cur.clone(), torch.ones(2), torch.ones(2, 3), beta=0.0)
        assert m["kl"] == 0.0

    def test_beta_without_reference_is_rejected(self):
        cur = _logps(n=2, t=3)
        with pytest.raises(ValueError, match="requires ref_per_token_logps"):
            grpo_policy_loss(cur, cur.clone(), torch.ones(2), torch.ones(2, 3), beta=0.1)


class TestAggregation:
    @pytest.fixture
    def uneven(self):
        """Two sequences: one 2 tokens long, one 6. Same per-token loss."""
        loss = torch.ones(2, 6)
        mask = torch.tensor([[1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]])
        return loss, mask

    def test_token_mean_weights_every_token_equally(self, uneven):
        loss, mask = uneven
        assert aggregate_per_token_loss(loss, mask, "token_mean").item() == pytest.approx(1.0)

    def test_seq_mean_over_weights_short_sequences(self, uneven):
        """The length bias in the paper's formulation, made visible."""
        loss, mask = uneven
        loss = loss.clone()
        loss[0] = 2.0  # the SHORT sequence has double the per-token loss

        token_level = aggregate_per_token_loss(loss, mask, "token_mean").item()
        seq_level = aggregate_per_token_loss(loss, mask, "seq_mean_token_mean").item()
        # 2 short tokens at 2.0 + 6 long at 1.0 -> 1.25 per token, but the short
        # sequence gets a full half of the sequence-level average.
        assert token_level == pytest.approx(1.25)
        assert seq_level == pytest.approx(1.5)
        assert seq_level > token_level

    def test_seq_mean_token_sum_uses_a_constant_denominator(self, uneven):
        loss, mask = uneven
        value = aggregate_per_token_loss(
            loss, mask, "seq_mean_token_sum", max_completion_length=12
        )
        assert value.item() == pytest.approx(((2 / 12) + (6 / 12)) / 2)

    def test_seq_mean_token_sum_requires_a_length_budget(self, uneven):
        loss, mask = uneven
        with pytest.raises(ValueError, match="needs max_completion_length"):
            aggregate_per_token_loss(loss, mask, "seq_mean_token_sum")

    @pytest.mark.parametrize(
        "mode,normalizer",
        [("token_mean", None), ("seq_mean_token_mean", 6), ("seq_mean_token_sum", 6)],
    )
    def test_gradient_accumulation_reproduces_the_full_batch(self, mode, normalizer):
        """Summing micro-batch losses must equal the undivided full-batch loss."""
        torch.manual_seed(0)
        loss = torch.rand(6, 5)
        mask = (torch.rand(6, 5) > 0.3).long()
        mask[:, 0] = 1  # no empty sequences
        norm = float(mask.sum()) if normalizer is None else float(normalizer)

        full = aggregate_per_token_loss(loss, mask, mode, max_completion_length=5)
        parts = sum(
            aggregate_per_token_loss(
                loss[i : i + 2], mask[i : i + 2], mode, max_completion_length=5, normalizer=norm
            )
            for i in range(0, 6, 2)
        )
        assert full.item() == pytest.approx(parts.item(), abs=1e-6)

    def test_unknown_aggregation_raises(self):
        with pytest.raises(ValueError, match="unknown aggregation"):
            aggregate_per_token_loss(torch.ones(1, 1), torch.ones(1, 1), "bogus")


class TestValidation:
    def test_shape_mismatch_between_current_and_old(self):
        with pytest.raises(ValueError, match="logp shape mismatch"):
            grpo_policy_loss(_logps(2, 3), _logps(2, 4), torch.ones(2), torch.ones(2, 3))

    def test_advantage_count_must_match_batch(self):
        with pytest.raises(ValueError, match="advantages has"):
            grpo_policy_loss(_logps(2, 3), _logps(2, 3), torch.ones(3), torch.ones(2, 3))
