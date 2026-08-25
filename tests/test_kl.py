"""The k3 KL estimator: non-negativity, unbiasedness, overflow safety."""

import pytest
import torch

from grpo.algo import kl_divergence_k3


def test_zero_when_policies_agree():
    logps = torch.log(torch.tensor([0.1, 0.5, 0.4]))
    assert torch.allclose(kl_divergence_k3(logps, logps), torch.zeros(3), atol=1e-7)


def test_non_negative_everywhere():
    """k3's defining property: every *sample* is >= 0, not just the mean.

    The naive estimator (ref - policy) is unbiased but signed, so individual
    tokens can contribute a negative penalty -- i.e. pay the policy to move away
    from the reference. k3 cannot.
    """
    torch.manual_seed(0)
    policy = torch.randn(2000) * 3 - 1
    ref = torch.randn(2000) * 3 - 1
    assert torch.all(kl_divergence_k3(policy, ref) >= 0.0)

    naive = ref - policy
    assert torch.any(naive < 0.0), "sanity: the naive estimator does go negative"


def test_unbiased_against_analytic_kl():
    """E_{x~p}[k3] == KL(p||q), computed exactly rather than sampled.

    Taking the expectation under p in closed form removes Monte-Carlo noise, so
    this asserts the identity itself and not a convergence rate.
    """
    torch.manual_seed(0)
    p = torch.softmax(torch.randn(64) * 2, dim=0)
    q = torch.softmax(torch.randn(64) * 2, dim=0)

    analytic = torch.sum(p * (p.log() - q.log()))
    expected_k3 = torch.sum(p * kl_divergence_k3(p.log(), q.log()))
    assert torch.allclose(expected_k3, analytic, atol=1e-5)
    assert analytic.item() > 0.01, "sanity: the two distributions really do differ"


def test_monte_carlo_convergence():
    """The same identity, the way it is actually used: averaged over samples."""
    torch.manual_seed(1234)
    p = torch.softmax(torch.randn(32), dim=0)
    q = torch.softmax(torch.randn(32), dim=0)
    draws = torch.multinomial(p, 200_000, replacement=True)

    estimate = kl_divergence_k3(p.log()[draws], q.log()[draws]).mean()
    analytic = torch.sum(p * (p.log() - q.log()))
    assert estimate.item() == pytest.approx(analytic.item(), rel=0.05)


def test_grows_with_divergence():
    ref = torch.zeros(1)
    near = kl_divergence_k3(torch.tensor([-0.1]), ref)
    far = kl_divergence_k3(torch.tensor([-2.0]), ref)
    assert far.item() > near.item()


def test_clamp_prevents_overflow():
    """One collapsed token must not NaN the whole batch's loss."""
    policy = torch.tensor([-1e4, -1.0])   # first token driven to ~zero probability
    ref = torch.tensor([0.0, -1.0])
    kl = kl_divergence_k3(policy, ref, max_log_ratio=20.0)
    assert torch.isfinite(kl).all()
    assert kl[0].item() == pytest.approx(torch.exp(torch.tensor(20.0)).item() - 21.0, rel=1e-5)


def test_clamp_does_not_perturb_normal_values():
    torch.manual_seed(0)
    policy, ref = torch.randn(500), torch.randn(500)
    clamped = kl_divergence_k3(policy, ref, max_log_ratio=20.0)
    unclamped = kl_divergence_k3(policy, ref, max_log_ratio=1e9)
    assert torch.allclose(clamped, unclamped, atol=1e-6)


def test_gradient_pushes_policy_toward_reference():
    policy = torch.tensor([-2.0], requires_grad=True)
    ref = torch.tensor([-0.5])
    kl_divergence_k3(policy, ref).backward()
    # policy logp is below ref: minimizing KL should raise it, so d(KL)/d(logp) < 0
    assert policy.grad.item() < 0
