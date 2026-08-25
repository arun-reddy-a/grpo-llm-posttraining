"""Group-relative advantage estimation."""

import pytest
import torch

from grpo.algo import compute_group_advantages, group_reward_stats


def test_group_norm_matches_hand_computation():
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0])
    adv = compute_group_advantages(rewards, group_size=4, mode="group_norm")
    mean, std = 0.25, rewards.std(unbiased=True)
    expected = (rewards - mean) / (std + 1e-4)
    assert torch.allclose(adv, expected, atol=1e-6)


def test_groups_are_independent():
    """Prompt 0's rewards must not influence prompt 1's advantages."""
    rewards = torch.tensor([0.0, 1.0, 100.0, 200.0])
    adv = compute_group_advantages(rewards, group_size=2)
    alone = compute_group_advantages(torch.tensor([0.0, 1.0]), group_size=2)
    assert torch.allclose(adv[:2], alone, atol=1e-6)


@pytest.mark.parametrize("mode", ["group_norm", "group_mean"])
def test_advantages_sum_to_zero_within_group(mode):
    torch.manual_seed(0)
    rewards = torch.rand(24)
    adv = compute_group_advantages(rewards, group_size=6, mode=mode)
    assert torch.allclose(adv.view(-1, 6).sum(dim=1), torch.zeros(4), atol=1e-5)


def test_zero_variance_group_gives_zero_advantage():
    """All-equal rewards carry no relative signal, so they must produce no gradient."""
    for value in (0.0, 1.0, -3.5):
        rewards = torch.full((8,), value)
        for mode in ("group_norm", "group_mean"):
            adv = compute_group_advantages(rewards, group_size=4, mode=mode)
            assert torch.all(adv == 0.0), f"{mode} at reward={value}"


def test_group_size_one_is_zero_not_nan():
    """A group of one has nothing to be relative to; std is undefined, not NaN."""
    adv = compute_group_advantages(torch.tensor([1.0, 2.0, 3.0]), group_size=1)
    assert not torch.isnan(adv).any()
    assert torch.all(adv == 0.0)


def test_group_mean_does_not_rescale():
    """group_mean centers only -- the raw reward spread must survive intact."""
    rewards = torch.tensor([0.0, 0.0, 0.0, 4.0])
    centered = compute_group_advantages(rewards, group_size=4, mode="group_mean")
    assert torch.allclose(centered, rewards - rewards.mean(), atol=1e-6)
    assert centered.max().item() == pytest.approx(3.0)


def test_group_norm_amplifies_near_collapsed_groups():
    """The difficulty bias Dr. GRPO objects to, made concrete.

    Four samples that essentially agree (spread 0.001) produce an order-1
    advantage under group_norm: dividing by a near-zero std turns a negligible
    disagreement into a full-strength gradient. group_mean is what turns this
    off, which is why it is a config option rather than a hard-coded choice.
    """
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.001])
    centered = compute_group_advantages(rewards, group_size=4, mode="group_mean")
    normed = compute_group_advantages(rewards, group_size=4, mode="group_norm")
    assert centered.abs().max().item() < 0.01
    assert normed.abs().max().item() > 1.0


def test_none_mode_passes_rewards_through():
    rewards = torch.tensor([0.5, 1.5, -2.0, 3.0])
    assert torch.allclose(compute_group_advantages(rewards, 2, mode="none"), rewards)


def test_normalization_makes_easy_and_hard_prompts_comparable():
    """The property group_norm exists for: equal advantage scale across prompts."""
    easy = torch.tensor([1.0, 1.0, 1.0, 0.0])   # model usually right
    hard = torch.tensor([0.0, 0.0, 0.0, 1.0])   # model usually wrong
    rewards = torch.cat([easy, hard])
    adv = compute_group_advantages(rewards, group_size=4, mode="group_norm")
    assert adv[:4].abs().max().item() == pytest.approx(adv[4:].abs().max().item(), rel=1e-4)


@pytest.mark.parametrize(
    "rewards,group_size,message",
    [
        (torch.rand(5), 2, "not divisible"),
        (torch.rand(2, 2), 2, "must be 1-D"),
        (torch.rand(4), 0, ">= 1"),
    ],
)
def test_invalid_shapes_raise(rewards, group_size, message):
    with pytest.raises(ValueError, match=message):
        compute_group_advantages(rewards, group_size)


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown advantage mode"):
        compute_group_advantages(torch.rand(4), 2, mode="bogus")


class TestGroupRewardStats:
    def test_detects_fully_collapsed_batch(self):
        stats = group_reward_stats(torch.ones(16), group_size=4)
        assert stats.frac_zero_variance == 1.0

    def test_detects_partially_collapsed_batch(self):
        rewards = torch.tensor([1.0, 1.0, 0.0, 1.0])  # group 0 dead, group 1 alive
        assert group_reward_stats(rewards, group_size=2).frac_zero_variance == 0.5

    def test_metric_keys_are_namespaced(self):
        metrics = group_reward_stats(torch.rand(8), 4).as_metrics()
        assert set(metrics) == {
            "reward/mean",
            "reward/std",
            "reward/group_std",
            "reward/frac_zero_variance_groups",
        }
