"""One runnable check per piece of non-trivial logic."""

import numpy as np

from rlsub.deli import (
    BUDGET,
    COMFORTABLE_ITEMS,
    OVER_BUDGET_RATE,
    price,
    true_satisfaction,
    verifies,
)
from rlsub.optim import group_advantages, is_degenerate_group, k3_kl
from rlsub.policy import BOS, Policy, sft_prior
from rlsub.reward import (
    VOCAB_SIZE,
    brier_score,
    expected_calibration_error,
    train_reward_model,
    verifier_reward,
)

BREAD, FILLING, SAUCE, EXTRA = 0, 3, 6, 8


def test_satisfaction_turns_negative_past_the_comfortable_size():
    growing = [true_satisfaction(tuple([EXTRA] * n)) for n in range(1, 9)]
    peak = int(np.argmax(growing)) + 1
    assert peak <= COMFORTABLE_ITEMS + 1
    assert growing[-1] < growing[peak - 1]


def test_budget_penalty_scales_with_overage():
    """A flat fee would charge these two the same; a per-pound rate must not."""
    mild = (FILLING,) * 3
    severe = (FILLING,) * 6
    assert BUDGET < price(mild) < price(severe)

    charged = lambda order: _unpenalised(order) - true_satisfaction(order)
    assert charged(severe) > charged(mild) * 2
    assert np.isclose(charged(mild), OVER_BUDGET_RATE * (price(mild) - BUDGET))


def _unpenalised(order):
    """Satisfaction with the budget term removed, for isolating that term."""
    return true_satisfaction(order) + OVER_BUDGET_RATE * max(0.0, price(order) - BUDGET)


def test_verifier_rejects_structural_and_budget_violations():
    assert verifies((BREAD, FILLING, EXTRA))
    assert not verifies((FILLING, EXTRA))            # no bread
    assert not verifies((BREAD, BREAD, FILLING))     # two breads
    assert not verifies((BREAD, EXTRA))              # no filling
    assert not verifies((BREAD,) + (FILLING,) * 3)   # over budget
    assert verifier_reward((BREAD, FILLING, EXTRA)) == 10.0
    assert verifier_reward((FILLING,)) == 0.0


def test_group_advantages_are_standardised():
    rewards = np.array([1.0, 2.0, 3.0, 10.0])
    advantages = group_advantages(rewards)
    assert np.isclose(advantages.mean(), 0.0, atol=1e-9)
    assert np.isclose(advantages.std(), 1.0, atol=1e-9)


def test_identical_rewards_produce_exactly_zero_gradient():
    rewards = np.full(8, 10.0)
    assert is_degenerate_group(rewards)
    assert not group_advantages(rewards).any()


def test_dr_grpo_variant_keeps_scale():
    rewards = np.array([0.0, 0.0, 10.0, 10.0])
    centred = group_advantages(rewards, divide_by_std=False)
    assert np.isclose(centred.mean(), 0.0)
    assert np.isclose(np.abs(centred).max(), 5.0)


def test_k3_is_non_negative_and_vanishes_at_equality():
    identical = np.array([-2.0, -3.5])
    assert np.allclose(k3_kl(identical, identical), 0.0)
    drifted = k3_kl(np.array([-2.0]), np.array([-4.0]))
    assert (drifted >= 0).all() and drifted[0] > 0


def test_grad_log_prob_matches_a_numerical_gradient():
    rng = np.random.default_rng(7)
    policy = Policy.initial(rng)
    order = (BREAD, FILLING, EXTRA)
    analytic = policy.grad_log_prob(order)

    epsilon = 1e-6
    for state, action in [(BOS, BREAD), (BREAD, FILLING), (FILLING, EXTRA)]:
        bumped = policy.logits.copy()
        bumped[state, action] += epsilon
        numerical = (Policy(bumped).log_prob(order) - policy.log_prob(order)) / epsilon
        assert np.isclose(analytic[state, action], numerical, atol=1e-4)


def test_reward_model_learns_the_trap():
    """Fit only on short orders, it concludes more items is always better."""
    model = train_reward_model(np.random.default_rng(0), n_pairs=2000, epochs=150)
    assert model.weights[VOCAB_SIZE] > 0.2
    lengthening = [model.score((FILLING,) * n) for n in range(1, 9)]
    assert lengthening == sorted(lengthening)


def test_perfect_forecasts_score_zero():
    outcomes = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(outcomes, outcomes) == 0.0
    assert expected_calibration_error(outcomes, outcomes) == 0.0


def test_sft_prior_underuses_bread_and_filling():
    bias = sft_prior()
    assert bias[BREAD] < 0 and bias[FILLING] < 0
    assert bias[SAUCE] > 0 and bias[EXTRA] > 0
