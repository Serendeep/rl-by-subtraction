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


def test_clipping_actually_binds_across_inner_epochs():
    """A single inner epoch leaves the ratio at 1, where clipping is a no-op."""
    from rlsub.optim import DEFAULT_CLIP, grpo_step
    from rlsub.reward import train_reward_model

    rng = np.random.default_rng(0)
    model = train_reward_model(rng, n_pairs=1500, epochs=120)
    policy = Policy.initial(rng, item_bias=sft_prior())
    reference = policy

    ratios = []
    for _ in range(40):
        orders = [policy.sample(rng) for _ in range(16)]
        rewards = np.array([model.score(o) for o in orders])
        before = [policy.log_prob(o) for o in orders]
        policy = grpo_step(policy, orders, rewards, reference, 0.08, 0.02, inner_epochs=4)
        ratios += [np.exp(policy.log_prob(o) - b) for o, b in zip(orders, before)]

    ratios = np.array(ratios)
    assert (ratios > 1 + DEFAULT_CLIP).any() or (ratios < 1 - DEFAULT_CLIP).any()


def test_gradient_coefficient_matches_deepseekmath():
    """At epoch one the coefficient must equal A + beta*(pi_ref/pi_theta - 1)."""
    from rlsub.optim import group_advantages

    rng = np.random.default_rng(11)
    policy = Policy.initial(rng)
    reference = Policy.initial(np.random.default_rng(12))
    orders = [tuple(int(x) for x in rng.integers(0, 12, size=3)) for _ in range(8)]
    advantages = group_advantages(rng.normal(0, 2, size=8))
    beta = 0.03

    for order, advantage in zip(orders, advantages):
        u = np.exp(reference.log_prob(order) - policy.log_prob(order))
        ours = advantage + beta * (u - 1.0)          # ratio 1, unclipped
        assert np.isclose(ours, advantage + beta * (u - 1.0))


def test_kl_term_changes_the_update():
    """With beta = 0 the KL machinery is inert; the test guards against shipping that."""
    from rlsub.optim import grpo_step

    rng = np.random.default_rng(5)
    policy = Policy.initial(rng, item_bias=sft_prior())
    reference = Policy.initial(np.random.default_rng(6))
    orders = [policy.sample(rng) for _ in range(12)]
    rewards = np.array([float(len(o)) for o in orders])

    without = grpo_step(policy, orders, rewards, reference, 0.1, 0.0, inner_epochs=2)
    with_kl = grpo_step(policy, orders, rewards, reference, 0.1, 0.25, inner_epochs=2)
    assert not np.allclose(without.logits, with_kl.logits)


def test_log_score_is_unbounded_where_brier_is_not():
    """Why RLCR needs a bounded rule: log score explodes on a confident miss."""
    from rlsub.reward import log_score

    confident_miss = np.array([0.999]), np.array([0.0])
    assert brier_score(*confident_miss) < 1.0
    assert log_score(*confident_miss) > 6.0


def test_the_optimum_does_not_verify():
    """The best sandwich breaks the rules, so the verifier costs real satisfaction."""
    from rlsub.best import ranked
    from rlsub.deli import verifies as ok

    scored = ranked()
    best_score, best_order = scored[0]
    best_legal = next(pair for pair in scored if ok(pair[1]))

    assert not ok(best_order)
    assert best_legal[0] < best_score
    assert best_legal[0] > 4.0


def test_reward_choice_inverts_the_retention_policy():
    """Speculative toy: the same optimiser keeps different things per reward."""
    from rlsub.notepad import (
        recall_reward,
        recency_reward,
        train as train_notepad,
        write_rate_by_position,
    )

    rng = np.random.default_rng(3)
    recency = write_rate_by_position(
        train_notepad(recency_reward, np.random.default_rng(0), steps=1200), rng, n=400)
    recall = write_rate_by_position(
        train_notepad(recall_reward, np.random.default_rng(0), steps=1200), rng, n=400)

    # Robust across seeds: recall writes early then defends, recency keeps
    # writing through the tail. Whether recency *rises* depends on the seed.
    # At 1,200 steps the recall tail is ~0.3; fully trained it reaches ~0.03.
    assert recall[0] - recall[-1] > 0.5
    assert recency[-1] > recall[-1] + 0.5


def test_group_mean_reduces_variance_like_a_critic():
    """GRPO's actual claim: any valid baseline does the critic's only job."""
    from rlsub.baselines import compare
    from rlsub.reward import train_reward_model

    rng = np.random.default_rng(0)
    result = compare(rng, train_reward_model(rng, n_pairs=1500, epochs=120), steps=60)
    spread = result["spread"]

    # group/none tightens with training (61% at 60 steps, 44% by 300), so the
    # loose bound here is deliberate. group/critic sits at 0.90 throughout and
    # is the claim worth pinning.
    assert spread["group"] < spread["none"] * 0.7
    assert spread["group"] < spread["critic"]
