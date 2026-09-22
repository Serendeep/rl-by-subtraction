"""What dropping the critic actually costs, measured three ways.

GRPO's claim is that the value network exists only to reduce variance, so any
unbiased baseline substitutes. That is checkable. Each step estimates the same
policy gradient with three baselines and records the spread across a group:

    none     A = r                 unbiased, high variance
    critic   A = r - V             a learned scalar, lags while the policy moves
    group    A = r - mean(r)       Monte Carlo, zero parameters

Reported as gradient standard deviation and as tracking error against the
expected reward each baseline is trying to estimate.
"""

from __future__ import annotations

import numpy as np

from .optim import ValueBaseline, grpo_step
from .policy import Policy, sft_prior
from .reward import RewardModel

GROUP_SIZE = 16


def _gradient_spread(policy: Policy, orders, advantages: np.ndarray) -> float:
    """Standard deviation across the group's per-sample gradient estimates."""
    flat = np.stack([
        (advantage * policy.grad_log_prob(order)).ravel()
        for order, advantage in zip(orders, advantages)
    ])
    return float(flat.std(axis=0).mean())


def compare(
    rng: np.random.Generator,
    reward_model: RewardModel,
    steps: int = 400,
    learning_rate: float = 0.08,
) -> dict:
    policy = Policy.initial(rng, item_bias=sft_prior())
    reference = policy
    critic = ValueBaseline()

    spreads = {"none": [], "critic": [], "group": []}
    tracking = {"critic": [], "group": []}

    for _ in range(steps):
        orders = [policy.sample(rng) for _ in range(GROUP_SIZE)]
        rewards = np.array([reward_model.score(o) for o in orders])

        # The quantity both baselines are estimating, measured on a wide sample.
        wide = [policy.sample(rng) for _ in range(512)]
        expected = float(np.mean([reward_model.score(o) for o in wide]))

        spreads["none"].append(_gradient_spread(policy, orders, rewards))
        spreads["critic"].append(_gradient_spread(policy, orders, rewards - critic.predict()))
        spreads["group"].append(_gradient_spread(policy, orders, rewards - rewards.mean()))

        tracking["critic"].append(abs(critic.predict() - expected))
        tracking["group"].append(abs(float(rewards.mean()) - expected))

        critic = critic.updated(rewards)
        policy = grpo_step(policy, orders, rewards, reference, learning_rate, 0.0)

    return {"spread": {k: float(np.mean(v)) for k, v in spreads.items()},
            "tracking": {k: float(np.mean(v)) for k, v in tracking.items()}}


def main() -> None:
    from .reward import train_reward_model

    rng = np.random.default_rng(0)
    result = compare(rng, train_reward_model(rng))

    none = result["spread"]["none"]
    print("gradient standard deviation across the group")
    for name in ("none", "critic", "group"):
        value = result["spread"][name]
        print(f"  {name:7} {value:7.4f}   {value / none:5.1%} of no baseline")
    print("\nmean error against the expected reward being estimated")
    for name in ("critic", "group"):
        print(f"  {name:7} {result['tracking'][name]:7.4f}")


if __name__ == "__main__":
    main()
