"""GRPO and the critic it replaced. One reward per order, so this is a bandit:
no GAE, no bootstrapping, leaving only the choice of baseline."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .policy import Policy

DEFAULT_CLIP = 0.2
# DAPO raises only the upper bound. Raising the lower one drives tokens to zero.
DAPO_CLIP_LOW = 0.2
DAPO_CLIP_HIGH = 0.28


def group_advantages(rewards: np.ndarray, divide_by_std: bool = True) -> np.ndarray:
    """`divide_by_std=True` is DeepSeekMath's form. Dr. GRPO argues it encodes a
    question-difficulty bias and should go."""
    centred = rewards - rewards.mean()
    if not divide_by_std:
        return centred
    spread = rewards.std()
    if spread < 1e-8:
        # DAPO's zero-advantage collapse. A real zero gradient, not a numerical one.
        return np.zeros_like(centred)
    return centred / spread


def is_degenerate_group(rewards: np.ndarray) -> bool:
    """Every sample scored the same, so the update is a no-op."""
    return bool(rewards.std() < 1e-8)


def clipped_surrogate(
    ratio: np.ndarray,
    advantage: np.ndarray,
    clip_low: float = DEFAULT_CLIP,
    clip_high: float | None = None,
) -> np.ndarray:
    """PPO's pessimistic bound."""
    high = clip_low if clip_high is None else clip_high
    return np.minimum(ratio * advantage, np.clip(ratio, 1 - clip_low, 1 + high) * advantage)


def k3_kl(log_prob_policy: np.ndarray, log_prob_reference: np.ndarray) -> np.ndarray:
    """Schulman's k3, (r - 1) - log r. Unbiased only at one inner epoch per rollout."""
    log_ratio = log_prob_reference - log_prob_policy
    return np.exp(np.clip(log_ratio, -60, 60)) - log_ratio - 1.0


@dataclass(frozen=True)
class ValueBaseline:
    """PPO's critic, which in a bandit degenerates to a running estimate of E[R]."""

    value: float = 0.0
    learning_rate: float = 0.1

    def updated(self, rewards: np.ndarray) -> "ValueBaseline":
        error = float(rewards.mean()) - self.value
        return replace(self, value=self.value + self.learning_rate * error)

    def predict(self) -> float:
        return self.value


def grpo_step(
    policy: Policy,
    orders: list[tuple[int, ...]],
    rewards: np.ndarray,
    reference: Policy,
    learning_rate: float,
    kl_coefficient: float,
    clip_low: float = DEFAULT_CLIP,
    clip_high: float | None = None,
    divide_by_std: bool = True,
) -> Policy:
    """One GRPO update over G orders. The ratio is 1 on the first inner epoch, so
    clipping never activates. The code computes it anyway to keep the term visible."""
    advantages = group_advantages(rewards, divide_by_std=divide_by_std)
    if not advantages.any():
        return policy

    gradient = np.zeros_like(policy.logits)
    for order, advantage in zip(orders, advantages):
        log_prob = policy.log_prob(order)
        ratio = 1.0  # single inner epoch
        surrogate = clipped_surrogate(
            np.array([ratio]), np.array([advantage]), clip_low, clip_high
        )[0]
        score = policy.grad_log_prob(order)

        # In the loss, not the reward. That keeps the advantage a pure function
        # of the group's rewards, and is the structural difference from PPO-RLHF.
        kl_penalty = k3_kl(
            np.array([log_prob]), np.array([reference.log_prob(order)])
        )[0]
        gradient += (surrogate - kl_coefficient * kl_penalty) * score

    return policy.updated(gradient / len(orders), learning_rate)


def critic_step(
    policy: Policy,
    orders: list[tuple[int, ...]],
    rewards: np.ndarray,
    baseline: ValueBaseline,
    learning_rate: float,
) -> tuple[Policy, ValueBaseline]:
    """Same update, learned baseline instead of the group mean."""
    advantages = rewards - baseline.predict()
    gradient = np.zeros_like(policy.logits)
    for order, advantage in zip(orders, advantages):
        gradient += advantage * policy.grad_log_prob(order)
    return policy.updated(gradient / len(orders), learning_rate), baseline.updated(rewards)
