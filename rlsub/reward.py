"""Three reward sources: learned (RLHF), verifier (RLVR), calibrated (RLCD)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .deli import VOCAB_SIZE, Order, accept_probability, true_satisfaction, verifies

VERIFIED_REWARD = 10.0  # Tulu 3 uses 10, not 1. Binary in structure, not magnitude

# The RM never sees orders longer than this, so it learns "more is better" and
# extrapolates off a cliff. This one number produces the over-optimization curve.
PREFERENCE_MAX_ITEMS = 4


def features(order: Order) -> np.ndarray:
    """Too weak to represent the crowding inflection. That weakness is the point."""
    counts = np.zeros(VOCAB_SIZE + 2)
    for token in order:
        counts[token] += 1.0
    counts[VOCAB_SIZE] = len(order)
    counts[VOCAB_SIZE + 1] = 1.0  # bias
    return counts


# --- RLVR ------------------------------------------------------------------


def verifier_reward(order: Order) -> float:
    return VERIFIED_REWARD if verifies(order) else 0.0


# --- RLHF ------------------------------------------------------------------


@dataclass(frozen=True)
class RewardModel:
    weights: np.ndarray

    def score(self, order: Order) -> float:
        return float(features(order) @ self.weights)


def sample_preference_pair(
    rng: np.random.Generator, noise: float
) -> tuple[Order, Order, int]:
    """Bradley-Terry. The label is sampled rather than argmax, so the fit is statistical."""
    a = _random_short_order(rng)
    b = _random_short_order(rng)
    gap = (true_satisfaction(a) - true_satisfaction(b)) / max(noise, 1e-6)
    prob_a_preferred = 1.0 / (1.0 + np.exp(-gap))
    return a, b, int(rng.random() < prob_a_preferred)


def _random_short_order(rng: np.random.Generator) -> Order:
    length = int(rng.integers(1, PREFERENCE_MAX_ITEMS + 1))
    return tuple(int(t) for t in rng.integers(0, VOCAB_SIZE, size=length))


def train_reward_model(
    rng: np.random.Generator,
    n_pairs: int = 4000,
    epochs: int = 200,
    learning_rate: float = 0.05,
    noise: float = 0.5,
) -> RewardModel:
    """Minimise -log sigmoid(r(win) - r(lose)). Ouyang Eq. 1 at K=2."""
    pairs = [sample_preference_pair(rng, noise) for _ in range(n_pairs)]
    win = np.stack([features(a if label else b) for a, b, label in pairs])
    lose = np.stack([features(b if label else a) for a, b, label in pairs])
    diff = win - lose

    weights = np.zeros(diff.shape[1])
    for _ in range(epochs):
        margin = diff @ weights
        weights += learning_rate * (_sigmoid(-margin)[:, None] * diff).mean(axis=0)
    return RewardModel(weights)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


# --- RLCD ------------------------------------------------------------------


@dataclass(frozen=True)
class DecisionHead:
    """Typed output: P(customer accepts)."""

    weights: np.ndarray

    def probability(self, order: Order) -> float:
        return float(_sigmoid(np.array(features(order) @ self.weights)))


def brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """Bounded and strictly proper. RLCR's Theorem 1 needs the boundedness."""
    return float(np.mean((probabilities - outcomes) ** 2))


def log_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """Strictly proper but unbounded, which is why RLCR uses Brier."""
    p = np.clip(probabilities, 1e-9, 1 - 1e-9)
    return float(-np.mean(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p)))


def expected_calibration_error(
    probabilities: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
) -> float:
    """Binned ECE, Guo et al. 2017."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (probabilities > lo) & (probabilities <= hi)
        if not in_bin.any():
            continue
        confidence = probabilities[in_bin].mean()
        accuracy = outcomes[in_bin].mean()
        total += in_bin.mean() * abs(accuracy - confidence)
    return float(total)


def train_decision_head(
    rng: np.random.Generator,
    orders: list[Order],
    epochs: int = 600,
    learning_rate: float = 0.35,
) -> DecisionHead:
    """Gradient descent on Brier. Outcomes are Bernoulli draws, so the head never
    sees the probability it must recover."""
    x = np.stack([features(o) for o in orders])
    y = np.array([rng.random() < accept_probability(o) for o in orders], dtype=float)

    weights = np.zeros(x.shape[1])
    for _ in range(epochs):
        p = _sigmoid(x @ weights)
        grad = (2.0 * (p - y) * p * (1.0 - p))[:, None] * x
        weights -= learning_rate * grad.mean(axis=0)
    return DecisionHead(weights)
