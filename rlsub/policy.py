"""Bigram softmax policy over deli orders. Each item is conditioned on the
previous one, so credit assignment is real. 169 parameters total."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .deli import MAX_ITEMS, VOCAB_SIZE, Order

BOS = VOCAB_SIZE
STOP = VOCAB_SIZE
N_STATES = VOCAB_SIZE + 1
N_ACTIONS = VOCAB_SIZE + 1


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


@dataclass(frozen=True)
class Policy:
    logits: np.ndarray

    @staticmethod
    def initial(
        rng: np.random.Generator,
        scale: float = 0.01,
        stop_bias: float = 3.6,
        item_bias: np.ndarray | None = None,
    ) -> "Policy":
        """`stop_bias` and `item_bias` approximate an SFT checkpoint. RLHF starts
        from a mediocre supervised model, never from uniform."""
        logits = rng.normal(0.0, scale, size=(N_STATES, N_ACTIONS))
        logits[:, STOP] += stop_bias
        if item_bias is not None:
            logits[:, :VOCAB_SIZE] += item_bias
        return Policy(logits)

    def log_probs(self) -> np.ndarray:
        return _log_softmax(self.logits)

    def sample(self, rng: np.random.Generator) -> Order:
        logp = self.log_probs()
        probs = np.exp(logp)
        order: list[int] = []
        state = BOS
        while len(order) < MAX_ITEMS:
            action = int(rng.choice(N_ACTIONS, p=probs[state]))
            if action == STOP:
                break
            order.append(action)
            state = action
        return tuple(order)

    def log_prob(self, order: Order) -> float:
        """Includes the STOP action, except when MAX_ITEMS forced termination."""
        logp = self.log_probs()
        total = 0.0
        state = BOS
        for token in order:
            total += logp[state, token]
            state = token
        if len(order) < MAX_ITEMS:
            total += logp[state, STOP]
        return float(total)

    def grad_log_prob(self, order: Order) -> np.ndarray:
        """Score function d log pi / d logits."""
        probs = np.exp(self.log_probs())
        grad = np.zeros_like(self.logits)
        state = BOS
        for token in order:
            grad[state] -= probs[state]
            grad[state, token] += 1.0
            state = token
        if len(order) < MAX_ITEMS:
            grad[state] -= probs[state]
            grad[state, STOP] += 1.0
        return grad

    def updated(self, gradient: np.ndarray, learning_rate: float) -> "Policy":
        return replace(self, logits=self.logits + learning_rate * gradient)

    def entropy(self) -> float:
        logp = self.log_probs()
        return float(-(np.exp(logp) * logp).sum(axis=-1).mean())


def sft_prior() -> np.ndarray:
    """Mediocre checkpoint. Over-uses extras, under-uses bread and filling, so it
    rarely earns the structure bonus. Fixing that is the early RL gain."""
    from .deli import MENU, Category

    bias = np.zeros(VOCAB_SIZE)
    for index, menu_item in enumerate(MENU):
        if menu_item.category in (Category.EXTRA, Category.SAUCE):
            bias[index] = 1.1
        else:
            bias[index] = -0.6
    return bias
