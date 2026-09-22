"""A fixed-size memory whose write policy is trained by reward rather than likelihood.

SPECULATIVE. No published work trains an SSM's write gate with RL, and this is a
tabular toy with three slots, not evidence about Mamba. It exists to make one
question concrete: if the reward decides what a model keeps, does it keep
something different from what next-token prediction would have kept?

The setup. Items stream past, each tagged with a category. A notepad holds K
items. At every step the policy either skips the item or writes it over one
slot. At the end a query names a category and asks for the FIRST item that
appeared in it, so anything worth keeping had to be kept early and defended.

Two rewards, same optimiser, same group baseline as the deli:

    recency   can the notepad reconstruct what came next
    recall    can the notepad answer the query

The policy is a lookup table indexed by what the model can see, so the learned
retention rule can be read directly instead of inferred.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

N_CATEGORIES = 3
N_TYPES = 6
SEQUENCE_LENGTH = 10
SLOTS = 3

EMPTY = N_CATEGORIES  # slot-category value meaning "nothing written yet"
N_ACTIONS = SLOTS + 1  # skip, or overwrite one of the slots
SKIP = 0


@dataclass(frozen=True)
class Episode:
    items: tuple[tuple[int, int], ...]  # (type, category) per step
    query: int
    answer: int  # type of the first item in the queried category


def sample_episode(rng: np.random.Generator) -> Episode:
    items = tuple(
        (int(rng.integers(0, N_TYPES)), int(rng.integers(0, N_CATEGORIES)))
        for _ in range(SEQUENCE_LENGTH)
    )
    # Query only categories that actually occurred, so every episode is answerable.
    present = sorted({c for _, c in items})
    query = int(rng.choice(present))
    answer = next(t for t, c in items if c == query)
    return Episode(items, query, answer)


def _state_index(item_category: int, slots: tuple[int, ...]) -> int:
    index = item_category
    for slot_category in slots:
        index = index * (N_CATEGORIES + 1) + slot_category
    return index


N_STATES = N_CATEGORIES * (N_CATEGORIES + 1) ** SLOTS


@dataclass(frozen=True)
class WritePolicy:
    logits: np.ndarray

    @staticmethod
    def initial(rng: np.random.Generator, scale: float = 0.01) -> "WritePolicy":
        return WritePolicy(rng.normal(0.0, scale, size=(N_STATES, N_ACTIONS)))

    def action_probs(self, state: int) -> np.ndarray:
        row = self.logits[state] - self.logits[state].max()
        exp = np.exp(row)
        return exp / exp.sum()

    def updated(self, gradient: np.ndarray, learning_rate: float) -> "WritePolicy":
        return replace(self, logits=self.logits + learning_rate * gradient)


def roll_out(
    policy: WritePolicy, episode: Episode, rng: np.random.Generator
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Returns the notepad contents and the (state, action) pairs taken."""
    notepad: list[tuple[int, int] | None] = [None] * SLOTS
    trace: list[tuple[int, int]] = []

    for item_type, item_category in episode.items:
        slots = tuple(EMPTY if s is None else s[1] for s in notepad)
        state = _state_index(item_category, slots)
        action = int(rng.choice(N_ACTIONS, p=policy.action_probs(state)))
        trace.append((state, action))
        if action != SKIP:
            notepad[action - 1] = (item_type, item_category)

    return [s for s in notepad if s is not None], trace


def recall_reward(notepad, episode: Episode) -> float:
    """Did the notepad keep the answer to the query."""
    return 1.0 if any(t == episode.answer and c == episode.query for t, c in notepad) else 0.0


def recency_reward(notepad, episode: Episode) -> float:
    """Did the notepad keep the most recent items, which is what next-item
    prediction rewards. Scored as the fraction of slots holding a late item."""
    if not notepad:
        return 0.0
    recent = set(episode.items[-SLOTS:])
    return sum(1.0 for entry in notepad if entry in recent) / SLOTS


def train(
    reward_fn,
    rng: np.random.Generator,
    steps: int = 4000,
    group_size: int = 16,
    learning_rate: float = 0.35,
) -> WritePolicy:
    """Same group-relative advantage as the deli, pointed at a memory policy."""
    from .optim import group_advantages

    policy = WritePolicy.initial(rng)
    for _ in range(steps):
        episode = sample_episode(rng)
        traces, rewards = [], []
        for _ in range(group_size):
            notepad, trace = roll_out(policy, episode, rng)
            traces.append(trace)
            rewards.append(reward_fn(notepad, episode))
        advantages = group_advantages(np.array(rewards))
        if not advantages.any():
            continue

        gradient = np.zeros_like(policy.logits)
        for trace, advantage in zip(traces, advantages):
            for state, action in trace:
                probs = policy.action_probs(state)
                gradient[state] -= advantage * probs
                gradient[state, action] += advantage
        policy = policy.updated(gradient / group_size, learning_rate)
    return policy


def evaluate(policy: WritePolicy, rng: np.random.Generator, n: int = 3000) -> dict:
    recall, recency, writes = [], [], []
    for _ in range(n):
        episode = sample_episode(rng)
        notepad, trace = roll_out(policy, episode, rng)
        recall.append(recall_reward(notepad, episode))
        recency.append(recency_reward(notepad, episode))
        writes.append(sum(1 for _, a in trace if a != SKIP))
    return {"recall": float(np.mean(recall)),
            "recency": float(np.mean(recency)),
            "writes_per_episode": float(np.mean(writes))}


def write_rate_by_position(policy: WritePolicy, rng: np.random.Generator, n: int = 2000) -> np.ndarray:
    """How often the policy writes at each position. Recency keeps the tail;
    keeping firsts means writing early and then defending."""
    counts = np.zeros(SEQUENCE_LENGTH)
    for _ in range(n):
        _, trace = roll_out(policy, sample_episode(rng), rng)
        for position, (_, action) in enumerate(trace):
            if action != SKIP:
                counts[position] += 1
    return counts / n


def main() -> None:
    rng = np.random.default_rng(0)
    print("Same optimiser, same group baseline, two different rewards.\n")
    for name, reward_fn in (("recency", recency_reward), ("recall", recall_reward)):
        policy = train(reward_fn, np.random.default_rng(0))
        stats = evaluate(policy, rng)
        rates = write_rate_by_position(policy, rng)
        print(f"  trained on {name:8} -> recall {stats['recall']:.1%}  "
              f"recency {stats['recency']:.2f}  writes/episode {stats['writes_per_episode']:.1f}")
        print(f"    write rate by position: "
              + " ".join(f"{r:.2f}" for r in rates))


if __name__ == "__main__":
    main()
