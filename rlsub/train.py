"""Three training runs over the same deli environment, one per reward source.

Each returns traces rather than printing, so charts.py and the tests can use
them without re-running training. `python -m rlsub.train` prints a summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .deli import Order, accept_probability, true_satisfaction, verifies
from .optim import (
    ValueBaseline,
    critic_step,
    grpo_step,
    group_advantages,
    is_degenerate_group,
)
from .policy import Policy, sft_prior
from .reward import (
    DecisionHead,
    RewardModel,
    brier_score,
    expected_calibration_error,
    train_decision_head,
    train_reward_model,
    verifier_reward,
)

GROUP_SIZE = 16


@dataclass
class Trace:
    steps: list[int] = field(default_factory=list)
    proxy: list[float] = field(default_factory=list)
    truth: list[float] = field(default_factory=list)
    kl: list[float] = field(default_factory=list)
    pass_rate: list[float] = field(default_factory=list)
    degenerate_fraction: list[float] = field(default_factory=list)


def _sampled_kl(policy: Policy, reference: Policy, orders: list[Order]) -> float:
    """Monte Carlo KL(policy || reference)."""
    return float(
        np.mean([policy.log_prob(o) - reference.log_prob(o) for o in orders])
    )


def run_rlhf(
    rng: np.random.Generator,
    reward_model: RewardModel,
    steps: int = 600,
    learning_rate: float = 0.18,
    kl_coefficient: float = 0.0,
) -> Trace:
    """KL defaults to 0, as in Gao et al.'s main runs. The penalty acts like early
    stopping rather than moving the frontier."""
    policy = Policy.initial(rng, item_bias=sft_prior())
    reference = policy
    trace = Trace()

    for step in range(steps):
        orders = [policy.sample(rng) for _ in range(GROUP_SIZE)]
        proxy = np.array([reward_model.score(o) for o in orders])
        policy = grpo_step(
            policy, orders, proxy, reference, learning_rate, kl_coefficient
        )

        if step % 10 == 0:
            evaluation = [policy.sample(rng) for _ in range(256)]
            trace.steps.append(step)
            trace.proxy.append(float(np.mean([reward_model.score(o) for o in evaluation])))
            trace.truth.append(float(np.mean([true_satisfaction(o) for o in evaluation])))
            trace.kl.append(_sampled_kl(policy, reference, evaluation))
    return trace


def run_rlvr(
    rng: np.random.Generator,
    steps: int = 900,
    learning_rate: float = 0.18,
    stop_bias: float = 2.0,
) -> Trace:
    """Tracks groups whose rewards are all identical. Those give zero gradient.

    `stop_bias` is lower here than the RLHF run uses. A colder start never
    produces a verifying order at all, so every group is uniformly zero and the
    policy never leaves the floor.
    """
    policy = Policy.initial(rng, stop_bias=stop_bias, item_bias=sft_prior())
    reference = policy
    trace = Trace()
    degenerate_window: list[bool] = []

    for step in range(steps):
        orders = [policy.sample(rng) for _ in range(GROUP_SIZE)]
        rewards = np.array([verifier_reward(o) for o in orders])
        degenerate_window.append(is_degenerate_group(rewards))
        policy = grpo_step(policy, orders, rewards, reference, learning_rate, 0.0)

        if step % 10 == 0:
            evaluation = [policy.sample(rng) for _ in range(256)]
            trace.steps.append(step)
            trace.pass_rate.append(float(np.mean([verifies(o) for o in evaluation])))
            trace.truth.append(float(np.mean([true_satisfaction(o) for o in evaluation])))
            trace.degenerate_fraction.append(float(np.mean(degenerate_window[-50:])))
    return trace


@dataclass
class CalibrationResult:
    probabilities: np.ndarray
    outcomes: np.ndarray
    true_probabilities: np.ndarray
    brier: float
    ece: float


def run_rlcd(
    rng: np.random.Generator,
    n_train: int = 4000,
    n_eval: int = 2000,
) -> CalibrationResult:
    """The environment defines true acceptance probability, so this is real
    calibration error, not a binned proxy of it."""
    train_orders = [_random_order(rng) for _ in range(n_train)]
    head: DecisionHead = train_decision_head(rng, train_orders)

    eval_orders = [_random_order(rng) for _ in range(n_eval)]
    probabilities = np.array([head.probability(o) for o in eval_orders])
    true_probabilities = np.array([accept_probability(o) for o in eval_orders])
    outcomes = (rng.random(n_eval) < true_probabilities).astype(float)

    return CalibrationResult(
        probabilities=probabilities,
        outcomes=outcomes,
        true_probabilities=true_probabilities,
        brier=brier_score(probabilities, outcomes),
        ece=expected_calibration_error(probabilities, outcomes),
    )


def _random_order(rng: np.random.Generator) -> Order:
    """Half structured, half arbitrary.

    Sampling uniformly gives almost entirely poor orders, so the decision head
    only ever sees low acceptance probabilities and the reliability diagram
    covers a narrow band. Mixing in well-formed orders spans the range.
    """
    from .deli import MAX_ITEMS, VOCAB_SIZE, Category, MENU

    if rng.random() < 0.5:
        breads = [i for i, m in enumerate(MENU) if m.category is Category.BREAD]
        fillings = [i for i, m in enumerate(MENU) if m.category is Category.FILLING]
        rest = [i for i, m in enumerate(MENU) if m.category not in (Category.BREAD, Category.FILLING)]
        order = [int(rng.choice(breads))]
        order += [int(rng.choice(fillings)) for _ in range(int(rng.integers(1, 3)))]
        order += [int(rng.choice(rest)) for _ in range(int(rng.integers(0, 3)))]
        return tuple(order)

    length = int(rng.integers(1, MAX_ITEMS + 1))
    return tuple(int(t) for t in rng.integers(0, VOCAB_SIZE, size=length))


def compare_baselines(
    rng: np.random.Generator,
    reward_model: RewardModel,
    steps: int = 300,
    learning_rate: float = 0.18,
) -> dict[str, list[float]]:
    """Group mean vs learned critic, estimating the same expected reward."""
    policy = Policy.initial(rng)
    baseline = ValueBaseline()
    group_trace: list[float] = []
    critic_trace: list[float] = []
    truth_trace: list[float] = []

    for _ in range(steps):
        orders = [policy.sample(rng) for _ in range(GROUP_SIZE)]
        rewards = np.array([reward_model.score(o) for o in orders])
        group_trace.append(float(rewards.mean()))
        critic_trace.append(baseline.predict())

        wide = [policy.sample(rng) for _ in range(512)]
        truth_trace.append(float(np.mean([reward_model.score(o) for o in wide])))

        policy, baseline = critic_step(policy, orders, rewards, baseline, learning_rate)

    return {"group_mean": group_trace, "critic": critic_trace, "expected": truth_trace}


def main() -> None:
    rng = np.random.default_rng(0)
    reward_model = train_reward_model(rng)

    rlhf = run_rlhf(rng, reward_model)
    peak = int(np.argmax(rlhf.truth))
    print("RLHF against a learned reward model")
    print(f"  proxy   {rlhf.proxy[0]:+.2f} -> {rlhf.proxy[-1]:+.2f}")
    print(f"  truth   {rlhf.truth[0]:+.2f} -> {rlhf.truth[-1]:+.2f}"
          f"  (peaked {rlhf.truth[peak]:+.2f} at step {rlhf.steps[peak]})")
    print(f"  KL      {rlhf.kl[-1]:.2f} nats")

    rlvr = run_rlvr(rng)
    print("\nRLVR against a verifier")
    print(f"  pass rate  {rlvr.pass_rate[0]:.1%} -> {rlvr.pass_rate[-1]:.1%}")
    print(f"  truth      {rlvr.truth[0]:+.2f} -> {rlvr.truth[-1]:+.2f}")
    print(f"  groups with zero advantage, final window: "
          f"{rlvr.degenerate_fraction[-1]:.1%}")

    rlcd = run_rlcd(rng)
    print("\nRLCD: a typed, calibrated decision")
    print(f"  Brier {rlcd.brier:.4f}   ECE {rlcd.ece:.4f}")


if __name__ == "__main__":
    main()
