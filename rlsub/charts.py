"""Renders the three figures into charts/:

    overoptimization.png  learned reward vs true satisfaction, against KL drift
    zero-advantage.png    verifier pass rate vs fraction of all-equal groups
    reliability.png       stated probability vs observed frequency

Run with `python -m rlsub.charts` or `make charts`.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .reward import train_reward_model  # noqa: E402
from .train import run_rlcd, run_rlhf, run_rlvr  # noqa: E402

OUTPUT = pathlib.Path(__file__).resolve().parent.parent / "charts"
PROXY_COLOUR = "#c2410c"
TRUTH_COLOUR = "#1d4ed8"


def _style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.15, linewidth=0.6)


def overoptimization(rng: np.random.Generator) -> None:
    trace = run_rlhf(rng, train_reward_model(rng), steps=2000, learning_rate=0.08)
    peak = int(np.argmax(trace.truth))

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(trace.kl, trace.proxy, color=PROXY_COLOUR, lw=2, label="Reward model says")
    ax.set_xlabel("KL from the starting policy (nats)")
    ax.set_ylabel("Reward model score", color=PROXY_COLOUR)
    ax.tick_params(axis="y", labelcolor=PROXY_COLOUR)

    # Separate axis because the two quantities are in different units. Sharing
    # one would squash the proxy curve flat against the collapse.
    twin = ax.twinx()
    twin.plot(trace.kl, trace.truth, color=TRUTH_COLOUR, lw=2, label="Customer actually thinks")
    twin.set_ylabel("True satisfaction", color=TRUTH_COLOUR)
    twin.tick_params(axis="y", labelcolor=TRUTH_COLOUR)
    twin.axvline(trace.kl[peak], color="#64748b", lw=0.9, ls="--")
    twin.annotate(
        f"truth peaks at KL {trace.kl[peak]:.1f}",
        xy=(trace.kl[peak], trace.truth[peak]),
        xytext=(trace.kl[peak] + max(trace.kl) * 0.05, min(trace.truth) * 0.55),
        fontsize=9,
        color="#475569",
        va="center",
    )

    ax.set_title("Optimising a learned reward, measured against the hidden truth")
    lines = ax.get_lines()[:1] + twin.get_lines()[:1]
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, loc="center right")
    _style(ax)
    twin.spines[["top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "overoptimization.png", dpi=160)
    plt.close(fig)


def zero_advantage(rng: np.random.Generator) -> None:
    trace = run_rlvr(rng)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(trace.steps, trace.pass_rate, color=TRUTH_COLOUR, lw=2, label="Orders that verify")
    ax.plot(
        trace.steps,
        trace.degenerate_fraction,
        color=PROXY_COLOUR,
        lw=2,
        label="Groups with zero advantage",
    )
    ax.set_xlabel("Training step")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1)
    ax.set_title("A binary verifier teaches nothing until the policy is sometimes right")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT / "zero-advantage.png", dpi=160)
    plt.close(fig)


def reliability(rng: np.random.Generator, n_bins: int = 10) -> None:
    result = run_rlcd(rng)
    edges = np.linspace(0, 1, n_bins + 1)
    centres, observed = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (result.probabilities > lo) & (result.probabilities <= hi)
        if mask.sum() < 20:
            continue
        centres.append(result.probabilities[mask].mean())
        observed.append(result.outcomes[mask].mean())

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot([0, 1], [0, 1], color="#94a3b8", ls="--", lw=1, label="Perfect calibration")
    ax.plot(centres, observed, "o-", color=TRUTH_COLOUR, lw=2, ms=6, label="Decision head")
    ax.set_xlabel("Probability the model stated")
    ax.set_ylabel("How often it actually happened")
    ax.set_title(f"Brier {result.brier:.3f}   ECE {result.ece:.3f}")
    ax.legend(frameon=False, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT / "reliability.png", dpi=160)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    overoptimization(np.random.default_rng(0))
    zero_advantage(np.random.default_rng(1))
    reliability(np.random.default_rng(2))
    for path in sorted(OUTPUT.glob("*.png")):
        print(f"wrote {path.relative_to(OUTPUT.parent)}")


if __name__ == "__main__":
    main()
