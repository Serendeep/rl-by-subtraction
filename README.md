# rl-by-subtraction

Toy implementations of three reward sources, learned, verifiable and calibrated, sharing one tiny environment so the comparison is fair. numpy and matplotlib only, no GPU, the whole thing runs in under a minute.

```bash
pip install -e ".[dev]"
make test      # 11 checks, including an analytic-vs-numerical gradient test
make run       # prints the three training summaries
make charts    # writes charts/*.png
```

## The environment

`rlsub/deli.py` is a deli counter. An order is a short sequence of menu items. The customer's true satisfaction is a fixed hidden function: tastiness per item, a bonus for a coherent sandwich, a quadratic crowding penalty past four items, and a charge per pound over budget.

Every training method here sees a different partial view of that function. In production you only see the proxy, which is why reward hacking is hard to catch. Here both curves fit on the same axes.

| Module | Implements |
|---|---|
| `reward.train_reward_model` | RLHF. A scalar fit to noisy pairwise preferences via Bradley-Terry |
| `reward.verifier_reward` | RLVR. A deterministic rule checker returning 10 or 0 |
| `reward.train_decision_head` | RLCD. A probability scored by a bounded proper scoring rule |
| `optim.grpo_step` | GRPO. Group-relative advantage, k3 KL in the loss |
| `optim.critic_step` | PPO's learned baseline, for contrast |

One reward arrives per completed order, so this is a contextual bandit. No discounting, no GAE, no bootstrapping. That leaves only the choice of baseline, which is the part worth comparing.

## What the figures show

`overoptimization.png` plots the learned reward against the hidden truth. The reward model trains only on orders of four items or fewer. In that range more items really is better, so it learns a positive length coefficient of +0.38 and extrapolates past the crowding inflection. True satisfaction rises to +0.83, peaks at 2.4 nats of KL, then collapses to -11.42 while the proxy climbs from +0.10 to +6.74 without interruption.

`zero-advantage.png` shows a binary verifier that cannot teach until the policy is already sometimes right. At the start, every one of the sixteen samples in a group fails, so the rewards are uniformly zero, the advantage is exactly zero, and no gradient exists. Degenerate groups run at 100% early and 0% late while the pass rate climbs from 1.6% to 58.6%. DAPO resamples until group accuracy sits strictly between 0 and 1, which covers this end and the saturated end alike.

`reliability.png` plots stated probability against observed frequency. The decision head emits P(customer accepts) and trains on Brier score, so a stated 0.30 should come true about 30% of the time. It reaches Brier 0.150 and ECE 0.043 across the full 0 to 1 range.

## Honest limitations

- `MAX_ITEMS` is 12, three times the comfortable order size. An earlier version used 8, where the cap bound. Once the policy could not lengthen further, the only way left to raise the proxy was better composition, which also raised true satisfaction. That produced a spurious recovery late in training. The cap should bound the environment, not shape the result.
- The two runs start from different `stop_bias` values. RLHF needs a short starting policy so there is room to improve before over-extending; RLVR needs a longer one, because a policy that never produces a verifying order sees uniformly zero rewards and never leaves the floor. Both are real regimes, and using one value for both would hide one of them.
- The menu's tastiness spread is wide, with fillings around +1.95 and cheap extras negative. A narrower spread makes the reward model's length coefficient dominate its item weights, and the policy then lengthens instead of fixing composition, which erases the early gain.
- `Policy.initial` takes a `stop_bias` and `item_bias` that approximate an SFT checkpoint. Starting from a uniform policy puts the run past the satisfaction peak before training begins.
- The reward model is linear over a bag-of-items feature vector, too weak to represent the crowding inflection. That weakness is the experiment, not an oversight.
- GRPO here runs one inner epoch per rollout, so the importance ratio is 1 and clipping never activates. The code computes the term anyway to keep it visible.

## Layout

```
rlsub/deli.py     environment and the hidden true reward
rlsub/policy.py   bigram softmax policy, 169 parameters
rlsub/reward.py   the three reward sources, plus Brier, log score and ECE
rlsub/optim.py    GRPO, k3 KL, and the critic it replaced
rlsub/train.py    the three runs
rlsub/charts.py   figure rendering
```
