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
| `optim.grpo_step` | GRPO. Group-relative advantage, clipped ratio, k3 KL in the loss |
| `optim.critic_step` | PPO's learned baseline, for contrast |
| `baselines.compare` | GRPO's own claim: the critic's only job is variance reduction |
| `notepad.train` | Speculative. The same optimiser pointed at a fixed-size memory |

One reward arrives per completed order, so this is a contextual bandit. No discounting, no GAE, no bootstrapping. That leaves only the choice of baseline, which is the part worth comparing.

## What the figures show

`overoptimization.png` plots the learned reward against the hidden truth. The reward model trains only on orders of four items or fewer. In that range more items really is better, so it learns a positive length coefficient of +0.38 and extrapolates past the crowding inflection. True satisfaction rises to +1.27, peaks at 4.3 nats of KL, then collapses to -9.71 while the proxy climbs from +0.11 to +8.13 without interruption.

`zero-advantage.png` shows a binary verifier going silent at both ends of training. When all sixteen samples in a group earn the same reward, the advantage is exactly zero and no gradient exists. Degenerate groups run at 100% at the start, where nothing verifies, fall to 0% at step 380 as the pass rate crosses the middle, then climb back to 92% once the policy passes almost everything. On seeds 2 to 4 the trough is 2 to 6% rather than 0%, at steps 340 to 480; the U holds on every seed. The pass rate goes 1.6% to 99.6%. A binary reward only teaches in the band where the policy sometimes fails, which is exactly why DAPO resamples until group accuracy sits strictly between 0 and 1.

`reliability.png` plots stated probability against observed frequency. The decision head emits P(customer accepts) and trains on Brier score, so a stated 0.30 should come true about 30% of the time. It reaches Brier 0.150 and ECE 0.043 across the full 0 to 1 range.

## Honest limitations

- `MAX_ITEMS` is 12, three times the comfortable order size. An earlier version used 8, where the cap bound. Once the policy could not lengthen further, the only way left to raise the proxy was better composition, which also raised true satisfaction. That produced a spurious recovery late in training. The cap should bound the environment, not shape the result.
- The two runs start from different `stop_bias` values. RLHF needs a short starting policy so there is room to improve before over-extending; RLVR needs a longer one, because a policy that never produces a verifying order sees uniformly zero rewards and never leaves the floor. Both are real regimes, and using one value for both would hide one of them.
- The menu's tastiness spread is wide, with fillings around +1.95 and cheap extras negative. A narrower spread makes the reward model's length coefficient dominate its item weights, and the policy then lengthens instead of fixing composition, which erases the early gain.
- `Policy.initial` takes a `stop_bias` and `item_bias` that approximate an SFT checkpoint. Starting from a uniform policy puts the run past the satisfaction peak before training begins.
- The reward model is linear over a bag-of-items feature vector, too weak to represent the crowding inflection. That weakness is the experiment, not an oversight.
- `grpo_step` reuses each rollout for `inner_epochs=4` updates. That is what makes the importance ratio diverge from 1 and gives clipping and the KL term something to do. At a single epoch the ratio is exactly 1, clipping is a no-op and the KL is multiplied by a coefficient every experiment sets to 0, which is worth checking before trusting any GRPO implementation. `test_clipping_actually_binds_across_inner_epochs` and `test_kl_term_changes_the_update` guard against regressing to that.
- The experiments run with `kl_coefficient=0`. That follows [Gao et al.](https://arxiv.org/abs/2210.10760), who set it to zero for their main runs after finding the penalty acts like early stopping, and DAPO, which removes it entirely. The KL path is exercised by tests rather than by the headline runs.
- At the first inner epoch the per-sample gradient coefficient equals DeepSeekMath's published form, $\hat{A} + \beta(\pi_{ref}/\pi_\theta - 1)$, verified in `test_gradient_coefficient_matches_deepseekmath`.

## Layout

```
rlsub/deli.py     environment and the hidden true reward
rlsub/policy.py   bigram softmax policy, 169 parameters
rlsub/reward.py   the three reward sources, plus Brier, log score and ECE
rlsub/optim.py    GRPO, k3 KL, and the critic it replaced
rlsub/train.py    the three runs
rlsub/charts.py   figure rendering
```

## Does dropping the critic cost anything

`python -m rlsub.baselines` estimates the same gradient three ways and reports the spread across the group.

| Baseline | Gradient standard deviation | Relative | Error against the expected reward |
|---|---|---|---|
| None | 0.1905 | 100% | |
| Learned critic | 0.0722 | 37.9% | 0.327 |
| Group mean | 0.0676 | 35.5% | 0.201 |

Those numbers are seed 0, the most favourable of five. Across seeds 0 to 4 the group baseline brings the spread to between 36% and 64% of no baseline and always sits at or just under the critic, 0.94 to 0.99 of it. It tracks the expected reward better in four of five seeds. The fair summary is that a zero-parameter group mean matches a learned critic, not that it beats one. This is a bandit, so the critic has no prefix to condition on and this is as favourable to the group as it gets.

## Writing to a memory with a reward instead of a likelihood

**Speculative.** No published work trains an SSM's write gate with RL. `rlsub/notepad.py` is a tabular toy with three slots, not evidence about Mamba.

Items stream past with category tags, a notepad holds three of them, and at the end a query asks for the first item in some category. Same GRPO, same group baseline, two different rewards.

| Trained on | Write rate, position 1 to 10 | Writes per episode | Recall |
|---|---|---|---|
| Recency | about 0.97 throughout (0.68 rising to 0.98 at seed 0) | 9.0 | 24.0% |
| Recall | 1.00 falling to 0.03 | 3.1 | 99.6% |

The curves invert. The recall-trained policy writes about three times, once per slot, early, and then defends what it has. Nothing about the architecture changed; only the reward did. `make charts` renders this as `retention.png`.

## What the policies actually ordered

Brute-forcing every order up to 12 items gives the environment's true optimum, which makes it possible to say exactly how far each method ends up from it.

| | Order | Cost | True satisfaction |
|---|---|---|---|
| Best possible | 5 x halloumi | £12.00 | +5.93, but fails the verifier |
| Best that verifies | sourdough + halloumi + halloumi | £6.00 | +4.95 |
| What RLHF learned | 12 x halloumi | £28.80 | -4.36 |
| What RLVR learned | focaccia + turkey | £4.20 | +3.00 |

The reward model scores twelve halloumi at +13.59, the highest value it ever assigns, because every slice adds tastiness and length and it has no term for crowding or budget. RLVR's policy lands on a real sandwich and then stops improving, since a binary reward carries no gradient above the threshold.

Reproduce with `python -m rlsub.best` .

## The Jev probe

`jev/probe.py` asks a live model two questions about the same support ticket: one answerable from the message, one that depends on a routing policy the model is never shown. It needs `TYPESAFE_API_KEY` in the environment and costs about two tenths of a cent for 120 tickets.

| Question | Accuracy | Brier | ECE | Mean confidence |
|---|---|---|---|---|
| Answerable, "is this customer unhappy?" | 100% | 0.006 | 0.054 | 0.95 |
| Unanswerable, "escalate to tier three?" | 40% | 0.449 | 0.430 | 0.84 |

Sixty of the 120 unanswerable answers came back below 0.1, near-certain. Only 19 sat in the 0.4 to 0.6 band where an honest "I cannot know" belongs. Mean P(escalate) was 0.287 for unhappy customers and 0.045 for pleased ones, so the model answered a question about tone rather than the one asked. A second run at seed 7 gives ECE 0.047 against 0.344.

The routing policy is arbitrary and uncorrelated with tone, so near-chance accuracy is expected and is not the point. The point is that the probability does not report the guess. Raw responses are in `jev/results.json` so the numbers can be rechecked without re-running the calls.
