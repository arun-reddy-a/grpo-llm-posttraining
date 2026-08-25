# GRPO, derived

Notation: a prompt `q`, a group of `G` sampled completions `o_1..o_G ~ pi_old(.|q)`,
a scalar reward `r_i = R(q, o_i)`, and `|o_i|` tokens in completion `i`.

---

## 1. Why the value network goes away

PPO's advantage needs a baseline. The standard choice is a learned value
function `V_phi(s)` trained alongside the policy, which for an LLM means a
second network of comparable size — more memory, more compute, and a second
optimization problem that can fail on its own.

GRPO's observation: if you are going to sample from `pi_old` anyway, sample `G`
times instead of once and use the group itself as the baseline.

```
A_i = r_i - mean(r_1..r_G)
```

`mean(r)` is an unbiased Monte-Carlo estimate of `E_{o~pi_old}[R(q,o)]`, which
is exactly what `V(q)` was approximating. It costs `G-1` extra rollouts and zero
extra parameters, and it cannot diverge — a `G`-sample mean is a `G`-sample
mean.

The trade is variance for memory. A learned `V_phi` amortizes across prompts and
across training; a group mean is re-estimated from scratch every step from `G`
samples. Small `G` therefore means a noisy baseline, which is the real reason
`G` lives in the 4–16 range rather than at 2.

**Where this fails.** If every sample in a group earns the same reward, the mean
equals every element, every advantage is 0, and the group contributes *no
gradient at all*. The rollout compute is spent and discarded. This is not an
edge case — it is the dominant failure mode on tasks that are too easy or too
hard for the current policy, and it is why `reward/frac_zero_variance_groups`
is logged on every step.

## 2. Normalizing by the group std

The published GRPO objective divides by the group standard deviation:

```
A_i = (r_i - mean(r)) / (std(r) + eps)
```

This puts every prompt on a comparable advantage scale regardless of how the
rewards happen to be spread, so no single prompt dominates a batch.

It also introduces a bias. Consider a prompt the model gets right 7 times out of
8. `std(r)` is small, so the one deviating sample is divided by a small number
and receives a large advantage. A prompt with an even 4/4 split has a large
`std(r)` and gets a *smaller* per-sample advantage — even though it carries far
more information about where the decision boundary is. Difficulty ends up
weighting the update in the wrong direction. This is the "difficulty bias"
identified in the Dr. GRPO analysis.

Both are implemented: `advantage_mode: group_norm` (published) and
`advantage_mode: group_mean` (centered only). Which wins is empirical.

## 3. The KL estimator

The regularizer is `KL(pi_theta || pi_ref)` for a frozen reference (typically
the SFT/instruct checkpoint the run starts from). We only ever have samples from
`pi_theta`, so it has to be estimated. Writing `d = log pi_ref(x) - log pi_theta(x)`
for a token `x ~ pi_theta`:

| estimator | formula | unbiased | non-negative per sample |
|---|---|---|---|
| `k1` | `-d` | yes | **no** |
| `k2` | `d^2 / 2` | no | yes |
| `k3` | `exp(d) - d - 1` | yes | yes |

`k1` is the textbook estimator and is unbiased, but a single sample can be
negative — so for that token the "penalty" term *pays* the policy to move away
from the reference. Over a batch it averages out; within a batch it adds
variance in precisely the direction you were trying to constrain.

`k3` is unbiased and non-negative for every sample. Unbiasedness:

```
E_{x~p}[exp(d) - d - 1]  =  sum_x p(x) [ q(x)/p(x) - log(q(x)/p(x)) - 1 ]
                         =  sum_x q(x)  +  sum_x p(x) log(p(x)/q(x))  -  1
                         =  1 + KL(p||q) - 1
                         =  KL(p||q)
```

Non-negativity is `exp(d) >= d + 1` for all real `d`. Both properties are
asserted in `tests/test_kl.py` — the first exactly, by taking the expectation in
closed form rather than sampling.

**Implementation note.** `exp(d)` overflows when the policy drives a token to
near-zero probability that the reference liked, and one `inf` NaNs the entire
batch's loss. `kl_divergence_k3` clamps `d` to `±20` by default, which bounds
the penalty instead of losing the step.

## 4. Clipping, and why it is inert on-policy

The objective is PPO's clipped surrogate with the group advantage in place of
the GAE one, minus the KL penalty:

```
rho_it = pi_theta(o_it) / pi_old(o_it)

J = E[ (1/G) sum_i  agg_t ( min(rho_it * A_i, clip(rho_it, 1-eps_lo, 1+eps_hi) * A_i)
                            - beta * k3_it ) ]
```

Note `A_i` carries no `t` index. The reward arrives only when the sequence
terminates, and GRPO learns no value function to distribute it across
timesteps, so every token in a completion is credited with the same advantage.

The clip exists to bound how far one update can move the policy from the
distribution the data was drawn from. With `num_inner_epochs = 1` there is no
optimizer step between sampling and the loss, so `pi_old` **is** `pi_theta`,
`rho = 1` exactly, and the clip never binds — `clip_frac` is `0.0` by
construction, not because something is broken. It starts doing real work when
`num_inner_epochs > 1` and each rollout batch is reused.

This is also why the trainer does not spend a forward pass computing `old`
log probs when `num_inner_epochs == 1`: taking them from the detached current
log probs is exact, not an approximation.

## 5. Loss aggregation is a length-bias knob

Three defensible ways to turn `[N, T]` per-token losses into one scalar:

| mode | formula | who gets equal weight |
|---|---|---|
| `seq_mean_token_mean` | `mean_i ( (1/\|o_i\|) sum_t L_it )` | each sequence |
| `token_mean` | `(sum_i sum_t L_it) / (sum_i \|o_i\|)` | each token |
| `seq_mean_token_sum` | `mean_i ( (1/T_max) sum_t L_it )` | each token, fixed scale |

The paper writes the first. Its `1/|o_i|` factor means a token inside a 20-token
completion pulls ten times harder on the gradient than a token inside a
200-token one — an implicit bias toward short outputs, which is the opposite of
what you want when the behaviour being trained is long-form reasoning.

`token_mean` (DAPO) removes it and is the default here. `seq_mean_token_sum`
(Dr. GRPO) divides by a constant instead, which also removes the length
dependence but ties the loss scale to the token budget.

**Gradient accumulation.** Under `token_mean` the denominator is a property of
the *whole* rollout batch, not of a micro-batch, so summing `K` micro-batch
means does not equal the mean over their union — micro-batches have different
token counts. The trainer counts the batch's unmasked tokens once and passes
that constant to every micro-batch, making accumulation exact. Asserted in
`tests/test_loss.py::TestAggregation::test_gradient_accumulation_reproduces_the_full_batch`
and, end to end, in
`tests/test_trainer.py::TestStepMechanics::test_gradient_accumulation_gives_the_same_update`.

## 6. What actually gets masked

`completion_mask` is 1 from the first completion token through the first EOS
**inclusive**, and 0 thereafter.

Including EOS matters: stopping is an action the policy chose, and if it carries
no gradient the model is never reinforced for ending a correct answer — a
documented path toward completions that ramble until they hit the token budget.

On top of that, `mask_truncated_completions` zeroes any completion that hit the
budget without emitting EOS. Such a completion is graded wrong by any
answer-extraction reward, but it was not necessarily *reasoning* wrong; it ran
out of room. Training on that signal teaches brevity rather than correctness.
Its reward still enters the group baseline — removing it would change the
effective group size per prompt and break the fixed-`G` reshape.

## 7. Weight precision is part of the algorithm

GRPO runs at `lr ≈ 1e-6`, two to three orders of magnitude below an SFT run,
because it is shifting an already-competent policy's distribution rather than
fitting one. That small learning rate interacts badly with 16-bit weights.

Adam normalizes the gradient, so the size of an update is set by `lr`, not by
the gradient magnitude:

```
Δw  =  lr · m̂ / (sqrt(v̂) + eps)   ≈   lr        (m̂/sqrt(v̂) is ~unit scale)
```

bf16 has an 8-bit mantissa, so near a weight of magnitude `w` the smallest
representable change is about `w · 2⁻⁸`. For a typical transformer weight
(`w ≈ 0.02`) that floor is `≈ 1.2e-4` — roughly **100× larger than a 1e-6
update**. The update does not lose precision; it rounds to exactly zero.

```
50 AdamW steps at lr=1e-6, w₀ = 0.02:
    bfloat16 →  mean |Δw| = 0.0            (nothing happened)
    float32  →  mean |Δw| = 5.0e-5         (as expected)
```

Nothing in the loss, the reward, or the gradient norm reveals this: gradients
are computed correctly, `grad_norm` is healthy, the KL stays at zero because the
policy never moves, and the reward simply never improves. It reads exactly like
a hyperparameter problem.

The fix is standard mixed precision — **fp32 master weights, bf16 math**:
`model.torch_dtype: float32` with `model.compute_dtype: bfloat16`, where
autocast recovers bf16's speed and activation savings while updates accumulate
in fp32. `Config.validate` rejects 16-bit master weights below `lr = 1e-4`.

LoRA is exempt, for a real reason rather than a convenience: its base weights
are frozen and never receive an update, so 16-bit storage there is harmless and
saves several GB. Only the adapters train, and `build_model_and_tokenizer`
upcasts exactly those to fp32.

Asserted in `tests/test_config.py::TestWeightPrecision`, which pins the numeric
fact and then checks that no shipped config violates it.

## 8. References

- Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (2024) — introduces GRPO.
- DeepSeek-AI, *DeepSeek-R1* (2025) — GRPO with rule-based rewards at scale.
- Liu et al., *Understanding R1-Zero-Like Training: A Critical Perspective* (2025) — Dr. GRPO; the std-normalization and length biases.
- Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale* (2025) — `token_mean`, clip-higher, dynamic sampling.
- Schulman, *Approximating KL Divergence* (2020) — the `k1`/`k2`/`k3` estimators.
