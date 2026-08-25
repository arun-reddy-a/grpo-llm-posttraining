# LLM Post-Training with GRPO

A from-scratch implementation of **Group Relative Policy Optimization** — the
algorithm behind DeepSeekMath and DeepSeek-R1 — for post-training open-source
language models. Grouped rollout generation, rule-based reward computation,
group-relative advantage estimation, and KL-regularized clipped policy updates,
written directly against PyTorch and Hugging Face rather than wrapped around an
existing RL trainer.

The point of writing it out is that the parts of GRPO that are easy to get wrong
are not the parts a wrapper exposes: which logit position predicts which token,
what the loss denominator should be under gradient accumulation, whether the
sampler's weights are still the policy's weights. Those decisions live in
`src/grpo/algo.py` and `src/grpo/logprobs.py`, depend on nothing but `torch`, and
are covered by **241 tests that run on CPU with no network access**.

```
                    ┌─────────────────────────────────────────────┐
   B prompts ──────▶│  rollout engine   (vLLM │ HF generate)      │
                    │  G samples per prompt, grouped contiguously │
                    └──────────────────────┬──────────────────────┘
                                           │  B×G completions
                    ┌──────────────────────▼──────────────────────┐
                    │  reward ensemble  (correctness + format)    │
                    └──────────────────────┬──────────────────────┘
                                           │  B×G scalars
                    ┌──────────────────────▼──────────────────────┐
                    │  A_i = (r_i − mean_group) / (std_group + ε) │  ← no value network
                    └──────────────────────┬──────────────────────┘
                                           │  B×G advantages
                    ┌──────────────────────▼──────────────────────┐
                    │  −min(ρA, clip(ρ)A) + β·KL_k3(π‖π_ref)      │
                    │  masked to completion tokens, accumulated   │
                    └──────────────────────┬──────────────────────┘
                                           │
                    ┌──────────────────────▼──────────────────────┐
                    │  AdamW step  ──▶  push weights to sampler   │
                    └─────────────────────────────────────────────┘
```

## Why GRPO rather than PPO

PPO needs a baseline to turn returns into advantages, and the standard baseline
is a learned value network — a second model of comparable size, with its own
optimizer state and its own way to fail.

GRPO removes it. Since you are sampling from the policy anyway, sample `G` times
per prompt and use the group's mean reward as the baseline. It is an unbiased
estimate of exactly the quantity `V(q)` was approximating, it costs `G−1` extra
rollouts and zero extra parameters, and it cannot diverge.

The cost is variance: a learned value function amortizes across prompts and
across training, while a group mean is re-estimated from `G` samples every step.
That is the real reason `group_size` sits in the 4–16 band rather than at 2.

It also creates a failure mode with no PPO analogue. **If every sample in a group
earns the same reward, every advantage is zero and the group contributes no
gradient at all** — the rollout compute is spent and thrown away. That is why
`reward/frac_zero_variance_groups` is logged on every step and why
`filter_zero_variance_groups` exists.

Full derivations, including the `k3` KL estimator and the length-bias analysis:
[`docs/algorithm.md`](docs/algorithm.md).

## Design decisions

### Algorithm

| Decision | Choice | Why |
|---|---|---|
| **Advantage baseline** | Group mean, `group_norm` by default | The defining GRPO move. `advantage_mode: group_mean` disables std-normalization — dividing by a near-zero std turns a negligible disagreement inside an easy group into a full-strength gradient, so difficulty ends up weighting the update backwards (Dr. GRPO's "difficulty bias"). Both ship; which wins is empirical. Demonstrated in `test_group_norm_amplifies_near_collapsed_groups`. |
| **KL estimator** | Schulman `k3`: `exp(d) − d − 1` | The naive `−d` estimator is unbiased but **signed**, so individual tokens can contribute a negative penalty — paying the policy to leave the reference. `k3` is unbiased *and* non-negative per sample. Both properties are asserted, unbiasedness in closed form rather than by sampling. |
| **KL overflow** | Clamp `d` to ±20 before `exp` | One token driven to near-zero probability produces `inf`, and one `inf` NaNs the whole batch. Clamping bounds the penalty instead of losing the step. |
| **Loss aggregation** | `token_mean` default; three modes | Not bookkeeping — a length-bias knob. The paper's `seq_mean_token_mean` carries a `1/\|o_i\|` factor that makes a token in a 20-token completion pull 10× harder than one in a 200-token completion, biasing *against* long reasoning. `token_mean` (DAPO) weights every token equally. |
| **Gradient accumulation** | Global denominator, passed in | Summing `K` micro-batch means ≠ the mean over their union, because micro-batches have different token counts. The trainer counts the batch's unmasked tokens once and hands that constant to every micro-batch, so accumulation is exact. Micro-batch size is a memory knob, never a hyperparameter — asserted end-to-end for sizes 1/2/4/8. |
| **`old` log probs** | Skipped when `num_inner_epochs == 1` | With one inner epoch no optimizer step happens between sampling and the loss, so `π_old` *is* `π_θ` and `ρ ≡ 1` exactly. Taking `old` from the detached current log probs is exact, not an approximation, and saves a full forward pass per step. Consequence: `clip_frac` is `0.0` by construction on-policy — expected, not a bug. |
| **EOS in the mask** | Included | Stopping is an action the policy chose. Excluded from the loss, the model is never reinforced for *ending* a correct answer — a documented path to completions that ramble into the token budget. |
| **Truncated completions** | Masked from the loss by default | A completion cut off at the budget is graded wrong by any answer extractor, but it was not necessarily reasoning wrong — it ran out of room. Training on that signal teaches brevity, not correctness. Its reward still enters the group baseline; removing it would change the effective group size per prompt. |
| **Weight precision** | fp32 master weights, bf16 *math* via autocast | The one that nearly shipped broken. bf16 has an 8-bit mantissa, so the smallest representable change to a weight of magnitude ~0.02 is ~1.2e-4 — while an AdamW step is ~`lr` in size because Adam normalizes the gradient. At GRPO's `lr=1e-6` **every update rounds to exactly zero and the run trains nothing**, with every logged metric still looking plausible. `torch_dtype` (storage) is now separate from `compute_dtype` (autocast), and config validation rejects 16-bit master weights below `lr=1e-4`. LoRA is exempt: its base is frozen and its adapters are upcast to fp32. |
| **Temperature** | Applied to logits before `log_softmax` | The ratio is only a valid importance weight against the *sampling* distribution. Forgetting this is silent: the loss stays finite and plausible while optimizing a different objective. Config validation also rejects `temperature: 0` — greedy decoding makes every sample in a group identical, hence zero variance and no signal. |

### Systems

| Decision | Choice | Why |
|---|---|---|
| **Rollout behind an interface** | `RolloutEngine` ABC, two backends | Generation is memory-bound autoregressive decode; training is a compute-bound teacher-forced pass. vLLM is built for the first and useless for the second. The trainer talks to the interface, and the backend is one config line. |
| **Backends** | vLLM (fast) and HF `generate` (simple) | Rollouts dominate wall-clock — 64 sequences × 512 tokens before a single gradient — and vLLM's paged KV cache is worth roughly an order of magnitude there. The HF backend has no extra dependency, no second copy of the weights, and **no sync step that can silently desynchronize**, which makes it the right tool for correctness work. |
| **Weight sync** | In-place `load_weights` after every step; fatal on failure | The classic silent GRPO failure: without it the sampler keeps drawing from the *initial* policy, the ratio drifts, clipping saturates, and reward flatlines with no error anywhere. Rebuilding the `LLM` object instead would re-profile the KV cache and re-capture CUDA graphs every step. |
| **vLLM version drift** | Probe known executor paths, fail loudly | vLLM has moved its executor internals more than once (V0→V1). `_MODEL_PATHS` tries each known layout and raises with the list it tried rather than pinning a narrow version range — and *never* silently no-ops, because a sync that reports success while updating nothing is the worst outcome available. |
| **Colocated engine** | Shares the training GPU, `gpu_memory_utilization: 0.35` | vLLM's 0.9 default assumes it owns the card. Here the policy, its gradients, and Adam's two moments have to fit alongside it. |
| **Rollouts return token ids** | Never strings | Detokenizing and re-tokenizing is not an identity round trip — byte-level BPE can merge across a boundary and change the token count. A completion shifted by one token trains on the wrong log probs, quietly. |
| **Log-prob memory** | `logsumexp` in fp32, row-by-row in bf16 | A `[N, T, V]` logit tensor at `V≈150k` dwarfs the model. In fp32, `log p = z − logsumexp(z)` is exact and avoids the intermediate; in bf16 that subtraction is too lossy to exponentiate, so we upcast one row at a time instead of the batch. |
| **transformers kwarg drift** | Sniff the `forward` signature | `num_logits_to_keep` was renamed `logits_to_keep` in 4.49. Detecting it keeps the trainer working across both, with a correct (if less memory-efficient) slicing fallback if neither exists. All three paths are tested to agree. |
| **LoRA path** | Reference model = adapters disabled | Two savings, not one. Adam's moments shrink ~200×, and `π_ref` becomes a context manager instead of a second full copy of the weights on the GPU. The single largest memory saving available to a single-GPU run. |

### Rewards

Rule-based, not a learned reward model. Because the advantage is computed
*within* a group for the same prompt, only the relative ordering inside that
group matters — which makes cheap deterministic rewards a first-class choice
rather than a fallback. They are free to evaluate, impossible to hack in the
ways a neural RM is, and testable without a GPU.

| Decision | Choice | Why |
|---|---|---|
| **Composable ensemble** | Weighted sum, components logged separately | A flat total reward cannot distinguish "not learning" from "learned the format and stopped improving at math". Those call for opposite responses, so the breakdown is logged every step. |
| **Format shaping** | `tag_count` gives partial credit | At step 0 the model has never seen the tags, so a pure-correctness reward is ~0 for every sample in every group → zero variance → zero gradient. A dense reward that separates "two tags" from "one tag" is what breaks the cold-start symmetry. |
| **Weighting** | Correctness 1.0, format 0.2, tags 0.1 | Format rewards are the easiest thing here to hack — emitting four tags is much cheaper than doing arithmetic. Keeping correctness dominant, and logging components separately, is what makes the hack visible if it happens. |
| **`None` ≠ 0** | A reward may return "not applicable" | A math reward on a row with no gold answer must not be read as "wrong" — that trains against nothing. `None` contributes 0 to the total and is excluded from that function's logged mean. |
| **Answer extraction** | `<answer>` → `\boxed{}` → last number | Explicit markers before heuristics, and the **last** match rather than the first so a model that reconsiders is graded on its final claim. A false negative here is worse than it looks: it does not just lose a sample, it *inverts* that sample's advantage relative to its group, actively training the model away from a correct behaviour. |
| **Numeric comparison** | Parse then compare with tolerance | `0.5`, `1/2`, `$0.50` and `.50` are the same answer. String equality would score three of them wrong. |
| **Config validation** | Unknown keys are errors | `beta_kl: 0.1` silently ignored because the field is `beta` means discovering after a multi-hour run that the KL penalty was never on. Every section rejects keys it does not recognize and lists the valid ones. |

## What is tested, and what that buys

241 tests, CPU-only, no network, no model downloads — so CI runs **everything**,
not a compile-only subset.

| File | Tests | Covers |
|---|---|---|
| `test_advantages.py` | 17 | Group independence, zero-variance collapse, `G=1` → 0 not NaN, the difficulty-bias amplification |
| `test_kl.py` | 8 | `k3` non-negativity, unbiasedness against analytic KL (closed form *and* Monte Carlo), overflow clamping |
| `test_loss.py` | 25 | Gradient signs, clip binding in all four ratio×advantage quadrants, mask inertness, aggregation modes, accumulation identity |
| `test_logprobs.py` | 21 | The off-by-one — plus a test that a one-position shift *would* be caught — temperature, bf16/fp32 agreement, all three transformers kwarg spellings |
| `test_masking.py` | 22 | EOS-inclusive masking, multi-EOS chat models, pad-equals-EOS confusion, left/right padding and truncation |
| `test_rewards.py` | 57 | Extraction precedence, numeric equivalence, format hacking, `None` propagation, registry |
| `test_config.py` | 50 | Every shipped config parses; typos rejected; each cross-field coherence rule; the 16-bit weight-precision guard |
| `test_data.py` | 24 | Chat templating, JSONL errors that name the line, sampler determinism and epoch wraparound |
| `test_trainer.py` | 17 | **The real trainer loop** against a tiny in-memory policy |

The trainer tests are the ones worth highlighting. They drive the actual
`GRPOTrainer` — real loss, real advantages, real accumulation — against a 16-token
toy LM and a fake sampler that replays one fixed batch. That determinism is what
makes the central assertion possible:

> **`test_rewarded_completions_become_more_likely`** — after training, the log
> probability of the completions that earned reward has gone *up* and the others
> have gone *down*. If the pipeline were inverted anywhere between reward
> computation and the optimizer step, this fails.

Alongside it: a collapsed reward batch produces no update at all, micro-batch
size 1/2/4/8 produce identical updates, a larger `beta` demonstrably restrains
the policy, and the sampler is synced exactly once per step.

## Repository layout

```
src/grpo/
├── algo.py              advantages, k3 KL, clipped surrogate, aggregation   (torch only)
├── logprobs.py          alignment, temperature, memory-efficient gather     (torch only)
├── config.py            typed YAML config; unknown keys and incoherent combos are errors
├── data.py              GSM8K / JSONL loading, chat templating, prompt sampler
├── trainer.py           the loop: rollout → reward → advantage → update → sync
├── evaluate.py          held-out greedy accuracy and pass@k
├── cli.py               `grpo train`, `grpo eval`, `--set section.field=value`
├── rewards/             registry, weighted ensemble, math + format rewards
├── rollout/             RolloutEngine ABC, HF backend, vLLM backend + weight sync
└── utils/               seeding, LR schedules, JSONL/W&B metric logging
configs/                 smoke · Qwen2.5-0.5B · Qwen2.5-1.5B · 1.5B-LoRA
docs/algorithm.md        derivations: the baseline, k3, clipping, length bias
tests/                   241 CPU-only tests
```

## Hardware

Memory is dominated by three things that must coexist on the card: the training
state, the frozen reference model, and the vLLM engine's weights plus KV cache.
Per trainable parameter, full fine-tuning costs **16 bytes** — 4 (fp32 weights)
+ 4 (fp32 grads) + 8 (Adam's two fp32 moments). Parameter counts below are
computed from each model's published architecture, not estimated.

| Config | Model | Train state | Ref model | vLLM | Total | Card |
|---|---|---|---|---|---|---|
| `smoke.yaml` | SmolLM2-135M (0.135B) | 2.2 GB | 0.3 GB | — (HF backend) | ~3 GB | any 8GB GPU, or CPU |
| `qwen2.5-0.5b-gsm8k.yaml` | Qwen2.5-0.5B (0.494B) | 7.9 GB | 1.0 GB | 8.4 GB | ~19 GB | **24GB** — RTX 3090/4090, L4, A10G |
| `qwen2.5-1.5b-gsm8k-lora.yaml` | Qwen2.5-1.5B (1.544B) | 3.7 GB | 0 GB | 10.8 GB | ~17 GB | **24GB** |
| `qwen2.5-1.5b-gsm8k.yaml` | Qwen2.5-1.5B (1.544B) | 24.7 GB | 3.1 GB | 9.6 GB | ~39 GB | **48GB** — A6000, L40S, A100 |

Two rows are worth reading against each other. The LoRA config trains a 1.5B
model on the same card as the 0.5B full fine-tune, because `r=32` on the seven
projection modules is 36.9M of 1.544B parameters (2.4%) — shrinking the Adam
state ~42× — *and* because the reference model disappears entirely when `π_ref`
is just the policy with its adapters switched off.

**Other requirements**

- **CUDA** for the vLLM backend (Linux only). The HF backend needs no GPU at all
  and runs on CPU, which is what makes `configs/smoke.yaml` usable anywhere.
- **Apple Silicon is CPU-only here.** The trainer checks `torch.cuda.is_available()`
  and does not use MPS, so an M-series Mac runs the tests fine and the smoke
  config slowly, but is not a training machine.
- **Disk**: a few GB for model weights and the HF cache.
- **No GPU needed for the test suite** — `pytest -q` is CPU-only and takes about
  a minute.

If you are renting: a single 24GB instance covers everything except the 1.5B
full fine-tune, and the LoRA config is the better first run regardless — it
reaches a larger model on smaller hardware.

## Usage

```bash
git clone https://github.com/arun-reddy-a/grpo-llm-posttraining
cd grpo-llm-posttraining

# CPU-only: the numerics and the whole test suite
pip install -r requirements-dev.txt && pip install -e . --no-deps
pytest -q

# Training environment (adds transformers/datasets/peft)
pip install -e ".[train]"

# GPU box, for the fast rollout backend (Linux + CUDA)
pip install vllm
```

**Smoke test first** — ten steps on a 135M model, exercising every code path
including the weight sync. It will not learn anything; that is not what it is
for.

```bash
grpo train --config configs/smoke.yaml
```

**A real run** — baseline eval, training, then the same eval again:

```bash
scripts/train_gsm8k.sh configs/qwen2.5-0.5b-gsm8k.yaml outputs/run1
```

The two evals are the point. Training reward always rises — it is what the run
optimizes — so the only honest measure is held-out accuracy under identical
decoding before and after.

**Individual commands**, with config overrides for anything in the YAML:

```bash
grpo train --config configs/qwen2.5-0.5b-gsm8k.yaml \
           --set algo.beta=0.0 --set rollout.group_size=16

grpo eval  --config configs/qwen2.5-0.5b-gsm8k.yaml \
           --model outputs/run1/final --split test --limit 500 --k 4
```

Every run writes `metrics.jsonl` (one JSON object per step),
`completions.jsonl` (sampled generations with their reward breakdown), and
`config.json` to the output directory.

## Reading a run

Metrics worth watching, and what they mean when they move:

| Metric | Healthy | What it means when it isn't |
|---|---|---|
| `reward/total` | rising, noisily | Flat from step 0 → check `frac_zero_variance_groups` before touching the LR |
| `reward/frac_zero_variance_groups` | well below 1.0 | → 1.0 means every group agrees with itself: the task is too easy, too hard, or the reward saturated. No gradient is flowing regardless of what the loss says |
| `reward/math_correctness` vs `reward/format` | correctness rising | Format rising while correctness is flat is reward hacking, visible only because they are logged apart |
| `kl` | small, slowly growing | Spiking → the policy is running from the reference; raise `beta` or lower the LR |
| `completion/mean_length` | stable or gently rising | Collapsing → length bias; check `algo.aggregation`. Pinned at the budget → check `frac_truncated` |
| `clip_frac` | `0.0` at `num_inner_epochs=1` | Nonzero there means the sampler and the policy have diverged — suspect the weight sync |
| `grad_norm` | steady | Spiking with `max_grad_norm` clipping constantly → LR too high for a policy-gradient objective (GRPO runs at ~1e-6, not SFT scale) |

## Results

This repository was developed and tested on a machine with **no NVIDIA GPU**, so
**no GSM8K accuracy numbers are asserted here**. A resume-facing number that was
not actually measured on the claimed setup is not worth including, and there is
no way to run a 0.5B model through 500 GRPO steps on CPU to get one honestly.

What *is* verified, on every commit and reproducible in under a minute:
241 tests covering the advantage estimator, the KL estimator, the clipped loss,
log-prob alignment, masking, rewards, config validation, and the real trainer
loop end to end — including that rewarded completions become more likely.

To fill this section in on a GPU box:

```bash
scripts/train_gsm8k.sh configs/qwen2.5-0.5b-gsm8k.yaml outputs/run1
```

and report `eval_before.json` against `eval_after.json` — same split, same
decoding, same `--limit` — alongside the run's `metrics.jsonl`.

| Model | Steps | GSM8K test acc. (before) | (after) | pass@4 (before → after) |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 500 | | | |
| Qwen2.5-1.5B-Instruct | 500 | | | |

*(Greedy decoding for accuracy; `--k 4` at T=0.7 for pass@4. Expect accuracy to
move considerably more than pass@4 — GRPO mostly sharpens the policy onto
solutions the base model could already reach, rather than teaching new ones.)*

## Scope and limitations

- **Single GPU.** No FSDP/DeepSpeed sharding and no multi-node rollout. The
  practical ceiling is a 0.5B full fine-tune or a 1.5B LoRA run on 24GB, and a
  1.5B full fine-tune on 48GB — see [Hardware](#hardware).
- **Rule-based rewards only.** No reward model, no preference data. That is a
  deliberate fit to verifiable-answer tasks (math, code, structured extraction);
  open-ended generation would need a different reward source, which is what the
  `RewardFunction` registry is there for.
- **vLLM is untested against every version.** The weight-sync path probes known
  executor layouts and fails loudly with the list it tried. It cannot silently
  no-op, but a new vLLM release may need a path added to `_MODEL_PATHS`.
- **`num_inner_epochs > 1` is implemented but off by default.** The clipping
  machinery only does real work off-policy; the default recipe is strictly
  on-policy, which is the more predictable starting point.
- **Determinism is bounded.** Seeding makes a run reproducible for a fixed
  sampler backend; vLLM and HF `generate` will not produce identical tokens from
  the same seed, and neither will the same backend across kernel versions.

## References

- Shao et al., [*DeepSeekMath*](https://arxiv.org/abs/2402.03300) (2024) — introduces GRPO
- DeepSeek-AI, [*DeepSeek-R1*](https://arxiv.org/abs/2501.12948) (2025) — GRPO with rule-based rewards at scale
- Liu et al., [*Understanding R1-Zero-Like Training*](https://arxiv.org/abs/2503.20783) (2025) — Dr. GRPO; the std-normalization and length biases
- Yu et al., [*DAPO*](https://arxiv.org/abs/2503.14476) (2025) — `token_mean`, clip-higher, dynamic sampling
- Schulman, [*Approximating KL Divergence*](http://joschu.net/blog/kl-approx.html) (2020) — the `k1`/`k2`/`k3` estimators

## License

[MIT](LICENSE)
