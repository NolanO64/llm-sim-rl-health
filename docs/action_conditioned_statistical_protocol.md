# Action-conditioned statistical simulator: fixed comparison protocol

This experiment addresses the missing non-LLM comparator in the StepCountJITAI
transfer and ranking study. The primary setting is `default`. The `weak` and
`strong` settings are declared response-strength sensitivities; report all three,
without selecting the setting with the best reference return.

This is a hand-specified statistical simulator that retains the qualitative
assumptions in the LLM prompt: context-matched messages can help more than generic
messages, repeated messages cause fatigue, and irrelevant/excessive messaging can
increase disengagement. It is not an independently calibrated clinical model,
an equal-data fitted baseline, or a copy of the reference transition equations.
It answers whether those assumptions can support transfer without LLM calls.
It does not establish the best performance achievable by statistical simulation.

## Information and parameter selection

- The simulator imports no StepCountJITAI implementation, source parameters,
  reference returns, real patient data, saved LLM corpora, or API clients.
- It samples the same persona distribution as the LLM world: age uniformly from
  20 through 79, and uniform baseline/reactivity/habituation categories. Age is
  retained in the persona but is not assigned an additional numeric effect.
- Numeric mappings below are explicit modeling choices on the synthetic 0-1
  outcome scale, not estimates from outcome data. Baseline levels reuse the level
  grid in the existing anchor-only control; category probabilities remain those
  of the shared LLM persona sampler. Response and fatigue levels increase in the
  order of the prompt's reactivity and habituation descriptions.
- The primary response multiplier is 1; 0.5 and 2 define weaker and stronger
  effects. Other settings are identical. There is no search over reference
  returns, and all selected variants finish synthetic training before reference
  evaluation starts.
- The runner writes a protocol snapshot and SHA-256 hashes of settings/source
  before any training or reference import. This is a local pre-evaluation
  record, not a formal external preregistration.

## Statistical model

Let c be the current binary context, a in {0,1,2,3} the action, f in [0,1]
fatigue, and d in [0,1] disengagement burden. Initial f=d=0; initial context is
balanced. Persona categories map to:

| Quantity | Values, in increasing trait order |
| --- | --- |
| Baseline b | 0.04, 0.08, 0.13, 0.20, 0.30 |
| Response size r | 0.02, 0.06, 0.12, 0.24 |
| Fatigue increment h | 0.04, 0.08, 0.16, 0.24 |

Define relevance u(a,c)=0 for no message or a mismatched tailored message,
u=0.5 for a generic message, and u=1 for a context-matched message. With response
multiplier m, the tendency for the current decision is:

    latent = clip(b * (1 - 0.5*f) + m*r*u(a,c)*exp(-f), 0, 1)

The outcome is sampled by the exact existing `TwoPartLognormalOutcomeModel`
(sigma=0.90, anchor_weight=0.50), using prior generated outcomes as its anchor
and history. The history starts empty, matching online LLM-world training; no
extra real or synthetic warm-start episodes are supplied. The sampled outcome
is the reward. For feedback into subsequent emissions it is rounded to three
decimals, as in the LLM training/scoring scripts.

After the action, fatigue updates as follows:

    f_next = 0.7*f                   if no message
    f_next = min(1, f+h)             otherwise

Let mismatch=1 for a tailored message addressed to the other context, and 0
otherwise. Burden and quitting are:

    d_next = min(1, 0.8*d + 0.15*(f_next + mismatch))
    p_quit = min(1, 0.001 + 0.05*d_next)

Quitting is a Bernoulli draw made after the current reward is counted. The next
context flips with probability 0.5, independently of action. Thus balanced
context transitions express an uninformative assumption, not an inferred
context-transition model. An episode ends on quitting or after 40 decisions.

The policy receives only the current context and consecutive-message count,
capped at three when indexing its Q table. Fatigue, burden and profile are
internal simulator variables and are never exposed to the policy. No additional
context-observation noise is imposed during synthetic training, matching the
existing LLM world. The reference evaluator instead supplies inferred context.

The random seed fixes persona, initial context, action exploration, outcome
emissions and dynamics. Persona/initial-context/exploration draws share the same
stream order as the original online LLM learner. Emissions use `seed+7` and
statistical dynamics use `seed+17`; policy scoring uses `seed+13` for stochastic
policy decisions. No API stochasticity is involved.

## Matched learner and budgets

`src/synthetic_q.py` contains the original growing-batch Q refit, extracted
without changing its algorithm; both LLM online scripts import this function.
For every variant and training seed (0,1,2):

1. Start with Q=0 over 2 contexts, 4 streak buckets, 4 actions.
2. Collect rounds of 40 full synthetic episodes with frozen Q during each round.
3. Set epsilon=max(0.1, 1 - transitions_at_round_start/8000).
4. Refit from Q=0 on the whole accumulated buffer: 60 shuffled passes,
   alpha=0.1, gamma=0.9, refit shuffle seed=0.
5. Stop after a complete round takes the buffer to at least 8000 transitions.

Actual transitions can exceed 8000 by less than a full round (at most 1600
transitions). This matches the current LLM runner's round stopping rule rather
than the older anchor-only script's episode stopping rule. Report actual counts.
The same target budget does not imply identical realized counts or identical
trajectory lengths. All final Q tables, round counts, exploration settings and
transition digests are saved before reference evaluation.

## Reference transfer evaluation

Frozen statistical policies, the three saved LLM-online policies, and the
no-message/random/context-matched baselines are evaluated with the existing
`src.stepcount.evaluate_real` function:

- Reference configuration `paper`, environment seeds 0-7, 80 episodes per seed,
  maximum 50 decisions per episode, policy RNG base seed 7000.
- The same evaluation schedule is used for every policy.
- Save every environment-seed mean, every training-seed mean, and mean/population
  SD across training seeds. Environment-seed variation and training-seed
  variation are different quantities and are not pooled together.
- Report all statistical variants, the predeclared primary setting, and the
  concurrently reevaluated LLM policies. Previously published rounded aggregate
  returns are historical context, not substituted for these matched evaluations.

These are synthetic-reference outcomes. A good result does not establish
clinical transfer, and a poor result does not establish LLM superiority over
all statistical simulators.

## Policy-ranking comparison

With `--rank`, score the original 49 deployable policies in each statistical
world, using the same 3 batches of 40 synthetic patients and seed schedule
`90000 + batch*1000 + patient_index` as the LLM benchmark. Exclude the privileged
observable-ceiling row, as in the existing analysis. Reconstruct policies using
the existing policy library, corpora and Q tables; store their input hashes.

Use the original saved rounded reference returns from `benchmark_robust.json`
for these same candidates, and explicitly label them as fixed estimates. Save
individual batch means and report pooled and per-batch rank correlation,
top-five overlap, the simulator-selected policy's reference return and selection
regret. Exact score ties are resolved in the original candidate order. This
ranking comparison does not include the newly trained statistical
policies, so the candidate set is unchanged from the original LLM comparison.

The experiment concerns the downstream transfer/ranking parts of the validation
framework. It does not apply the new simulator to HeartSteps or HPTN or claim
full validation of the statistical simulator on real datasets.

## Commands

No API keys are required. From the repository root:

```powershell
.\.venv\Scripts\python.exe experiments/statistical_world_benchmark.py --smoke

# Full training only; does not import the reference environment.
.\.venv\Scripts\python.exe experiments/statistical_world_benchmark.py

# Then evaluate frozen fits and the original ranking set.
$env:PYTHONPATH = '/path/to/StepCountJITAI'
.\.venv\Scripts\python.exe experiments/statistical_world_benchmark.py --evaluate-only --rank
```

Alternatively, `--evaluate --rank` performs all training first, followed by both
evaluations. `--resume` skips completed seeds/evaluations and requires unchanged
source/protocol hashes. Give smoke runs a different output path (the default
smoke path already differs). Existing artifacts are never replaced silently;
use a new `--out`, or explicit `--force` to rerun from scratch.
