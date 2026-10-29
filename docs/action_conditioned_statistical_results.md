# Action-conditioned statistical comparator: completed results

Run completed on 25 October 2026 using the fixed protocol in
`action_conditioned_statistical_protocol.md`. All three settings were declared
before reference evaluation; `default` is the primary setting. No parameters
were changed after observing reference returns. No API requests were made.

## Transfer

Each statistical setting uses training seeds 0,1,2 and the same growing-batch Q
learner, target 8000 transitions, and shared persona/emission components. Actual
counts range from 8301 to 8819 because full rounds are completed. All nine fits
were saved before importing the reference environment.

Frozen statistical and saved LLM-online policies were evaluated together using
the original reference configuration, environment seeds 0-7 and 80 episodes per
seed (640 episodes per policy). Returns below are mean +/- population SD across
the three training seeds, not confidence intervals.

| Training simulator | Reference return | Per-training-seed returns |
| --- | --- | --- |
| Statistical, weaker response | 617 +/- 150 | 588.263, 813.295, 450.571 |
| Statistical, default (primary) | 745 +/- 114 | 888.909, 736.238, 611.274 |
| Statistical, stronger response | 932 +/- 14 | 912.928, 945.610, 936.971 |
| LLM world, saved Qwen policies reevaluated | 1013 +/- 70 | 1108.651, 989.089, 942.066 |

The concurrently evaluated no-message, random and context-matched heuristic
returns are 22.240, 575.058 and 780.633, respectively. Thus the new evaluator
reproduces the published rounded reference baselines and LLM transfer summary.

The statistical model can support useful transfer. The strong setting exceeds
the context-matched heuristic and comes closer to the LLM transfer result.
The primary setting is lower and more variable. Three seeds do not establish
general LLM superiority, equivalence, or a statistically significant difference.
The range across settings shows that the quantitative strength assigned to the
qualitative prompt assumptions matters.

## Ranking the original 49 policies

The same 49 deployable candidates were scored in each statistical world, using
3 batches of 40 patients with the original patient-seed schedule. Reference
scores are the original saved rounded values, held fixed for this comparison.
Newly trained statistical policies are not added to this ranking set.

| Scoring simulator | Overall Spearman | Reference-top-third Spearman | Selection regret | Top-5 overlap |
| --- | --- | --- | --- | --- |
| Statistical, weaker response | 0.088 | 0.176 | 381 (34.4%) | 0/5 |
| Statistical, default (primary) | 0.274 | -0.038 | 381 (34.4%) | 0/5 |
| Statistical, stronger response | 0.807 | 0.056 | 295 (26.6%) | 0/5 |
| LLM world, existing saved benchmark | 0.661 | -0.026 | 202 (18.2%) | 1/5 |

The weak and default settings select `online-eps0.4 s1`, with reference return
728 versus the best candidate's 1109. The strong setting selects `heur-mix0.70`,
with reference return 814; the best reference candidate is ranked 16th by that
statistical simulator.

The strong statistical setting has higher overall correlation than the LLM
benchmark but worse selection regret and no top-five overlap. This provides an
additional example of why global ranking correlation alone is insufficient for
final policy selection. It does not show that every statistical model has this
failure or that the LLM has better ranking quality by every metric.

Individual scoring batches also vary. Statistical-default regret is 20.9%,
32.3% and 34.4%; strong-setting regret is 26.6%, 17.7% and 15.1%. Those are
descriptive batch results conditional on saved reference estimates, not a full
uncertainty analysis. The JSON contains all batch scores and metrics.

## Verification and provenance

- 18 focused unit/integration tests passed, including original reproducibility
  guards, isolated statistical imports, hidden-state restrictions, terminal
  transitions, fatigue/recovery, deterministic seeding, safe resume and atomic
  checkpoint retries.
- The extracted shared Q refit reproduces a numeric fingerprint captured from
  the original function before editing. The shared persona sampler reproduces
  the original fixed-seed descriptions and random-draw order.
- The full 8000-target-transition primary seed 0 was rerun independently. Its
  final Q table, transition digest, round metadata and exact count (8451)
  matched the original run.
- All source hashes in the saved protocol matched the implementation after the
  experiment completed. Reference implementation and input-artifact hashes
  are included in the results.
- The completed run contains 9 statistical training fits, 15 reference-policy
  evaluations and 147 statistical ranking rows (49 for each setting).

Machine-readable output:

- `data/results/action_conditioned_statistical.json`
- `data/results/action_conditioned_statistical.protocol.json`

## Suggested reviewer-response paragraph

We added a non-LLM, action-conditioned statistical comparator that retains the
qualitative assumptions supplied to the LLM: context matching, heterogeneous
responsiveness, fatigue/recovery and disengagement. It uses the same persona
distribution, emission layer and growing-batch Q refit, with three training
seeds and an 8000-transition target budget. We fixed a primary setting and
weaker/stronger response settings before reference evaluation and report all
three. Their reference returns are 745 +/- 114 (primary), 617 +/- 150 (weaker)
and 932 +/- 14 (stronger), compared with 1013 +/- 70 for the saved LLM-trained
policies reevaluated under the same schedule. Thus the qualitative assumptions
can support useful transfer without an LLM, and we do not attribute all transfer
to LLM-specific capabilities. In the original 49-policy ranking task, the
stronger statistical setting obtains overall correlation 0.807 but incurs
26.6% selection regret and has top-five overlap 0/5. This extends the observed
gap between broad agreement and reliable policy selection to an explicit
statistical comparator. These results concern the declared statistical model
and do not establish superiority over all action-conditioned simulators.

## Suggested Git scope

One focused PR can contain the statistical world, runner, protocol, tests and
the small behavior-preserving extraction of the shared persona sampler and
Q refit. Save complete experiment results in a separate commit within that PR
so implementation and evidence can be reviewed independently. The repository
already contains other uncommitted edits and artifacts; do not stage them as
part of this change without reviewing their scope.
