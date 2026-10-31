# Frozen-policy dynamics sweep: completed results

Completed 29 October 2026. Machine-readable artifact:
`data/results/reference_dynamics_sweep_w4.json`. Its separate `.protocol.json`
records settings and source/input hashes before evaluation. No new LLM calls,
participant-data queries or online policy training were performed.

## Design and interpretation

Keep the original 49 deployable candidates and saved LLM ranking scores fixed.
Separately evaluate three saved Qwen policies and three corrected GPT-5-mini
policies. The Qwen policies are already in the ranking set; GPT-5-mini is not
added retrospectively to it. Offline candidates are deterministically rebuilt
from their original fixed corpora with the same learner.

Each policy/configuration uses eight environment seeds and 80 episodes/seed,
with the original 50-decision cap and policy RNG convention. Seven settings vary
one factor at a time: baseline, half/double message-effect strength, half/double
habituation growth per message, and half/double recovery on a no-message action.
The exact coefficients and fixed noise settings are in
`docs/reference_dynamics_sweep_protocol.md`. The ranges are transparent stress
tests, not clinically estimated bounds. They do not change the LLM prompt or
optimize policies for shifted dynamics.

## Ranking the original 49 policies

The selected policy is always `tabQ c0 s1` because its LLM score is held fixed.
The best reference candidate is recomputed in each world. Loss is the difference
between that world's best candidate return and selected-policy return, divided
by that best return. It is not loss against a known globally optimal policy.

| Reference setting | Rank correlation | Selection shortfall | Top-five overlap |
| --- | --- | --- | --- |
| Original baseline | .661 | 18.1% | 1/5 |
| Half message strength | .637 | 16.4% | 1/5 |
| Double message strength | .663 | 18.4% | 1/5 |
| Half habituation growth | .655 | 15.2% | 0/5 |
| Double habituation growth | .535 | 27.0% | 1/5 |
| Half habituation recovery | .697 | 11.9% | 2/5 |
| Double habituation recovery | .617 | 21.1% | 1/5 |

The ranking-selection failure persists in these settings, but its magnitude is
not invariant: shortfall ranges from 11.9% to 27.0%. Faster habituation growth
gives the lowest overall correlation and largest loss. The reference-best
candidate changes to `online-eps0.8 s0` in that setting; in the other six it is
`online-tabQ s0`. These are conditional case-study results, not generality over
all environments or clinical evidence.

## Transfer of the frozen trained policies

Values are mean +/- population SD across three training seeds, after averaging
640 reference episodes per policy. Higher return is better. The random baseline
is evaluated in each corresponding world; it is not a single constant comparator.

| Reference setting | Qwen-trained | GPT-5-mini-trained | Random |
| --- | --- | --- | --- |
| Original baseline | 1013 +/- 70 | 647 +/- 201 | 575 |
| Half message strength | 506 +/- 32 | 320 +/- 102 | 281 |
| Double message strength | 2016 +/- 138 | 1269 +/- 400 | 1122 |
| Half habituation growth | 1184 +/- 112 | 859 +/- 294 | 729 |
| Double habituation growth | 693 +/- 24 | 384 +/- 121 | 337 |
| Half habituation recovery | 913 +/- 27 | 633 +/- 206 | 532 |
| Double habituation recovery | 1134 +/- 115 | 658 +/- 194 | 626 |

The Qwen mean exceeds both random and the GPT-5-mini mean in all seven settings.
This is descriptive evidence for these frozen policies; three training seeds and
SD overlap do not establish general superiority, equivalence or an isolated
effect of model choice. Scores across worlds have different reward scales, so
compare policies within a row rather than treating larger cross-row raw return
as a better simulator. The historical direct-training reference 1005 +/- 61
applies only to the original world and its different training/update schedule;
it is not an optimum for the shifted worlds.

## Uncertainty and provenance checks

- There are 364 policy/configuration records and 232,960 episode returns.
  Of these, 31,360 baseline candidate episodes reuse the verified historical
  eight-seed audit; 201,600 are new frozen-policy evaluations.
- The baseline cache checks source/input hashes, candidate identities/order,
  fixed LLM scores, seed schedule, episode counts and raw-episode/mean equality.
- All source/input hashes were rechecked after completion and match.
- All 364 row means were checked against their stored episode returns.
- Every no-message episode return/length matches across all seven settings,
  as expected when only message effects and habituation are varied.
- All 61 overlapping policy/configuration evaluations in the interrupted serial
  pilot match the parallel run's episode records exactly. All 21 smoke-test
  policy/configuration records also match exactly between serial and parallel.
- Three reference-dynamics unit tests pass. They check baseline equivalence to
  the original episode evaluator, one-factor-at-a-time parameter construction,
  and rejection of a changed-input baseline cache. A Windows path-normalization
  issue in the new cache guard was caught and fixed before the full run.
- Four CPU workers change scheduling only; evaluation RNG seeds are explicit.
  The serial pilot's incomplete output is retained separately and not used as
  an additional replicate or selected for reporting.

The JSON includes approximate reference-seed bootstrap intervals for each
setting (2000 resamples). They condition on fixed simulator scores and trained
policies, use only eight clusters and do not propagate LLM scoring uncertainty.
The earlier 32-seed result (18.0%, interval 16.4-19.7%) remains a separate,
original-world precision check, not an interval across these seven worlds.

## Scope and reproduction

This extends the earlier context-noise ranking analysis to
frozen-policy transfer and ranking under changed message strength, habituation
growth and recovery. It does not add a second downstream task, retrained policies
under shifted dynamics, an independent LLM simulator or clinical validation.

Run `python experiments/restore_evidence.py` to restore the compressed completed
episode artifact and the two corrected GPT-5-mini input artifacts. These contain
synthetic experiment data, not newly uploaded participant records.

Historical hashes are preserved. For a fresh checkout, first run the original
eight-seed reference audit to a new filename, then supply that cache explicitly:

```powershell
python experiments/ranking_reference_uncertainty.py --out data/results/ranking_reference_reproduced.json
python experiments/reference_dynamics_sweep.py --baseline-cache data/results/ranking_reference_reproduced.json --out data/results/dynamics_reproduced.json
```

This avoids resuming historical files with different source bytes or line
endings. The published runner adds only the configurable cache path; the exact
executed runner is retained in `docs/provenance/reference_dynamics_sweep_20261029.py`
as a provenance snapshot, not as a second experiment or an alternative result.
