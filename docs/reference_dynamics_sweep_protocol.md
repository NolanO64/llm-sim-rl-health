# Frozen-policy reference-dynamics sensitivity

29 October 2026. This protocol is fixed before evaluating the modified dynamics.
It does not request LLM inference, train new online policies or access study records.

## Design

Use the original 49 deployable candidates, original fixed LLM ranking scores,
three saved Qwen-trained policies, and three corrected GPT-5-mini policies.
The Qwen policies already belong to the 49 candidates. GPT-5-mini policies are
evaluated for transfer only, not added retrospectively to the ranking set.
Offline candidates are reconstructed deterministically from their fixed corpora
with the existing learner. No policy is optimized against the modified worlds.
Up to four CPU worker processes evaluate distinct fixed policy/configuration
pairs. RNG seeds are explicit per evaluation, independent of scheduling. The
serial pilot was interrupted for runtime reasons; its partial output is retained
separately and is not used for reporting. The same grid is used by the full run.

All worlds use the paper context-noise setting sigma=2, outcome SD=20 and
Uniform transition spread=0.3. Each policy uses environment seeds 0-7 and
80 episodes per seed; the maximum length is 50 decisions and policy RNG is
seeded at 7000 plus environment seed. StepCountJITAI's termination and reward
rules are unchanged. Store episode returns/lengths and seed-level means.

Run seven settings, changing one quantity at a time:

| Setting | Generic/matched reward coefficients | Habituation increase per message | No-message recovery fraction |
| --- | --- | --- | --- |
| Baseline | 50 / 200 | .05 | .10 |
| Half intervention strength | 25 / 100 | .05 | .10 |
| Double intervention strength | 100 / 400 | .05 | .10 |
| Half habituation growth | 50 / 200 | .025 | .10 |
| Double habituation growth | 50 / 200 | .10 | .10 |
| Half habituation recovery | 50 / 200 | .05 | .05 |
| Double habituation recovery | 50 / 200 | .05 | .20 |

Strength scales both positive message-effect coefficients, retaining a generic
to matched ratio of 1:4 and zero mismatched-message effect. The baseline activity,
disengagement parameters, initial state and outcome-noise SD stay unchanged.
Changing strength therefore also changes relative signal-to-noise ratio.
Habituation growth and recovery use the actual constructor parameters epsilon_h
and delta_h; chosen_a is transition spread, not habituation or action strength.
Halving/doubling are transparent local stress tests, not clinically calibrated
plausibility bounds. These are separate from the statistical comparator's
weak/default/strong settings. No post-result choice of settings is allowed.

## Provenance and analysis

Snapshot protocol, source and input hashes before evaluation. Refuse output
overwrite and reject resume after changed source/settings. Baseline candidate
scores may reuse the completed eight-seed episode audit only after verifying
input hashes, exact candidate order, score identity, seed schedule, episode
counts and episode/mean consistency. Smoke-test baseline episodes against the
canonical evaluator before running all variants.

Report all seven settings: overall rank correlation, selected-policy shortfall,
top-five overlap, and mean/SD across the three training seeds for each model.
All reference rankings concern the original 49 frozen candidates. Bootstrap
reference-seed clusters jointly across candidates (2,000 resamples) with fixed
LLM scores; label uncertainty as reference-only and approximate (eight clusters).
Do not imply that varying dynamics validates clinical transfer, a second task,
newly trained performance under shifted dynamics, or LLM-side uncertainty.

The 1,005 +/- 61 direct-training reference describes the original configuration
and training schedule only; do not reuse it as an optimum for modified worlds.

## Commands

Set PYTHONPATH to the sibling StepCountJITAI directory. From this repository:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_reference_dynamics -v
.\.venv\Scripts\python.exe experiments/reference_dynamics_sweep.py --smoke --out data/results/reference_dynamics_sweep_smoke_w4.json
.\.venv\Scripts\python.exe experiments/reference_dynamics_sweep.py --out data/results/reference_dynamics_sweep_w4.json
```
