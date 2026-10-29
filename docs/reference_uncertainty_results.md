# Reference-only policy-ranking uncertainty

## Question and method

Does reference-evaluation Monte Carlo noise explain the original policy-selection
failure? Keep all 49 candidate policies and their saved LLM scores fixed. Rebuild
offline candidates deterministically from their fixed corpora and load saved
online Q tables. Evaluate using the canonical paper reference setting, 80 episodes
per seed, a 50-decision cap and policy RNG base 7000. Store individual episode
returns, lengths and termination flags, with input/source hashes.

The eight-seed audit contains 31,360 episode records. All 49 rounded means
reproduce the original benchmark. A 32-seed extension contains 125,440 episodes;
its first eight seeds reproduce every original-schedule episode exactly.
No new LLM inference or online policy training is involved.

## Results

| Quantity | Eight reference seeds | 32 reference seeds |
| --- | --- | --- |
| Ranking correlation | .6605 | .6601 |
| Selected-policy shortfall | 18.15% | 17.97% |
| Approximate 95% shortfall interval | 15.04-22.59% | 16.36-19.74% |
| Top-five overlap | 1/5 | 2/5 |
| Selected policy's return | 907.486 | 889.131 |
| Best candidate's reference return | 1108.651 | 1083.923 |

The fixed simulator choice is `tabQ c0 s1`. In the extended evaluation its gap
is 194.79 reference-score points. Small changes in exact percentages and overlap
are expected when reference estimates change. The original 18.2% used rounded
saved returns; unrounded eight-seed returns give 18.15%.

Uncertainty resamples complete environment-seed clusters jointly across policies
(10,000 bootstrap draws), not candidate identities or individual dependent
episodes. The JSON also reports approximate simultaneous candidate-difference
bands. Winner/rank statistics are nonsmooth, especially near ties, so these
bootstrap summaries are approximate, not exact-coverage optimality certificates.

## Scope

Reference noise alone does not remove the observed selection failure. These
intervals condition on the frozen policies and saved LLM scores. They do not
propagate simulator-rollout noise, training-population uncertainty, task choice
or clinical validity. This is not the complete joint ranking-uncertainty analysis.

## Reproduction

Install the pinned dependencies and make the sibling StepCountJITAI package
available on PYTHONPATH. Use a new output name when rerunning historical evidence:

```powershell
python experiments/restore_evidence.py
python experiments/ranking_reference_uncertainty.py --out data/results/ranking_reference_reproduced.json
python experiments/ranking_reference_uncertainty.py --seeds 32 --out data/results/ranking_reference_32_reproduced.json
python -m unittest tests.test_reference_uncertainty -v
```

Large episode-level JSON files are distributed as deterministic `.json.gz`
archives to keep code review manageable. The restore command reproduces their
original bytes and refuses to overwrite differing local results. No human
participant trajectories are added by these archives.

Historical raw-byte source hashes are preserved, not rewritten to match release
comments, line endings or a different dependency checkout. Resume is intentionally
strict; generate fresh outputs rather than trying to resume completed historical
files from a different source snapshot.
