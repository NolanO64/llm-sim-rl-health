# HeartSteps source reconstruction and paired-person effect audit

## Source reconstruction

The local audit pins the public source snapshot and SHA-256 hashes of
suggestions.csv, users.csv and its CC BY 4.0 license. It exports aggregate counts
and diagnostics, not participant IDs, records, profiles or trajectories. Source
CSV downloads remain local and gitignored; they are not added by this commit.

| Step | Records |
| --- | --- |
| Raw decision rows | 8274 |
| Exclude unavailable decisions | -1684 |
| Exclude missing usable outcomes | -423 |
| Exclude first 35 observed-history records per person | -1295 |
| Evaluated observations | 4872 |

All 37 people are retained. Every analyzed outcome is a 30-minute source: 3047
Google Fit and 1825 Jawbone observations. Longer-window fallbacks exist in the
loader but are unused here. The 4872 action/outcome pairs match the two saved
emission variants to 1e-12. An uncapped sequence or cap >=250 reproduces them;
the current default cap 210 does not. The exact historical cap argument was not
saved, so reproduction must state the sufficiently high cap explicitly.

## Paired-person effect uncertainty

Reanalyse the saved real/simulated observations with 2000 whole-person bootstrap
resamples, retaining each pair's complete history together. Use the same pooled,
unweighted contrast on clip(30-minute steps/1500,0,1), excluding initial observed
history supplied to initialize the simulator. No new LLM requests are made.

- For the two-part outcome sampler, the activity-suggestion contrast is overstated
  by +.0367, paired-person 95% interval [.0166,.0566]. Equal-person weighting also
  retains overstatement (real +.0022 versus simulated +.0370).
- With the history/action-noise prompt, the error is +.0172, interval
  [-.0031,.0399]. The interval includes zero: a definite residual-bias claim is
  not supported by this analysis.
- Including the initial-history observations changes the observed contrast.
  Report the exact evaluation subset rather than interpreting the diagnostic as
  the full trial's causal estimate. No propensity or missing-outcome weighting
  is applied; independent people, not independent decisions, are assumed.

The source audit identifies four non-monotonic naive time intervals and missing
future outcomes. It assesses delayed-effect/subgroup feasibility only; it does
not validate delayed causal effects or subgroup policy values. Existing results
cannot retrospectively become an independent held-out validation set.

## Commands

Obtain the licensed source snapshot identified in the result/protocol, retaining
it under the ignored dataset directory. Then:

```powershell
python experiments/heartsteps_source_audit.py --dataset datasets/heartsteps_v1 --compact data/realism/heartsteps_emission_two_part.json --out data/results/heartsteps_source_reproduced.json
python experiments/heartsteps_effect_audit.py --out data/results/heartsteps_effect_reproduced.json
python -m unittest tests.test_heartsteps_audit -v
```

The source and paired-effect results are separate artifacts. Public availability
and a dataset license do not alone establish privacy clearance for future LLM
processing. Nothing in this local audit certifies provider authorization or
retention arrangements.
