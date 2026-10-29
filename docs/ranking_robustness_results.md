# Ranking robustness across batches and policy families

Reanalyse the original saved scores, not new simulator rollouts. The 49 candidates
are grouped into six related families: elementary baselines, fixed-message
policies, heuristics, offline Q, online Q and online Q/no-message blends. Three
original scoring batches each contain 40 simulated people. Reference values in
this analysis are the historical rounded means, held fixed.

## Results

- Selected-policy shortfall is 18.2%, 13.4% and 17.7% across the three scoring
  batches; corresponding top-five overlap is 1/5, 2/5 and 1/5.
- Removing each family and recomputing the best candidate within the remaining
  set leaves shortfall between 15.1% and 18.2%. The broad finding is not driven
  solely by one included family.
- Performance is not uniformly poor within every family. Within offline Q,
  correlation is .755 and shortfall is 6/913 = 0.66% of that family's own best
  reference return. Within online/no-message blends, correlation is .406 and
  shortfall is 12.2%. Small families with only three candidates support little
  precision; the output identifies the actual overlap denominator k=min(5,n).
- The previously saved context-noise sweep has overall correlations .729, .702,
  .674, .665 and .661 for sigma=.4,.8,1.2,1.6,2.0. This is a summary of existing
  frozen-policy results, not a new sweep or evidence of invariant ranking.

Enumerating all 27 ordered resamples of three batch means gives a regret range
120-202 and top-five overlap 1-2. These are exploratory sensitivity ranges, not
reliable confidence intervals or a substitute for individual rollout records.
Related candidates are not 49 independent algorithms. This analysis does not
propagate reference evaluation uncertainty; see the separate reference audit.

Reproduce with `python experiments/ranking_robustness_audit.py --out
data/results/ranking_robustness_reproduced.json`. The aggregate output contains
no participant profiles or raw study data and no API calls are required.
