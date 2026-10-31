# Trajectory mean-alignment sensitivity

Question: does shifting simulated trajectories to each person's observed mean
hide differences in activity levels? Refit the existing classifier to the same
37 real/simulated patient pairs, keeping patient pairs in the same disjoint folds.
No new trajectories are generated and no LLM calls are made.

| Simulator variant | Aligned shape features | Unaligned shape features | Unaligned, mean included |
| --- | --- | --- | --- |
| Prompt only | .973 | .973 | .986 |
| Outcome sampler | .743 | .743 | .757 |
| Sampler plus history/action-noise prompt | .716 | .716 | .730 |

Lower classifier accuracy means simulated trajectories are harder to distinguish
from real ones under the selected features. All results remain above chance
(.5). The outcome sampler improves this diagnostic, but does not establish full
trajectory or action-conditioned distributional realism.

Alignment is a post hoc diagnostic transformation, not deployable calibration.
The original shift includes clipping, so unchanged shape-only accuracies here
are an empirical observation, not a mathematical invariance claim. The reanalysis
omits the classifier's naive binomial p-value because paired/dependent records
are not independent Bernoulli observations. No participant-level features or
new study records are published in the result file.

Reproduce with `python experiments/trajectory_alignment_audit.py --out
data/results/trajectory_alignment_reproduced.json`. Compare the pooled classifier
accuracies with `data/results/trajectory_alignment_audit.json`. The code refuses
to overwrite an existing result.
