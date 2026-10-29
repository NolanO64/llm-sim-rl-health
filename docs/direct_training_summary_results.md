# Direct-training mean versus best-run reporting

This reanalysis reads saved run summaries only. It does not train another policy,
evaluate the environment again, or make LLM requests.

The six saved observable-state direct-training returns are 1002, 1027, 1067,
1080, 917 and 939. Their mean is 1005.33, population SD 60.64, and maximum 1080.
The saved rounded Qwen simulator-trained summary is 1013 +/- 70 over three runs.

The previous 94% headline was 1013/1080: a mean divided by a best run, not a
comparison of means. Report the direct mean (1005 +/- 61) and best separately.
The numerical correction does not prove simulator training matches or exceeds
direct training: direct training uses 25000 episodes and a different update
schedule, versus an 8000-transition target for simulator training. The empirical
reference is not a proven upper bound. Historical direct-run seed IDs are absent
from the saved artifact; do not invent them.

The same reporting audit confirms the corrected GPT-5-mini returns
825.193, 750.667 and 365.477. Their mean is 647.112 and population SD 201.457;
sample SD is 246.733. Use 647 +/- 201 when following the paper's population-SD
convention. Changing the SD convention changes reporting, not the seed returns.
Neither SD is a confidence interval or an equivalence/significance test.

Reproduce with `python experiments/direct_training_summary.py --out
data/results/direct_training_summary_reproduced.json`. The loader accepts the
compressed saved GPT-5-mini inputs or their byte-identical restored JSON files.
