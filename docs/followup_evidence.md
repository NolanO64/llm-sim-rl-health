# Follow-up evidence: seven separate checks

Each check has its own commit, runnable code, saved results and interpretation.
No new LLM calls are required. Earlier anchor/action-label controls remain in
their existing commits; these follow-ups do not republish unfinished model runs.

| Check | Explanation and results | Main script |
| --- | --- | --- |
| Non-LLM action-conditioned comparator | [Protocol/results](action_conditioned_statistical_results.md): 617/745/932 versus Qwen 1013; three seeds per setting | `statistical_world_benchmark.py` |
| Reference ranking uncertainty | [Results](reference_uncertainty_results.md): 32-seed shortfall 17.97%, conditional interval 16.36-19.74% | `ranking_reference_uncertainty.py` |
| Frozen-policy dynamics | [Results](reference_dynamics_sweep_results.md): seven settings; selection shortfall 11.9-27.0% | `reference_dynamics_sweep.py` |
| Trajectory mean alignment | [Results](trajectory_alignment_results.md): unaligned mean-inclusive accuracies .986/.757/.730 | `trajectory_alignment_audit.py` |
| Batches and policy families | [Results](ranking_robustness_results.md): batch losses 18.2/13.4/17.7%; offline-family loss 0.66% | `ranking_robustness_audit.py` |
| HeartSteps source/effect audit | [Results](heartsteps_audit_results.md): 4872 reconstructed 30-minute outcomes; paired-person effect-error intervals | `heartsteps_source_audit.py`, `heartsteps_effect_audit.py` |
| Mean versus best/SD reporting | [Results](direct_training_summary_results.md): direct 1005 +/- 61, best 1080; mini 647 +/- 201 | `direct_training_summary.py` |

## Inspect and test

Use the repository's pinned dependencies. Make the external StepCountJITAI
package available on PYTHONPATH as described in the main README. Then:

```powershell
python experiments/restore_evidence.py
python -m unittest discover -s tests -v
```

The deterministic gzip files preserve original JSON bytes. Restoration refuses
to overwrite differing local evidence. These archives contain synthetic
reference episodes or saved synthetic-policy inputs, not newly published study
records. HeartSteps downloads remain gitignored; only aggregate audits are added.

## Reanalyse saved outputs

Use new filenames so the scripts do not replace the historical evidence:

```powershell
python experiments/trajectory_alignment_audit.py --out data/results/trajectory_alignment_reproduced.json
python experiments/ranking_robustness_audit.py --out data/results/ranking_robustness_reproduced.json
python experiments/heartsteps_effect_audit.py --out data/results/heartsteps_effect_reproduced.json
python experiments/direct_training_summary.py --out data/results/direct_summary_reproduced.json
```

## Rerun local simulations

The historical manifests retain their actual run/source hashes. Release changes
and platform line endings can alter byte hashes without changing the numerical
algorithm. Do not bypass strict resume checks or rewrite the old hashes. Create
fresh output/protocol files from the current checkout instead:

```powershell
python experiments/statistical_world_benchmark.py --out data/results/statistical_reproduced.json --evaluate --rank
python experiments/ranking_reference_uncertainty.py --out data/results/ranking_reference_reproduced.json
python experiments/ranking_reference_uncertainty.py --seeds 32 --out data/results/ranking_reference_32_reproduced.json
python experiments/reference_dynamics_sweep.py --baseline-cache data/results/ranking_reference_reproduced.json --out data/results/dynamics_reproduced.json
```

The configurable baseline-cache path is a release convenience; the exact runner
used for the completed seven-setting experiment is also retained under
`docs/provenance/`. Settings, candidate sets, reference seeds and limitations are
documented per check. The original 49 candidates remain the ranking set; the
saved GPT-5-mini policies are additional transfer evaluations, not retroactive
ranking candidates.

## Interpretation limits

Useful training does not certify final policy selection. Qwen has higher observed
scores than this hand-set statistical comparator, but three seeds do not prove
general superiority or identify an isolated causal LLM gain. Reference-only
intervals omit LLM-score noise. Frozen-policy stress tests are not a second task,
clinical transfer, independent LLM-baseline evaluation, or delayed/subgroup
validation. Data-source audits do not certify historical provider authorization.
