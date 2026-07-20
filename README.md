# An LLM-Based Simulation Framework for Benchmarking RL in Health Interventions

[Thesis PDF](Nolan_Otam_MSc_Thesis_Final.pdf)

## Repository structure

- `simulator/` - dataset loaders, prompts, LLM backends, and outcome models.
- `experiments/` - experiment generation and analysis scripts.
- `src/` - policy learning, StepCountJITAI evaluation, and statistical utilities.
- `data/realism/` - compact outputs from the HeartSteps and HPTN067 experiments.
- `data/corpora/` - LLM-world training corpora.
- `data/online_q/` - trained online Q-tables.
- `data/results/` - aggregate policy-transfer and benchmark results.

Raw third-party datasets and external environment implementations are not included.

## Environment

Python 3.12:

```bash
python -m venv .venv
pip install -r requirements.txt
```

LLM backends use `NEBULA_API_KEY`, `OPENAI_API_KEY`, or `DEEPSEEK_API_KEY`.

## External dependencies

- [HeartSteps V1](https://github.com/klasnja/HeartStepsV1)
- [StepCountJITAI](https://github.com/reml-lab/StepCountJITAI)
- [HPTN 067/ADAPT](https://doi.org/10.7910/DVN/VYXMNJ)

HeartSteps and HPTN067 locations are supplied with `--dataset-root`.
The StepCountJITAI repository root is supplied through `PYTHONPATH`.

## HeartSteps simulation

```bash
python -m simulator.validate_dataset \
  --dataset heartsteps \
  --dataset-root /path/to/HeartStepsV1 \
  --require-real-data \
  --mode autoregressive \
  --seed 42 \
  --episode-length 250 \
  --max-steps 250 \
  --patients 37 \
  --backend nebula \
  --model "SURF.Qwen3.5 122B A10B NVFP4" \
  --temperature 0.6 \
  --top-p 0.9 \
  --history-window 35 \
  --prefix-format hybrid \
  --anchor-days 35 \
  --patient-memory anchor-summary \
  --prompt-variant heartsteps-context-faithful \
  --outcome-decoder two_part_lognormal \
  --patient-workers 4 \
  -o data/realism/heartsteps_run.json
```

CLI options:

```bash
python -m simulator.validate_dataset --help
```

## Analysis

```bash
python experiments/realism_c2st.py
python experiments/feature_group_analysis.py
python experiments/intervention_effect.py
python experiments/no_llm_ablation.py
python experiments/realism_hptn.py
python experiments/emission_sweep.py
python experiments/reference_ladder.py
python experiments/benchmark_validity.py
python experiments/benchmark_analysis.py
python experiments/benchmark_sigma_sweep.py
```

## LLM-world policy experiments

```bash
python experiments/generate_corpus.py 400 data/corpora/llmworld_corpus_new.json
python experiments/train_online_llm.py --budget 8000 --seeds 0,1,2 --out data/online_q/online_qtables_new.json
python experiments/score_policies_llm.py --batches 3 --patients 40 --out data/results/benchmark_new.json
```

Online LLM runs use the VU Nebula OpenAI-compatible gateway by default and
require:

```bash
NEBULA_API_KEY=...
```

The default model is `SURF.Qwen3.5 122B A10B NVFP4`, configured in
`src/llm_world.py`.

## Online action-label controls

These controls test whether the online transfer result depends on the fine-grained
action label seen by the LLM. The learner's chosen action and Q update are left
unchanged, but the action label shown to the LLM and recorded in its simulated
history is corrupted.

```bash
# none stays none; every nonzero message is shown as generic
NEBULA_API_KEY=... python experiments/train_online_action_control.py \
  --mode neutralized --budget 8000 --seeds 0,1,2

# each chosen action is replaced by a random label before the LLM sees it
NEBULA_API_KEY=... python experiments/train_online_action_control.py \
  --mode random-label --budget 8000 --seeds 0,1,2
```

Add `--evaluate` and set `PYTHONPATH=/path/to/StepCountJITAI` to score the
resulting Q-tables on the reference environment in the same run.

These are API-heavy experiments. Use `--smoke` only to check connectivity. For
full runs, prefer `--resume` so partially completed seeds are not repeated. To
score existing Q-tables without API calls, use:

```bash
PYTHONPATH=/path/to/StepCountJITAI python experiments/train_online_action_control.py \
  --mode neutralized --evaluate-only --out data/results/online_action_control_neutralized.json
```

The full neutralized online control in
`data/results/online_action_control_neutralized.json` gives online Q =
`579 +/- 166` on the paper StepCountJITAI configuration. This is below the full
LLM-world online result (`1013 +/- 70`) but above the anchor-only online control
(`322 +/- 44`), suggesting that fine-grained action semantics matter but do not
explain all transfer by themselves.

## Anchor-only control

The anchor-only control removes the LLM and action semantics from the synthetic
training environment. It samples a patient baseline, generates a short warm-start
anchor, and then uses the anchor mean as the constant latent passed through the
same two-part emission layer. This tests whether baseline/emission dynamics
alone can explain the train-on-synthetic transfer result.

```bash
PYTHONPATH=/path/to/StepCountJITAI python experiments/anchor_only_benchmark.py
```

The committed `data/results/anchor_only_benchmark.json` was produced with the
default paper configuration. It gives anchor-only offline Q = `547 +/- 232` and
anchor-only online Q = `322 +/- 44`, well below the LLM-world online Q result
reported in `data/results/reference_ladder.json` (`1013 +/- 70`).

## Action-label controls

The action-label controls reuse the saved LLM-world corpora and do not make new
LLM calls. Patient histories, contexts, outcomes, and quit flags are preserved,
but the action labels used by offline Q-learning are corrupted:

- `patient-shuffled`: action labels are permuted within each patient's trajectory.
- `neutralized`: all nonzero actions are mapped to `generic`.

```bash
PYTHONPATH=/path/to/StepCountJITAI python experiments/action_shuffled_benchmark.py
```

The committed `data/results/action_shuffled_benchmark.json` gives original
offline Q = `667 +/- 273`, patient-shuffled offline Q = `457 +/- 297`, and
neutralized offline Q = `657 +/- 233`. Thus fully breaking action assignment
hurts transfer, while preserving the message/no-message distinction but removing
context-matching semantics leaves the high-variance offline result largely
unchanged.
