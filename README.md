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
