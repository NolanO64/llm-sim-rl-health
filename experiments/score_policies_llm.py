"""Score the benchmark policy set inside the LLM world (produces the benchmark file).

Each policy is rolled out on a number of synthetic patients in the LLM world; its
LLM-world score is the mean realised activity, reported as several independent
batches of patients (different patients per batch, the same patients across
policies within a batch) so the run-to-run stability of the induced ranking can be
measured. Each policy's real return is also computed on StepCountJITAI. The output
is the benchmark file consumed by experiments/benchmark_validity.py.

Requires a Nebula API key. A full run scores ~50 policies over several batches of
patients and takes hours; use --smoke for a quick end-to-end check.

  python experiments/score_policies_llm.py --smoke
  python experiments/score_policies_llm.py --batches 3 --patients 40 --out data/results/benchmark_new.json
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.artifacts import load_corpora, load_online_qtables
from src.llm_client import build_client
from src.llm_world import rollout_policy
from src.paths import RESULTS_DIR
from src.policies import build_policy_zoo, matched_heuristic
from src.stepcount import evaluate_real
from src.tabular_q import greedy_policy, train_online_q


def score_policy(client, policy, n_batches, patients_per_batch, base_seed=90000, workers=4):
    batch_means = []
    for batch in range(n_batches):
        seeds = [base_seed + batch * 1000 + i for i in range(patients_per_batch)]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            totals = list(pool.map(lambda s: rollout_policy(client, policy, s), seeds))
        batch_means.append(round(float(np.mean(totals)), 3))
    return round(float(np.mean(batch_means)), 3), round(float(np.std(batch_means)), 3), batch_means


def train_observable_ceiling(config="paper", seeds=range(4)):
    candidates = [train_online_q(config, seed) for seed in seeds]
    best = max(candidates, key=lambda Q: evaluate_real(greedy_policy(Q), config))
    return greedy_policy(best)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--patients", type=int, default=40)
    parser.add_argument("--out", default=str(RESULTS_DIR / "benchmark_new.json"))
    parser.add_argument("--smoke", action="store_true", help="one cheap policy, three patients")
    args = parser.parse_args()

    client = build_client()

    if args.smoke:
        zoo = {"heuristic": matched_heuristic}
        args.batches, args.patients = 1, 3
    else:
        corpora = load_corpora()
        online_qs = load_online_qtables()
        ceiling = train_observable_ceiling("paper")
        zoo = build_policy_zoo(corpora, online_qs, ceiling_policy=ceiling)

    rows = []
    for name, policy in zoo.items():
        real = round(evaluate_real(policy, "paper"))
        mean, sd, batch_means = score_policy(client, policy, args.batches, args.patients)
        rows.append({"policy": name, "real_new": real, "llm_mean": mean,
                     "llm_sd": sd, "batch_means": batch_means})
        print("%-18s real=%-5d LLM=%.3f +/- %.3f" % (name, real, mean, sd), flush=True)
        with open(args.out, "w") as f:
            json.dump({"batches": args.batches, "patients": args.patients, "rows": rows}, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
