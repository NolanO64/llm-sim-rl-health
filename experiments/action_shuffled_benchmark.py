"""Action-label controls for the RQ3 offline train-on-synthetic benchmark.

These controls reuse the saved LLM-world corpora, so they require no new LLM
calls. Patient histories, contexts, outcomes, and quit flags are kept, but the
action labels seen by offline Q-learning are corrupted:

* patient-shuffled: permute action labels within each patient's trajectory;
* neutralized: map every nonzero message action to generic.

The benchmark asks whether offline transfer survives when the policy learner no
longer sees the original action semantics. If transfer remains strong, the LLM
corpus may be useful mainly through baseline/context structure. If it drops, the
original action labels contain decision-relevant signal.

Example:

  PYTHONPATH=/path/to/StepCountJITAI python experiments/action_shuffled_benchmark.py
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.anchor_only_benchmark import (  # local helpers avoid importing StepCount at module load
    corpus_to_transitions,
    greedy_policy,
    mean_sd,
    train_offline_q,
)
from src.artifacts import load_corpora
from src.paths import RESULTS_DIR


def patient_shuffle_actions(corpus, seed):
    """Permute each patient's action labels while preserving all other fields."""
    rng = np.random.default_rng(seed)
    shuffled = copy.deepcopy(corpus)
    for patient in shuffled:
        trajectory = patient.get("trajectory", [])
        actions = [int(record["action"]) for record in trajectory]
        if len(actions) <= 1:
            continue
        permuted = list(rng.permutation(actions))
        for record, action in zip(trajectory, permuted):
            record["action"] = int(action)
    return shuffled


def neutralize_message_actions(corpus):
    """Remove context-matching semantics: all nonzero message actions become generic."""
    neutralized = copy.deepcopy(corpus)
    for patient in neutralized:
        for record in patient.get("trajectory", []):
            action = int(record["action"])
            record["action"] = 0 if action == 0 else 1
    return neutralized


def train_and_eval(corpora, config, n_episodes):
    from src.stepcount import evaluate_real

    values = []
    policies = []
    for corpus_id, corpus in corpora.items():
        transitions = corpus_to_transitions(corpus)
        for seed in (0, 1, 2):
            Q = train_offline_q(transitions, seed)
            policy = greedy_policy(Q)
            values.append(evaluate_real(policy, config, n_episodes=n_episodes))
            policies.append({
                "corpus": int(corpus_id),
                "seed": int(seed),
                "greedy_actions": np.asarray(Q).argmax(axis=2).tolist(),
                "return": round(float(values[-1]), 3),
            })
    return mean_sd(values), policies


def evaluate_baselines(config, n_episodes):
    from src.policies import matched_heuristic, no_action, random_action
    from src.stepcount import evaluate_real

    return {
        "floor": round(evaluate_real(no_action, config, n_episodes=n_episodes)),
        "random": round(evaluate_real(random_action, config, n_episodes=n_episodes)),
        "heuristic": round(evaluate_real(matched_heuristic, config, n_episodes=n_episodes)),
    }


def print_rows(rows):
    print("\n==== action-label controls (%s) ====" % rows["config"])
    print("  floor                 %d" % rows["floor"])
    print("  random                %d" % rows["random"])
    print("  heuristic             %d" % rows["heuristic"])
    print("  original offline Q    %d +/- %d" % tuple(rows["original_offline_q"]))
    print("  action-shuffled Q     %d +/- %d" % tuple(rows["action_shuffled_offline_q"]))
    print("  neutralized Q         %d +/- %d" % tuple(rows["neutralized_offline_q"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="paper", choices=["paper", "clean"])
    parser.add_argument("--n-episodes", type=int, default=80)
    parser.add_argument("--out", default=str(RESULTS_DIR / "action_shuffled_benchmark.json"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_episodes = 5

    original = load_corpora()
    shuffled = {
        corpus_id: patient_shuffle_actions(corpus, seed=10_000 + corpus_id)
        for corpus_id, corpus in original.items()
    }
    neutralized = {
        corpus_id: neutralize_message_actions(corpus)
        for corpus_id, corpus in original.items()
    }

    rows = {
        "control": "action_label_controls",
        "config": args.config,
        "n_episodes": args.n_episodes,
        **evaluate_baselines(args.config, args.n_episodes),
    }
    rows["original_offline_q"], original_policies = train_and_eval(original, args.config, args.n_episodes)
    rows["action_shuffled_offline_q"], shuffled_policies = train_and_eval(shuffled, args.config, args.n_episodes)
    rows["neutralized_offline_q"], neutralized_policies = train_and_eval(neutralized, args.config, args.n_episodes)
    rows["policy_details"] = {
        "original": original_policies,
        "action_shuffled": shuffled_policies,
        "neutralized": neutralized_policies,
    }

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    print_rows(rows)
    print("wrote", output)


if __name__ == "__main__":
    main()
