"""Train-on-synthetic, test-on-reference transfer ladder (Table 3).

By default this prints the saved results. Use ``--retrain`` to run a fresh
comparison and write ``data/results/reference_ladder_new.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.artifacts import load_corpora, load_online_qtables, load_result
from src.paths import RESULTS_DIR
N_EPISODES = 80


def mean_sd(values):
    return round(float(np.mean(values))), round(float(np.std(values)))


def deployable_ladder(config, corpora, online_qs):
    from src.policies import matched_heuristic, no_action, random_action
    from src.stepcount import evaluate_real
    from src.tabular_q import corpus_to_transitions, greedy_policy, train_offline_q

    rows = {
        "floor": round(evaluate_real(no_action, config, n_episodes=N_EPISODES)),
        "random": round(evaluate_real(random_action, config, n_episodes=N_EPISODES)),
        "heuristic": round(evaluate_real(matched_heuristic, config, n_episodes=N_EPISODES)),
    }
    offline = []
    for corpus_id in (0, 1, 2):
        transitions = corpus_to_transitions(corpora[corpus_id])
        for seed in (0, 1, 2):
            offline.append(
                evaluate_real(greedy_policy(train_offline_q(transitions, seed)), config,
                              n_episodes=N_EPISODES)
            )
    rows["offline_q"] = mean_sd(offline)
    online = [
        evaluate_real(greedy_policy(online_qs[seed]), config, n_episodes=N_EPISODES)
        for seed in (0, 1, 2)
    ]
    rows["online_q"] = mean_sd(online)
    return rows


def ceilings(config, observable_seeds=range(4), oracle_seeds=range(3)):
    from src.stepcount import evaluate_real
    from src.tabular_q import (
        greedy_fullstate_policy,
        greedy_policy,
        make_state_bucketers,
        train_fullstate_q,
        train_online_q,
    )

    observable = [
        evaluate_real(greedy_policy(train_online_q(config, seed)), config,
                      n_episodes=N_EPISODES)
        for seed in observable_seeds
    ]
    h_bucket, d_bucket = make_state_bucketers(config)
    oracle = [
        evaluate_real(
            greedy_fullstate_policy(
                train_fullstate_q(config, h_bucket, d_bucket, seed), h_bucket, d_bucket
            ),
            config,
            n_episodes=N_EPISODES,
        )
        for seed in oracle_seeds
    ]
    return {
        "observable_ceiling": round(max(observable)),
        "observable_per_seed": [round(x) for x in observable],
        "full_state_oracle": round(max(oracle)),
        "oracle_per_seed": [round(x) for x in oracle],
    }


def print_rows(config, rows, saved=False):
    label = "saved results" if saved else "new run"
    print(f"\n==== {config} configuration ({label}) ====")
    print("  floor                 %d" % rows["floor"])
    print("  random                %d" % rows["random"])
    print("  heuristic             %d" % rows["heuristic"])
    print("  offline Q             %d +/- %d" % tuple(rows["offline_q"]))
    print("  online Q              %d +/- %d" % tuple(rows["online_q"]))
    print("  observable ceiling    %d  (per seed %s)"
          % (rows["observable_ceiling"], rows["observable_per_seed"]))
    print("  full-state oracle     %d  (per seed %s)"
          % (rows["full_state_oracle"], rows["oracle_per_seed"]))
    if "online_fraction_of_observable" in rows:
        print("  online / observable   %.1f%%" % (100 * rows["online_fraction_of_observable"]))


def print_saved():
    data = load_result("reference_ladder.json")
    print_rows("paper", data["paper"], saved=True)


def retrain():
    corpora = load_corpora()
    online_qs = load_online_qtables()
    result = {}
    for config in ("paper", "clean"):
        print(f"\ntraining fresh ceilings for {config} (this takes a few minutes)...")
        rows = {**deployable_ladder(config, corpora, online_qs), **ceilings(config)}
        result[config] = rows
        print_rows(config, rows)
    output = RESULTS_DIR / "reference_ladder_new.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print("\nwrote", output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true",
                        help="train a new stochastic comparison")
    args = parser.parse_args()
    retrain() if args.retrain else print_saved()


if __name__ == "__main__":
    main()
