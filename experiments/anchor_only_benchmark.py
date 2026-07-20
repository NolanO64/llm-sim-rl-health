"""Anchor-only control for the RQ3 train-on-synthetic benchmark.

This experiment removes the LLM from the synthetic training environment. Each
patient has a sampled baseline, a short generated warm-start anchor, and a
constant anchor-mean latent passed through the same two-part emission layer as
the LLM world. Actions and contexts have no semantic effect.

The script trains offline and online Q-learning on this control world and, when
StepCountJITAI is available on PYTHONPATH, evaluates transfer to the reference
environment. It is designed to answer the reviewer question: does the headline
transfer result require the LLM, or do baseline/emission dynamics already suffice?

Examples:

  python experiments/anchor_only_benchmark.py --train-only --smoke
  python experiments/anchor_only_benchmark.py --smoke
  python experiments/anchor_only_benchmark.py --budget 8000 --corpus-patients 400
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.anchor_world import EPISODE_LENGTH, collect_anchor_episode, generate_anchor_trajectory
from src.paths import RESULTS_DIR


GAMMA = 0.9


def mean_sd(values):
    return [round(float(np.mean(values))), round(float(np.std(values)))]


def refit_q(buffer, passes=60, alpha=0.1):
    Q = np.zeros((2, 4, 4))
    rng = np.random.default_rng(0)
    order = list(range(len(buffer)))
    for _ in range(passes):
        rng.shuffle(order)
        for j in order:
            c, b, a, r, next_c, next_b, done = buffer[j]
            target = r if done else r + GAMMA * Q[next_c, next_b].max()
            Q[c, b, a] += alpha * (target - Q[c, b, a])
    return Q


def message_bucket(streak):
    return min(streak, 3)


def corpus_to_transitions(corpus):
    transitions = []
    context_index = {"A": 0, "B": 1}
    for patient in corpus:
        steps = []
        streak = 0
        for record in patient.get("trajectory", []):
            context = context_index.get(str(record.get("context", "A")).upper()[:1], 0)
            action = int(record["action"])
            reward = float(record["outcome"])
            steps.append((context, message_bucket(streak), action, reward))
            streak = 0 if action == 0 else streak + 1
        for i, (c, b, a, r) in enumerate(steps):
            if i + 1 < len(steps):
                next_c, next_b = steps[i + 1][0], steps[i + 1][1]
                transitions.append((c, b, a, r, next_c, next_b, False))
            else:
                transitions.append((c, b, a, r, 0, 0, True))
    return transitions


def train_offline_q(transitions, seed, passes=120, alpha=0.1, gamma=GAMMA):
    rng = np.random.default_rng(seed)
    Q = np.zeros((2, 4, 4))
    order = list(range(len(transitions)))
    for _ in range(passes):
        rng.shuffle(order)
        for j in order:
            c, b, a, r, next_c, next_b, done = transitions[j]
            target = r if done else r + gamma * Q[next_c, next_b].max()
            Q[c, b, a] += alpha * (target - Q[c, b, a])
    return Q


def greedy_policy(Q):
    Q = np.asarray(Q)

    def policy(obs, streak, rng):
        return int(Q[int(obs["c_infer"]), message_bucket(streak)].argmax())

    return policy


def make_anchor_corpus(seed, patients, episode_length=EPISODE_LENGTH):
    rng = np.random.default_rng(seed)
    corpus = []
    for patient_id in range(patients):
        actions = rng.integers(0, 4, size=episode_length)
        trajectory = generate_anchor_trajectory(actions, seed * 1_000_000 + patient_id)
        corpus.append({"patient": patient_id, "trajectory": trajectory})
    return corpus


def train_online_anchor(seed, budget, episodes_per_round=40):
    buffer = []
    Q = np.zeros((2, 4, 4))
    steps = 0
    episode_id = 0
    while steps < budget:
        eps = max(0.1, 1.0 - steps / budget)
        for _ in range(episodes_per_round):
            transitions = collect_anchor_episode(Q, eps, seed * 1_000_000 + episode_id)
            episode_id += 1
            buffer += transitions
            steps += len(transitions)
            if steps >= budget:
                break
        Q = refit_q(buffer)
    return Q, steps


def evaluate_anchor_policies(config, corpora, online_qs, n_episodes):
    from src.policies import matched_heuristic, no_action, random_action
    from src.stepcount import evaluate_real

    rows = {
        "floor": round(evaluate_real(no_action, config, n_episodes=n_episodes)),
        "random": round(evaluate_real(random_action, config, n_episodes=n_episodes)),
        "heuristic": round(evaluate_real(matched_heuristic, config, n_episodes=n_episodes)),
    }

    offline = []
    for corpus_id, corpus in enumerate(corpora):
        transitions = corpus_to_transitions(corpus)
        for seed in (0, 1, 2):
            policy = greedy_policy(train_offline_q(transitions, seed))
            offline.append(evaluate_real(policy, config, n_episodes=n_episodes))
    rows["anchor_offline_q"] = mean_sd(offline)

    online = [
        evaluate_real(greedy_policy(Q), config, n_episodes=n_episodes)
        for Q in online_qs
    ]
    rows["anchor_online_q"] = mean_sd(online)
    rows["anchor_online_fraction_of_heuristic"] = (
        float(np.mean(online)) / rows["heuristic"] if rows["heuristic"] else None
    )
    return rows


def print_rows(config, rows):
    print(f"\n==== anchor-only control ({config}) ====")
    print("  floor                 %d" % rows["floor"])
    print("  random                %d" % rows["random"])
    print("  heuristic             %d" % rows["heuristic"])
    print("  anchor offline Q      %d +/- %d" % tuple(rows["anchor_offline_q"]))
    print("  anchor online Q       %d +/- %d" % tuple(rows["anchor_online_q"]))
    if rows.get("anchor_online_fraction_of_heuristic") is not None:
        print("  online / heuristic    %.1f%%" % (100 * rows["anchor_online_fraction_of_heuristic"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=8000, help="anchor-world transitions per online seed")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--corpus-patients", type=int, default=400)
    parser.add_argument("--config", default="paper", choices=["paper", "clean"])
    parser.add_argument("--n-episodes", type=int, default=80)
    parser.add_argument("--out", default=str(RESULTS_DIR / "anchor_only_benchmark.json"))
    parser.add_argument("--train-only", action="store_true",
                        help="write trained Q-tables/corpus summary without importing StepCountJITAI")
    parser.add_argument("--smoke", action="store_true", help="tiny fast run")
    args = parser.parse_args()

    if args.smoke:
        args.budget = 400
        args.corpus_patients = 20
        args.n_episodes = 5
        args.seeds = "0,1"

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    corpora = [
        make_anchor_corpus(corpus_id, args.corpus_patients)
        for corpus_id in range(3)
    ]
    online = []
    online_steps = {}
    for seed in seeds:
        Q, steps = train_online_anchor(seed, args.budget)
        online.append(Q)
        online_steps[str(seed)] = steps

    result = {
        "control": "anchor_only",
        "definition": (
            "sampled patient baseline + generated warm-start anchor + constant "
            "anchor-mean latent; no LLM calls and no action/context semantics"
        ),
        "config": args.config,
        "budget": args.budget,
        "seeds": seeds,
        "corpus_patients": args.corpus_patients,
        "online_steps": online_steps,
        "online_qtables": [Q.round(5).tolist() for Q in online],
    }

    if not args.train_only:
        rows = evaluate_anchor_policies(args.config, corpora, online, args.n_episodes)
        result["rows"] = rows
        print_rows(args.config, rows)
    else:
        print("trained anchor-only controls; skipped StepCountJITAI evaluation")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print("wrote", output)


if __name__ == "__main__":
    main()
