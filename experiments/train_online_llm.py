"""Online tabular Q-learning inside the LLM world (produces data/online_q).

Growing-batch training: in each round the current policy (epsilon-greedy on the
tabular Q) chooses every action, the language model simulates the day, and the
Q-table is refit from scratch on the growing buffer of transitions. The real
environment is never touched during training -- an assertion enforces that the
StepCountJITAI module is not even imported -- so the resulting policies are
trained purely on synthetic experience. Their real-environment evaluation is done
afterwards by experiments/reference_ladder.py.

The committed data/online_q/online_qtables.json was produced by this procedure
(budget 8000 transitions per seed, seeds 0/1/2). The language model is stochastic,
so a fresh run gives policies of the same quality, not bit-identical Q-tables.

Requires NEBULA_API_KEY. A full-budget run is hours of model calls;
--smoke does one tiny round to check the loop end-to-end.

  python experiments/train_online_llm.py --smoke
  python experiments/train_online_llm.py --budget 8000 --seeds 0,1,2 --out data/online_q/online_qtables_new.json
"""
import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm_client import build_client
from src.llm_world import ACTIONS, EPISODE_LENGTH, emit_outcome, llm_day_step, make_emitter, persona
from src.paths import ONLINE_Q_DIR

GAMMA = 0.9
EPISODES_PER_ROUND = 40
WORKERS = 4  # concurrent patients; stays under the gateway's request limit


def collect_episode(client, Q, eps, seed):
    """One interactive episode under the current policy; returns its transitions."""
    rng = np.random.default_rng(seed)
    emit_rng = random.Random(seed + 7)
    emitter = make_emitter()
    person = persona(rng)
    context = "A" if rng.random() < 0.5 else "B"
    streak = 0
    rows = []
    seq = []
    for _ in range(EPISODE_LENGTH):
        c = 0 if context == "A" else 1
        # same bucketing as stepcount.message_bucket; not imported from there because
        # src.stepcount loads the real environment, which must stay out of training
        b = min(streak, 3)
        if rng.random() < eps:
            action = int(rng.integers(4))
        else:
            action = int(Q[c, b].argmax())
        latent, quit_, next_context = llm_day_step(client, person, rows, context, ACTIONS[action])
        outcome = emit_outcome(emitter, latent, ACTIONS[action], [r["outcome"] for r in rows], emit_rng)
        rows.append({"context": context, "action": action, "outcome": round(outcome, 3), "quit": quit_})
        seq.append((c, b, action, outcome))
        streak = 0 if action == 0 else streak + 1
        if quit_:
            break
        context = next_context
    transitions = []
    for i, (c, b, a, r) in enumerate(seq):
        if i + 1 < len(seq):
            transitions.append((c, b, a, r, seq[i + 1][0], seq[i + 1][1], False))
        else:
            transitions.append((c, b, a, r, 0, 0, True))
    return transitions


def refit_q(buffer, passes=60, alpha=0.1):
    """Refit the Q-table from scratch on the whole buffer (growing-batch)."""
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


def train_seed(client, seed, budget, episodes_per_round=EPISODES_PER_ROUND):
    start = time.time()
    buffer = []
    Q = np.zeros((2, 4, 4))
    steps = 0
    round_no = 0
    episode_id = 0
    while steps < budget:
        round_no += 1
        eps = max(0.1, 1.0 - steps / budget)
        seeds = [seed * 1_000_000 + episode_id + i for i in range(episodes_per_round)]
        episode_id += episodes_per_round
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for transitions in pool.map(lambda s: collect_episode(client, Q, eps, s), seeds):
                buffer += transitions
                steps += len(transitions)
        Q = refit_q(buffer)
        print("  seed %d round %2d | steps %5d/%d | eps %.2f | %.0fs"
              % (seed, round_no, steps, budget, eps, time.time() - start), flush=True)
    return Q, steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=8000, help="transitions per seed")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--out", default=str(ONLINE_Q_DIR / "online_qtables_new.json"))
    parser.add_argument("--smoke", action="store_true", help="one tiny round on one seed")
    args = parser.parse_args()

    if args.smoke:
        args.budget, args.seeds = 30, "0"
        episodes_per_round = 2
    else:
        episodes_per_round = EPISODES_PER_ROUND

    # the whole point: training must never touch the real environment
    assert "StepCountJITAI" not in sys.modules, "real environment imported before training"

    client = build_client()
    per_seed = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        Q, steps = train_seed(client, seed, args.budget, episodes_per_round)
        per_seed.append({"seed": seed, "final_Q": Q.round(5).tolist(), "timesteps": steps})
        with open(args.out, "w") as f:
            json.dump({"per_seed": per_seed}, f)
        print("seed %d done (%d transitions), saved -> %s" % (seed, steps, args.out), flush=True)

    assert "StepCountJITAI" not in sys.modules, "real environment imported during training"
    print("done; evaluate the policies on the real environment with experiments/reference_ladder.py")


if __name__ == "__main__":
    main()
