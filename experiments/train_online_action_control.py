"""Online LLM-world controls with corrupted action labels.

The original online benchmark lets the learner choose an action and sends that
same action label to the LLM environment. This script keeps the learner and Q
update unchanged, but corrupts the action label seen by the LLM:

* neutralized: none stays none; every nonzero message is shown as generic;
* random-label: each chosen action is replaced by a random action label.

The Q-table is still updated under the learner's chosen action. The LLM history
also records the corrupted label, so the simulated patient never sees the true
fine-grained action semantics in the control environment.

Requires NEBULA_API_KEY and the openai/python-dotenv packages. A full run has the
same order of cost as experiments/train_online_llm.py.

Examples:

  python experiments/train_online_action_control.py --smoke
  python experiments/train_online_action_control.py --mode neutralized --budget 8000 --seeds 0,1,2
  python experiments/train_online_action_control.py --mode neutralized --resume
  PYTHONPATH=/path/to/StepCountJITAI python experiments/train_online_action_control.py --evaluate
  PYTHONPATH=/path/to/StepCountJITAI python experiments/train_online_action_control.py --evaluate-only --out data/results/online_action_control_neutralized.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm_client import build_client
from src.llm_world import ACTIONS, EPISODE_LENGTH, emit_outcome, llm_day_step, make_emitter, persona
from src.paths import RESULTS_DIR


GAMMA = 0.9
EPISODES_PER_ROUND = 40
WORKERS = 4


def corrupt_action(action, mode, rng):
    """Return the action label shown to the LLM control environment."""
    action = int(action)
    if mode == "canonical":
        return action
    if mode == "neutralized":
        return 0 if action == 0 else 1
    if mode == "random-label":
        return int(rng.integers(4))
    raise ValueError(f"unknown action-label mode: {mode}")


def collect_episode(client, Q, eps, seed, mode):
    rng = np.random.default_rng(seed)
    label_rng = np.random.default_rng(seed + 31)
    emit_rng = random.Random(seed + 7)
    emitter = make_emitter()
    person = persona(rng)
    context = "A" if rng.random() < 0.5 else "B"
    streak = 0
    rows = []
    seq = []
    for _ in range(EPISODE_LENGTH):
        c = 0 if context == "A" else 1
        b = min(streak, 3)
        if rng.random() < eps:
            action = int(rng.integers(4))
        else:
            action = int(Q[c, b].argmax())

        shown_action = corrupt_action(action, mode, label_rng)
        latent, quit_, next_context = llm_day_step(
            client, person, rows, context, ACTIONS[shown_action]
        )
        outcome = emit_outcome(
            emitter, latent, ACTIONS[shown_action], [r["outcome"] for r in rows], emit_rng
        )
        rows.append({
            "context": context,
            "action": shown_action,
            "chosen_action": action,
            "outcome": round(outcome, 3),
            "quit": quit_,
        })
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


def _serializable_buffer(buffer):
    return [
        [int(c), int(b), int(a), float(r), int(next_c), int(next_b), bool(done)]
        for c, b, a, r, next_c, next_b, done in buffer
    ]


def _restore_buffer(rows):
    return [
        (int(c), int(b), int(a), float(r), int(next_c), int(next_b), bool(done))
        for c, b, a, r, next_c, next_b, done in rows
    ]


def train_seed(client, seed, budget, mode, episodes_per_round=EPISODES_PER_ROUND,
               state=None, checkpoint=None):
    start = time.time()
    if state:
        buffer = _restore_buffer(state.get("buffer", []))
        Q = np.array(state.get("current_Q", np.zeros((2, 4, 4))), dtype=float)
        steps = int(state.get("timesteps", len(buffer)))
        round_no = int(state.get("round_no", 0))
        episode_id = int(state.get("episode_id", 0))
    else:
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
            for transitions in pool.map(lambda s: collect_episode(client, Q, eps, s, mode), seeds):
                buffer += transitions
                steps += len(transitions)
        Q = refit_q(buffer)
        print("  %s seed %d round %2d | steps %5d/%d | eps %.2f | %.0fs"
              % (mode, seed, round_no, steps, budget, eps, time.time() - start), flush=True)
        if checkpoint:
            checkpoint({
                "seed": seed,
                "status": "partial",
                "current_Q": Q.round(5).tolist(),
                "timesteps": steps,
                "round_no": round_no,
                "episode_id": episode_id,
                "buffer": _serializable_buffer(buffer),
            })
    return Q, steps


def evaluate_qtables(qtables, config, n_episodes):
    from src.stepcount import evaluate_real

    def message_bucket(streak):
        return min(streak, 3)

    values = []
    for Q in qtables:
        Q = np.asarray(Q)

        def policy(obs, streak, rng):
            return int(Q[int(obs["c_infer"]), message_bucket(streak)].argmax())

        values.append(evaluate_real(policy, config, n_episodes=n_episodes))
    return [round(float(np.mean(values))), round(float(np.std(values)))], [round(float(v), 3) for v in values]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="neutralized",
                        choices=["neutralized", "random-label", "canonical"])
    parser.add_argument("--budget", type=int, default=8000, help="transitions per seed")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--out", default=None)
    parser.add_argument("--evaluate", action="store_true",
                        help="evaluate trained Q-tables on StepCountJITAI after training")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="evaluate existing Q-tables in --out without making LLM calls")
    parser.add_argument("--resume", action="store_true",
                        help="skip seeds already present in --out")
    parser.add_argument("--config", default="paper", choices=["paper", "clean"])
    parser.add_argument("--n-episodes", type=int, default=80)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.budget, args.seeds = 30, "0"
        episodes_per_round = 2
        args.n_episodes = 2
    else:
        episodes_per_round = EPISODES_PER_ROUND

    if args.out is None:
        args.out = str(RESULTS_DIR / f"online_action_control_{args.mode}.json")

    existing = None
    per_seed = []
    output_path = Path(args.out)
    if output_path.exists():
        with output_path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        per_seed = list(existing.get("per_seed", []))

    if args.evaluate_only:
        done_entries = [
            entry for entry in per_seed
            if entry.get("status", "done") == "done" and "final_Q" in entry
        ]
        if not done_entries:
            raise RuntimeError(f"no per_seed Q-tables found in {args.out}")
        qtables = [np.array(entry["final_Q"]) for entry in done_entries]
        summary, values = evaluate_qtables(qtables, args.config, args.n_episodes)
        payload = existing or {}
        payload["evaluation"] = {
            "config": args.config,
            "n_episodes": args.n_episodes,
            "online_q": summary,
            "per_seed_return": values,
        }
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print("%s online Q on %s: %d +/- %d" % (args.mode, args.config, summary[0], summary[1]))
        return

    assert "StepCountJITAI" not in sys.modules, "real environment imported before training"
    if "NEBULA_API_KEY" not in os.environ:
        raise RuntimeError("NEBULA_API_KEY is required for online LLM controls")

    client = build_client()
    def write_payload(entries):
        payload = {
            "control": "online_action_label_control",
            "mode": args.mode,
            "definition": (
                "Q-learning action is unchanged, but the action label shown to the "
                "LLM and recorded in its history is corrupted according to mode"
            ),
            "budget": args.budget,
            "per_seed": entries,
        }
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def replace_seed_entry(entry):
        nonlocal per_seed
        per_seed = [old for old in per_seed if int(old["seed"]) != int(entry["seed"])]
        per_seed.append(entry)
        per_seed.sort(key=lambda item: int(item["seed"]))
        write_payload(per_seed)

    completed = {
        int(entry["seed"]) for entry in per_seed
        if entry.get("status", "done") == "done" and "final_Q" in entry
    }
    for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
        if args.resume and seed in completed:
            print("seed %d already present in %s; skipping" % (seed, args.out), flush=True)
            continue
        seed_state = None
        if args.resume:
            seed_state = next(
                (entry for entry in per_seed
                 if int(entry["seed"]) == seed and entry.get("status") == "partial"),
                None,
            )
            if seed_state:
                print("resuming seed %d from %d transitions in %s"
                      % (seed, int(seed_state.get("timesteps", 0)), args.out), flush=True)

        def checkpoint(entry):
            replace_seed_entry(entry)

        Q, steps = train_seed(
            client, seed, args.budget, args.mode, episodes_per_round,
            state=seed_state, checkpoint=checkpoint,
        )
        replace_seed_entry({
            "seed": seed,
            "status": "done",
            "final_Q": Q.round(5).tolist(),
            "timesteps": steps,
        })
        print("seed %d done (%d transitions), saved -> %s" % (seed, steps, args.out), flush=True)

    assert "StepCountJITAI" not in sys.modules, "real environment imported during training"

    if args.evaluate:
        done_entries = [
            entry for entry in per_seed
            if entry.get("status", "done") == "done" and "final_Q" in entry
        ]
        qtables = [np.array(entry["final_Q"]) for entry in done_entries]
        summary, values = evaluate_qtables(qtables, args.config, args.n_episodes)
        with open(args.out, encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["evaluation"] = {
            "config": args.config,
            "n_episodes": args.n_episodes,
            "online_q": summary,
            "per_seed_return": values,
        }
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print("%s online Q on %s: %d +/- %d" % (args.mode, args.config, summary[0], summary[1]))


if __name__ == "__main__":
    main()
