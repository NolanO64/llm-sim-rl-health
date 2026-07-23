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
from src.paths import ONLINE_Q_DIR

GAMMA = 0.9
EPISODES_PER_ROUND = 40
WORKERS = 4  # concurrent patients; stays under the gateway's request limit
Q_REFIT_PASSES = 60
Q_ALPHA = 0.1


def collect_episode(client, Q, eps, seed, model, backend, temperature, max_tokens):
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
        latent, quit_, next_context = llm_day_step(
            client, person, rows, context, ACTIONS[action], model=model,
            backend=backend, temperature=temperature, max_tokens=max_tokens
        )
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


def refit_q(buffer, passes=Q_REFIT_PASSES, alpha=Q_ALPHA):
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


def train_seed(client, seed, budget, model, backend, temperature, max_tokens,
               episodes_per_round=EPISODES_PER_ROUND, state=None, checkpoint=None):
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
            for transitions in pool.map(
                lambda s: collect_episode(
                    client, Q, eps, s, model, backend, temperature, max_tokens
                ),
                seeds
            ):
                buffer += transitions
                steps += len(transitions)
        Q = refit_q(buffer)
        print("  seed %d round %2d | steps %5d/%d | eps %.2f | %.0fs"
              % (seed, round_no, steps, budget, eps, time.time() - start), flush=True)
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

    values = []
    for Q in qtables:
        Q = np.asarray(Q)

        def policy(obs, streak, rng):
            return int(Q[int(obs["c_infer"]), min(streak, 3)].argmax())

        values.append(evaluate_real(policy, config, n_episodes=n_episodes))
    return [round(float(np.mean(values))), round(float(np.std(values)))], [round(float(v), 3) for v in values]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=8000, help="transitions per seed")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--out", default=str(ONLINE_Q_DIR / "online_qtables_new.json"))
    parser.add_argument("--backend", default="nebula", choices=["nebula", "openai"])
    parser.add_argument("--model", default=None, help="LLM deployment name for the online world")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--evaluate", action="store_true",
                        help="evaluate trained Q-tables on StepCountJITAI after training")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="evaluate existing Q-tables in --out without making LLM calls")
    parser.add_argument("--resume", action="store_true",
                        help="skip completed seeds and resume partial checkpoints in --out")
    parser.add_argument("--config", default="paper", choices=["paper", "clean"])
    parser.add_argument("--n-episodes", type=int, default=80)
    parser.add_argument("--smoke", action="store_true", help="one tiny round on one seed")
    args = parser.parse_args()
    if args.model is None:
        if args.backend == "openai":
            args.model = "gpt-5-mini"
        else:
            from src.llm_world import MODEL
            args.model = MODEL

    if args.smoke:
        args.budget, args.seeds = 30, "0"
        episodes_per_round = 2
        args.n_episodes = 2
    else:
        episodes_per_round = EPISODES_PER_ROUND

    output_path = Path(args.out)
    existing = None
    per_seed = []
    if output_path.exists():
        with output_path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        per_seed = list(existing.get("per_seed", []))

    if args.evaluate_only:
        payload = existing or {}
        done_entries = [
            entry for entry in per_seed
            if entry.get("status", "done") == "done" and "final_Q" in entry
        ]
        qtables = [np.array(entry["final_Q"]) for entry in done_entries]
        if not qtables:
            raise RuntimeError(f"no per_seed Q-tables found in {args.out}")
        summary, values = evaluate_qtables(qtables, args.config, args.n_episodes)
        payload["evaluation"] = {
            "config": args.config,
            "n_episodes": args.n_episodes,
            "online_q": summary,
            "per_seed_return": values,
        }
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print("online Q on %s: %d +/- %d" % (args.config, summary[0], summary[1]))
        return

    # the whole point: training must never touch the real environment
    assert "StepCountJITAI" not in sys.modules, "real environment imported before training"
    required_key = "OPENAI_API_KEY" if args.backend == "openai" else "NEBULA_API_KEY"
    if required_key not in os.environ:
        raise RuntimeError(f"{required_key} is required for online LLM training")

    client = build_client(args.backend)

    def write_payload(entries):
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({
                "backend": args.backend,
                "model": args.model,
                "budget": args.budget,
                "budget_stopping_rule": "continue full episodes until timesteps >= budget",
                "gamma": GAMMA,
                "episodes_per_round": episodes_per_round,
                "workers": WORKERS,
                "q_refit_passes": Q_REFIT_PASSES,
                "q_alpha": Q_ALPHA,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "strict_json_validation": True,
                "per_seed": entries,
            }, f, indent=2)

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
            client, seed, args.budget, args.model, args.backend,
            args.temperature, args.max_tokens, episodes_per_round,
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
        summary, values = evaluate_qtables(
            [
                np.array(entry["final_Q"]) for entry in per_seed
                if entry.get("status", "done") == "done" and "final_Q" in entry
            ],
            args.config,
            args.n_episodes,
        )
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
        print("online Q on %s: %d +/- %d" % (args.config, summary[0], summary[1]))
    print("done; evaluate the policies on the real environment with experiments/reference_ladder.py")


if __name__ == "__main__":
    main()
