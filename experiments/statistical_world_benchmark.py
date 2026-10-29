"""Train and evaluate a declared action-conditioned statistical simulator.

No LLM calls. All selected variants are trained and saved before reference
evaluation. See docs/action_conditioned_statistical_protocol.md for assumptions.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import ranks, spearman, tertile_spearman
from src.paths import REPO_ROOT, RESULTS_DIR
from src.statistical_world import PRESETS, collect_episode, rollout_policy
from src.synthetic_q import EPISODES_PER_ROUND, GAMMA, Q_ALPHA, Q_REFIT_PASSES, refit_q


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_payload(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # Cloud-sync clients may briefly lock the destination during replacement.
    for attempt in range(8):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.05 * (attempt + 1))


def assert_training_isolated():
    if any(name == "StepCountJITAI" or name.startswith("StepCountJITAI.") for name in sys.modules):
        raise RuntimeError("reference environment must not be imported during synthetic training")


def train_seed(seed, budget, config, episodes_per_round=EPISODES_PER_ROUND):
    assert_training_isolated()
    if seed < 0 or budget < 1 or episodes_per_round < 1:
        raise ValueError("seed must be nonnegative and budget/round size positive")
    Q = np.zeros((2, 4, 4))
    buffer, rounds = [], []
    episode_id = 0
    while len(buffer) < budget:
        before = len(buffer)
        eps = max(0.1, 1.0 - before / budget)
        # Match the LLM runner: complete the whole round before testing the budget.
        for _ in range(episodes_per_round):
            buffer.extend(collect_episode(Q, eps, seed * 1_000_000 + episode_id, config))
            episode_id += 1
        Q = refit_q(buffer)
        rounds.append({"round": len(rounds) + 1, "epsilon": eps, "timesteps": len(buffer)})
    assert_training_isolated()
    return {
        "seed": seed, "status": "done", "timesteps": len(buffer),
        "episodes": episode_id, "rounds": rounds, "final_Q": Q.tolist(),
        "transitions_sha256": json_hash(buffer),
    }


def greedy_policy(Q):
    Q = np.asarray(Q, dtype=float)

    def policy(obs, streak, rng):
        return int(Q[int(obs["c_infer"]), min(streak, 3)].argmax())

    return policy


def reference_summary(policy, config, seeds, n_episodes):
    from src.stepcount import evaluate_real

    values = [evaluate_real(policy, config, seeds=[seed], n_episodes=n_episodes) for seed in seeds]
    return {"mean": float(np.mean(values)), "environment_seed_means": values}


def evaluate_transfer(payload, args):
    # These imports are intentionally deferred until every synthetic fit is saved.
    from src.artifacts import load_online_qtables
    from src.policies import matched_heuristic, no_action, random_action
    from src.stepcount import CONFIGS
    from StepCountJITAI import envs as reference_module

    settings = {
        "config": args.config, "parameters": CONFIGS[args.config],
        "environment_seeds": args.reference_seeds, "episodes_per_environment_seed": args.n_episodes,
        "policy_rng_base_seed": 7000, "max_episode_length": 50,
        "reference_wrapper_sha256": file_hash(REPO_ROOT / "src/stepcount.py"),
        "reference_implementation_sha256": file_hash(Path(reference_module.__file__)),
        "llm_qtables_sha256": file_hash(REPO_ROOT / "data/online_q/online_qtables.json"),
    }
    evaluation = payload.setdefault("evaluation", {"settings": settings, "policies": {}})
    if evaluation["settings"] != settings:
        raise ValueError("saved reference evaluation settings differ; use a separate output artifact")
    policies = {"no_message": no_action, "random": random_action, "matched_heuristic": matched_heuristic}
    for seed, Q in load_online_qtables().items():
        policies["llm_online_seed_%d" % seed] = greedy_policy(Q)
    for variant, entries in payload["training"].items():
        for entry in entries:
            policies["statistical_%s_seed_%d" % (variant, entry["seed"])] = greedy_policy(entry["final_Q"])
    for name, policy in policies.items():
        if name not in evaluation["policies"]:
            evaluation["policies"][name] = reference_summary(
                policy, args.config, args.reference_seeds, args.n_episodes,
            )
            write_payload(args.out, payload)
        print("reference %-34s %.3f" % (name, evaluation["policies"][name]["mean"]), flush=True)
    summary = {}
    for group in ["llm_online"] + ["statistical_" + v for v in payload["training"]]:
        values = [r["mean"] for name, r in evaluation["policies"].items() if name.startswith(group + "_seed_")]
        summary[group] = {"n_training_seeds": len(values), "mean": float(np.mean(values)),
                          "sd_across_training_seeds": float(np.std(values)), "per_training_seed": values}
    evaluation["summary"] = summary
    write_payload(args.out, payload)


def ranking_metrics(rows):
    real = [r["reference_return"] for r in rows]
    scores = [r["statistical_mean"] for r in rows]
    order_r = sorted(range(len(rows)), key=lambda i: -real[i])
    order_s = sorted(range(len(rows)), key=lambda i: -scores[i])
    best, selected = order_r[0], order_s[0]
    k = min(5, len(rows))
    regret = real[best] - real[selected]
    return {
        "n_policies": len(rows), "spearman": spearman(real, scores),
        "tertile_spearman_top_middle_bottom": list(tertile_spearman(real, scores)),
        "selected_policy": rows[selected]["policy"], "selected_reference_return": real[selected],
        "best_reference_policy": rows[best]["policy"], "best_reference_return": real[best],
        "best_reference_simulator_rank": ranks(scores)[best],
        "selection_regret": regret, "relative_regret_percent": 100 * regret / real[best],
        "top5_overlap_count": len(set(order_r[:k]) & set(order_s[:k])),
    }


def evaluate_ranking(payload, args):
    from src.artifacts import load_corpora, load_online_qtables, load_result
    from src.policies import build_policy_zoo

    saved = [r for r in load_result("benchmark_robust.json")["rows"] if r["policy"] != "observable-ceiling"]
    zoo = build_policy_zoo(load_corpora(), load_online_qtables())
    if set(zoo) != {r["policy"] for r in saved}:
        raise ValueError("reconstructed policy labels do not match the saved reference benchmark")
    settings = {
        "batches": args.batches, "patients_per_batch": args.patients,
        "base_seed": 90000, "reference_source": "data/results/benchmark_robust.json",
        "reference_sha256": file_hash(RESULTS_DIR / "benchmark_robust.json"),
        "reference_values": "saved rounded reference returns; held fixed for ranking metrics",
        "tie_breaking": "canonical policy order for exact score ties",
        "input_sha256": {str(p.relative_to(REPO_ROOT)).replace("\\", "/"): file_hash(p) for p in [
            REPO_ROOT / "src/policies.py", REPO_ROOT / "src/tabular_q.py",
            *sorted((REPO_ROOT / "data/corpora").glob("llmworld_corpus_c[012].json")),
            REPO_ROOT / "data/online_q/online_qtables.json",
        ]},
    }
    ranking = payload.setdefault("ranking", {"settings": settings, "variants": {}})
    if ranking["settings"] != settings:
        raise ValueError("saved ranking settings/inputs differ; use a separate output artifact")
    for variant in payload["protocol"]["variants"]:
        output = ranking["variants"].setdefault(variant, {"rows": []})
        completed = {r["policy"] for r in output["rows"]}
        for row in saved:
            name = row["policy"]
            if name in completed:
                continue
            means = []
            for batch in range(args.batches):
                means.append(float(np.mean([
                    rollout_policy(zoo[name], 90000 + batch * 1000 + i, PRESETS[variant])
                    for i in range(args.patients)
                ])))
            output["rows"].append({
                "policy": name, "reference_return": float(row["real_new"]),
                "statistical_mean": float(np.mean(means)), "batch_means": means,
            })
            write_payload(args.out, payload)
        output["pooled"] = ranking_metrics(output["rows"])
        output["per_batch"] = [ranking_metrics([
            {**r, "statistical_mean": r["batch_means"][b]} for r in output["rows"]
        ]) for b in range(args.batches)]
        print("ranking %s: %s" % (variant, json.dumps(output["pooled"])), flush=True)
        write_payload(args.out, payload)


def parse_seeds(value):
    seeds = [int(v.strip()) for v in value.split(",") if v.strip()]
    if not seeds or any(s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("provide distinct nonnegative seeds separated by commas")
    return seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=list(PRESETS), default=["weak", "default", "strong"])
    parser.add_argument("--budget", type=int, default=8000)
    parser.add_argument("--seeds", type=parse_seeds, default=[0, 1, 2])
    parser.add_argument("--episodes-per-round", type=int, default=EPISODES_PER_ROUND)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "action_conditioned_statistical.json")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--rank", action="store_true", help="score the canonical 49 policies in each statistical world")
    parser.add_argument("--config", choices=["paper", "clean"], default="paper")
    parser.add_argument("--reference-seeds", type=parse_seeds, default=list(range(8)))
    parser.add_argument("--n-episodes", type=int, default=80)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--patients", type=int, default=40)
    existing = parser.add_mutually_exclusive_group()
    existing.add_argument("--evaluate-only", action="store_true")
    existing.add_argument("--resume", action="store_true")
    existing.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.budget, args.seeds, args.episodes_per_round = 80, [0], 2
        args.reference_seeds, args.n_episodes = [0, 1], 2
        args.batches, args.patients = 1, 3
        if args.out == RESULTS_DIR / "action_conditioned_statistical.json":
            args.out = RESULTS_DIR / "action_conditioned_statistical_smoke.json"
    if min(args.budget, args.episodes_per_round, args.n_episodes, args.batches, args.patients) < 1:
        parser.error("budgets, round size, episodes, batches and patients must be positive")
    if len(set(args.variants)) != len(args.variants):
        parser.error("variant labels must be unique")
    if args.rank and args.config != "paper":
        parser.error("saved canonical ranking reference scores use the paper configuration")
    if args.evaluate_only and not args.out.exists():
        parser.error("--evaluate-only requires an existing training artifact")
    started = time.time()
    source_names = [
        "src/statistical_world.py", "src/synthetic_patient.py", "src/synthetic_q.py", "src/metrics.py",
        "simulator/adherence_simulator/outcome_models.py", "experiments/statistical_world_benchmark.py",
        "docs/action_conditioned_statistical_protocol.md",
    ]
    protocol = {
        "version": 1, "primary_variant": "default" if "default" in args.variants else None,
        "variants": {v: PRESETS[v].to_dict() for v in args.variants},
        "training_seeds": args.seeds, "budget": args.budget, "episodes_per_round": args.episodes_per_round,
        "budget_stopping_rule": "complete full rounds until transitions >= budget",
        "epsilon": "max(0.1, 1 - transitions_at_round_start / budget)",
        "gamma": GAMMA, "q_refit_passes": Q_REFIT_PASSES, "q_alpha": Q_ALPHA, "q_shuffle_seed": 0,
        "parameter_selection": "fixed qualitative-prior baseline; no reference-return or LLM-data fitting",
        "source_sha256": {name: file_hash(REPO_ROOT / name) for name in source_names},
    }
    if args.out.exists() and not (args.resume or args.evaluate_only or args.force):
        parser.error("output exists; use --resume, --evaluate-only, --force, or another --out")
    if args.evaluate_only or (args.resume and args.out.exists()):
        payload = json.loads(args.out.read_text(encoding="utf-8"))
        if payload.get("protocol_sha256") != json_hash(protocol):
            parser.error("protocol/source differs from saved run; use matching arguments and source or a new --out")
    else:
        payload = {
            "experiment": "action_conditioned_statistical", "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "software": {"python": platform.python_version(), "numpy": np.__version__},
            "protocol": protocol, "protocol_sha256": json_hash(protocol),
            "stage": "training", "training": {v: [] for v in args.variants},
        }
        # The protocol snapshot is written before training or reference imports.
        write_payload(args.out.with_suffix(".protocol.json"), {
            "created_utc": payload["created_utc"], "protocol": protocol, "protocol_sha256": json_hash(protocol),
        })
        write_payload(args.out, payload)
    if not args.evaluate_only:
        for variant in args.variants:
            done = {e["seed"] for e in payload["training"][variant] if e.get("status") == "done"}
            for seed in args.seeds:
                if seed in done:
                    continue
                entry = train_seed(seed, args.budget, PRESETS[variant], args.episodes_per_round)
                payload["training"][variant].append(entry)
                print("trained %s seed %d: %d transitions" % (variant, seed, entry["timesteps"]), flush=True)
                write_payload(args.out, payload)
        payload["stage"] = "trained"
        write_payload(args.out, payload)
    for variant in args.variants:
        if {e["seed"] for e in payload["training"][variant] if e.get("status") == "done"} != set(args.seeds):
            parser.error("all requested training seeds must be complete before reference evaluation")
    if args.evaluate or args.evaluate_only:
        evaluate_transfer(payload, args)
    if args.rank:
        evaluate_ranking(payload, args)
    payload["stage"] = "complete" if (args.evaluate or args.evaluate_only or args.rank) else "trained"
    payload["last_invocation_seconds"] = round(time.time() - started, 3)
    write_payload(args.out, payload)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
