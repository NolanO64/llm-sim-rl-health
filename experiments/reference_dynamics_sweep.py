"""Frozen-policy reference sensitivity; no LLM calls and no policy retraining."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.ranking_reference_uncertainty import checkpoint, digest, summarize
from src.artifacts import load_corpora, load_online_qtables, load_result
from src.paths import REPO_ROOT, RESULTS_DIR
from src.policies import build_policy_zoo
from src.stepcount import CONFIGS, MAX_EPISODE_LENGTH, OBSERVATIONS
from src.tabular_q import greedy_policy
from StepCountJITAI import envs as reference_module


MINI_FILES = (
    "online_qtables_openai_gpt5mini_minimal_seed0_full8000.json",
    "online_qtables_openai_gpt5mini_minimal_seeds1_2_full8000.json",
)
_WORKER_POLICIES = None


def settings_grid():
    base = {**CONFIGS["paper"], "effect_multiplier": 1.0,
            "habituation_increment": .05, "habituation_recovery": .1}
    settings = {"baseline": dict(base)}
    for parameter, label in (("effect_multiplier", "effect"),
                             ("habituation_increment", "habituation_growth"),
                             ("habituation_recovery", "habituation_recovery")):
        for multiplier, suffix in ((.5, "half"), (2.0, "double")):
            settings[f"{label}_{suffix}"] = {**base, parameter: base[parameter] * multiplier}
    return settings


def make_reference(seed, settings):
    return reference_module.StepCountJITAI(
        sigma=settings["sigma"], chosen_obs_names=OBSERVATIONS,
        n_version=1, b_using_uniform=True, chosen_a=settings["chosen_a"],
        seed=seed, max_episode_length=MAX_EPISODE_LENGTH, b_display=False,
        **{"\u03c3s": settings["sigma_s"],
           "\u03c11": 50.0 * settings["effect_multiplier"],
           "\u03c12": 200.0 * settings["effect_multiplier"], "\u03c13": 0.0,
           "\u03b5h": settings["habituation_increment"],
           "\u03b4h": settings["habituation_recovery"]},
    )


def evaluate(policy, settings, seeds, episodes):
    seed_rows = []
    for seed in seeds:
        env = make_reference(seed, settings)
        rng = np.random.default_rng(7000 + seed)
        returns, lengths, ended = [], [], 0
        try:
            for _ in range(episodes):
                env.reset()
                streak, total = 0, 0.0
                for step in range(MAX_EPISODE_LENGTH):
                    action = int(policy(env.unpack_obs_array(env.current_state), streak, rng))
                    if action not in range(4):
                        raise ValueError("Invalid policy action")
                    streak = 0 if action == 0 else streak + 1
                    _, reward, terminated, truncated, _ = env.step(action)
                    total += float(reward)
                    if terminated or truncated:
                        ended += 1
                        break
                if not np.isfinite(total):
                    raise ValueError("Nonfinite return")
                returns.append(total)
                lengths.append(step + 1)
        finally:
            env.close()
        seed_rows.append({"seed": seed, "policy_rng_seed": 7000 + seed,
                          "returns": returns, "lengths": lengths,
                          "environment_ended_episodes": ended})
    means = [float(np.mean(row["returns"])) for row in seed_rows]
    return {"mean": float(np.mean(means)), "seed_means": means, "seed_rows": seed_rows}


def cached_baseline(cache, saved, source_hashes, seeds, episodes):
    source_hashes = {name.replace("\\", "/"): value for name, value in source_hashes.items()}
    if cache["stage"] != "complete" or cache["settings"]["environment_seeds"] != seeds:
        raise ValueError("Baseline cache has wrong seed schedule/stage")
    if cache["settings"]["episodes_per_seed"] != episodes or cache["settings"]["config"] != CONFIGS["paper"]:
        raise ValueError("Baseline cache has wrong episodes/config")
    if cache["settings"]["policy_rng_base"] != 7000 or cache["settings"]["horizon"] != MAX_EPISODE_LENGTH:
        raise ValueError("Baseline cache has wrong policy RNG/horizon")
    for name, old_hash in cache["source_sha256"].items():
        name = name.replace("\\", "/")
        if name not in source_hashes or source_hashes[name] != old_hash:
            raise ValueError(f"Changed baseline input: {name}")
    if [r["policy"] for r in cache["rows"]] != [r["policy"] for r in saved]:
        raise ValueError("Baseline candidate identity/order mismatch")
    rows = []
    for old, candidate in zip(cache["rows"], saved):
        if old["llm_mean_held_fixed"] != candidate["llm_mean"]:
            raise ValueError("Changed simulator score")
        seed_rows = []
        for seed in seeds:
            records = [e for e in old["episodes"] if e["environment_seed"] == seed]
            if len(records) != episodes or [e["episode"] for e in records] != list(range(episodes)):
                raise ValueError("Incomplete baseline episode sequence")
            seed_rows.append({"seed": seed, "policy_rng_seed": 7000 + seed,
                              "returns": [e["return"] for e in records],
                              "lengths": [e["length"] for e in records],
                              "environment_ended_episodes": sum(e["terminated"] or e["truncated"] for e in records)})
        means = [float(np.mean(e["returns"])) for e in seed_rows]
        if not np.allclose(means, old["environment_seed_means"], rtol=0, atol=1e-12):
            raise ValueError("Baseline episode/summary mismatch")
        rows.append({"policy": old["policy"], "mean": float(np.mean(means)),
                     "seed_means": means, "seed_rows": seed_rows, "origin": "verified_baseline_cache"})
    return rows


def policy_summary(rows, names):
    values = [next(r["mean"] for r in rows if r["policy"] == name) for name in names]
    return {"mean": float(np.mean(values)), "sd_training_seeds": float(np.std(values)),
            "per_training_seed": values, "n_training_seeds": len(values)}


def load_mini(paths):
    policies = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("model") != "gpt-5-mini" or data.get("budget") != 8000:
            raise ValueError("Unexpected mini artifact model or budget")
        for row in data["per_seed"]:
            name = f"gpt5mini s{row['seed']}"
            q = np.asarray(row["final_Q"], dtype=float)
            if name in policies or q.shape != (2, 4, 4) or not np.isfinite(q).all() or row["timesteps"] < 8000:
                raise ValueError("Invalid/duplicate/incomplete mini policy")
            policies[name] = greedy_policy(q)
    if set(policies) != {f"gpt5mini s{s}" for s in range(3)}:
        raise ValueError("Need all three mini training seeds")
    return policies


def initialize_worker(smoke):
    global _WORKER_POLICIES
    if smoke:
        from src.policies import no_action, random_action, matched_heuristic
        _WORKER_POLICIES = {"none": no_action, "random": random_action, "heuristic": matched_heuristic}
    else:
        _WORKER_POLICIES = build_policy_zoo(load_corpora(), load_online_qtables())
        _WORKER_POLICIES.update(load_mini([REPO_ROOT / "data/online_q" / name for name in MINI_FILES]))


def worker_evaluate(task):
    label, name, config, seeds, episodes = task
    return label, {"policy": name, **evaluate(_WORKER_POLICIES[name], config, seeds, episodes),
                   "origin": "new_frozen_policy_evaluation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "reference_dynamics_sweep.json")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--baseline-cache", type=Path,
                        default=RESULTS_DIR / "ranking_reference_uncertainty.json",
                        help="verified eight-seed cache from this source/dependency checkout")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.seeds, args.episodes, args.bootstrap = 2, 2, 100
        if args.out == RESULTS_DIR / "reference_dynamics_sweep.json":
            args.out = RESULTS_DIR / "reference_dynamics_sweep_smoke.json"
    if args.seeds < 2 or args.episodes < 1 or args.bootstrap < 100:
        parser.error("Need >=2 seeds, >=1 episode and >=100 bootstrap draws")
    if args.workers not in range(1, 5):
        parser.error("Use between one and four CPU workers")
    seeds = list(range(args.seeds))
    saved = [r for r in load_result("benchmark_robust.json")["rows"] if r["policy"] != "observable-ceiling"]
    mini_paths = [REPO_ROOT / "data/online_q" / name for name in MINI_FILES]
    cache_path = args.baseline_cache.resolve()
    inputs = [REPO_ROOT / "data/results/benchmark_robust.json", cache_path,
              REPO_ROOT / "data/online_q/online_qtables.json", *mini_paths,
              *sorted((REPO_ROOT / "data/corpora").glob("llmworld_corpus_c[012].json")),
              *[REPO_ROOT / p for p in ("src/policies.py", "src/tabular_q.py", "src/stepcount.py",
                                       "src/metrics.py", "src/artifacts.py",
                                       "experiments/ranking_reference_uncertainty.py",
                                       "experiments/reference_dynamics_sweep.py",
                                       "docs/reference_dynamics_sweep_protocol.md")]]
    hashes = {p.relative_to(REPO_ROOT).as_posix(): digest(p) for p in inputs}
    hashes["external/StepCountJITAI/envs.py"] = digest(Path(reference_module.__file__))
    settings = {"grid": settings_grid(), "seeds": seeds, "episodes_per_seed": args.episodes,
                "horizon": MAX_EPISODE_LENGTH, "policy_rng_base": 7000,
                "bootstrap": args.bootstrap, "smoke": args.smoke, "cpu_workers": args.workers,
                "candidate_names": [r["policy"] for r in saved]}
    if args.out.exists():
        if not args.resume:
            parser.error("Output exists; use --resume or a new output")
        payload = json.loads(args.out.read_text(encoding="utf-8"))
        if payload["settings"] != settings or payload["source_sha256"] != hashes:
            raise ValueError("Changed source/settings; cannot resume")
    else:
        payload = {"stage": "evaluation", "created_utc": datetime.now(timezone.utc).isoformat(),
                   "python": platform.python_version(), "numpy": np.__version__,
                   "settings": settings, "source_sha256": hashes, "variants": {}}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        checkpoint(args.out.with_suffix(".protocol.json"), payload)
        checkpoint(args.out, payload)
    print("Reconstructing fixed candidates; no new online training or API calls.", flush=True)
    if args.smoke:
        from src.policies import no_action, random_action, matched_heuristic
        zoo = {"none": no_action, "random": random_action, "heuristic": matched_heuristic}
        keep = set(zoo)
        saved = [r for r in saved if r["policy"] in keep]
        mini = {}
    else:
        zoo = build_policy_zoo(load_corpora(), load_online_qtables())
        if list(zoo) != settings["candidate_names"]:
            raise ValueError("Candidate identity/order changed")
        mini = load_mini(mini_paths)
    tasks = []
    for label, config in settings["grid"].items():
        output = payload["variants"].setdefault(label, {"rows": []})
        if label == "baseline" and not args.smoke and args.seeds == 8 and args.episodes == 80 and not output["rows"]:
            output["rows"] = cached_baseline(json.loads(cache_path.read_text()), saved, hashes, seeds, args.episodes)
            checkpoint(args.out, payload)
        for name, policy in {**zoo, **mini}.items():
            if any(r["policy"] == name for r in output["rows"]):
                continue
            tasks.append((label, name, config, seeds, args.episodes))
    def save_result(result):
        label, row = result
        output = payload["variants"][label]
        output["rows"].append(row)
        checkpoint(args.out, payload)
        print(f"{label}: {len(output['rows'])}/{len(zoo)+len(mini)} {row['policy']}: {row['mean']:.3f}", flush=True)
    if args.workers == 1:
        initialize_worker(args.smoke)
        for task in tasks:
            save_result(worker_evaluate(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                                 initargs=(args.smoke,)) as executor:
            futures = [executor.submit(worker_evaluate, task) for task in tasks]
            for future in as_completed(futures):
                save_result(future.result())
    for label in settings["grid"]:
        output = payload["variants"][label]
        rows = [next(r for r in output["rows"] if r["policy"] == s["policy"]) for s in saved]
        output["ranking"] = summarize([r["seed_means"] for r in rows],
                                      [r["llm_mean"] for r in saved], [r["policy"] for r in saved], args.bootstrap)
        if not args.smoke:
            output["qwen_transfer"] = policy_summary(output["rows"], [f"online-tabQ s{s}" for s in range(3)])
            output["mini_transfer"] = policy_summary(output["rows"], [f"gpt5mini s{s}" for s in range(3)])
        output["stage"] = "complete"
        checkpoint(args.out, payload)
        print(label, json.dumps(output["ranking"]["point"]), flush=True)
    payload["stage"] = "complete"
    for path in inputs:
        if digest(path) != hashes[path.relative_to(REPO_ROOT).as_posix()]:
            raise RuntimeError("An input changed during evaluation")
    if digest(Path(reference_module.__file__)) != hashes["external/StepCountJITAI/envs.py"]:
        raise RuntimeError("Reference source changed during evaluation")
    payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
    checkpoint(args.out, payload)
    print("Completed", args.out, flush=True)


if __name__ == "__main__":
    main()
