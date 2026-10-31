"""CPU-only reference ranking audit; LLM scores and candidate policies held fixed."""
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

from src.artifacts import load_corpora, load_online_qtables, load_result
from src.metrics import spearman
from src.paths import REPO_ROOT, RESULTS_DIR
from src.policies import build_policy_zoo
from src.stepcount import CONFIGS, MAX_EPISODE_LENGTH, evaluate_real, make_env


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.05 * (attempt + 1))


def evaluate_episodes(policy, seeds, n_episodes=80, base_seed=7000):
    """Instrument the canonical evaluation loop without changing its frozen source."""
    records = []
    for seed in seeds:
        env = make_env(seed, "paper")
        rng = np.random.default_rng(base_seed + seed)
        for episode in range(n_episodes):
            env.reset()
            streak, total = 0, 0.
            for step in range(MAX_EPISODE_LENGTH):
                obs = env.unpack_obs_array(env.current_state)
                action = int(policy(obs, streak, rng))
                if action not in range(4):
                    raise ValueError("Policy emitted an invalid action")
                streak = 0 if action == 0 else streak + 1
                _, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                if terminated or truncated:
                    break
            if not np.isfinite(total):
                raise ValueError("Nonfinite episode return")
            records.append({"environment_seed": seed, "policy_rng_seed": base_seed + seed,
                            "episode": episode, "return": total, "length": step + 1,
                            "terminated": bool(terminated), "truncated": bool(truncated)})
        env.close()
    return records


def ranking(reference, scores):
    reference, scores = np.asarray(reference), np.asarray(scores)
    top_r, top_s = np.argsort(-reference, kind="stable"), np.argsort(-scores, kind="stable")
    selected, best = int(top_s[0]), int(top_r[0])
    regret = float(reference[best] - reference[selected])
    return {"rho": spearman(reference, scores), "selected_index": selected, "best_index": best,
            "regret": regret, "relative_regret_percent": 100 * regret / reference[best],
            "top5_overlap": len(set(top_r[:5]) & set(top_s[:5])),
            "best_candidate_simulator_rank": int(np.where(top_s == best)[0][0]) + 1}


def summarize(seed_means, scores, names, n_boot=10000, seed=20261025):
    """Resample reference-seed clusters jointly; never resample candidate identities."""
    a = np.asarray(seed_means, dtype=float)
    if a.ndim != 2 or a.shape[0] != len(names) or a.shape[1] < 2:
        raise ValueError("Need policies x at least two reference-seed means")
    rng = np.random.default_rng(seed)
    indices = rng.integers(a.shape[1], size=(n_boot, a.shape[1]))
    boot = a[:, indices].mean(axis=2).T
    means = a.mean(axis=1)
    point = ranking(means, scores)
    metrics = [ranking(row, scores) for row in boot]
    point.update({"selected": names[point["selected_index"]], "best": names[point["best_index"]],
                  "selected_return": float(means[point["selected_index"]]),
                  "best_return": float(means.max())})
    intervals = {key: np.quantile([m[key] for m in metrics], [.025, .975]).tolist()
                 for key in ("rho", "regret", "relative_regret_percent", "top5_overlap")}
    differences = boot - boot[:, [point["selected_index"]]]
    point_differences = means - means[point["selected_index"]]
    # A joint max-absolute-deviation band complements nonsmooth winner bootstrap summaries.
    radius = float(np.quantile(np.max(np.abs(differences - point_differences), axis=1), .95))
    per_policy = []
    for i, name in enumerate(names):
        lo, hi = np.quantile(boot[:, i], [.025, .975])
        per_policy.append({"policy": name, "mean": float(means[i]), "conditional_ci95": [lo, hi],
                           "reference_best_frequency": float(np.mean([m["best_index"] == i for m in metrics])),
                           "gap_vs_llm_selected": float(point_differences[i]),
                           "joint_gap_band95": [float(point_differences[i] - radius),
                                                float(point_differences[i] + radius)]})
    return {"scope": "Reference Monte Carlo uncertainty conditional on saved simulator scores and frozen candidates.",
            "bootstrap_unit": "environment-seed clusters; common resample indices across policies",
            "n_seed_clusters": a.shape[1], "bootstrap_resamples": n_boot, "bootstrap_seed": seed,
            "point": point, "conditional_percentile_intervals95": intervals,
            "joint_gap_band_method": "95% bootstrap max absolute centered deviation; approximate, not exact coverage",
            "regret_joint_band95": [max(0., float(point_differences.max()) - radius),
                                    max(0., float(point_differences.max()) + radius)],
            "policies": per_policy,
            "limitations": ["Simulator rollout noise and policy training randomness are not propagated.",
                            "Only eight clusters in the original schedule; bootstrap coverage is approximate.",
                            "Near ties make winner/rank/overlap distributions nonsmooth."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "ranking_reference_uncertainty.json")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.seeds < 2 or args.episodes < 1 or args.bootstrap < 100:
        parser.error("Need at least two seeds, one episode and 100 bootstrap samples")
    saved = [r for r in load_result("benchmark_robust.json")["rows"]
             if r["policy"] != "observable-ceiling"]
    inputs = [REPO_ROOT / "data/results/benchmark_robust.json",
              REPO_ROOT / "data/online_q/online_qtables.json",
              *sorted((REPO_ROOT / "data/corpora").glob("llmworld_corpus_c[012].json")),
              *[REPO_ROOT / p for p in ("src/policies.py", "src/tabular_q.py", "src/stepcount.py",
                                       "experiments/ranking_reference_uncertainty.py")]]
    import StepCountJITAI.envs as environment
    source_hashes = {str(p.relative_to(REPO_ROOT)): digest(p) for p in inputs}
    source_hashes["external/StepCountJITAI/envs.py"] = digest(Path(environment.__file__))
    settings = {"config": CONFIGS["paper"], "environment_seeds": list(range(args.seeds)),
                "episodes_per_seed": args.episodes, "policy_rng_base": 7000,
                "horizon": MAX_EPISODE_LENGTH, "smoke": args.smoke}
    if args.smoke:
        from src.policies import no_action, random_action
        zoo = {"none": no_action, "random": random_action}
        saved = [r for r in saved if r["policy"] in zoo]
    else:
        print("Reconstructing the original offline candidates from fixed corpora and loading online Q tables.", flush=True)
        zoo = build_policy_zoo(load_corpora(), load_online_qtables())
    if list(zoo) != [r["policy"] for r in saved]:
        raise ValueError("Candidate order/identity differs from saved benchmark")
    if args.out.exists():
        if not args.resume:
            parser.error("Output exists; use --resume or a new output path")
        payload = json.loads(args.out.read_text(encoding="utf-8"))
        if payload["source_sha256"] != source_hashes or payload["settings"] != settings:
            raise ValueError("Cannot resume with changed inputs, code or settings")
    else:
        payload = {"created_utc": datetime.now(timezone.utc).isoformat(), "settings": settings,
                   "source_sha256": source_hashes, "python": platform.python_version(),
                   "numpy": np.__version__, "stage": "evaluation", "rows": []}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        checkpoint(args.out, payload)
    completed = {r["policy"] for r in payload["rows"]}
    for old in saved:
        name = old["policy"]
        if name in completed:
            continue
        records = evaluate_episodes(zoo[name], range(args.seeds), args.episodes)
        mean = float(np.mean([r["return"] for r in records]))
        original_schedule = args.seeds == 8 and args.episodes == 80
        row = {"policy": name, "mean": mean, "saved_rounded_return": old["real_new"],
               "rounded_reproduction": round(mean) == old["real_new"] if original_schedule else None,
               "llm_mean_held_fixed": old["llm_mean"], "episodes": records,
               "environment_seed_means": [float(np.mean([r["return"] for r in records
                                                           if r["environment_seed"] == s]))
                                           for s in range(args.seeds)]}
        payload["rows"].append(row)
        checkpoint(args.out, payload)
        print(f"{len(payload['rows'])}/{len(saved)} {name}: {mean:.6f}; saved={old['real_new']}; reproduce={row['rounded_reproduction']}", flush=True)
        if original_schedule and not row["rounded_reproduction"]:
            raise RuntimeError("Historical reference mismatch: saved checkpoint; investigate before continuing")
    payload["analysis"] = summarize([r["environment_seed_means"] for r in payload["rows"]],
                                    [r["llm_mean_held_fixed"] for r in payload["rows"]],
                                    [r["policy"] for r in payload["rows"]], args.bootstrap)
    payload["stage"] = "complete"
    payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
    checkpoint(args.out, payload)
    print(json.dumps(payload["analysis"]["point"], indent=2), flush=True)


if __name__ == "__main__":
    main()
