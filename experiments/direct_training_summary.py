"""Correct mean-versus-best and SD reporting from saved runs; no new training."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import REPO_ROOT, RESULTS_DIR


def summary(values):
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError("Need at least two finite saved returns")
    return {"values": a.tolist(), "n": len(a), "mean": float(a.mean()),
            "population_sd": float(a.std(ddof=0)), "sample_sd": float(a.std(ddof=1)),
            "maximum": float(a.max())}


def analyze():
    hashes = {}
    def load(relative):
        path = REPO_ROOT / relative
        raw = path.read_bytes() if path.exists() else gzip.decompress(path.with_suffix(path.suffix+'.gz').read_bytes())
        hashes[relative] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    ref = load("data/results/reference_ladder.json")["paper"]
    direct = summary(ref["observable_per_seed"])
    mini_paths = ("online_qtables_openai_gpt5mini_minimal_seed0_full8000.json",
                  "online_qtables_openai_gpt5mini_minimal_seeds1_2_full8000.json")
    mini = [value for name in mini_paths for value in load("data/online_q/"+name)["evaluation"]["per_seed_return"]]
    return {"scope": "Saved-run reporting audit only; no new policy training or reference/LLM calls.",
            "direct_observable_training": direct,
            "qwen_saved_rounded_summary": {"mean": ref["online_q"][0], "population_sd": ref["online_q"][1]},
            "old_mean_to_best_ratio": ref["online_q"][0] / direct["maximum"],
            "mean_to_mean_ratio_descriptive_only": ref["online_q"][0] / direct["mean"],
            "gpt5mini": summary(mini), "source_sha256": hashes,
            "limits": ["Direct-training returns are saved rounded estimates; historical seed identities are absent.",
                       "Direct training uses 25000 episodes and different updates; simulator training targets 8000 transitions.",
                       "No matched-budget superiority, equivalence or proven-ceiling claim follows.",
                       "Population SD is reported consistently; it is not a confidence interval."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "direct_training_summary.json")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output exists; use a new --out")
    result = analyze()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name in ("direct_observable_training", "gpt5mini"):
        print(name, result[name]["mean"], result[name]["population_sd"])


if __name__ == "__main__":
    main()
