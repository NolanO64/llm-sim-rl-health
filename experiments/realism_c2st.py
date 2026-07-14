"""Distributional realism on HeartSteps (Tables: baselines, models, discrete, emission).

For each committed HeartSteps trajectory file this recomputes:

  * the residual-feature, patient-disjoint five-fold C2ST accuracy (mean +/- sd
    across folds) with a one-sided binomial test against 0.5 -- the realism
    headline (0.5 indistinguishable, 1.0 separable);
  * the between-patient descriptive metrics reported alongside it in the tables:
    the Spearman rank correlation of per-patient mean activity, the heterogeneity
    ratio, and the pooled mean activity (simulated / real).

The bundled files cover the anchor-length baselines, the model comparison, the
prompt-context ablation, the discrete-target reformulation and the emission-layer
factorial. The HPTN cross-dataset table and the emission-parameter sweep are in
realism_hptn.py and emission_sweep.py. CPU only, no LLM calls.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import REALISM_DIR
from src.artifacts import load_result
from src.realism import c2st_cv, descriptive_metrics, extract_patient_series, residual_features

PANELS = {
    "anchor-length baselines (Qwen, prompt-only)": [
        ("heartsteps_baseline_K0", "K=0"),
        ("heartsteps_baseline_K14", "K=14"),
        ("heartsteps_baseline_K35", "K=35"),
        ("heartsteps_baseline_K70", "K=70"),
    ],
    "model comparison (K=35)": [
        ("heartsteps_baseline_K35", "Qwen3.5-122B (n=37)"),
        ("heartsteps_model_qwen12", "Qwen3.5-122B (same 12)"),
        ("heartsteps_model_deepseek", "DeepSeek-V4-pro"),
        ("heartsteps_model_gpt5mini", "GPT-5.4-mini (12)"),
        ("heartsteps_model_gpt5", "GPT-5.4 (12)"),
        ("heartsteps_model_gptoss", "gpt-oss-120b"),
        ("heartsteps_model_gemma3", "gemma3-12b"),
    ],
    "prompt-context ablation (K=35)": [
        ("heartsteps_baseline_K35", "full context"),
        ("heartsteps_context_no_domain", "no domain description"),
        ("heartsteps_context_no_persona", "no persona"),
        ("heartsteps_context_no_weather", "no weather / location"),
        ("heartsteps_context_no_intervention", "no intervention text"),
        ("heartsteps_context_structured", "structured fields only"),
        ("heartsteps_context_history_only", "history only"),
        ("heartsteps_context_first_person", "first-person framing"),
    ],
    "discrete-target reformulation": [
        ("heartsteps_baseline_K35", "continuous (control)"),
        ("heartsteps_discrete_relative5", "relative 5-bin"),
        ("heartsteps_discrete_relative3", "relative 3-bin (12)"),
        ("heartsteps_discrete_binary", "binary activity (12)"),
        ("heartsteps_discrete_zeropos", "zero / positive (12)"),
    ],
    "emission layer (vs prompt-only control)": [
        ("heartsteps_baseline_K35", "prompt-only (control)"),
        ("heartsteps_emission_two_part", "two-part emission"),
        ("heartsteps_emission_stateful", "  + stateful state"),
        ("heartsteps_emission_hint", "  + action-noise hint"),
        ("heartsteps_emission_stateful_hint", "  + stateful + hint"),
    ],
}

def evaluate(stem):
    with open(REALISM_DIR / (stem + ".json")) as f:
        data = json.load(f)
    records = [extract_patient_series(p) for p in data["patients"]]
    records = [r for r in records if r[0]]
    real_feats, sim_feats = residual_features(records)
    desc = descriptive_metrics(records)
    summary = data.get("summary") or {}
    if desc and summary:
        # The tables report the point estimate with bootstrap sd, not the mean of
        # the bootstrap distribution. The runner summaries retain those points.
        if summary.get("spearman_patient") is not None:
            desc["spearman"] = (summary["spearman_patient"], desc["spearman"][1])
        if summary.get("heterogeneity_ratio") is not None:
            desc["heterogeneity"] = (summary["heterogeneity_ratio"], desc["heterogeneity"][1])
        if summary.get("autocorr_abs_err") is not None:
            desc["autocorr_err"] = (summary["autocorr_abs_err"], desc["autocorr_err"][1])
        if summary.get("mean_actual") is not None:
            desc["mean_real"] = summary["mean_actual"]
        if summary.get("mean_simulated") is not None:
            desc["mean_sim"] = summary["mean_simulated"]
    return c2st_cv(real_feats, sim_feats), desc


def main():
    diagnostics = load_result("discrete_diagnostics.json")
    header = "%-26s %-5s %-15s %-14s %-13s %-14s %-13s %s" % (
        "configuration", "n", "C2ST acc+/-sd", "Spearman", "heterog.",
        "autocorr err", "class acc.", "mean sim/real")
    for title, rows in PANELS.items():
        print("\n=== %s ===" % title)
        print(header)
        for stem, label in rows:
            if not (REALISM_DIR / (stem + ".json")).exists():
                continue
            c2st, desc = evaluate(stem)
            spear = "%.3f+/-%.3f" % desc["spearman"] if desc else "--"
            heter = "%.3f+/-%.3f" % desc["heterogeneity"] if desc else "--"
            autocorr = "%.3f+/-%.3f" % desc["autocorr_err"] if desc else "--"
            means = "%.3f/%.3f" % (desc["mean_sim"], desc["mean_real"]) if desc else "--"
            class_acc = diagnostics.get(stem, {}).get("class_accuracy")
            class_text = "%.3f" % class_acc if class_acc is not None else "--"
            n = desc["n_pat"] if desc else 0
            if c2st is None:
                acc = "-- (collapsed)"
            else:
                acc = "%.3f+/-%.3f" % (c2st["pooled"], c2st["sd"])
            print("%-26s n=%-3d %-15s %-14s %-13s %-14s %-13s %s"
                  % (label, n, acc, spear, heter, autocorr, class_text, means))


if __name__ == "__main__":
    main()
