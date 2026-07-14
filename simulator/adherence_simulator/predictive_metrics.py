"""Predictive metrics for simulator validation runs.

Single source of truth for the four "is the simulator actually predictive?"
metrics attached to every validation run's summary: `compute_from_payload`
computes them from the saved per-step records, so they can also be recomputed
for any saved validation JSON that lacks them.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np


# ── Helpers ────────────────────────────────────────────────────────────────

def _step_field(step: Any, field: str, default: Any = None) -> Any:
    """Read a step field from either a typed StepLog or a serialized dict."""
    if hasattr(step, field):
        return getattr(step, field)
    if hasattr(step, "get"):
        return step.get(field, default)
    return default


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


# ── Core ───────────────────────────────────────────────────────────────────

def compute_predictive_metrics(
    patient_pairs: Iterable[tuple[list[float], list[float]]],
) -> dict[str, Any]:
    """Compute the four predictive metrics from per-patient (actual, sim) day arrays.

    Each pair is two equal-length lists of floats: the observed actual adherence
    and the simulator's predicted adherence on the SAME observed days. Missing
    observations should already be filtered out by the caller.

    Returns a flat dict — see `_blank()` for keys.
    """
    per_patient_actual: list[float] = []
    per_patient_sim: list[float] = []
    all_actual_days: list[float] = []
    all_sim_days: list[float] = []

    for actual_days, sim_days in patient_pairs:
        if not actual_days:
            continue
        per_patient_actual.append(float(np.mean(actual_days)))
        per_patient_sim.append(float(np.mean(sim_days)))
        all_actual_days.extend(actual_days)
        all_sim_days.extend(sim_days)

    out = _blank()
    out["n_patients"] = len(per_patient_actual)
    out["n_paired_days"] = len(all_actual_days)

    if not all_actual_days:
        return out

    actual_days_arr = np.array(all_actual_days, dtype=float)
    sim_days_arr = np.array(all_sim_days, dtype=float)
    mu = float(actual_days_arr.mean())
    out["population_mean_actual"] = mu

    # (1) Population-mean baseline at PATIENT level
    if per_patient_actual:
        pa = np.array(per_patient_actual, dtype=float)
        ps = np.array(per_patient_sim, dtype=float)
        out["baseline_mae"] = float(np.mean(np.abs(pa - mu)))
        out["baseline_rmse"] = float(np.sqrt(np.mean((pa - mu) ** 2)))
        out["sim_mae"] = float(np.mean(np.abs(ps - pa)))
        out["sim_rmse"] = float(np.sqrt(np.mean((ps - pa) ** 2)))
        out["mae_lift_vs_baseline"] = out["baseline_mae"] - out["sim_mae"]
        out["rmse_lift_vs_baseline"] = out["baseline_rmse"] - out["sim_rmse"]

    # (2) Day-level Brier (MSE)
    out["brier_sim"] = float(np.mean((sim_days_arr - actual_days_arr) ** 2))
    out["brier_baseline"] = float(np.mean((mu - actual_days_arr) ** 2))
    out["brier_lift_vs_baseline"] = out["brier_baseline"] - out["brier_sim"]

    # (3) Day-level ROC-AUC
    y_true = (actual_days_arr > 0).astype(int)
    if 0 < int(y_true.sum()) < len(y_true):
        out["day_auc"] = _roc_auc(y_true, sim_days_arr)

    # (4) Spearman rank correlation across patients
    if len(per_patient_actual) >= 3:
        out["spearman_patient_rates"] = _spearman(per_patient_actual, per_patient_sim)

    return out


def compute_from_payload(payload: dict) -> dict[str, Any]:
    """Recompute predictive metrics from a saved validation-run JSON payload.

    Expects the saved validation-run schema:
      payload["patients"] = [
        {"actual_daily": [...], "steps": [{"adherence": float, ...}, ...], ...},
        ...
      ]

    Returns the dict as-is from `compute_predictive_metrics`. Missing/empty
    inputs return zeros (caller should treat None values as "not computable").
    """
    pairs: list[tuple[list[float], list[float]]] = []
    for p in payload.get("patients", []) or []:
        actual_daily = p.get("actual_daily") or []
        steps = p.get("steps") or []
        n = min(len(actual_daily), len(steps))
        a_days, s_days = [], []
        for d in range(n):
            if _step_field(steps[d], "gt_override", False):
                continue
            a = actual_daily[d]
            if a is None:
                continue
            try:
                a_val = float(a)
            except (TypeError, ValueError):
                continue
            s_val = _finite_float(_step_field(steps[d], "adherence", 0.0)) or 0.0
            a_days.append(a_val)
            s_days.append(s_val)
        if a_days:
            pairs.append((a_days, s_days))
    return compute_predictive_metrics(pairs)


# ── Internals ──────────────────────────────────────────────────────────────

def _blank() -> dict[str, Any]:
    return {
        "n_patients": 0,
        "n_paired_days": 0,
        "population_mean_actual": None,
        "baseline_mae": None,
        "baseline_rmse": None,
        "sim_mae": None,
        "sim_rmse": None,
        "mae_lift_vs_baseline": None,
        "rmse_lift_vs_baseline": None,
        "brier_sim": None,
        "brier_baseline": None,
        "brier_lift_vs_baseline": None,
        "day_auc": None,
        "spearman_patient_rates": None,
    }


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    """ROC-AUC via Mann–Whitney U with tie-corrected ranks. No sklearn dep."""
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, scores))
    except Exception:
        pass
    try:
        n = len(scores)
        # Average-rank of tied scores.
        order = np.argsort(scores, kind="mergesort")
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(1, n + 1)
        uniq, inv, counts = np.unique(scores, return_inverse=True,
                                      return_counts=True)
        if (counts > 1).any():
            sums = np.zeros(len(counts), dtype=float)
            np.add.at(sums, inv, ranks)
            mean_ranks = sums / counts
            ranks = mean_ranks[inv]
        n_pos = int(y_true.sum())
        n_neg = n - n_pos
        if n_pos == 0 or n_neg == 0:
            return None
        pos_rank_sum = float(ranks[y_true == 1].sum())
        u = pos_rank_sum - n_pos * (n_pos + 1) / 2.0
        return float(u / (n_pos * n_neg))
    except Exception:
        return None


def _spearman(a: list[float], b: list[float]) -> float | None:
    """Spearman ρ. Falls back to Pearson on plain ranks if scipy is missing."""
    try:
        from scipy import stats as sp_stats
        rho, _ = sp_stats.spearmanr(a, b)
        return _finite_float(rho)
    except Exception:
        pass
    arr_a = np.array(a, dtype=float)
    arr_b = np.array(b, dtype=float)
    ra = arr_a.argsort().argsort().astype(float)
    rb = arr_b.argsort().argsort().astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return _finite_float(np.corrcoef(ra, rb)[0, 1])
