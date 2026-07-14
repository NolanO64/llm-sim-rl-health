"""Dataset-backed validation runner used by the thesis experiments.

Real trajectories supply the observed interventions and evaluation targets;
the LLM predicts the corresponding normalized behavioral outcome at each step.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from simulator.adherence_simulator.aggregator import (
    aggregate_adherence,
    aggregate_certainty,
    aggregate_reason_codes,
)
from simulator.adherence_simulator.config import SimulatorConfig
from simulator.adherence_simulator.dataset_prompts import (
    DATASET_PREFIX_FORMATS,
    DATASET_PROFILES,
    build_dataset_step_prompt,
    build_dataset_system_instruction,
    compute_dataset_anchor_stats,
)
from simulator.adherence_simulator.models import StepResponse
from simulator.adherence_simulator.outcome_models import (
    OutcomeModelState,
    build_outcome_model,
    is_online_outcome_model,
)
from simulator.adherence_simulator.predictive_metrics import compute_predictive_metrics
from simulator.adherence_simulator.llm_engine import create_engine


# LLM-necessity ablation: fixed latent used when --latent-source constant.
# A reasonable HeartSteps-scale adherence/activity value applied to every
# patient/step when the LLM is bypassed entirely.
CONSTANT_LATENT_VALUE = 0.1


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    name: str
    domain: str
    intervention: str
    action_structure: str
    trajectory: str
    outcome: str
    outcome_scale: str
    rl_fit: str
    required_files: tuple[str, ...]
    public_data_note: str
    source_url: str
    default_patients: int
    default_steps: int
    actions: tuple[str, ...]
    baseline_action: str


@dataclass(frozen=True)
class ObservedStep:
    day: int
    action: str
    actual: float
    context: dict[str, Any]
    raw_outcome: dict[str, Any]
    observed_action: str | None = None
    reward: float | None = None


@dataclass(frozen=True)
class ObservedPatient:
    record_id: str
    vignette: dict[str, Any]
    steps: list[ObservedStep]


@dataclass(frozen=True)
class EngineTotals:
    total_calls: int = 0
    prompt_token_count: int = 0
    output_token_count: int = 0
    thinking_token_count: int = 0


SPECS: dict[str, DatasetSpec] = {
    "heartsteps": DatasetSpec(
        id="heartsteps",
        name="HeartSteps V1",
        domain="Physical activity JITAI",
        intervention="Activity suggestions delivered at repeated phone decision points.",
        action_structure=(
            "Observed repeated decisions: no suggestion, active suggestion, "
            "or sedentary-break suggestion."
        ),
        trajectory="37 participants over about 6 weeks, up to 5 decision points per day.",
        outcome="Proximal 30-minute step-count response after the decision point.",
        outcome_scale=(
            "0 means no proximal activity response; 1 approximates >=1500 "
            "post-decision steps. Google Fit phone steps are used when present, "
            "falling back to Jawbone tracker aggregates when Google Fit is missing."
        ),
        rl_fit="Strong: repeated context-action-outcome structure.",
        required_files=("suggestions.csv", "users.csv"),
        public_data_note="Public HeartSteps V1 files from the study GitHub repository.",
        source_url="https://github.com/klasnja/HeartStepsV1",
        default_patients=37,
        default_steps=210,
        actions=("no_suggestion", "active_suggestion", "sedentary_break_suggestion"),
        baseline_action="no_suggestion",
    ),
    "hptn067": DatasetSpec(
        id="hptn067",
        name="HPTN 067 / ADAPT",
        domain="PrEP adherence",
        intervention="Randomized PrEP dosing regimen: daily, time-driven, or event-driven.",
        action_structure="Observed regimen assignment, then repeated weekly pill-use interviews.",
        trajectory="Post-randomization self-administered phase, typically up to 24 weeks.",
        outcome="Weekly regimen coverage estimated from pill-use interview events.",
        outcome_scale="0 means no estimated regimen coverage; 1 means observed pill use covers expected regimen pills.",
        rl_fit="Moderate: real alternative interventions, but assignment is fixed rather than adaptive.",
        required_files=("WI_nofmt.tab", "RAN_nofmt.tab", "DEM_nofmt.tab"),
        public_data_note="Harvard Dataverse files require the HPTN guestbook response before download.",
        source_url="https://doi.org/10.7910/DVN/VYXMNJ",
        default_patients=96,
        default_steps=24,
        actions=("daily_regimen", "time_driven_regimen", "event_driven_regimen"),
        baseline_action="daily_regimen",
    ),
    "stepcountjitai": DatasetSpec(
        id="stepcountjitai",
        name="StepCountJITAI (synthetic)",
        domain="Physical activity JITAI (synthetic RL benchmark)",
        intervention="Activity suggestions at repeated decision points: none, generic, or context-specific.",
        action_structure=(
            "Repeated decisions: no intervention, a generic suggestion, or one "
            "of two context-specific suggestions."
        ),
        trajectory="100 synthetic participants, up to 70 decision points (episodes end early on disengagement).",
        outcome="Proximal step-count response after the decision point.",
        outcome_scale=(
            "0 means no proximal activity response; 1 approximates a strong "
            "proximal response (raw step-count of about 200)."
        ),
        rl_fit="Strong: known non-zero, context-dependent intervention effect (designed RL benchmark).",
        required_files=("episodes.csv",),
        public_data_note="Synthetic trajectories generated from StepCountJITAI (reml-lab, NeurIPS 2024 workshop).",
        source_url="https://github.com/reml-lab/StepCountJITAI",
        default_patients=100,
        default_steps=70,
        actions=("no_intervention", "generic_suggestion", "context_a_suggestion", "context_b_suggestion"),
        baseline_action="no_intervention",
    ),
}

ACTION_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "heartsteps": {
        "no_suggestion": "No activity suggestion is sent at this decision point.",
        "active_suggestion": "An active walking/activity suggestion is sent.",
        "sedentary_break_suggestion": "A sedentary-break suggestion is sent.",
    },
    "hptn067": {
        "daily_regimen": "Daily oral PrEP regimen assignment.",
        "time_driven_regimen": "Time-driven PrEP regimen assignment: scheduled weekly doses plus post-exposure boost.",
        "event_driven_regimen": "Event-driven PrEP regimen assignment: doses around potential HIV exposure.",
    },
    "stepcountjitai": {
        "no_intervention": "No activity suggestion is sent at this decision point.",
        "generic_suggestion": "A generic activity suggestion not tailored to the current context.",
        "context_a_suggestion": "An activity suggestion tailored to context A.",
        "context_b_suggestion": "An activity suggestion tailored to context B.",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(SPECS))
    parser.add_argument("--dataset-root")
    parser.add_argument("--episode-length", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        default="real-history",
        choices=["inspect", "real-history", "autoregressive"],
        help=(
            "real-history uses observed previous outcomes in the prompt; "
            "autoregressive feeds previous simulated outcomes."
        ),
    )
    parser.add_argument("--patients", type=int)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--require-real-data", action="store_true")
    parser.add_argument("--backend", default="nebula", choices=["nebula", "openai", "deepseek"])
    parser.add_argument("--model", default="SURF.Qwen3.5 122B A10B NVFP4")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--aggregation", default="mean", choices=["mean", "median", "majority_vote"])
    parser.add_argument("--history-window", type=int, default=7)
    parser.add_argument("--full-history", action="store_true")
    parser.add_argument(
        "--prefix-format",
        default="raw",
        choices=list(DATASET_PREFIX_FORMATS),
        help=(
            "How prior step history is rendered in the warm-start prefix: "
            "raw / raw-actions / summary / hybrid / pattern-typed."
        ),
    )
    parser.add_argument(
        "--anchor-days",
        type=int,
        default=0,
        help=(
            "K-step anchored mode: first K steps of each patient use the real "
            "observed outcome in the prompt; from step K+1 onwards the prompt "
            "switches to the LLM's own prior predictions (autoregressive). "
            "0 disables it -- behaviour then follows --mode."
        ),
    )
    parser.add_argument(
        "--anchor-prior",
        default="off",
        choices=["off", "population"],
        help=(
            "When 'population' (and --anchor-days > 0), inject a cohort-level "
            "descriptive prior block (mean outcome, distribution, per-action "
            "effects across the anchor window) into every step prompt."
        ),
    )
    parser.add_argument("--prompt-variant", default="dataset-v1")
    parser.add_argument(
        "--prediction-target",
        default="continuous",
        choices=[
            "continuous",
            "heartsteps_relative_5bin",
            "heartsteps_relative_3bin",
            "heartsteps_binary_activity",
            "heartsteps_zero_positive",
        ],
        help=(
            "What the LLM is asked to predict. continuous uses the "
            "numeric 0-1 target. heartsteps_relative_* asks for a discrete "
            "activity class relative to visible patient-local history, then "
            "maps that class back to a numeric outcome without cohort/test "
            "calibration."
        ),
    )
    parser.add_argument(
        "--patient-memory",
        default="none",
        choices=["none", "anchor-summary"],
        help=(
            "Optional stable patient memory injected into every step prompt. "
            "'anchor-summary' derives a compact patient state from static "
            "profile fields plus the visible anchored warm-start only."
        ),
    )
    parser.add_argument(
        "--outcome-decoder",
        default="none",
        choices=[
            "none",
            "two_part_lognormal",
        ],
        help=(
            "Optional outcome-emission model applied after LLM prediction. "
            "'two_part_lognormal' is an online, dataset-agnostic simulator "
            "emission model whose generated outcome is fed into subsequent "
            "autoregressive history."
        ),
    )
    parser.add_argument(
        "--decoder-sigma", type=float, default=None,
        help="Override two_part_lognormal emission sigma (default 0.90); used by the emission-parameter sensitivity sweep.",
    )
    parser.add_argument(
        "--decoder-anchor-weight", type=float, default=None,
        help="Override two_part_lognormal anchor_weight (default 0.50); used by the emission-parameter sensitivity sweep.",
    )
    parser.add_argument(
        "--latent-source",
        default="llm",
        choices=["llm", "anchor_mean", "constant"],
        help=(
            "LLM-necessity ablation. 'llm' (default): use the LLM's "
            "per-step prediction as the latent (unchanged baseline behaviour). "
            "'anchor_mean': SKIP the LLM entirely and use the patient's warm-start "
            "anchor mean as a constant latent for every step (no LLM calls). "
            "'constant': SKIP the LLM and use a single global constant latent "
            f"({CONSTANT_LATENT_VALUE}) for everyone. The non-'llm' modes feed a "
            "constant latent into --outcome-decoder to test whether the decoder + "
            "anchors do the work without the LLM."
        ),
    )
    parser.add_argument(
        "--stateful-llm",
        action="store_true",
        help=(
            "Stateful-simulation variant: feed the model's own state_update back into the next step's "
            "prompt as an evolving latent state (stress/momentum/etc. the LLM "
            "maintains). Makes the action effect "
            "state-dependent and noisier. LLM-driven, dataset-agnostic."
        ),
    )
    parser.add_argument(
        "--action-noise-hint",
        action="store_true",
        help=(
            "Action-noise-hint variant: append a system-instruction hint that the intervention "
            "effect is small AND noisy (often absent or slightly negative), to "
            "discourage a clean deterministic action shift."
        ),
    )
    parser.add_argument("--no-stochastic-sampling", action="store_true")
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--no-reasoning", action="store_true")
    parser.add_argument("--patient-workers", type=int, default=1, help="Run this many patients concurrently.")
    parser.add_argument("--thinking", action="store_true", help="Enable model thinking with backend default budget.")
    parser.add_argument("--thinking-budget", type=int, default=0)
    parser.add_argument("--show-prompt", action="store_true")
    parser.add_argument("--mock-llm", action="store_true", help="Deterministic test backend; not used by UI templates.")
    parser.add_argument("--api-key")
    parser.add_argument("-o", "--output", required=True)
    return parser


def run_dataset_experiment(args: argparse.Namespace) -> dict[str, Any]:
    _normalize_args(args)
    start = time.perf_counter()
    spec = SPECS[args.dataset]
    root = _default_dataset_root(spec) if not args.dataset_root else Path(args.dataset_root)
    readiness = inspect_dataset(spec, root)

    if args.mode == "inspect":
        result = _build_inspect_result(spec, args, readiness)
    else:
        if not readiness["ready"]:
            missing = [f["path"] for f in readiness["files"] if not f["present"]]
            raise SystemExit(
                f"{spec.id} real data is not ready. Missing files in dataset_root "
                f"{readiness['dataset_root']}: {', '.join(missing)}"
            )
        observed = load_observed_patients(spec, args, root)
        result = _run_llm_validation(spec, args, readiness, observed)

    elapsed = time.perf_counter() - start
    result.setdefault("run", {})
    result["run"]["total_seconds"] = elapsed
    result.setdefault("summary", {})["total_seconds"] = elapsed
    return result


def inspect_dataset(spec: DatasetSpec, dataset_root: Path | None) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    present = 0
    root = dataset_root.resolve() if dataset_root else None
    for rel in spec.required_files:
        path = root / rel if root else None
        exists = bool(path and path.exists())
        present += int(exists)
        entry: dict[str, Any] = {"path": rel, "present": exists}
        if exists and path is not None:
            entry["size_bytes"] = path.stat().st_size
            entry.update(_peek_table(path))
        files.append(entry)
    return {
        "dataset_root": str(root) if root else None,
        "required_files_present": present,
        "required_files_total": len(spec.required_files),
        "ready": present == len(spec.required_files),
        "files": files,
    }


def _load_stepcountjitai(spec: DatasetSpec, args: argparse.Namespace, root: Path) -> list[ObservedPatient]:
    rows = _read_csv(root / "episodes.csv")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("patient") or "unknown"), []).append(row)
    max_steps = _resolved_steps(spec, args)
    patients: list[ObservedPatient] = []
    for pid in sorted(grouped, key=_natural_key):
        prows = sorted(grouped[pid], key=lambda r: _safe_float(r.get("decision_index"), 0.0) or 0.0)[:max_steps]
        if not prows:
            continue
        vignette = {"dataset": spec.id, "record_id": f"scjitai-{pid}", "n_doses": 1}
        steps: list[ObservedStep] = []
        for idx, row in enumerate(prows, start=1):
            s_raw = _safe_float(row.get("s_raw"), 0.0) or 0.0
            action = str(row.get("action") or "no_intervention")
            actual = _clip(s_raw / 200.0, 0.0, 1.0)
            # Latent habituation/disengagement (h, d) and the TRUE context are
            # deliberately withheld: the simulator must infer fatigue/response
            # from the action history, exactly as for HeartSteps motivation.
            steps.append(
                ObservedStep(
                    day=idx,
                    action=action,
                    observed_action=action,
                    actual=actual,
                    reward=actual,
                    context={
                        "decision_point_number": idx,
                        "context_label": row.get("context_label") or "unknown",
                        "context_confidence": _safe_float(row.get("context_confidence")),
                    },
                    raw_outcome={
                        "step_count": s_raw,
                        "target": "normalized_step_count",
                        "scale": 200.0,
                    },
                )
            )
        patients.append(ObservedPatient(record_id=f"scjitai-{pid}", vignette=vignette, steps=steps))
    return patients


def load_observed_patients(spec: DatasetSpec, args: argparse.Namespace, root: Path) -> list[ObservedPatient]:
    if spec.id == "heartsteps":
        patients = _load_heartsteps(spec, args, root)
    elif spec.id == "hptn067":
        patients = _load_hptn067(spec, args, root)
    elif spec.id == "stepcountjitai":
        patients = _load_stepcountjitai(spec, args, root)
    else:
        raise ValueError(f"Unknown dataset: {spec.id}")

    if args.patients:
        patients = patients[: int(args.patients)]
    if not patients:
        raise SystemExit(f"{spec.id} files are present but no usable trajectories were parsed.")
    print(
        f"[DATASET] {spec.id}: parsed {len(patients)} patients, "
        f"{sum(len(p.steps) for p in patients)} observed steps",
        flush=True,
    )
    return patients


def _run_llm_validation(
    spec: DatasetSpec,
    args: argparse.Namespace,
    readiness: dict[str, Any],
    observed_patients: list[ObservedPatient],
) -> dict[str, Any]:
    print(
        f"[RUN] dataset={spec.id} mode={args.mode} "
        f"backend={args.backend} model={args.model} ensemble={args.ensemble_size} "
        f"history={'full' if args.full_history else args.history_window}",
        flush=True,
    )
    print(f"[RUN] outcome={spec.outcome}", flush=True)
    if str(getattr(args, "prediction_target", "continuous") or "continuous") != "continuous":
        print(f"[RUN] prediction_target={args.prediction_target}", flush=True)

    profile = DATASET_PROFILES.get(spec.id)
    system_instruction = build_dataset_system_instruction(
        profile,
        prompt_variant=args.prompt_variant,
        include_reasoning=not args.no_reasoning,
    ) if profile is not None else _dataset_system_instruction(spec, args)
    system_instruction = _augment_system_for_prediction_target(system_instruction, args)
    if getattr(args, "stateful_llm", False):
        system_instruction = (
            system_instruction
            + "\n\nStateful simulation: you are simulating ONE person across "
            "consecutive decision points. Maintain a compact evolving LATENT state "
            "that is NOT in the observed data — e.g. stress, motivation_momentum, "
            "fatigue, recent_disruption — each a short scalar or word. EVERY step you "
            "MUST return a non-empty state_update reflecting how these latent factors "
            "changed given the context and the latest outcome, and carry them forward "
            "coherently. Let this latent state modulate the response: high stress / "
            "fatigue / disruption suppress activity; positive momentum slightly raises "
            "it. Crucially the intervention's effect should DEPEND on this state — it "
            "helps sometimes and does nothing or backfires when stress/fatigue are high "
            "— rather than producing a constant positive shift."
        )
    if getattr(args, "action_noise_hint", False):
        system_instruction = (
            system_instruction
            + "\n\nIntervention-effect realism: the effect of the current "
            "intervention is small AND noisy. For many decision points the "
            "intervention has no effect, and sometimes a slightly negative one. "
            "Do NOT apply a consistent positive shift whenever an intervention "
            "is active; let the outcome be driven mostly by context, history, and "
            "the person's state, with the intervention as a weak, often-absent nudge."
        )
    action_index = {name: idx for idx, name in enumerate(spec.actions)}
    patients: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    # --show-prompt previews the prompt at a few checkpoints in the FIRST
    # patient's trajectory so the user can see both the first-step prompt
    # (no prior history yet) and a later prompt with the warm-start prefix
    # populated. Set is mutated in the inner loop.
    shown_prompt_steps: set[int] = set()
    # Population anchor-window prior: computed once across the cohort's first
    # K observed steps and injected into every step prompt. Only active when
    # both --anchor-days > 0 and --anchor-prior == "population".
    anchor_days = max(0, int(getattr(args, "anchor_days", 0) or 0))
    anchor_prior_mode = str(getattr(args, "anchor_prior", "off") or "off")
    anchor_prior_stats: dict | None = None
    if profile is not None and anchor_days > 0 and anchor_prior_mode == "population":
        anchor_prior_stats = compute_dataset_anchor_stats(
            observed_patients, anchor_days, profile
        )
        if anchor_prior_stats:
            print(
                f"[RUN] anchor_prior=population anchor_days={anchor_days} "
                f"n_patients={anchor_prior_stats['n_patients']} "
                f"n_obs={anchor_prior_stats['n_obs']} "
                f"pop_mean={anchor_prior_stats['pop_mean']:.3f}",
                flush=True,
            )
    if anchor_days > 0:
        print(
            f"[RUN] anchored_mode: first {anchor_days} step(s) per patient "
            "use real outcome history; remaining steps use simulated history",
            flush=True,
        )
    population_anchor_values: tuple[float, ...] = tuple(
        _clip(float(obs.actual), 0.0, 1.0)
        for patient in observed_patients
        for obs in patient.steps[:anchor_days]
    ) if anchor_days > 0 else ()
    if is_online_outcome_model(args.outcome_decoder):
        print(
            f"[RUN] online_outcome_model={args.outcome_decoder} "
            "applied inside the rollout before simulated history updates",
            flush=True,
        )
    if str(getattr(args, "patient_memory", "none") or "none") != "none":
        print(
            f"[RUN] patient_memory={args.patient_memory} "
            "derived from profile + visible warm-start anchors only",
            flush=True,
        )
    run_started = time.perf_counter()
    n_workers = max(1, int(args.patient_workers or 1))
    n_workers = min(n_workers, len(observed_patients))
    log_lock = None

    def make_engine(worker_idx: int) -> Any:
        if args.mock_llm or args.backend == "mock":
            return MockDatasetEngine(seed=(args.seed or 0) + worker_idx)
        return create_engine(_simulator_config(args), api_key=args.api_key)

    if n_workers == 1:
        engine = make_engine(0)
        worker_engines = [engine]
        for pidx, patient in enumerate(observed_patients, start=1):
            patient_payload, patient_records = _run_patient_trajectory(
                spec=spec,
                args=args,
                patient=patient,
                pidx=pidx,
                total_patients=len(observed_patients),
                engine=engine,
                system_instruction=system_instruction,
                profile=profile,
                action_index=action_index,
                anchor_days=anchor_days,
                anchor_prior_stats=anchor_prior_stats,
                population_anchor_values=population_anchor_values,
                shown_prompt_steps=shown_prompt_steps,
                log_lock=None,
            )
            patients.append(patient_payload)
            records.extend(patient_records)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from queue import Queue
        import threading

        log_lock = threading.Lock()
        worker_engines = [make_engine(i) for i in range(n_workers)]
        engine_pool: Queue[Any] = Queue()
        for engine in worker_engines:
            engine_pool.put(engine)

        print(
            f"\nRunning {len(observed_patients)} patients with {n_workers} workers...",
            flush=True,
        )

        def task(pidx: int, patient: ObservedPatient) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
            engine = engine_pool.get()
            try:
                patient_payload, patient_records = _run_patient_trajectory(
                    spec=spec,
                    args=args,
                    patient=patient,
                    pidx=pidx,
                    total_patients=len(observed_patients),
                    engine=engine,
                    system_instruction=system_instruction,
                    profile=profile,
                    action_index=action_index,
                    anchor_days=anchor_days,
                    anchor_prior_stats=anchor_prior_stats,
                    population_anchor_values=population_anchor_values,
                    shown_prompt_steps=shown_prompt_steps,
                    log_lock=log_lock,
                )
                return pidx - 1, patient_payload, patient_records
            finally:
                engine_pool.put(engine)

        ordered_patients: list[dict[str, Any] | None] = [None] * len(observed_patients)
        ordered_records: list[list[dict[str, Any]] | None] = [None] * len(observed_patients)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(task, pidx, patient): pidx
                for pidx, patient in enumerate(observed_patients, start=1)
            }
            for future in as_completed(futures):
                idx, patient_payload, patient_records = future.result()
                ordered_patients[idx] = patient_payload
                ordered_records[idx] = patient_records
        patients.extend(p for p in ordered_patients if p is not None)
        for patient_records in ordered_records:
            if patient_records:
                records.extend(patient_records)

    elapsed = time.perf_counter() - run_started
    engine_totals = _sum_engine_totals(worker_engines)
    result = _assemble_result(
        spec=spec,
        args=args,
        readiness=readiness,
        patients=patients,
        records=records,
        data_mode=f"real_{spec.id}_llm_validation",
        engine=engine_totals,
        elapsed=elapsed,
    )
    _print_validation_summary(result)
    return result


def _run_patient_trajectory(
    *,
    spec: DatasetSpec,
    args: argparse.Namespace,
    patient: ObservedPatient,
    pidx: int,
    total_patients: int,
    engine: Any,
    system_instruction: str,
    profile: Any,
    action_index: dict[str, int],
    anchor_days: int,
    anchor_prior_stats: dict | None,
    population_anchor_values: tuple[float, ...],
    shown_prompt_steps: set[int],
    log_lock: Any | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    p_started = time.perf_counter()
    rng = random.Random((args.seed or 0) + pidx * 100_003)
    calls_before = int(getattr(engine, "total_calls", 0) or 0)
    prompt_tokens_before = int(getattr(engine, "prompt_token_count", 0) or 0)
    output_tokens_before = int(getattr(engine, "output_token_count", 0) or 0)

    _print_log(
        log_lock,
        f"[START {pidx}/{total_patients}] Patient {patient.record_id} "
        f"(observed_steps={len(patient.steps)})",
    )
    simulated_history: list[float] = []
    actual_history: list[float] = []
    actions_history: list[str] = []
    context_history: list[dict[str, Any]] = []
    step_logs: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    outcome_model = build_outcome_model(args.outcome_decoder, sigma=getattr(args, "decoder_sigma", None), anchor_weight=getattr(args, "decoder_anchor_weight", None))

    warm_start_count = min(max(0, int(anchor_days or 0)), len(patient.steps))
    anchor_prefix: list[dict[str, Any]] = []
    if warm_start_count > 0:
        for sidx, obs in enumerate(patient.steps[:warm_start_count], start=1):
            action = obs.observed_action or obs.action
            actual = float(_clip(float(obs.actual), 0.0, 1.0))
            reward = obs.reward if obs.reward is not None else _reward_for_step(spec, action, obs.actual)
            actual_history.append(actual)
            actions_history.append(action)
            context_history.append(obs.context)
            anchor_prefix.append(
                {
                    "day": obs.day,
                    "action_label": action,
                    "observed_action_label": obs.observed_action or obs.action,
                    "actual": actual,
                    "reward": reward,
                    "contextual_factors": obs.context,
                    "data_mode": "anchored_warm_start",
                }
            )
        _print_log(
            log_lock,
            f"[ANCHOR] patient={pidx}/{total_patients} rid={patient.record_id} "
            f"warm_start_steps={warm_start_count} "
            f"first_predicted_day={warm_start_count + 1 if warm_start_count < len(patient.steps) else 'none'}",
        )

    patient_memory = _build_patient_memory(spec, patient, anchor_prefix, args)

    # LLM-necessity ablation: when the latent source is not the LLM, precompute a constant
    # override latent for this patient and skip all LLM calls in the loop below.
    latent_source = str(getattr(args, "latent_source", "llm") or "llm")
    latent_override: float | None = None
    if latent_source == "constant":
        latent_override = _clip(float(CONSTANT_LATENT_VALUE), 0.0, 1.0)
    elif latent_source == "anchor_mean":
        anchor_vals = actual_history[:warm_start_count]
        anchor_mean = _mean(anchor_vals)
        if anchor_mean is None:
            # No warm-start anchors for this patient: fall back to the
            # population anchor mean, then to a sane default.
            anchor_mean = _mean(population_anchor_values)
        if anchor_mean is None:
            anchor_mean = 0.1
        latent_override = _clip(float(anchor_mean), 0.0, 1.0)

    predicted_steps = patient.steps[warm_start_count:]
    outcome_state = OutcomeModelState(
        dataset_id=spec.id,
        patient_id=patient.record_id,
        anchor_values=tuple(float(prefix["actual"]) for prefix in anchor_prefix),
        population_anchor_values=population_anchor_values,
    )
    total_predicted_steps = len(predicted_steps)
    llm_latent_state: dict[str, Any] = {}  # stateful variant: carried-forward LLM state
    for loop_idx, obs in enumerate(predicted_steps, start=1):
        sidx = warm_start_count + loop_idx
        action = obs.action
        if latent_override is not None:
            # LLM-necessity ablation: bypass the LLM entirely. Use the precomputed
            # constant latent and supply sane stubs for the downstream fields
            # (response_dicts, certainty, reason_code) that the rest of the loop
            # reads, so no LLM call is made for this step.
            response_dicts = []
            adherence_raw = float(latent_override)
            uncertainty = 0.0
            certainty = 0.5
            reason_code = "no_change"
        else:
            uses_real = _uses_real_history(args.mode)
            # When anchor_days == 0 this picks the unanchored single-source mode.
            # When anchor_days > 0 we are past the warm-start window; the prompt
            # builder composes real[:K] + simulated post-anchor history.
            history_source = "observed" if uses_real else "simulated"
            history_window = None if args.full_history else int(args.history_window)
            if profile is not None:
                prompt = build_dataset_step_prompt(
                    profile=profile,
                    patient=patient,
                    obs=obs,
                    action=action,
                    observation_history=actual_history,
                    simulated_history=simulated_history,
                    prefix_format=args.prefix_format,
                    history_window=history_window,
                    history_source=history_source,
                    anchor_prior_stats=anchor_prior_stats,
                    history_actions=actions_history,
                    history_contexts=context_history,
                    anchor_days=anchor_days,
                    prompt_variant=args.prompt_variant,
                    patient_memory=patient_memory,
                    include_reasoning=not args.no_reasoning,
                )
            else:
                prompt_history = actual_history if uses_real else simulated_history
                prompt = _build_dataset_prompt(
                    spec=spec,
                    patient=patient,
                    obs=obs,
                    action=action,
                    history=prompt_history,
                    history_source=history_source,
                    history_window=history_window,
                )
            prompt = _augment_prompt_for_prediction_target(
                prompt=prompt,
                spec=spec,
                args=args,
                actual_history=actual_history[:warm_start_count],
                simulated_history=simulated_history,
            )
            if getattr(args, "stateful_llm", False):
                state_section = _render_llm_latent_state(llm_latent_state)
                if not state_section:
                    state_section = (
                        "Your evolving latent state is currently EMPTY — initialize it now: "
                        "in state_update set a few latent factors (e.g. stress, "
                        "motivation_momentum, fatigue) you will carry forward and update "
                        "every step from now on."
                    )
                prompt = f"{prompt}\n\n{state_section}"
            if args.show_prompt and pidx == 1:
                n_steps = len(patient.steps)
                first_predicted = min(n_steps, warm_start_count + 1)
                preview_steps = {
                    first_predicted,
                    min(n_steps, first_predicted + 5),
                    max(first_predicted, n_steps // 2),
                }
                if sidx in preview_steps and sidx not in shown_prompt_steps:
                    header = (
                        f"\n[DATASET PROMPT PREVIEW -- patient 1, step "
                        f"{sidx}/{n_steps}]"
                    )
                    if log_lock is None:
                        print(header + "\n" + prompt + "\n", flush=True)
                        shown_prompt_steps.add(sidx)
                    else:
                        with log_lock:
                            if sidx not in shown_prompt_steps:
                                print(header + "\n" + prompt + "\n", flush=True)
                                shown_prompt_steps.add(sidx)

            seed = args.seed + (pidx * 100_000) + (sidx * 100)
            responses = _sample_responses(engine, prompt, system_instruction, args, seed, obs, action)
            response_dicts = [r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in responses]
            adherence_raw, uncertainty = aggregate_adherence(response_dicts, method=args.aggregation)
            certainty = aggregate_certainty(response_dicts)
            reason_code = aggregate_reason_codes(response_dicts)
            if getattr(args, "stateful_llm", False) and response_dicts:
                _merge_llm_state_update(llm_latent_state, response_dicts[0].get("state_update"))

        adherence_prob = _clip(float(adherence_raw), 0.0, 1.0)
        discrete_result = _decode_prediction_target(
            spec=spec,
            args=args,
            response_dicts=response_dicts,
            latent_value=adherence_prob,
            anchor_values=actual_history[:warm_start_count],
            simulated_history=simulated_history,
            rng=rng,
        )
        actual_activity_bin = _actual_activity_bin(
            spec=spec,
            args=args,
            value=obs.actual,
            anchor_values=actual_history[:warm_start_count],
        )
        n_doses = _n_doses_for_step(spec, obs)
        if discrete_result is not None:
            adherence = discrete_result["value"]
            adherence_prob = discrete_result["expected_value"]
            doses_taken = None
        elif args.no_stochastic_sampling:
            adherence = adherence_prob
            doses_taken = None
        else:
            doses_taken = int(rng.binomial(n_doses, adherence_prob)) if hasattr(rng, "binomial") else None
            if doses_taken is None:
                doses_taken = sum(1 for _ in range(n_doses) if rng.random() < adherence_prob)
            adherence = doses_taken / n_doses

        outcome_result = None
        if outcome_model is not None:
            outcome_result = outcome_model.decode(
                latent=adherence_prob,
                state=outcome_state,
                action=action,
                context=obs.context,
                simulated_history=simulated_history,
                rng=rng,
                deterministic=bool(args.no_stochastic_sampling),
                certainty=float(certainty if certainty is not None else 0.5),
            )
            adherence = outcome_result.value

        reward = obs.reward if obs.reward is not None else _reward_for_step(spec, action, obs.actual)
        step_log = {
            "day": obs.day,
            "action": [action_index.get(action, 0)],
            "action_flat": action_index.get(action, 0),
            "action_label": action,
            "observed_action_label": obs.observed_action or obs.action,
            "active_factors": [action],
            "adherence_raw": adherence_raw,
            "adherence_prob": outcome_result.expected_value if outcome_result is not None else adherence_prob,
            "adherence": float(_clip(adherence, 0.0, 1.0)),
            "stochastic_sampling": not args.no_stochastic_sampling,
            "doses_taken": doses_taken,
            "n_doses": n_doses,
            "certainty": certainty,
            "reason_code": reason_code,
            "llm_state_update": (response_dicts[0].get("state_update") if (getattr(args, "stateful_llm", False) and response_dicts) else None),
            "llm_latent_state": (dict(llm_latent_state) if getattr(args, "stateful_llm", False) else None),
            "took_medication": adherence >= 0.5,
            "uncertainty": uncertainty,
            "contextual_factors": obs.context,
            "ensemble_values": [float(r.get("adherence", 0.0)) for r in response_dicts],
            "gt_adherence": obs.actual,
            "gt_override": False,
            "reward": reward,
            "raw_outcome": obs.raw_outcome,
        }
        if discrete_result is not None:
            step_log["prediction_target"] = args.prediction_target
            step_log["activity_bin"] = discrete_result["activity_bin"]
            step_log["actual_activity_bin"] = actual_activity_bin
            step_log["activity_bin_votes"] = discrete_result["votes"]
            step_log["llm_adherence_before_discrete_mapping"] = adherence_raw
            step_log["discrete_mapping"] = discrete_result["metadata"]
        if outcome_result is not None:
            step_log["llm_adherence_before_decoder"] = adherence_prob
            step_log["outcome_decoder"] = outcome_model.name
            step_log["outcome_model"] = outcome_model.name
            step_log["outcome_model_expected"] = outcome_result.expected_value
            if outcome_result.p_positive is not None:
                step_log["outcome_model_p_positive"] = outcome_result.p_positive
            if outcome_result.positive_mean is not None:
                step_log["outcome_model_positive_mean"] = outcome_result.positive_mean
            if outcome_result.metadata:
                step_log["outcome_model_metadata"] = outcome_result.metadata
        step_logs.append(step_log)
        records.append(
            {
                "patient_id": patient.record_id,
                "day": obs.day,
                "action": action,
                "observed_action": obs.observed_action or obs.action,
                "action_index": action_index.get(action, 0),
                "actual": obs.actual,
                "simulated": step_log["adherence"],
                "sim_prob": step_log["adherence_prob"],
                "reward": reward,
                "outcome_decoder": outcome_model.name if outcome_model is not None else None,
                "prediction_target": str(getattr(args, "prediction_target", "continuous") or "continuous"),
                "activity_bin": step_log.get("activity_bin"),
                "actual_activity_bin": step_log.get("actual_activity_bin"),
            }
        )
        actual_history.append(obs.actual)
        simulated_history.append(step_log["adherence"])
        actions_history.append(action)
        context_history.append(obs.context)

        _print_log(
            log_lock,
            f"[PROGRESS] patient={pidx}/{total_patients} rid={patient.record_id} "
            f"day={loop_idx}/{total_predicted_steps} calls={getattr(engine, 'total_calls', 0)} "
            f"actual_day={sidx}/{len(patient.steps)}",
        )

    actual_daily = [float(s["gt_adherence"]) for s in step_logs]
    simulated_daily = [float(s["adherence"]) for s in step_logs]
    actual_rate = _mean(actual_daily) or 0.0
    simulated_rate = _mean(simulated_daily) or 0.0
    elapsed = time.perf_counter() - p_started
    calls_used = int(getattr(engine, "total_calls", 0) or 0) - calls_before
    prompt_tokens_used = int(getattr(engine, "prompt_token_count", 0) or 0) - prompt_tokens_before
    output_tokens_used = int(getattr(engine, "output_token_count", 0) or 0) - output_tokens_before
    patient_summary = {
        "record_id": patient.record_id,
        "total_days": len(step_logs),
        "mean_actual": actual_rate,
        "mean_adherence": simulated_rate,
        "cumulative_adherence": simulated_rate,
        "mean_certainty": _mean(s["certainty"] for s in step_logs) or 0.0,
        "mean_uncertainty": _mean(s["uncertainty"] for s in step_logs) or 0.0,
        "total_reward": sum(float(s.get("reward") or 0.0) for s in step_logs),
        "elapsed_seconds": elapsed,
        "total_llm_calls": calls_used,
        "prompt_tokens": prompt_tokens_used,
        "output_tokens": output_tokens_used,
        "trained": False,
    }
    patient_payload = {
        "record_id": patient.record_id,
        "data_mode": f"real_{spec.id}_llm_validation",
        "actual_rate": actual_rate,
        "actual_daily": actual_daily,
        "matched_days": len(actual_daily),
        "simulated_rate": simulated_rate,
        "simulated_rate_all_days": simulated_rate,
        "delta": simulated_rate - actual_rate,
        "delta_all_days": simulated_rate - actual_rate,
        "vignette": patient.vignette,
        "anchor_prefix": anchor_prefix,
        "anchor_days": warm_start_count,
        "patient_memory": patient_memory,
        "steps": step_logs,
        "summary": patient_summary,
        "episode_summary": patient_summary,
    }
    _print_log(
        log_lock,
        f"[COMPLETE] rid={patient.record_id} i={pidx}/{total_patients} "
        f"predicted={simulated_rate:.4f} "
        f"actual={actual_rate:.4f} delta={simulated_rate - actual_rate:+.4f} "
        f"elapsed={elapsed:.1f} calls={calls_used}",
    )
    return patient_payload, records


def _print_log(lock: Any | None, message: str) -> None:
    if lock is None:
        print(message, flush=True)
        return
    with lock:
        print(message, flush=True)


def _build_patient_memory(
    spec: DatasetSpec,
    patient: ObservedPatient,
    anchor_prefix: list[dict[str, Any]],
    args: argparse.Namespace,
) -> str | None:
    mode = str(getattr(args, "patient_memory", "none") or "none")
    if mode == "none":
        return None
    if mode != "anchor-summary":
        raise ValueError(f"Unknown patient_memory mode: {mode}")

    lines: list[str] = []
    vignette = patient.vignette or {}
    lines.append(f"- Patient ID: {patient.record_id}")

    if spec.id == "heartsteps":
        if vignette.get("selfeff_intake") is not None:
            lines.append(f"- Static exercise self-efficacy: {vignette.get('selfeff_intake')}")
        if vignette.get("conscientiousness") is not None:
            lines.append(f"- Static conscientiousness: {vignette.get('conscientiousness')}")
        if vignette.get("walk_intake") not in (None, ""):
            lines.append(f"- Intake walking report: {vignette.get('walk_intake')}")
        if vignette.get("occupation") not in (None, ""):
            lines.append(f"- Occupation context: {vignette.get('occupation')}")
    else:
        for key in ("age", "gender", "sex", "assigned_regimen", "assigned_support"):
            if vignette.get(key) is not None:
                lines.append(f"- Static {key}: {vignette.get(key)}")

    values = [float(_clip(float(item["actual"]), 0.0, 1.0)) for item in anchor_prefix]
    if values:
        mean_value = _mean(values) or 0.0
        recent_values = values[-min(5, len(values)):]
        recent_mean = _mean(recent_values) or 0.0
        near_zero = sum(1 for value in values if value <= 0.03)
        high = sum(1 for value in values if value >= 0.25)
        positive_values = [value for value in values if value > 0.03]
        positive_mean = _mean(positive_values) or 0.0
        first_half = values[: max(1, len(values) // 2)]
        second_half = values[len(first_half):] or values[-1:]
        trend = (_mean(second_half) or 0.0) - (_mean(first_half) or 0.0)
        if trend > 0.05:
            trend_label = "recently increasing"
        elif trend < -0.05:
            trend_label = "recently decreasing"
        else:
            trend_label = "roughly stable"
        lines.extend(
            [
                f"- Visible warm-start length: {len(values)} step(s)",
                f"- Warm-start mean outcome: {mean_value:.3f}",
                f"- Recent warm-start mean outcome: {recent_mean:.3f}",
                f"- Near-zero warm-start windows: {near_zero}/{len(values)}",
                f"- Clearly active warm-start windows (>=0.25): {high}/{len(values)}",
                f"- Mean among positive warm-start windows: {positive_mean:.3f}",
                f"- Warm-start trend: {trend_label}",
            ]
        )

        action_values: dict[str, list[float]] = {}
        slot_values: dict[str, list[float]] = {}
        for item in anchor_prefix:
            action = str(item.get("action_label") or "unknown")
            action_values.setdefault(action, []).append(float(item["actual"]))
            context = item.get("contextual_factors") or {}
            slot = context.get("slot") or context.get("time_of_day")
            if slot not in (None, ""):
                slot_values.setdefault(str(slot), []).append(float(item["actual"]))
        if action_values:
            action_bits = [
                f"{name}={(_mean(vals) or 0.0):.3f} over {len(vals)}"
                for name, vals in sorted(action_values.items())
            ]
            lines.append("- Warm-start action pattern: " + "; ".join(action_bits))
        if slot_values:
            slot_bits = [
                f"{name}={(_mean(vals) or 0.0):.3f} over {len(vals)}"
                for name, vals in sorted(slot_values.items())[:6]
            ]
            lines.append("- Warm-start time-slot pattern: " + "; ".join(slot_bits))
    else:
        lines.append("- No warm-start anchors are visible; memory uses static profile only.")

    return "\n".join(lines)


def _sum_engine_totals(engines: Iterable[Any]) -> EngineTotals:
    return EngineTotals(
        total_calls=sum(int(getattr(engine, "total_calls", 0) or 0) for engine in engines),
        prompt_token_count=sum(int(getattr(engine, "prompt_token_count", 0) or 0) for engine in engines),
        output_token_count=sum(int(getattr(engine, "output_token_count", 0) or 0) for engine in engines),
        thinking_token_count=sum(int(getattr(engine, "thinking_token_count", 0) or 0) for engine in engines),
    )


def _render_llm_latent_state(state: dict[str, Any]) -> str:
    """Stateful variant: render the LLM's carried-forward latent state for the prompt."""
    items = [
        (str(k), v) for k, v in state.items()
        if v is not None and str(v).strip() != ""
    ]
    if not items:
        return ""
    lines = [
        "Your evolving latent state (running notes you carried from prior steps; "
        "these are latent factors NOT in the observed data, e.g. stress, momentum, "
        "fatigue, motivation — keep them coherent and update them via state_update):",
    ]
    for key, value in items[:12]:
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def _merge_llm_state_update(state: dict[str, Any], state_update: Any) -> None:
    """Merge an LLM state_update into the carried latent state (None removes)."""
    if not isinstance(state_update, dict):
        return
    for key, value in state_update.items():
        k = str(key)
        if value is None:
            state.pop(k, None)
        else:
            state[k] = value


def _sample_responses(
    engine: Any,
    prompt: str,
    system_instruction: str,
    args: argparse.Namespace,
    seed: int,
    obs: ObservedStep,
    action: str,
) -> list[StepResponse]:
    if isinstance(engine, MockDatasetEngine):
        return engine.sample(obs=obs, action=action, k=args.ensemble_size, base_seed=seed)
    return engine.sample_prompt(
        prompt,
        system_instruction=system_instruction,
        k=args.ensemble_size,
        base_seed=seed,
    )


_RELATIVE_5BINS = ("much_lower", "lower", "typical", "higher", "much_higher")
_RELATIVE_3BINS = ("lower", "typical", "higher")
_BINARY_ACTIVITY_BINS = ("inactive", "active")
_ZERO_POSITIVE_BINS = ("near_zero", "positive")


def _activity_bins_for_target(target: str) -> tuple[str, ...]:
    if target == "heartsteps_binary_activity":
        return _BINARY_ACTIVITY_BINS
    if target == "heartsteps_zero_positive":
        return _ZERO_POSITIVE_BINS
    if target == "heartsteps_relative_3bin":
        return _RELATIVE_3BINS
    return _RELATIVE_5BINS


def _augment_system_for_prediction_target(system_instruction: str, args: argparse.Namespace) -> str:
    target = str(getattr(args, "prediction_target", "continuous") or "continuous")
    if target == "continuous":
        return system_instruction
    if target in {
        "heartsteps_relative_5bin",
        "heartsteps_relative_3bin",
        "heartsteps_binary_activity",
        "heartsteps_zero_positive",
    }:
        labels = ", ".join(_activity_bins_for_target(target))
        return (
            f"{system_instruction}\n\n"
            "Discrete target mode: for HeartSteps, decide the participant's "
            "next 30-minute activity class relative to the visible patient-local "
            f"history. Return activity_bin as one of: {labels}. Still include adherence as a rough "
            "0-1 estimate for compatibility; the evaluator will reconstruct "
            "the numeric outcome from activity_bin using only visible "
            "patient-local history."
        )
    return system_instruction


def _augment_prompt_for_prediction_target(
    *,
    prompt: str,
    spec: DatasetSpec,
    args: argparse.Namespace,
    actual_history: list[float],
    simulated_history: list[float],
) -> str:
    target = str(getattr(args, "prediction_target", "continuous") or "continuous")
    if target == "continuous":
        return prompt
    if spec.id != "heartsteps":
        return prompt

    visible = _visible_patient_local_values(actual_history, simulated_history)
    if visible:
        stats = (
            f"visible_n={len(visible)}, mean={_mean(visible):.3f}, "
            f"min={min(visible):.3f}, median={_quantile(visible, 0.5):.3f}, "
            f"max={max(visible):.3f}"
        )
    else:
        stats = "visible_n=0; no patient-local numeric baseline yet"

    return (
        f"{prompt}\n\n"
        "Discrete prediction target for this run:\n"
        "- Do not try to output the exact normalized step count directly.\n"
        "- First classify this next 30-minute window relative to this "
        "participant's visible patient-local history.\n"
        f"- Patient-local baseline summary: {stats}.\n"
        f"{_activity_bin_definition_block(target)}\n"
        "- Use only the profile, current pre-decision context, current action, "
        "and visible prior history. Do not infer hidden future outcomes.\n"
        "- Return strict JSON with the usual keys plus activity_bin.\n"
        'Example key: "activity_bin": "typical".'
    )


def _activity_bin_definition_block(target: str) -> str:
    if target == "heartsteps_binary_activity":
        return (
            "- activity_bin definitions:\n"
            "  - inactive: below this participant's visible median activity level.\n"
            "  - active: at or above this participant's visible median activity level."
        )
    if target == "heartsteps_zero_positive":
        return (
            "- activity_bin definitions:\n"
            "  - near_zero: this 30-minute window has essentially no proximal steps "
            "(normalized outcome around 0.03 or less).\n"
            "  - positive: this 30-minute window has a nonzero proximal step response."
        )
    if target == "heartsteps_relative_3bin":
        return (
            "- activity_bin definitions:\n"
            "  - lower: below this participant's recent/visible level.\n"
            "  - typical: close to this participant's recent/visible level.\n"
            "  - higher: above this participant's recent/visible level."
        )
    return (
        "- activity_bin definitions:\n"
        "  - much_lower: clearly below this participant's recent/visible level.\n"
        "  - lower: somewhat below this participant's recent/visible level.\n"
        "  - typical: close to this participant's recent/visible level.\n"
        "  - higher: somewhat above this participant's recent/visible level.\n"
        "  - much_higher: clearly above this participant's recent/visible level."
    )


def _decode_prediction_target(
    *,
    spec: DatasetSpec,
    args: argparse.Namespace,
    response_dicts: list[dict[str, Any]],
    latent_value: float,
    anchor_values: list[float],
    simulated_history: list[float],
    rng: random.Random,
) -> dict[str, Any] | None:
    target = str(getattr(args, "prediction_target", "continuous") or "continuous")
    if target == "continuous":
        return None
    if spec.id != "heartsteps":
        return None
    if target not in {
        "heartsteps_relative_5bin",
        "heartsteps_relative_3bin",
        "heartsteps_binary_activity",
        "heartsteps_zero_positive",
    }:
        raise ValueError(f"Unknown prediction target: {target}")

    reference = _stable_reference_values(anchor_values, simulated_history)
    activity_bin, votes, source_counts = _aggregate_activity_bin(response_dicts, target=target)
    if not reference:
        # No patient-local baseline exists in K0 first-step cases. Keep this
        # non-leaky by falling back to the LLM's numeric compatibility output.
        value = _clip(latent_value, 0.0, 1.0)
        return {
            "value": value,
            "expected_value": value,
            "activity_bin": activity_bin,
            "votes": votes,
            "metadata": {
                "method": f"{target.replace('heartsteps_', '')}_no_reference_fallback",
                "reference_n": 0,
                "activity_bin_source_counts": source_counts,
            },
        }

    value, metadata = _map_relative_bin_to_value(
        reference=reference,
        activity_bin=activity_bin,
        target=target,
        rng=rng,
        deterministic=bool(args.no_stochastic_sampling),
    )
    value = _clip(value, 0.0, 1.0)
    metadata["activity_bin_source_counts"] = source_counts
    return {
        "value": value,
        "expected_value": value,
        "activity_bin": activity_bin,
        "votes": votes,
        "metadata": metadata,
    }


def _visible_patient_local_values(anchor_values: list[float], simulated_history: list[float]) -> list[float]:
    values = []
    values.extend(float(v) for v in anchor_values if v is not None)
    values.extend(float(v) for v in simulated_history if v is not None)
    return [_clip(v, 0.0, 1.0) for v in values]


def _stable_reference_values(anchor_values: list[float], simulated_history: list[float]) -> list[float]:
    anchor_reference = [_clip(float(v), 0.0, 1.0) for v in anchor_values if v is not None]
    if anchor_reference:
        return anchor_reference
    # Only K0 has no anchor reference. In that case simulated history is the
    # only patient-local baseline available, but anchored runs never feed
    # simulated values into the bin mapping. This avoids self-reinforcing
    # collapse where one low mapped value lowers every later mapped value.
    return [_clip(float(v), 0.0, 1.0) for v in simulated_history if v is not None]


def _aggregate_activity_bin(
    response_dicts: list[dict[str, Any]],
    *,
    target: str = "heartsteps_relative_5bin",
) -> tuple[str, dict[str, float], dict[str, int]]:
    labels = _activity_bins_for_target(target)
    votes = {name: 0.0 for name in labels}
    source_counts = {"explicit": 0, "missing_or_invalid": 0}
    for raw in response_dicts:
        label = str(raw.get("activity_bin") or "").strip().lower().replace("-", "_")
        if label not in votes:
            source_counts["missing_or_invalid"] += 1
            continue
        source_counts["explicit"] += 1
        weight = _clip(float(raw.get("adherence_certainty", 0.5) or 0.5), 0.0, 1.0)
        votes[label] += max(weight, 0.05)
    if not any(votes.values()):
        raise ValueError(
            "Discrete prediction target requires an explicit valid activity_bin "
            "from the LLM; refusing to derive it from the numeric adherence field."
        )
    best = max(labels, key=lambda name: (votes[name], -labels.index(name)))
    return best, votes, source_counts


def _latent_to_activity_bin(value: float, reference: list[float] | None = None) -> str:
    value = _clip(value, 0.0, 1.0)
    if reference:
        return _value_to_relative_bin(value, reference)
    if value < 0.2:
        return "much_lower"
    if value < 0.4:
        return "lower"
    if value < 0.6:
        return "typical"
    if value < 0.8:
        return "higher"
    return "much_higher"


def _value_to_relative_bin(value: float, reference: list[float]) -> str:
    baseline = max(float(_mean(reference) or 0.0), 1e-6)
    ratio = _clip(float(value), 0.0, 1.0) / baseline
    if ratio < 0.50:
        return "much_lower"
    if ratio < 0.85:
        return "lower"
    if ratio <= 1.15:
        return "typical"
    if ratio <= 1.75:
        return "higher"
    return "much_higher"


def _value_to_relative_3bin(value: float, reference: list[float]) -> str:
    label = _value_to_relative_bin(value, reference)
    if label in {"much_lower", "lower"}:
        return "lower"
    if label in {"higher", "much_higher"}:
        return "higher"
    return "typical"


def _value_to_activity_bin(value: float, reference: list[float], target: str) -> str:
    if target == "heartsteps_binary_activity":
        return "active" if _clip(float(value), 0.0, 1.0) >= _quantile(reference, 0.5) else "inactive"
    if target == "heartsteps_zero_positive":
        return "positive" if _clip(float(value), 0.0, 1.0) > 0.03 else "near_zero"
    if target == "heartsteps_relative_3bin":
        return _value_to_relative_3bin(value, reference)
    return _value_to_relative_bin(value, reference)


def _relative_bin_bounds(reference: list[float], activity_bin: str, target: str) -> tuple[float, float]:
    baseline = max(float(_mean(reference) or 0.0), 1e-6)
    if target == "heartsteps_binary_activity":
        median = _quantile(reference, 0.5)
        ranges = {
            "inactive": (0.0, median),
            "active": (median, 1.0),
        }
        return ranges.get(activity_bin, ranges["inactive"])
    if target == "heartsteps_zero_positive":
        ranges = {
            "near_zero": (0.0, 0.03),
            "positive": (0.03, 1.0),
        }
        return ranges.get(activity_bin, ranges["near_zero"])
    if target == "heartsteps_relative_3bin":
        ranges = {
            "lower": (0.0, 0.85),
            "typical": (0.85, 1.15),
            "higher": (1.15, max(2.50, 1.15 + (0.20 / baseline))),
        }
    else:
        ranges = {
            "much_lower": (0.0, 0.50),
            "lower": (0.50, 0.85),
            "typical": (0.85, 1.15),
            "higher": (1.15, 1.75),
            "much_higher": (1.75, max(2.50, 1.75 + (0.20 / baseline))),
        }
    rlo, rhi = ranges.get(activity_bin, ranges["typical"])
    return _clip(rlo * baseline, 0.0, 1.0), _clip(rhi * baseline, 0.0, 1.0)


def _map_relative_bin_to_value(
    *,
    reference: list[float],
    activity_bin: str,
    target: str,
    rng: random.Random,
    deterministic: bool,
) -> tuple[float, dict[str, Any]]:
    clean_reference = [_clip(float(v), 0.0, 1.0) for v in reference if v is not None]
    bucket = [
        value
        for value in clean_reference
        if _value_to_activity_bin(value, clean_reference, target) == activity_bin
    ]
    if bucket:
        value = _mean(bucket) if deterministic else rng.choice(bucket)
        return float(value or 0.0), {
            "method": "patient_anchor_relative_bin_bucket",
            "target": target,
            "reference_n": len(clean_reference),
            "reference_mean": _mean(clean_reference),
            "bucket_n": len(bucket),
            "deterministic": deterministic,
        }

    lo, hi = _relative_bin_bounds(clean_reference, activity_bin, target)
    if hi < lo:
        lo, hi = hi, lo
    value = (lo + hi) / 2.0 if deterministic or hi - lo < 1e-6 else rng.uniform(lo, hi)
    return float(value), {
        "method": "patient_anchor_relative_bin_bounds",
        "target": target,
        "reference_n": len(clean_reference),
        "reference_mean": _mean(clean_reference),
        "bucket_n": 0,
        "bin_lo": lo,
        "bin_hi": hi,
        "deterministic": deterministic,
    }


def _actual_activity_bin(
    *,
    spec: DatasetSpec,
    args: argparse.Namespace,
    value: float | None,
    anchor_values: list[float],
) -> str | None:
    target = str(getattr(args, "prediction_target", "continuous") or "continuous")
    if (
        target not in {
            "heartsteps_relative_5bin",
            "heartsteps_relative_3bin",
            "heartsteps_binary_activity",
            "heartsteps_zero_positive",
        }
        or spec.id != "heartsteps"
        or value is None
    ):
        return None
    reference = _stable_reference_values(anchor_values, [])
    if not reference:
        return None
    return _value_to_activity_bin(float(value), reference, target)


def _quantile(values: list[float], q: float) -> float:
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    pos = _clip(float(q), 0.0, 1.0) * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _evaluation_patients(patients: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return patient payloads restricted to actual LLM-predicted steps.

    Anchored warm-start steps have ``gt_override=True`` and ``adherence`` equal
    to the ground-truth outcome by construction. They should remain in the raw
    output for traceability, but headline validation metrics must not score
    them as model predictions.
    """
    filtered: list[dict[str, Any]] = []
    for patient in patients:
        eval_steps = [
            step for step in (patient.get("steps") or [])
            if not step.get("gt_override")
        ]
        if not eval_steps:
            continue

        actual_daily = [
            float(step["gt_adherence"])
            for step in eval_steps
            if step.get("gt_adherence") is not None
        ]
        simulated_daily = [
            float(step["adherence"])
            for step in eval_steps
            if step.get("adherence") is not None
        ]
        if not actual_daily or not simulated_daily:
            continue

        payload = dict(patient)
        payload["steps"] = eval_steps
        payload["actual_daily"] = actual_daily
        payload["matched_days"] = len(eval_steps)
        payload["actual_rate"] = _mean(actual_daily)
        payload["simulated_rate"] = _mean(simulated_daily)
        payload["simulated_rate_all_days"] = payload["simulated_rate"]
        payload["delta"] = (
            payload["simulated_rate"] - payload["actual_rate"]
            if payload["simulated_rate"] is not None and payload["actual_rate"] is not None
            else None
        )
        payload["delta_all_days"] = payload["delta"]
        filtered.append(payload)
    return filtered


def _activity_bin_metrics(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    labels = _activity_bins_for_records(records)
    pairs = [
        (str(r.get("actual_activity_bin")), str(r.get("activity_bin")))
        for r in records
        if r.get("actual_activity_bin") in labels
        and r.get("activity_bin") in labels
    ]
    if not pairs:
        return None

    pred_counts = {name: 0 for name in labels}
    actual_counts = {name: 0 for name in labels}
    confusion = {
        actual: {pred: 0 for pred in labels}
        for actual in labels
    }
    ordinal_errors: list[int] = []
    exact = 0
    for actual, pred in pairs:
        actual_counts[actual] += 1
        pred_counts[pred] += 1
        confusion[actual][pred] += 1
        if actual == pred:
            exact += 1
        ordinal_errors.append(abs(labels.index(pred) - labels.index(actual)))

    n = len(pairs)
    return {
        "activity_bin_n": n,
        "activity_bin_accuracy": exact / n,
        "activity_bin_ordinal_mae": _mean(ordinal_errors),
        "activity_bin_pred_distribution": {k: v / n for k, v in pred_counts.items()},
        "activity_bin_actual_distribution": {k: v / n for k, v in actual_counts.items()},
        "activity_bin_confusion": confusion,
        "activity_bin_note": (
            "Observed outcomes are converted to the same patient-local relative "
            "bins using anchor-only history. These are diagnostic metrics for "
            "discrete-target runs; numeric trajectory metrics remain primary."
        ),
    }


def _activity_bins_for_records(records: list[dict[str, Any]]) -> tuple[str, ...]:
    labels = {
        str(r.get("activity_bin"))
        for r in records
        if r.get("activity_bin") is not None
    }
    if labels and labels.issubset(set(_BINARY_ACTIVITY_BINS)):
        return _BINARY_ACTIVITY_BINS
    if labels and labels.issubset(set(_ZERO_POSITIVE_BINS)):
        return _ZERO_POSITIVE_BINS
    if labels and labels.issubset(set(_RELATIVE_3BINS)):
        return _RELATIVE_3BINS
    return _RELATIVE_5BINS


def _assemble_result(
    *,
    spec: DatasetSpec,
    args: argparse.Namespace,
    readiness: dict[str, Any],
    patients: list[dict[str, Any]],
    records: list[dict[str, Any]],
    data_mode: str,
    engine: Any,
    elapsed: float,
) -> dict[str, Any]:
    # Evaluation metrics only score post-prefix LLM predictions. New dataset
    # runs store warm-start observations in
    # patient["anchor_prefix"], outside patients[].steps and records[].
    # The gt_override filter remains for older outputs produced before that
    # contract was tightened.
    eval_patients = _evaluation_patients(patients)
    eval_records = [r for r in records if not r.get("anchored")]
    warm_start_points = (
        sum(len(p.get("anchor_prefix") or []) for p in patients)
        + sum(1 for r in records if r.get("anchored"))
    )

    actual_rates = [_as_float(p.get("actual_rate")) for p in eval_patients]
    sim_rates = [_as_float(p.get("simulated_rate")) for p in eval_patients]
    paired_rates = [(a, s) for a, s in zip(actual_rates, sim_rates) if a is not None and s is not None]
    actual_rates = [a for a, _ in paired_rates]
    sim_rates = [s for _, s in paired_rates]
    errors = [s - a for a, s in paired_rates]

    all_actual = [float(r["actual"]) for r in eval_records if r.get("actual") is not None]
    all_sim = [float(r["simulated"]) for r in eval_records if r.get("simulated") is not None]
    day_errors = [s - a for a, s in zip(all_actual, all_sim)]
    predictive_pairs = [
        (
            [float(x) for x in (p.get("actual_daily") or [])],
            [float((p.get("steps") or [])[i].get("adherence", 0.0)) for i in range(len(p.get("actual_daily") or []))],
        )
        for p in eval_patients
        if p.get("actual_daily")
    ]
    predictive = compute_predictive_metrics(predictive_pairs)
    predictive["pearson_patient_rates"] = _pearson(actual_rates, sim_rates)
    activity_bin_summary = _activity_bin_metrics(eval_records)

    if args.policy == "observed":
        action_effects = _action_effect_table(spec, eval_records)
        lift_corr = _lift_correlation(action_effects)
        action_effect_note = "observed_action_validation"
    else:
        action_effects = _action_effect_table(spec, eval_records)
        lift_corr = None
        action_effect_note = (
            "lift_correlation_disabled_for_policy_override; actual outcomes "
            "were generated under observed dataset actions, not the resolved "
            f"{args.policy!r} policy actions."
        )
    mean_actual = _mean(actual_rates)
    mean_sim = _mean(sim_rates)
    corr = _pearson(actual_rates, sim_rates)
    corr_days = _pearson(all_actual, all_sim)
    summary = {
        "dataset": spec.id,
        "dataset_name": spec.name,
        "data_mode": data_mode,
        "mode": args.mode,
        "policy": args.policy,
        "outcome_decoder": args.outcome_decoder,
        "prediction_target": str(getattr(args, "prediction_target", "continuous") or "continuous"),
        "ready": readiness["ready"],
        "n_patients": len(eval_patients),
        "n_total_patients": len(patients),
        "n_decision_points": len(eval_records),
        "n_total_decision_points": len(eval_records) + warm_start_points,
        "n_warm_start_points": warm_start_points,
        "decision_points_per_patient": _mean(len(p.get("steps") or []) for p in eval_patients),
        "action_space_size": len(spec.actions),
        "mean_actual": mean_actual,
        "mean_simulated": mean_sim,
        "mean_simulated_all_days": _mean(all_sim),
        "mean_reward": _mean(r.get("reward") for r in eval_records),
        "mae": _mean(abs(e) for e in errors),
        "mae_all_days": _mean(abs(e) for e in day_errors),
        "rmse": math.sqrt(_mean(e * e for e in errors) or 0.0) if errors else None,
        "rmse_all_days": math.sqrt(_mean(e * e for e in day_errors) or 0.0) if day_errors else None,
        "correlation": corr,
        "correlation_all_days": corr_days,
        "patient_level_correlation": corr,
        "spearman_patient": _spearman(actual_rates, sim_rates),
        "heterogeneity_ratio": _heterogeneity_ratio(actual_rates, sim_rates),
        "per_patient_ks_pass_rate": _per_patient_ks_pass_rate(eval_patients),
        "per_patient_ks_mean_p": _per_patient_ks_mean_p(eval_patients),
        "autocorr_abs_err": _autocorr_abs_err(eval_patients),
        "factor_lift_correlation": lift_corr,
        "factor_lift_note": action_effect_note,
        "evaluation_note": (
            "headline metrics score post-anchor LLM predictions only; "
            "anchored warm-start observations are conditioning context."
        ),
        "rl_fit": spec.rl_fit,
        "total_seconds": elapsed,
        "total_llm_calls": getattr(engine, "total_calls", 0),
        "prompt_tokens": getattr(engine, "prompt_token_count", 0),
        "output_tokens": getattr(engine, "output_token_count", 0),
        "thinking_tokens": getattr(engine, "thinking_token_count", 0),
    }
    if activity_bin_summary:
        summary.update(activity_bin_summary)

    return {
        "kind": "dataset_validation_experiment",
        "run": {
            "dataset": spec.id,
            "mode": args.mode,
            "policy": args.policy,
            "backend": args.backend,
            "model": args.model,
            "temperature": args.temperature,
            "ensemble_size": args.ensemble_size,
            "aggregation": args.aggregation,
            "episode_length": args.episode_length,
            "patients": args.patients,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "prompt_variant": args.prompt_variant,
            "patient_memory": args.patient_memory,
            "outcome_decoder": args.outcome_decoder,
            "prediction_target": str(getattr(args, "prediction_target", "continuous") or "continuous"),
            "prefix_format": args.prefix_format,
            "anchor_days": int(getattr(args, "anchor_days", 0) or 0),
            "anchor_prior": str(getattr(args, "anchor_prior", "off") or "off"),
            "history_window": None if args.full_history else args.history_window,
            "full_history": args.full_history,
            "stochastic_sampling": not args.no_stochastic_sampling,
            "total_seconds": elapsed,
            "total_llm_calls": getattr(engine, "total_calls", 0),
            "prompt_tokens": getattr(engine, "prompt_token_count", 0),
            "output_tokens": getattr(engine, "output_token_count", 0),
            "thinking_tokens": getattr(engine, "thinking_token_count", 0),
        },
        "summary": summary,
        "dataset_spec": asdict(spec),
        "readiness": readiness,
        "config": _config(spec, args, data_mode),
        "predictive_metrics": predictive,
        "day_level_metrics": {
            "intervention_effects": action_effects,
            "intervention_lift_correlation": lift_corr,
            "intervention_lift_note": action_effect_note,
            "activity_bin_metrics": activity_bin_summary,
            "ks_test": _simple_ks_summary(eval_records),
        },
        "patients": patients,
        "action_space": list(spec.actions),
        "note": (
            "Dataset-backed LLM validation using observed dataset actions."
        ),
    }


def _build_inspect_result(spec: DatasetSpec, args: argparse.Namespace, readiness: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "dataset_inspection",
        "summary": {
            "dataset": spec.id,
            "dataset_name": spec.name,
            "ready": readiness["ready"],
            "mode": "inspect",
            "rl_fit": spec.rl_fit,
            "action_structure": spec.action_structure,
            "trajectory": spec.trajectory,
            "required_files_present": readiness["required_files_present"],
            "required_files_total": readiness["required_files_total"],
        },
        "dataset_spec": asdict(spec),
        "readiness": readiness,
        "config": _config(spec, args, "inspect"),
        "next_step": (
            "Files found. Launch real-history/autoregressive validation to produce LLM scoring."
            if readiness["ready"]
            else "Download the required public dataset files into dataset_root, then launch validation."
        ),
    }


def _parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    # HeartSteps exports naive local timestamps with optional ".0" seconds.
    if text.endswith(".0"):
        text = text[:-2]
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _positive_elapsed_minutes(current: datetime | None, previous: datetime | None) -> float | None:
    if current is None or previous is None:
        return None
    minutes = (current - previous).total_seconds() / 60.0
    if minutes < 0:
        return None
    return round(minutes, 1)


def _clock_time(value: datetime | None) -> str | None:
    return value.strftime("%H:%M") if value is not None else None


def _time_of_day(value: datetime | None) -> str | None:
    if value is None:
        return None
    hour = value.hour + (value.minute / 60.0)
    if hour < 11:
        return "morning"
    if hour < 14:
        return "midday"
    if hour < 17:
        return "afternoon"
    if hour < 21:
        return "evening"
    return "late_evening"


def _first_float_with_source(
    row: dict[str, Any],
    candidates: Iterable[tuple[str, str]],
) -> tuple[float | None, str | None]:
    for column, source in candidates:
        value = _as_float(row.get(column))
        if value is not None:
            return value, source
    return None, None


def _load_heartsteps(spec: DatasetSpec, args: argparse.Namespace, root: Path) -> list[ObservedPatient]:
    suggestions = _read_csv(root / "suggestions.csv")
    users = {str(row.get("user.index")): row for row in _read_csv(root / "users.csv")}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in suggestions:
        if not _truthy(row.get("avail", "true")):
            continue
        post = _first_float(row, ("gfsteps30", "jbsteps30", "gfsteps40", "jbsteps40", "gfsteps60", "jbsteps60"))
        if post is None:
            continue
        user = str(row.get("user.index") or "unknown")
        grouped.setdefault(user, []).append(row)

    max_steps = _resolved_steps(spec, args)
    patients: list[ObservedPatient] = []
    for user in sorted(grouped, key=_natural_key):
        rows = sorted(grouped[user], key=lambda r: _safe_float(r.get("decision.index"), 0.0) or 0.0)[:max_steps]
        if not rows:
            continue
        user_row = users.get(user, {})
        vignette = {
            "dataset": spec.id,
            "record_id": f"heartsteps-user-{user}",
            "age": _as_float(user_row.get("age")),
            "gender": user_row.get("gender") or "unknown",
            "education": user_row.get("education") or "unknown",
            "occupation": user_row.get("occupation") or "unknown",
            "selfeff_intake": _as_float(user_row.get("selfeff.intake")),
            "conscientiousness": _as_float(user_row.get("consc")),
            "walk_intake": user_row.get("walk.intake") or "unknown",
            "n_doses": 1,
        }
        parsed_times = [
            _parse_datetime(row.get("sugg.decision.utime"))
            or _parse_datetime(row.get("sugg.select.utime"))
            for row in rows
        ]
        first_dt = next((dt for dt in parsed_times if dt is not None), None)
        same_day_counts: dict[str, int] = {}
        previous_dt: datetime | None = None
        previous_dt_by_action: dict[str, datetime] = {}
        steps: list[ObservedStep] = []
        for idx, row in enumerate(rows, start=1):
            decision_dt = parsed_times[idx - 1]
            local_date = decision_dt.date().isoformat() if decision_dt else None
            if local_date is not None:
                same_day_counts[local_date] = same_day_counts.get(local_date, 0) + 1
            post_steps_raw, post_source = _first_float_with_source(
                row,
                (
                    ("gfsteps30", "google_fit_30m"),
                    ("jbsteps30", "jawbone_30m"),
                    ("gfsteps40", "google_fit_40m"),
                    ("jbsteps40", "jawbone_40m"),
                    ("gfsteps60", "google_fit_60m"),
                    ("jbsteps60", "jawbone_60m"),
                ),
            )
            pre_steps_raw, pre_source = _first_float_with_source(
                row,
                (
                    ("gfsteps30pre", "google_fit_30m_pre"),
                    ("jbsteps30pre", "jawbone_30m_pre"),
                    ("gfsteps40pre", "google_fit_40m_pre"),
                    ("jbsteps40pre", "jawbone_40m_pre"),
                    ("gfsteps60pre", "google_fit_60m_pre"),
                    ("jbsteps60pre", "jawbone_60m_pre"),
                ),
            )
            post_steps = post_steps_raw if post_steps_raw is not None else 0.0
            pre_steps = pre_steps_raw if pre_steps_raw is not None else 0.0
            action = _heartsteps_action(row)
            actual = _clip(post_steps / 1500.0, 0.0, 1.0)
            reward = _clip((post_steps - 0.20 * pre_steps) / 1500.0, 0.0, 1.0)
            minutes_since_previous = _positive_elapsed_minutes(decision_dt, previous_dt)
            minutes_since_same_action = _positive_elapsed_minutes(
                decision_dt,
                previous_dt_by_action.get(action),
            )
            if decision_dt is not None:
                previous_dt = decision_dt
                previous_dt_by_action[action] = decision_dt
            study_day = (
                (decision_dt.date() - first_dt.date()).days + 1
                if decision_dt is not None and first_dt is not None
                else None
            )
            steps.append(
                ObservedStep(
                    day=idx,
                    action=action,
                    observed_action=action,
                    actual=actual,
                    reward=reward,
                    context={
                        "decision_index": _safe_float(row.get("decision.index")),
                        "decision_point_number": idx,
                        "study_day": study_day,
                        "same_day_decision_number": same_day_counts.get(local_date) if local_date else None,
                        "local_decision_time": _clock_time(decision_dt),
                        "clock_hour": (
                            round(decision_dt.hour + decision_dt.minute / 60.0, 2)
                            if decision_dt is not None else None
                        ),
                        "time_of_day": _time_of_day(decision_dt),
                        "slot": row.get("sugg.select.slot") or row.get("sugg.decision.slot"),
                        "minutes_since_previous_decision": minutes_since_previous,
                        "minutes_since_previous_same_action": minutes_since_same_action,
                        "hours_since_phone_use": _as_float(row.get("sugg.device.since")),
                        "prefetched_context": _truthy(row.get("is.prefetch")),
                        "in_transit": _truthy(row.get("intransit")),
                        "recognized_activity": row.get("recognized.activity") or "unknown",
                        "location_category": row.get("dec.location.category") or "unknown",
                        "weather": row.get("dec.weather.condition") or "unknown",
                        "temperature": _as_float(row.get("dec.temperature")),
                        "pre_steps": pre_steps,
                        "interaction_count": _as_float(row.get("interaction.count")),
                    },
                    raw_outcome={
                        "post_steps": post_steps,
                        "pre_steps": pre_steps,
                        "post_steps_source": post_source,
                        "pre_steps_source": pre_source,
                        "target": "normalized_30_min_steps",
                    },
                )
            )
        patients.append(ObservedPatient(record_id=f"heartsteps-user-{user}", vignette=vignette, steps=steps))
    return patients


def _load_hptn067(spec: DatasetSpec, args: argparse.Namespace, root: Path) -> list[ObservedPatient]:
    wi_rows = _read_tab(root / "WI_nofmt.tab")
    ran_rows = _read_tab(root / "RAN_nofmt.tab")
    dem_rows = {str(row.get("uid")): row for row in _read_tab(root / "DEM_nofmt.tab")}
    arm_by_uid = {
        str(row.get("uid")): str(row.get("RANarm"))
        for row in ran_rows
        if str(row.get("RANran")) == "1" and str(row.get("RANarm") or "").strip()
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in wi_rows:
        uid = str(row.get("uid"))
        if uid not in arm_by_uid:
            continue
        visit = _safe_float(row.get("visitno"), 0.0) or 0.0
        if visit <= 8.0:
            continue
        length = _as_float(row.get("WIlength"))
        if length is None or length <= 0:
            continue
        grouped.setdefault(uid, []).append(row)

    max_steps = _resolved_steps(spec, args)
    patients: list[ObservedPatient] = []
    for uid in sorted(grouped, key=_natural_key):
        arm_code = arm_by_uid[uid]
        action = _hptn067_action(arm_code)
        rows = sorted(grouped[uid], key=lambda r: _safe_float(r.get("visitno"), 0.0) or 0.0)[:max_steps]
        dem = dem_rows.get(uid, {})
        # Decode CDISC-standard sex code (1 = male, 2 = female); fall back to
        # raw value if unrecognised. HPTN067 cohort mixes MSM/TGW and
        # cisgender women across the 3 sites, so both values appear.
        sex_raw = str(dem.get("DEMsex") or "").strip()
        sex_decoded = {"1": "male (1)", "2": "female (2)"}.get(sex_raw, sex_raw or "unknown")
        # Student status (1 = student, 2 = non-student) inferred from the
        # cohort distribution; documented as such in the label.
        student_raw = str(dem.get("DEMstdnt") or "").strip()
        student_decoded = {"1": "student (1)", "2": "non-student (2)"}.get(student_raw, student_raw or "unknown")
        vignette = {
            "dataset": spec.id,
            "record_id": f"hptn067-{uid}",
            "age": _as_float(dem.get("DEMage")) or _as_float(dem.get("DEMbthdt")),
            "sex": sex_decoded,
            "study_site_code": dem.get("DEMstid") or "unknown",
            "student_status": student_decoded,
            # employment_code / education_code: codebook integer values (1-3 and
            # 2-9 ranges respectively) whose mapping is not encoded inline.
            # Labels in the profile flag them as opaque graded codes so the LLM
            # treats them as identifiers rather than inventing meaning.
            "employment_code": dem.get("DEMemp") or "unknown",
            "education_code": dem.get("DEMedu") or "unknown",
            "assigned_regimen": action,
            "n_doses": 1,
        }
        steps: list[ObservedStep] = []
        for idx, row in enumerate(rows, start=1):
            length = int(round(_safe_float(row.get("WIlength"), 7.0) or 7.0))
            pills = _hptn067_pill_count(row)
            sex_events = _hptn067_sex_event_count(row)
            expected = _hptn067_expected_pills(action, length, sex_events)
            actual = _clip(pills / max(1, expected), 0.0, 1.0)
            visit = _safe_float(row.get("visitno"), idx) or idx
            steps.append(
                ObservedStep(
                    day=idx,
                    action=action,
                    observed_action=action,
                    actual=actual,
                    reward=actual,
                    context={
                        # LEAKAGE FIX: pill_events, expected_pills_for_regimen, and
                        # no_pill_flag have been moved out of context entirely.
                        #
                        # pill_events is the numerator of the outcome
                        #   (actual = clip(pill_events / expected_pills, 0, 1)).
                        # expected_pills_for_regimen is the denominator.
                        # no_pill_flag (WInopill) directly signals zero pill-taking.
                        # All three would let the model reconstruct the exact outcome
                        # before it is revealed -- outcome leakage.
                        #
                        # sex_events is also withheld from prompt context because
                        # it is measured over the same interview interval and is
                        # part of the expected-pill denominator for non-daily arms.
                        "visitno": visit,
                        "interval_days": length,
                        "wi_type": row.get("WItype"),
                    },
                    raw_outcome={
                        # raw_outcome is logged for evaluation but never injected
                        # into prompts by build_dataset_step_prompt.
                        "pill_events": pills,
                        "expected_pills": expected,
                        "no_pill_flag": row.get("WInopill"),
                        "sex_events": sex_events,
                        "target": "weekly_regimen_coverage",
                    },
                )
            )
        if steps:
            patients.append(ObservedPatient(record_id=f"hptn067-{uid}", vignette=vignette, steps=steps))
    return patients


def _dataset_system_instruction(spec: DatasetSpec, args: argparse.Namespace) -> str:
    reason_line = (
        'Use one reason_code from: "routine", "motivated", "intervention_response", '
        '"forgot", "side_effects", "stress", "disruption", "low_motivation", '
        '"fatigue", "social_support", "no_change".'
        if not args.no_reasoning
        else 'Use "no_change" for reason_code.'
    )
    return (
        "You are an adherence/outcome simulator used for dataset-backed validation. "
        "Predict the observed step outcome from the patient vignette, prior history, "
        "current context, and current intervention. "
        f"Dataset: {spec.name}. Domain: {spec.domain}. Target: {spec.outcome}. "
        f"Scale: {spec.outcome_scale} "
        "Return only a compact JSON object with keys adherence, adherence_certainty, "
        "reason_code, and state_update. adherence must be a float in [0, 1]. "
        "adherence_certainty must be a float in [0, 1]. "
        f"{reason_line} "
        "state_update may be an empty object if no latent state change is needed."
    )


def _build_dataset_prompt(
    *,
    spec: DatasetSpec,
    patient: ObservedPatient,
    obs: ObservedStep,
    action: str,
    history: list[float],
    history_source: str,
    history_window: int | None,
) -> str:
    if history_window is not None:
        shown_history = history[-history_window:]
    else:
        shown_history = history
    hist_summary = _history_summary(shown_history)
    action_text = ACTION_DESCRIPTIONS.get(spec.id, {}).get(action, action)
    return "\n".join(
        [
            f"Dataset: {spec.name}",
            f"Domain: {spec.domain}",
            f"Outcome target: {spec.outcome}",
            f"Outcome scale: {spec.outcome_scale}",
            "",
            "Patient vignette:",
            json.dumps(_compact_dict(patient.vignette), ensure_ascii=True, sort_keys=True),
            "",
            f"Prior {history_source} history ({len(shown_history)} values):",
            json.dumps([round(float(v), 4) for v in shown_history], ensure_ascii=True),
            f"History summary: {json.dumps(hist_summary, ensure_ascii=True, sort_keys=True)}",
            "",
            "Current intervention/action:",
            f"- action_label: {action}",
            f"- description: {action_text}",
            f"- observed_dataset_action: {obs.observed_action or obs.action}",
            "",
            "Current context:",
            json.dumps(_compact_dict(obs.context), ensure_ascii=True, sort_keys=True),
            "",
            "Task:",
            (
                "Predict the current step's adherence-like outcome on the dataset scale. "
                "Do not reveal or use the hidden observed outcome; infer it from the vignette, "
                "history, action, and context."
            ),
            'Output JSON example: {"adherence": 0.62, "adherence_certainty": 0.70, '
            '"reason_code": "intervention_response", "state_update": {}}',
        ]
    )


class MockDatasetEngine:
    """Small deterministic backend for tests and CLI smoke runs."""

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)
        self._calls = 0
        self._prompt_tokens = 0
        self._output_tokens = 0
        self._thinking_tokens = 0

    def sample(self, *, obs: ObservedStep, action: str, k: int, base_seed: int) -> list[StepResponse]:
        responses: list[StepResponse] = []
        for idx in range(k):
            rng = random.Random(base_seed + idx)
            action_shift = {
                "active_suggestion": 0.04,
                "sedentary_break_suggestion": 0.025,
                "daily_regimen": 0.03,
                "time_driven_regimen": 0.015,
                "event_driven_regimen": 0.005,
                "enhanced_feedback_support": 0.035,
            }.get(action, 0.0)
            value = _clip(obs.actual * 0.78 + 0.11 + action_shift + rng.gauss(0.0, 0.04), 0.0, 1.0)
            responses.append(
                StepResponse(
                    adherence=value,
                    adherence_certainty=_clip(0.72 - abs(value - obs.actual) * 0.3, 0.25, 0.95),
                    reason_code="intervention_response" if action_shift > 0 else "routine",
                    state_update={},
                )
            )
            self._calls += 1
            self._prompt_tokens += 220
            self._output_tokens += 35
        return responses

    @property
    def total_calls(self) -> int:
        return self._calls

    @property
    def prompt_token_count(self) -> int:
        return self._prompt_tokens

    @property
    def output_token_count(self) -> int:
        return self._output_tokens

    @property
    def thinking_token_count(self) -> int:
        return self._thinking_tokens


def _simulator_config(args: argparse.Namespace) -> SimulatorConfig:
    budget = int(args.thinking_budget or 0)
    if args.thinking and budget == 0:
        budget = -1
    return SimulatorConfig(
        backend=args.backend,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        ensemble_size=args.ensemble_size,
        thinking_budget=budget,
        parallel=not args.no_parallel,
        no_reasoning=args.no_reasoning,
    )


def _config(spec: DatasetSpec, args: argparse.Namespace, data_mode: str) -> dict[str, Any]:
    return {
        "dataset": spec.id,
        "dataset_name": spec.name,
        "dataset_root": str(_default_dataset_root(spec) if not args.dataset_root else Path(args.dataset_root).resolve()),
        "data_mode": data_mode,
        "mode": args.mode,
        "policy": args.policy,
        "episode_length": args.episode_length,
        "seed": args.seed,
        "patients": args.patients,
        "max_steps": args.max_steps,
        "require_real_data": args.require_real_data,
        "backend": args.backend,
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "ensemble_size": args.ensemble_size,
        "aggregation": args.aggregation,
        "history_window": None if args.full_history else args.history_window,
        "full_history": args.full_history,
        "prefix_format": args.prefix_format,
        "anchor_days": int(getattr(args, "anchor_days", 0) or 0),
        "anchor_prior": str(getattr(args, "anchor_prior", "off") or "off"),
        "prompt_variant": args.prompt_variant,
        "patient_memory": args.patient_memory,
        "outcome_decoder": args.outcome_decoder,
        "prediction_target": str(getattr(args, "prediction_target", "continuous") or "continuous"),
        "stochastic_sampling": not args.no_stochastic_sampling,
    }


def _normalize_args(args: argparse.Namespace) -> None:
    defaults = {
        "dataset_root": None,
        "episode_length": "auto",
        "seed": 42,
        "mode": "real-history",
        "policy": "observed",
        "patients": None,
        "max_steps": 0,
        "require_real_data": True,
        "backend": "nebula",
        "model": "SURF.Qwen3.5 122B A10B NVFP4",
        "temperature": 0.6,
        "top_p": None,
        "ensemble_size": 1,
        "aggregation": "mean",
        "history_window": 7,
        "full_history": False,
        "prefix_format": "raw",
        "anchor_days": 0,
        "anchor_prior": "off",
        "prompt_variant": "dataset-v1",
        "prediction_target": "continuous",
        "patient_memory": "none",
        "outcome_decoder": "none",
        "no_stochastic_sampling": False,
        "no_parallel": False,
        "no_reasoning": False,
        "patient_workers": 1,
        "thinking": False,
        "thinking_budget": 0,
        "show_prompt": False,
        "mock_llm": False,
        "api_key": None,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    if args.prediction_target != "continuous" and args.dataset != "heartsteps":
        raise SystemExit("--prediction-target is currently HeartSteps-only unless set to continuous")
    if args.patients is not None and int(args.patients) < 1:
        raise SystemExit("--patients must be >= 1")
    if args.max_steps is not None and int(args.max_steps) < 0:
        raise SystemExit("--max-steps must be >= 0")


def _default_dataset_root(spec: DatasetSpec) -> Path:
    root_name = "heartsteps_v1" if spec.id == "heartsteps" else spec.id
    return Path(__file__).resolve().parent.parent / "datasets" / root_name


def _peek_table(path: Path) -> dict[str, Any]:
    if path.suffix.lower() not in {".csv", ".tab", ".tsv"}:
        return {}
    delimiter = "\t" if path.suffix.lower() in {".tab", ".tsv"} else ","
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader, [])
            rows = sum(1 for _ in reader)
    except OSError:
        return {}
    return {"columns": header, "n_rows": rows}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def _read_tab(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _resolved_steps(spec: DatasetSpec, args: argparse.Namespace) -> int:
    n_steps = _episode_steps(spec, args.episode_length)
    if args.max_steps and int(args.max_steps) > 0:
        n_steps = min(n_steps, int(args.max_steps))
    return n_steps


def _episode_steps(spec: DatasetSpec, raw: Any) -> int:
    value = str(raw or "auto").strip().lower()
    if value in {"auto", "default"}:
        return spec.default_steps
    if value.endswith(("w", "m")):
        return max(1, int(round(float(value[:-1] or 1))))
    if value.endswith("d"):
        days = int(float(value[:-1] or 1))
        return days * 5 if spec.id == "heartsteps" else days
    try:
        return max(1, int(float(value)))
    except ValueError:
        return spec.default_steps


def _heartsteps_action(row: dict[str, Any]) -> str:
    if not _truthy(row.get("send")):
        return "no_suggestion"
    if _truthy(row.get("send.active")):
        return "active_suggestion"
    if _truthy(row.get("send.sedentary")):
        return "sedentary_break_suggestion"
    return "active_suggestion"


def _hptn067_action(code: str) -> str:
    return {
        "1": "daily_regimen",
        "2": "time_driven_regimen",
        "3": "event_driven_regimen",
    }.get(str(code).strip(), "daily_regimen")


def _hptn067_pill_count(row: dict[str, Any]) -> int:
    return sum(1 for i in range(1, 15) if str(row.get(f"WIp{i}dt") or "").strip())


def _hptn067_sex_event_count(row: dict[str, Any]) -> int:
    count = 0
    for i in range(1, 25):
        if str(row.get(f"WIs{i}dt") or "").strip():
            count += 1
            continue
        if any(str(row.get(f"WIs{i}{suffix}") or "").strip() == "1" for suffix in ("ar", "ai", "or", "oi")):
            count += 1
    return count


def _hptn067_expected_pills(action: str, interval_days: int, sex_events: int) -> int:
    days = max(1, int(interval_days))
    if action == "daily_regimen":
        return days
    if action == "time_driven_regimen":
        return max(1, math.ceil(days * 2 / 7) + sex_events)
    if sex_events > 0:
        return max(1, 2 * sex_events)
    return max(1, math.ceil(days / 7))


def _uses_real_history(mode: str) -> bool:
    return mode == "real-history"


def _n_doses_for_step(spec: DatasetSpec, obs: ObservedStep) -> int:
    if spec.id == "hptn067":
        return max(1, int(obs.raw_outcome.get("expected_pills") or 1))
    return 1


def _reward_for_step(spec: DatasetSpec, action: str, actual: float) -> float:
    cost = 0.0
    if spec.id == "heartsteps" and action != "no_suggestion":
        cost = 0.025
    elif spec.id == "hptn067" and action == "daily_regimen":
        cost = 0.03
    return _clip(actual - cost, 0.0, 1.0)


def _history_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "recent": None, "trend": None}
    recent = values[-3:]
    first = values[: max(1, len(values) // 2)]
    second = values[max(1, len(values) // 2):]
    trend = None
    if first and second:
        trend = _mean(second) - _mean(first)
    return {
        "n": len(values),
        "mean": _mean(values),
        "recent_mean": _mean(recent),
        "min": min(values),
        "max": max(values),
        "trend": trend,
    }


def _action_effect_table(spec: DatasetSpec, records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    baseline = spec.baseline_action
    base_actual = _mean(r["actual"] for r in records if r["action"] == baseline)
    base_sim = _mean(r["simulated"] for r in records if r["action"] == baseline)
    table: dict[str, dict[str, Any]] = {}
    for action in spec.actions:
        actual = _mean(r["actual"] for r in records if r["action"] == action)
        sim = _mean(r["simulated"] for r in records if r["action"] == action)
        table[action] = {
            "n": sum(1 for r in records if r["action"] == action),
            "actual_mean": actual,
            "sim_mean": sim,
            "actual_lift": (actual - base_actual) if actual is not None and base_actual is not None else None,
            "sim_lift": (sim - base_sim) if sim is not None and base_sim is not None else None,
        }
    return table


def _lift_correlation(table: dict[str, dict[str, Any]]) -> float | None:
    actual: list[float] = []
    sim: list[float] = []
    for row in table.values():
        a = _as_float(row.get("actual_lift"))
        s = _as_float(row.get("sim_lift"))
        if a is not None and s is not None:
            actual.append(a)
            sim.append(s)
    return _pearson(actual, sim)


def _simple_ks_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    actual = sorted(v for v in (_as_float(r.get("actual")) for r in records) if v is not None)
    sim = sorted(v for v in (_as_float(r.get("simulated")) for r in records) if v is not None)
    if not actual or not sim:
        return {"mean_p_value": None, "pct_similar": None}
    dist = _ks_distance(actual, sim)
    pseudo_p = math.exp(-2.0 * (dist * math.sqrt(min(len(actual), len(sim)))) ** 2)
    return {
        "ks_distance": dist,
        "mean_p_value": pseudo_p,
        "pct_similar": 100.0 if pseudo_p >= 0.05 else 0.0,
    }


def _per_patient_ks_values(patients: list[dict[str, Any]]) -> list[float]:
    vals: list[float] = []
    for patient in patients:
        actual = sorted(float(v) for v in patient.get("actual_daily") or [])
        sim = sorted(float(s.get("adherence", 0.0)) for s in patient.get("steps") or [])
        if len(actual) >= 2 and len(sim) >= 2:
            dist = _ks_distance(actual, sim)
            pseudo_p = math.exp(-2.0 * (dist * math.sqrt(min(len(actual), len(sim)))) ** 2)
            vals.append(pseudo_p)
    return vals


def _per_patient_ks_pass_rate(patients: list[dict[str, Any]]) -> float | None:
    vals = _per_patient_ks_values(patients)
    if not vals:
        return None
    return 100.0 * sum(1 for v in vals if v >= 0.05) / len(vals)


def _per_patient_ks_mean_p(patients: list[dict[str, Any]]) -> float | None:
    return _mean(_per_patient_ks_values(patients))


def _autocorr_abs_err(patients: list[dict[str, Any]]) -> float | None:
    errs: list[float] = []
    for patient in patients:
        actual = [float(v) for v in patient.get("actual_daily") or []]
        sim = [float(s.get("adherence", 0.0)) for s in patient.get("steps") or []]
        a = _autocorr1(actual)
        s = _autocorr1(sim)
        if a is not None and s is not None:
            errs.append(abs(s - a))
    return _mean(errs)


def _heterogeneity_ratio(actual_rates: list[float], sim_rates: list[float]) -> float | None:
    a = _std(actual_rates)
    s = _std(sim_rates)
    if a is None or s is None or a == 0:
        return None
    return s / a


def _ks_distance(a: list[float], b: list[float]) -> float:
    vals = sorted(set(a + b))
    if not vals:
        return 0.0
    i = j = 0
    best = 0.0
    for v in vals:
        while i < len(a) and a[i] <= v:
            i += 1
        while j < len(b) and b[j] <= v:
            j += 1
        best = max(best, abs(i / len(a) - j / len(b)))
    return best


def _autocorr1(xs: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mean = sum(xs) / len(xs)
    den = sum((x - mean) ** 2 for x in xs)
    if den == 0.0:
        return None
    num = sum((xs[i] - mean) * (xs[i - 1] - mean) for i in range(1, len(xs)))
    return num / den


def _first_float(row: dict[str, Any], names: Iterable[str]) -> float | None:
    for name in names:
        value = _as_float(row.get(name))
        if value is not None:
            return value
    return None


def _safe_float(value: Any, default: float | None = None) -> float | None:
    parsed = _as_float(value)
    return default if parsed is None else parsed


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() in {"", "."}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _natural_key(value: str) -> tuple[int, str]:
    try:
        return (int(value), value)
    except ValueError:
        return (10**9, value)


def _mean(values: Iterable[float | int | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _std(values: Iterable[float | int | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    if len(vals) < 2:
        return None
    return statistics.stdev(vals)


def _pearson(a: list[float], b: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0.0 or deny == 0.0:
        return None
    return num / (denx * deny)


def _spearman(a: list[float], b: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    return _pearson(_ranks(xs), _ranks(ys))


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _compact_dict(raw: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in raw.items() if v not in (None, "", [], {})}


def _print_validation_summary(result: dict[str, Any]) -> None:
    summary = result.get("summary", {})
    print("\n=== Dataset Validation Summary ===", flush=True)
    print(f"Dataset: {summary.get('dataset_name')} ({summary.get('dataset')})", flush=True)
    print(f"Patients: {summary.get('n_patients')}  Decision points: {summary.get('n_decision_points')}", flush=True)
    print(
        "Mean actual={:.1%}  Mean sim={:.1%}  MAE={:.3f}  RMSE={:.3f}".format(
            summary.get("mean_actual") or 0.0,
            summary.get("mean_simulated") or 0.0,
            summary.get("mae") or 0.0,
            summary.get("rmse") or 0.0,
        ),
        flush=True,
    )
    print(
        f"Patient corr={summary.get('correlation')}  "
        f"Day corr={summary.get('correlation_all_days')}  "
        f"LLM calls={summary.get('total_llm_calls')}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_dataset_experiment(args)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
