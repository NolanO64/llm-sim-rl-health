"""Prompt framework for the datasets used in the thesis experiments.

- Skeptical / calibration-focused system instructions with no-leakage guard.
- Literature-grounded priors baked into the prompt.
- Vignette section + execution-capacity-style framing summary.
- Observation-window warm-start prefix with immutable-guard and prefix-format variants.
- Strict context whitelist: only pre-decision fields safe to show.
- Per-dataset reason codes appropriate to the domain.

Extension point: add a new DatasetPromptProfile instance for each dataset.
The profile is the ONLY thing a new dataset must provide; all builder functions
are dataset-agnostic.

Currently defined profiles:
  HEARTSTEPS_PROFILE  -- HeartSteps V1 (physical-activity JITAI), the main testbed
  HPTN067_PROFILE     -- HPTN 067 (PrEP adherence), used for the cross-dataset check
  STEPCOUNTJITAI_PROFILE -- synthetic StepCountJITAI benchmark

To implement a new dataset, define a profile with all fields filled in and
register it in DATASET_PROFILES.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# DatasetPromptProfile -- the ONLY thing a new dataset must provide
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetPromptProfile:
    """All dataset-specific prompt configuration.

    Fields
    ------
    dataset_id : str
        Matches DatasetSpec.id in validate_dataset.py.
    domain_label : str
        Human-readable domain (e.g. "physical-activity JITAI").
    outcome_name : str
        Short name for the target quantity (e.g. "proximal activity response").
    outcome_definition : str
        One sentence defining what the outcome measures.
    outcome_scale_description : str
        Sentence describing the 0-1 scale and what endpoints mean.
    literature_priors : tuple[str, ...]
        Grounded bullet strings (cite real papers) anchoring base rates.
    reason_codes : tuple[str, ...]
        Domain-appropriate reason codes; must NOT include medication-specific
        codes like "side_effects" or "forgot a dose" for non-medication domains.
    vignette_field_labels : tuple[tuple[str, str], ...]
        Ordered (key, human_label) pairs for patient profile rendering.
        Keys that are absent from the vignette dict are silently skipped.
    context_field_labels : tuple[tuple[str, str], ...]
        WHITELIST of (key, human_label) pairs for per-step context.
        ONLY pre-decision fields known BEFORE the outcome may appear here.
        Anything that encodes or proxies the outcome must be excluded.
    action_descriptions : tuple[tuple[str, str], ...]
        (action_id, description) for rendering the current intervention.
    baseline_action : str
        The no-intervention action id (matches DatasetSpec.baseline_action).
    """

    dataset_id: str
    domain_label: str
    outcome_name: str
    outcome_definition: str
    outcome_scale_description: str
    literature_priors: tuple[str, ...]
    reason_codes: tuple[str, ...]
    vignette_field_labels: tuple[tuple[str, str], ...]
    context_field_labels: tuple[tuple[str, str], ...]
    action_descriptions: tuple[tuple[str, str], ...]
    baseline_action: str


DATASET_PROMPT_VARIANT_DEFAULT = "dataset-v1"
HEARTSTEPS_PROMPT_VARIANTS = (
    DATASET_PROMPT_VARIANT_DEFAULT,
    "heartsteps-zero-inflated",
    "heartsteps-conservative-action",
    "heartsteps-soft-zero-calibrated",
    "heartsteps-mean-preserving-mixture",
    "heartsteps-hurdle-calibrated",
    "heartsteps-context-faithful",
    "heartsteps-human-noise",
    "heartsteps-structured-forecast",
    "heartsteps-persona-simple",
    "heartsteps-persona-measured",
    "heartsteps-persona-small-steps",
    "heartsteps-persona-rhythm",
    "heartsteps-persona-balanced",
    "heartsteps-stochastic-realized",
    "heartsteps-stochastic-bursty",
    # Context ablations: each drops one context component from the
    # default prompt to test which parts help/hurt C2ST. Model held fixed.
    "heartsteps-ablate-no-domain",
    "heartsteps-ablate-no-persona",
    "heartsteps-ablate-no-weather",
    "heartsteps-ablate-no-intervention",
    "heartsteps-ablate-structured-only",
    "heartsteps-ablate-history-only",
)

# Context-ablation variants -> which prompt sections to drop. Components:
#   framing/literature = dataset-domain explanation; vignette/capacity = persona/
#   profile; context = current weather/location; intervention_text = the action's
#   descriptive message (bare action id kept); guidance = history-anchoring prose.
# "simple first-person" is already covered by the existing heartsteps-persona-simple.
HEARTSTEPS_ABLATION_DROP = {
    "heartsteps-ablate-no-domain": {"framing", "literature"},
    "heartsteps-ablate-no-persona": {"vignette", "capacity"},
    "heartsteps-ablate-no-weather": {"context"},
    "heartsteps-ablate-no-intervention": {"intervention_text"},
    "heartsteps-ablate-structured-only": {"framing", "literature", "capacity", "guidance"},
    "heartsteps-ablate-history-only": {
        "framing", "vignette", "capacity", "literature", "context", "guidance",
    },
}

HEARTSTEPS_PERSONA_VARIANTS = {
    "heartsteps-persona-simple",
    "heartsteps-persona-measured",
    "heartsteps-persona-small-steps",
    "heartsteps-persona-rhythm",
    "heartsteps-persona-balanced",
}

HEARTSTEPS_STOCHASTIC_CUE_VARIANTS = {
    "heartsteps-stochastic-realized",
    "heartsteps-stochastic-bursty",
}


def normalize_dataset_prompt_variant(
    profile: DatasetPromptProfile,
    prompt_variant: str | None = None,
) -> str:
    """Normalize and validate dataset prompt variants.

    Dataset prompts default to dataset-v1 for all datasets. HeartSteps gets
    extra variants because its outcome is a zero-inflated continuous activity
    response and the current baseline prompt smooths too aggressively.
    """
    variant = str(prompt_variant or DATASET_PROMPT_VARIANT_DEFAULT).strip()
    if not variant:
        variant = DATASET_PROMPT_VARIANT_DEFAULT
    if profile.dataset_id == "heartsteps":
        if variant not in HEARTSTEPS_PROMPT_VARIANTS:
            raise ValueError(
                "HeartSteps prompt_variant must be one of "
                f"{', '.join(HEARTSTEPS_PROMPT_VARIANTS)}, got {variant!r}"
            )
        return variant
    if variant != DATASET_PROMPT_VARIANT_DEFAULT:
        raise ValueError(
            f"{profile.dataset_id} prompt_variant must be "
            f"{DATASET_PROMPT_VARIANT_DEFAULT!r}, got {variant!r}"
        )
    return variant


# ---------------------------------------------------------------------------
# HeartSteps V1 -- fully specified profile
# ---------------------------------------------------------------------------

HEARTSTEPS_PROFILE = DatasetPromptProfile(
    dataset_id="heartsteps",
    domain_label="physical-activity just-in-time adaptive intervention (JITAI)",
    outcome_name="proximal 30-minute step-count response",
    outcome_definition=(
        "The number of steps taken in the 30 minutes immediately AFTER a "
        "phone decision point, normalized so that 1500 steps maps to 1.0 "
        "and 0 steps maps to 0.0. The current loader uses Google Fit phone "
        "steps when present and falls back to Jawbone tracker steps when the "
        "Google Fit aggregate is missing."
    ),
    outcome_scale_description=(
        "0.0 = no detectable walking response in the 30-minute window; "
        "1.0 approximates 1500 or more steps in that window (brisk ~15-minute walk)."
    ),
    # NOTE: priors are GENERAL behavioural-science domain knowledge only. They must
    # NOT state this dataset's measured outcome distribution, per-arm effects, or
    # any result obtained from the trial itself -- that would leak the answer into
    # a simulator we are validating against this very dataset.
    literature_priors=(
        "Nahum-Shani I et al. (2018), 'Just-in-Time Adaptive Interventions (JITAIs) "
        "in Mobile Health,' Annals of Behavioral Medicine 52(6):446-462. A JITAI "
        "prompt aims to reach a person at a moment of opportunity, but the proximal "
        "effect of any single prompt on near-term behaviour is generally small and "
        "highly variable. Whether a prompt produces activity depends far more on the "
        "person's momentary situation than on the prompt itself.",

        "In everyday life most short (30-minute) windows contain little or no "
        "walking; sustained activity bursts are the exception. A decision point "
        "should not be assumed to produce meaningful activity merely because a "
        "suggestion was delivered.",

        "Context dominates the proximal activity response. Being in transit, "
        "indoors, in a sedentary workplace, or facing cold or inclement weather "
        "strongly suppresses any walking response even when a prompt is sent. "
        "Activity is bursty and time-of-day dependent: morning and early-evening "
        "windows tend to see more walking than midday windows.",

        "Stable traits such as exercise self-efficacy and conscientiousness shift a "
        "person's overall activity tendency but do not reliably determine the "
        "outcome of any single decision-point window, which is driven mostly by "
        "momentary context.",

        "Activity-suggestion interventions, including personalized ones, change "
        "behaviour only modestly on average. Do not assume that sending a suggestion "
        "causes a large activity increase at a given decision point; treat the "
        "prompt as a weak nudge whose effect is frequently absent.",
    ),
    reason_codes=(
        "context_favorable",
        "suggestion_response",
        "baseline_activity",
        "context_unfavorable",
        "in_transit",
        "sedentary_setting",
        "weather_barrier",
        "low_engagement",
        "time_slot_effect",
        "fatigue",
        "habit_walk",
        "no_change",
    ),
    vignette_field_labels=(
        ("record_id", "Participant ID"),
        ("age", "Age"),
        ("gender", "Gender"),
        ("education", "Education level"),
        ("occupation", "Occupation"),
        ("selfeff_intake", "Exercise self-efficacy at intake (higher = more confident)"),
        ("conscientiousness", "Conscientiousness (personality trait, higher = more disciplined)"),
        ("walk_intake", "Reported walking at intake"),
        # totaldays (study days completed) intentionally excluded: it is only
        # known at study end and leaks retention/dropout for this participant.
    ),
    # WHITELIST: only pre-decision fields safe to expose.
    # EXCLUDED (would leak outcome): post_steps, raw_outcome fields.
    # pre_steps is the step count in the window BEFORE the decision -- safe.
    context_field_labels=(
        ("decision_point_number", "Decision point number in this participant's available trajectory"),
        ("study_day", "Study day (1 = first available HeartSteps decision day for this participant)"),
        ("same_day_decision_number", "Decision point number within this local day"),
        ("local_decision_time", "Local decision clock time"),
        ("time_of_day", "Time-of-day category"),
        ("slot", "Decision-point time slot (1=morning ... 5=evening)"),
        ("minutes_since_previous_decision", "Elapsed minutes since previous available decision point"),
        ("minutes_since_previous_same_action", "Elapsed minutes since this participant last received the same action"),
        ("hours_since_phone_use", "Hours since most recent phone use before the decision"),
        ("prefetched_context", "Decision context was prefetched 30 minutes before the decision"),
        ("in_transit", "In transit at decision time (true = strong barrier)"),
        ("recognized_activity", "Activity recognized by phone at decision time"),
        ("location_category", "Location category at decision time"),
        ("weather", "Weather condition at decision time"),
        ("temperature", "Temperature at decision time (Celsius)"),
        ("pre_steps", "Steps in the 30 minutes BEFORE this decision point"),
        ("interaction_count", "Number of prior app interactions (engagement proxy)"),
    ),
    action_descriptions=(
        ("no_suggestion", "No activity suggestion is sent at this decision point."),
        ("active_suggestion", "An active walking/activity suggestion is sent via phone."),
        (
            "sedentary_break_suggestion",
            "A sedentary-break suggestion is sent (prompt to stand or move briefly).",
        ),
    ),
    baseline_action="no_suggestion",
)


# ---------------------------------------------------------------------------
# HPTN 067 / ADAPT -- fully specified profile
#
# HPTN 067 (the ADAPT study) randomized cisgender women (Bangkok, Harlem,
# Cape Town sites) to three oral PrEP dosing schedules:
#   Arm 1 -- daily dosing (one tablet every day)
#   Arm 2 -- time-driven (twice weekly + post-sex dose)
#   Arm 3 -- event-driven (two pills around each sex event)
#
# The outcome is weekly regimen coverage: the fraction of expected regimen
# pills that the participant actually took, estimated from pill-event
# interview records collected roughly weekly.
#
# LEAKAGE FIX (applied to _load_hptn067 in validate_dataset.py):
#   pill_events, expected_pills_for_regimen, and no_pill_flag have been
#   removed from ObservedStep.context. All three fields directly encode the
#   outcome (actual = clip(pill_events / expected_pills, 0, 1); no_pill_flag
#   signals zero pill-taking). They now live only in raw_outcome, which is
#   logged for evaluation but never injected into prompts.
#   sex_events is also withheld from context: it is measured over the same
#   interview interval and participates in the expected-pill denominator for
#   non-daily arms. The whitelist below provides a second layer of defence
#   against any future re-introduction of leaking fields.
# ---------------------------------------------------------------------------

HPTN067_PROFILE = DatasetPromptProfile(
    dataset_id="hptn067",
    domain_label="PrEP adherence (oral HIV pre-exposure prophylaxis, HPTN 067 / ADAPT)",
    outcome_name="weekly regimen coverage",
    outcome_definition=(
        "Fraction of the expected regimen pills actually taken during the "
        "interview interval, estimated from pill-event records collected at "
        "weekly adherence interviews. For daily arms, expected pills equals "
        "interval days; for time-driven arms, expected pills is "
        "twice-per-week plus post-sex doses; for event-driven arms, expected "
        "pills is two doses per sex event (or at minimum one weekly dose)."
    ),
    outcome_scale_description=(
        "0.0 = no pills taken in the interval (zero regimen coverage); "
        "1.0 = full regimen coverage -- every expected dose taken. "
        "Daily-arm participants face a higher absolute pill burden, so "
        "achieving 1.0 is harder than in event-driven arms where few sex "
        "events yield a low expected count. Values above 0.8 indicate good "
        "adherence; values below 0.4 indicate high regimen non-adherence."
    ),
    # NOTE: priors are GENERAL PrEP behavioural-science domain knowledge only.
    # They must NOT state this dataset's measured outcome distribution, its
    # per-arm results, or any figure obtained from the trial itself -- doing so
    # would leak the answer into a simulator validated against this dataset.
    literature_priors=(
        "Haberer JE (2016), 'Current concepts for PrEP adherence in the PrEP "
        "revolution,' Current Opinion in HIV and AIDS 11(1):10-17. Oral PrEP "
        "adherence is highly variable between people and over time within a "
        "person. It tends to be higher near enrollment and to decline as novelty "
        "fades, with intermittent rather than stable day-to-day patterns.",

        "Perceived HIV risk is a primary driver of PrEP dosing: intervals with no "
        "anticipated or recent sexual exposure are commonly associated with "
        "reduced pill-taking in every regimen, while an anticipated exposure can "
        "transiently raise adherence.",

        "Non-daily regimens (time-driven, event-driven) are cognitively harder "
        "than daily dosing: they require the person to anticipate exposures and "
        "correctly link doses to events, so regimen confusion and mistimed doses "
        "are common. Daily dosing is simpler to execute but carries a higher "
        "absolute pill burden.",

        "Side effects of oral tenofovir/emtricitabine (GI upset, nausea, "
        "headache) concentrate in the first few weeks and are a common early "
        "reason for missed doses or discontinuation. Stigma around carrying or "
        "being seen taking pills, and travel or schedule disruption, are "
        "recurring real-world barriers.",

        "Self-reported pill-use measures are known to overstate true adherence "
        "relative to objective drug-level measures. An interview-based coverage "
        "figure should be treated as an upper bound on real dosing, not a precise "
        "value.",
    ),
    reason_codes=(
        "routine_dosing",        # habitual, consistent pill-taking without special trigger
        "regimen_confusion",     # participant unclear on when/how many pills to take
        "anticipated_exposure",  # upcoming or recent sex event motivates dosing (event-driven)
        "no_perceived_risk",     # no sex events anticipated, dosing deprioritized
        "side_effects",          # GI upset, nausea, headache suppressing pill-taking
        "travel_disruption",     # travel or schedule disruption caused missed doses
        "pill_fatigue",          # fatigue with daily routine, declining adherence over time
        "stigma_concern",        # concern about carrying or taking pills in front of others
        "ran_out",               # supply gap -- pills unavailable for some of interval
        "study_engagement",      # heightened adherence near clinic visits or interviews
        "low_motivation",        # low perceived benefit or HIV risk perception reducing effort
        "no_change",             # no identifiable change driver; continuation of prior pattern
    ),
    # Vignette field labels explain each column. Sex is decoded in the loader
    # via the CDISC convention (1 = male, 2 = female); HPTN067 enrolled
    # MSM/TGW at two sites plus cisgender women at the third site, so both
    # values appear. Student status is decoded the same way (1 = student,
    # 2 = non-student). employment_code and education_code are exposed as
    # raw codebook integers because the HPTN067 codebook is not embedded in
    # the prompt; the labels flag them as graded codes so the model treats
    # them as identifiers rather than inventing precise meanings.
    vignette_field_labels=(
        ("record_id", "Participant ID"),
        ("age", "Age at enrollment (years)"),
        ("sex", "Recorded sex (HPTN067 used CDISC codes 1=male, 2=female; the cohort mixes MSM/TGW and cisgender women across the three sites)"),
        ("study_site_code", "Study site letter code (HPTN067 ran three sites - Bangkok, Harlem, Cape Town - each with a different enrolled population; the exact A/B/C-to-city mapping is in the HPTN067 codebook and not encoded here)"),
        ("student_status", "Student status at enrollment (1=student, 2=non-student per the HPTN067 DEM convention)"),
        ("employment_code", "Employment-status code (HPTN067 DEM file; graded integer code 1-3, higher values indicating different employment categories per the HPTN067 codebook)"),
        ("education_code", "Highest education attained (HPTN067 DEM file; graded integer code 2-9 where higher generally means more schooling, per the HPTN067 codebook)"),
        ("assigned_regimen", "Randomly assigned PrEP dosing regimen"),
    ),
    # WHITELIST: only pre-outcome fields safe to expose.
    # EXCLUDED (leaks outcome):
    #   pill_events             -- numerator of outcome formula
    #   expected_pills_for_regimen -- denominator of outcome formula
    #   no_pill_flag (WInopill) -- directly signals zero pill-taking = outcome 0
    # All three were removed from the loader's context dict; this whitelist
    # provides defense-in-depth in case they are ever re-added.
    # sex_events is also excluded: it is measured over the same interview
    # interval and participates in the expected-pill denominator for non-daily
    # regimens, so it is not a safe pre-outcome prompt field.
    context_field_labels=(
        ("visitno", "Visit number (proxy for time in study / engagement decay)"),
        ("interval_days", "interval_days: days covered by the current interview interval"),
        ("wi_type", "Interview type code (scheduled vs unscheduled)"),
        # pill_events, expected_pills_for_regimen, no_pill_flag, sex_events intentionally absent.
    ),
    action_descriptions=(
        ("daily_regimen", "Daily oral PrEP: one tablet every day regardless of sex activity."),
        (
            "time_driven_regimen",
            "Time-driven oral PrEP: approximately two fixed doses per week plus an "
            "additional dose after sex events.",
        ),
        (
            "event_driven_regimen",
            "Event-driven oral PrEP: two doses around each sex event (one pre-exposure, "
            "one post-exposure), with no required dosing on non-sex days.",
        ),
    ),
    baseline_action="daily_regimen",
)


# ---------------------------------------------------------------------------
# Profile registry -- the single lookup used by validate_dataset.py
# ---------------------------------------------------------------------------

STEPCOUNTJITAI_PROFILE = DatasetPromptProfile(
    dataset_id="stepcountjitai",
    domain_label="physical-activity JITAI (synthetic StepCountJITAI benchmark)",
    outcome_name="proximal step-count response",
    outcome_definition=(
        "Step-count response in the short window following a decision point, "
        "for a person using an activity-coaching app that may send a suggestion."
    ),
    outcome_scale_description=(
        "0.0 = no proximal activity response after the decision point; "
        "1.0 = a strong proximal response. The value reflects how much the "
        "person moved following the (possibly absent) suggestion."
    ),
    literature_priors=(
        "Just-in-time adaptive interventions (JITAIs) aim to deliver the right "
        "support at the right moment; a suggestion that matches the person's "
        "current context tends to help more than a generic one, and an "
        "ill-timed or mismatched prompt may do little.",
        "Responsiveness to repeated prompts is not constant: people can "
        "habituate when prompted too often, so the same suggestion may produce "
        "a smaller response after many recent prompts than when it is fresh.",
        "Activity responses are noisy and vary within a person over time; "
        "many decision points produce little or no measurable proximal "
        "activity even when a suggestion is sent.",
    ),
    reason_codes=(
        "responded_to_suggestion",
        "context_match",
        "context_mismatch",
        "habituated",
        "disengaged",
        "no_suggestion_sent",
        "spontaneous_activity",
        "no_change",
    ),
    vignette_field_labels=(
        ("record_id", "Participant ID (synthetic)"),
    ),
    context_field_labels=(
        ("decision_point_number", "Decision point number within the trajectory"),
        ("context_label", "Inferred current context for this decision point"),
        ("context_confidence", "Confidence (0-1) that the inferred context is correct"),
    ),
    action_descriptions=(
        ("no_intervention", "No activity suggestion is sent at this decision point."),
        ("generic_suggestion", "A generic activity suggestion not tailored to the current context."),
        ("context_a_suggestion", "An activity suggestion tailored to context A."),
        ("context_b_suggestion", "An activity suggestion tailored to context B."),
    ),
    baseline_action="no_intervention",
)


DATASET_PROFILES: dict[str, DatasetPromptProfile] = {
    "heartsteps": HEARTSTEPS_PROFILE,
    "hptn067": HPTN067_PROFILE,
    "stepcountjitai": STEPCOUNTJITAI_PROFILE,
}


# ---------------------------------------------------------------------------
# Prefix format constants (mirrors config.PREFIX_FORMATS for dataset path)
#   raw           : day/step-by-step observed outcomes, no action info
#   raw-actions   : same, but each step also shows the action that was taken
#   summary       : prior steps collapsed into a stat summary
#   hybrid        : early steps as summary + most-recent steps as raw-actions
#   pattern-typed : summary + a categorical trajectory tag
# ---------------------------------------------------------------------------

DATASET_PREFIX_FORMATS: tuple[str, ...] = (
    "raw",
    "raw-actions",
    "summary",
    "hybrid",
    "pattern-typed",
)

# Number of most-recent steps rendered verbatim in the "hybrid" format.
_HYBRID_RECENT_STEPS = 5


# ---------------------------------------------------------------------------
# Pattern classifier for continuous 0..1 outcomes
# ---------------------------------------------------------------------------

def _classify_continuous_pattern(values: list[float]) -> str:
    """Categorical trajectory tag for a continuous 0..1 outcome sequence."""
    if not values:
        return "no history"
    n = len(values)
    mean_val = sum(values) / n
    zeros = sum(1 for v in values if v <= 0.05)
    highs = sum(1 for v in values if v >= 0.80)

    if zeros == n:
        return "all-zero"
    if highs == n:
        return "all-high"
    if mean_val >= 0.75:
        return "stable-high"
    if mean_val <= 0.15:
        return "near-zero"

    if n >= 4:
        half = n // 2
        first_mean = sum(values[:half]) / half
        second_mean = sum(values[half:]) / (n - half)
        delta = second_mean - first_mean
        if delta < -0.20:
            return "declining"
        if delta > 0.20:
            return "improving"

    if zeros > 0 and highs > 0:
        return "intermittent"
    return "moderate"


# ---------------------------------------------------------------------------
# Prefix-block renderers
# ---------------------------------------------------------------------------

def _format_history_timing(context: dict[str, Any] | None) -> str:
    """Compact, pre-outcome timing/context tag for verbatim history lines."""
    if not context:
        return ""
    parts: list[str] = []
    if context.get("study_day") is not None:
        parts.append(f"study_day={context['study_day']}")
    if context.get("same_day_decision_number") is not None:
        parts.append(f"same_day_point={context['same_day_decision_number']}")
    if context.get("local_decision_time") not in (None, ""):
        parts.append(f"time={context['local_decision_time']}")
    if context.get("slot") not in (None, ""):
        parts.append(f"slot={context['slot']}")
    gap = context.get("minutes_since_previous_decision")
    if gap is not None:
        parts.append(f"gap={_format_minutes(float(gap))}")
    pre_steps = context.get("pre_steps")
    if pre_steps is not None:
        try:
            parts.append(f"pre_steps={float(pre_steps):.0f}")
        except (TypeError, ValueError):
            parts.append(f"pre_steps={pre_steps}")
    return f" [{', '.join(parts)}]" if parts else ""


def _format_minutes(minutes: float) -> str:
    if minutes >= 24 * 60:
        return f"{minutes / (24 * 60):.1f}d"
    if minutes >= 60:
        return f"{minutes / 60:.1f}h"
    return f"{minutes:.0f}min"


def _render_prefix_raw(
    values: list[float],
    *,
    start_index: int = 1,
    contexts: list[dict[str, Any]] | None = None,
) -> str:
    """Step-by-step observed outcome listing (no action info)."""
    lines = []
    for i, v in enumerate(values, start=start_index):
        context = contexts[i - start_index] if contexts and i - start_index < len(contexts) else None
        lines.append(f"  Step {i}{_format_history_timing(context)}: outcome={v:.3f}")
    return "\n".join(lines)


def _render_prefix_raw_actions(
    values: list[float],
    actions: list[str] | None,
    *,
    start_index: int = 1,
    contexts: list[dict[str, Any]] | None = None,
) -> str:
    """Step-by-step listing with the action taken at each prior step.

    Falls back to plain raw rendering when no actions are available.
    """
    if not actions or len(actions) < len(values):
        return _render_prefix_raw(values, start_index=start_index, contexts=contexts)
    lines = []
    for i, (v, a) in enumerate(zip(values, actions), start=start_index):
        context = contexts[i - start_index] if contexts and i - start_index < len(contexts) else None
        lines.append(f"  Step {i}{_format_history_timing(context)}: action={a} -> outcome={v:.3f}")
    return "\n".join(lines)


def _render_prefix_summary(values: list[float], *, with_pattern: bool) -> str:
    """Collapsed statistics for the observed prefix."""
    if not values:
        return ""
    n = len(values)
    mean_val = sum(values) / n
    low = sum(1 for v in values if v <= 0.10)
    mid = sum(1 for v in values if 0.10 < v < 0.70)
    high = sum(1 for v in values if v >= 0.70)

    streak = 0
    cur = 0
    for v in values:
        if v <= 0.10:
            cur += 1
            streak = max(streak, cur)
        else:
            cur = 0

    last_k = values[-min(5, n):]
    last_k_mean = sum(last_k) / len(last_k)

    lines = [
        f"Observed prefix summary ({n} step(s)):",
        f"- Overall mean outcome: {mean_val:.3f}",
        f"- Distribution: {low} near-zero, {mid} mid-range, {high} high-outcome step(s)",
        f"- Longest near-zero streak: {streak} step(s)",
        f"- Recent {len(last_k)}-step mean: {last_k_mean:.3f}",
    ]
    if with_pattern:
        lines.append(f"- Pattern type: {_classify_continuous_pattern(values)}")
    return "\n".join(lines)


def _build_prefix_block(
    values: list[float],
    prefix_format: str,
    actions: list[str] | None = None,
    contexts: list[dict[str, Any]] | None = None,
    *,
    start_index: int = 1,
) -> str:
    """Render the observed prefix according to ``prefix_format``.

    ``start_index`` controls the numbering of the verbatim per-step lines
    used by the raw / raw-actions / hybrid formats. The aggregate formats
    (summary / pattern-typed) ignore it.
    """
    fmt = prefix_format if prefix_format in DATASET_PREFIX_FORMATS else "raw"
    if fmt == "raw":
        return _render_prefix_raw(values, start_index=start_index, contexts=contexts)
    if fmt == "raw-actions":
        return _render_prefix_raw_actions(
            values, actions, start_index=start_index, contexts=contexts,
        )
    if fmt == "pattern-typed":
        return _render_prefix_summary(values, with_pattern=True)
    if fmt == "hybrid":
        n = len(values)
        if n <= _HYBRID_RECENT_STEPS:
            return _render_prefix_raw_actions(values, actions, start_index=start_index)
        split = n - _HYBRID_RECENT_STEPS
        early_summary = _render_prefix_summary(values[:split], with_pattern=False)
        recent_actions = actions[split:] if actions else None
        recent_contexts = contexts[split:] if contexts else None
        recent_block = _render_prefix_raw_actions(
            values[split:],
            recent_actions,
            start_index=start_index + split,
            contexts=recent_contexts,
        )
        return (
            f"{early_summary}\n"
            f"Most recent {_HYBRID_RECENT_STEPS} step(s), verbatim:\n"
            f"{recent_block}"
        )
    # summary
    return _render_prefix_summary(values, with_pattern=False)


# ---------------------------------------------------------------------------
# Public builder functions
# ---------------------------------------------------------------------------

def build_dataset_system_instruction(
    profile: DatasetPromptProfile,
    *,
    prompt_variant: str | None = None,
    include_reasoning: bool = True,
) -> str:
    """Return a skeptical, calibration-focused system instruction.

    Parameters
    ----------
    profile:
        The dataset's DatasetPromptProfile.
    include_reasoning:
        If False, keeps reason-code semantics disabled for the compact schema.
    """
    variant = normalize_dataset_prompt_variant(profile, prompt_variant)
    # Context-ablation variants share the context-faithful system instruction;
    # they differ from baseline only in the dropped user-prompt component.
    if variant in HEARTSTEPS_ABLATION_DROP:
        variant = "heartsteps-context-faithful"
    if profile.dataset_id == "heartsteps" and variant in HEARTSTEPS_PERSONA_VARIANTS:
        return _build_heartsteps_persona_system_instruction(
            profile,
            prompt_variant=variant,
            include_reasoning=include_reasoning,
        )
    if include_reasoning:
        reason_codes_str = ", ".join(f'"{r}"' for r in profile.reason_codes)
        reason_line = f"Use one reason_code from: {reason_codes_str}."
    else:
        reason_line = (
            'Compact no-reasoning mode omits reason_code from the output schema; '
            'if a backend returns one anyway, use "no_change".'
        )

    # Calibration-focused, execution-first forecaster
    # that treats friction as strong evidence, traits/intentions as weak
    # evidence, expects gradual change rather than flips, preserves observed
    # intermittency, and uses intermediate values + lower certainty under
    # mixed evidence.
    variant_instruction = _build_system_variant_instruction(profile, variant)
    variant_block = f"\n\n{variant_instruction}" if variant_instruction else ""

    return (
        f"You are a calibration-focused behavioral forecaster predicting one "
        f"step of {profile.domain_label} for dataset-backed validation. "
        f"Estimate what an external monitor would record for this decision "
        f"point, given the participant's profile, prior outcome history, "
        f"current context, and current intervention.\n\n"
        "The outcome is execution behaviour, not intent. Concrete contextual "
        "friction (poor location, bad weather, transit, side effects, fatigue, "
        "disruption, low engagement) outweighs positive-sounding traits, study "
        "participation, or a supportive intervention. Behaviour change is "
        "usually gradual; a single intervention rarely flips a low-outcome "
        "step into a high-outcome one.\n\n"
        "If the visible history shows mixed outcomes (both high and low "
        "values), preserve that intermittency in your prediction; do not "
        "collapse it into deterministic 0.0 or 1.0. When evidence is mixed, "
        "prefer intermediate outcome values and lower adherence_certainty. "
        "Reserve high adherence_certainty for cases where prior history, "
        "current context, and the current intervention all point the same way.\n\n"
        "Predict what an external observer would record -- NOT the ideal "
        "behaviour, NOT what the participant intended, NOT the most socially "
        "desirable answer.\n\n"
        "Leakage guard: the prompt will NOT show you the actual outcome for "
        "the current step. Never fabricate or guess the hidden observed value "
        "directly. Infer the likely outcome from the pre-decision context, "
        "prior history, and intervention only.\n\n"
        f"Domain: {profile.domain_label}\n"
        f"Outcome: {profile.outcome_name}\n"
        f"Definition: {profile.outcome_definition}\n"
        f"Scale: {profile.outcome_scale_description}"
        f"{variant_block}\n\n"
        "Return strict JSON only, no markdown fences, no explanation. "
        "Keys: adherence (float 0-1), adherence_certainty (float 0-1), "
        f"{'reason_code, ' if include_reasoning else ''}state_update (object). "
        f"{reason_line}"
    )


def build_dataset_step_prompt(
    profile: DatasetPromptProfile,
    patient: Any,  # ObservedPatient from validate_dataset
    obs: Any,       # ObservedStep from validate_dataset
    action: str,
    observation_history: list[float],
    simulated_history: list[float],
    *,
    prefix_format: str = "raw",
    history_window: int | None = 7,
    history_source: str = "observed",
    history_actions: list[str] | None = None,
    history_contexts: list[dict[str, Any]] | None = None,
    anchor_prior_stats: dict | None = None,
    anchor_days: int = 0,
    prompt_variant: str | None = None,
    patient_memory: str | None = None,
    include_reasoning: bool = True,
) -> str:
    """Assemble the full user prompt for one simulation step.

    Sections (in order):
    1. Dataset / outcome framing
    2. Participant profile (vignette) with ordered human labels
    3. Participant activity framing summary (execution-capacity analogue)
    4. Literature priors (grounded base-rate anchors)
    5. Output schema with domain-appropriate reason codes
    6. Current pre-decision context (WHITELISTED fields only -- no leakage)
    7. Observation-window warm-start prefix (real measured history) with
       immutable guard and prefix-format rendering
    8. Recent simulated history (if autoregressive mode)
    9. Current intervention

    Parameters
    ----------
    profile:
        The dataset's DatasetPromptProfile.
    patient:
        ObservedPatient instance (from validate_dataset).
    obs:
        ObservedStep for the current decision point.
    action:
        Resolved action label (may differ from obs.action under policy override).
    observation_history:
        Real observed outcomes from prior steps (used in real-history mode).
    simulated_history:
        LLM-simulated outcomes from prior steps (used in autoregressive mode).
    prefix_format:
        One of "raw", "raw-actions", "summary", "hybrid", "pattern-typed".
        Controls how prior history is rendered in the observation-window block.
    history_window:
        Maximum number of prior steps to show. None = show all.
    history_source:
        "observed" = use observation_history; "simulated" = use simulated_history.
    history_actions:
        Action label taken at each prior step, parallel to the history list.
        Required for the "raw-actions" and "hybrid" formats; ignored otherwise.
    history_contexts:
        Safe pre-outcome context for each prior step, parallel to the history
        list. Used only for compact timing labels in verbatim history lines.
    include_reasoning:
        If False, omit reason_code from the requested JSON object.
    """
    variant = normalize_dataset_prompt_variant(profile, prompt_variant)
    if profile.dataset_id == "heartsteps" and variant in HEARTSTEPS_PERSONA_VARIANTS:
        return _build_heartsteps_persona_prompt(
            profile=profile,
            patient=patient,
            obs=obs,
            action=action,
            observation_history=observation_history,
            simulated_history=simulated_history,
            history_window=history_window,
            history_actions=history_actions,
            history_contexts=history_contexts,
            anchor_days=anchor_days,
            prompt_variant=variant,
            include_reasoning=include_reasoning,
        )

    # Context ablation: drop one context component (model held fixed).
    # Ablation variants behave exactly as heartsteps-context-faithful for every
    # variant-dependent section (output schema, step-variant, guidance) so they
    # match the #465 baseline and differ ONLY by the dropped component.
    drop = HEARTSTEPS_ABLATION_DROP.get(variant, set())
    eff_variant = "heartsteps-context-faithful" if variant in HEARTSTEPS_ABLATION_DROP else variant

    sections: list[str] = []

    # 1. Dataset / outcome framing
    if "framing" not in drop:
        sections.append(_build_framing_section(profile))

    # 2. Participant profile (vignette)
    if "vignette" not in drop:
        sections.append(_build_vignette_section(profile, patient.vignette))

    # 3. Participant activity framing (execution-capacity analogue)
    if "capacity" not in drop:
        cap_section = _build_capacity_section(profile, patient.vignette)
        if cap_section:
            sections.append(cap_section)
    if patient_memory:
        sections.append(_build_patient_memory_section(patient_memory))

    # 4. Literature priors
    if "literature" not in drop:
        sections.append(_build_literature_priors_section(profile))

    # 5. Output schema
    sections.append(
        _build_output_schema_section(
            profile,
            prompt_variant=eff_variant,
            include_reasoning=include_reasoning,
        )
    )
    variant_section = _build_step_variant_section(profile, eff_variant)
    if variant_section:
        sections.append(variant_section)
    stochastic_section = _build_stochastic_cue_section(profile, eff_variant, patient, obs, action)
    if stochastic_section:
        sections.append(stochastic_section)

    # 6. Current pre-decision context (whitelisted)
    if "context" not in drop:
        sections.append(_build_context_section(profile, obs))

    # 7. Two-section prefix (anchored) OR single-section history (unanchored).
    #
    # Anchored mode (anchor_days K > 0): two distinct sections.
    #   7a. Observation window: K real measured outcomes (immutable warm-start
    #       prefix), rendered with the user-chosen ``prefix_format``.
    #   7b. Recent simulated history: the LLM's own autoregressive predictions
    #       for steps K+1, K+2, ..., rendered day-by-day (with actions when
    #       available). Starts empty at step K+1 and grows by one per LLM call.
    # Unanchored single-section mode (anchor_days == 0): pick observation_history
    # or simulated_history per ``history_source`` and render with prefix_format.
    actions_all = list(history_actions or [])
    contexts_all = list(history_contexts or [])
    if anchor_days and anchor_days > 0:
        k = int(anchor_days)
        window_outcomes = list(observation_history[:k])
        window_actions = actions_all[:k] if actions_all else None
        window_contexts = contexts_all[:k] if contexts_all else None

        # Population anchor-window prior (optional ablation knob) -- shown
        # BETWEEN the immutable observation window and the autoregressive
        # history so the cohort prior is read after the patient's own
        # warm-start but before the LLM's own simulated trajectory.
        sections.append(
            _build_observation_window_section(
                profile, window_outcomes, prefix_format, window_actions,
                window_contexts,
                anchor_days=k,
            )
        )
        if anchor_prior_stats:
            prior_section = build_dataset_anchor_prior_section(anchor_prior_stats, profile)
            if prior_section is not None:
                sections.append(prior_section)

        sim_outcomes = list(simulated_history)
        # Actions at simulated steps live in actions_all[K : K+len(sim)].
        sim_actions = actions_all[k:k + len(sim_outcomes)] if actions_all else None
        sim_contexts = contexts_all[k:k + len(sim_outcomes)] if contexts_all else None
        if history_window is not None:
            if history_window <= 0:
                sim_outcomes = []
                sim_actions = [] if sim_actions else None
                sim_contexts = [] if sim_contexts else None
            else:
                sim_outcomes = sim_outcomes[-history_window:]
                if sim_actions:
                    sim_actions = sim_actions[-history_window:]
                if sim_contexts:
                    sim_contexts = sim_contexts[-history_window:]
        sections.append(
            _build_simulated_history_section(
                profile, sim_outcomes, sim_actions, sim_contexts, anchor_days=k,
            )
        )
        # For the guidance section, anchor on the visible composite (window +
        # recent sim) so the "history is mixed / sticky / gradual" rules read
        # against the same evidence the model is being shown.
        anchoring_view = window_outcomes + sim_outcomes
    else:
        history = observation_history if history_source == "observed" else simulated_history
        if history_window is not None:
            if history_window <= 0:
                shown = []
                shown_actions = [] if actions_all else None
                shown_contexts = [] if contexts_all else None
            else:
                shown = history[-history_window:]
                shown_actions = actions_all[-history_window:] if actions_all else None
                shown_contexts = contexts_all[-history_window:] if contexts_all else None
        else:
            shown = list(history)
            shown_actions = actions_all or None
            shown_contexts = contexts_all or None
        sections.append(
            _build_observation_window_section(
                profile, shown, prefix_format, shown_actions,
                shown_contexts,
                anchor_days=0, source_label=history_source,
            )
        )
        if anchor_prior_stats:
            prior_section = build_dataset_anchor_prior_section(anchor_prior_stats, profile)
            if prior_section is not None:
                sections.append(prior_section)
        anchoring_view = shown

    # 8. History-anchoring guidance (variant-2 discipline)
    if "guidance" not in drop:
        sections.append(_build_history_guidance_section(profile, anchoring_view, eff_variant))

    # 9. Current intervention (bare action id when the message text is ablated)
    if "intervention_text" in drop:
        sections.append(f"Current intervention (action id only): {action}")
    else:
        sections.append(_build_intervention_section(profile, action))

    return "\n\n".join(sections)


def _build_heartsteps_persona_system_instruction(
    profile: DatasetPromptProfile,
    *,
    prompt_variant: str = "heartsteps-persona-simple",
    include_reasoning: bool = True,
) -> str:
    reason_codes = ", ".join(profile.reason_codes)
    reason_line = (
        f'Use one reason_code from: {reason_codes}.'
        if include_reasoning else
        "Do not include reason_code."
    )
    return (
        "You are the person described in the user message. Answer as yourself "
        "at one phone decision moment.\n\n"
        "Think about what you would actually do in the next 30 minutes, not "
        "what would be healthiest or what the study wants. A phone suggestion "
        "can help, but you might ignore it, postpone it, be busy, be tired, "
        "or not walk at all.\n\n"
        f"{_heartsteps_persona_system_variant_text(prompt_variant)}\n\n"
        "Return strict JSON only. No markdown, no explanation. "
        "Keys: adherence, adherence_certainty, "
        f"{'reason_code, ' if include_reasoning else ''}state_update. "
        f"In this task, adherence means your 0-1 activity score for the next "
        f"30 minutes. {reason_line}"
    )


def _build_heartsteps_persona_prompt(
    profile: DatasetPromptProfile,
    patient: Any,
    obs: Any,
    action: str,
    observation_history: list[float],
    simulated_history: list[float],
    *,
    history_window: int | None = 7,
    history_actions: list[str] | None = None,
    history_contexts: list[dict[str, Any]] | None = None,
    anchor_days: int = 0,
    prompt_variant: str = "heartsteps-persona-simple",
    include_reasoning: bool = True,
) -> str:
    """Short first-person HeartSteps prompt.

    This intentionally avoids the dense "dataset-backed simulator" framing.
    It follows the ARMMAN-style prompt shape: "you are this person", then
    profile, past behavior, current situation, and a direct question.
    """
    actions_all = list(history_actions or [])
    contexts_all = list(history_contexts or [])
    k = max(0, int(anchor_days or 0))
    anchor_outcomes = list(observation_history[:k]) if k > 0 else []
    anchor_actions = actions_all[:k] if actions_all else None
    anchor_contexts = contexts_all[:k] if contexts_all else None

    sim_outcomes = list(simulated_history)
    sim_actions = actions_all[k:k + len(sim_outcomes)] if actions_all else None
    sim_contexts = contexts_all[k:k + len(sim_outcomes)] if contexts_all else None
    if history_window is not None:
        if history_window <= 0:
            sim_outcomes = []
            sim_actions = [] if sim_actions else None
            sim_contexts = [] if sim_contexts else None
        else:
            sim_outcomes = sim_outcomes[-history_window:]
            if sim_actions:
                sim_actions = sim_actions[-history_window:]
            if sim_contexts:
                sim_contexts = sim_contexts[-history_window:]

    action_desc = dict(profile.action_descriptions).get(action, action)
    schema = {
        "adherence": "<float 0.0-1.0: your next-30-minute activity score>",
        "adherence_certainty": "<float 0.0-1.0>",
        "state_update": {},
    }
    if include_reasoning:
        schema["reason_code"] = f"<one of: {', '.join(profile.reason_codes)}>"

    sections = [
        (
            "You are this participant right now. Decide what you actually do "
            "in the next 30 minutes after this phone decision."
        ),
        _build_persona_profile_section(profile, patient),
        _build_persona_current_context_section(profile, obs),
    ]

    if anchor_outcomes:
        sections.append(
            "Your earlier real records in this study:\n"
            + _render_persona_history_lines(
                profile,
                anchor_outcomes,
                anchor_actions,
                anchor_contexts,
                start_index=1,
            )
        )
    else:
        sections.append("Your earlier real records in this study: none shown.")

    if sim_outcomes:
        start_index = max(k + 1, k + len(simulated_history) - len(sim_outcomes) + 1)
        sections.append(
            "Your recent previous answers in this continued run:\n"
            + _render_persona_history_lines(
                profile,
                sim_outcomes,
                sim_actions,
                sim_contexts,
                start_index=start_index,
            )
        )
    else:
        sections.append("Your recent previous answers in this continued run: none yet.")

    sections.extend(
        [
            (
                "What happens now:\n"
                f"- Phone action: {action}\n"
                f"- Meaning: {action_desc}"
            ),
            _heartsteps_persona_prompt_variant_text(prompt_variant),
            (
                "Question: during the next 30 minutes, how much do you actually "
                "walk or move?\n"
                f"{_heartsteps_persona_scale_text(prompt_variant)}"
            ),
            "Return JSON only:\n" + json.dumps(schema, indent=2),
        ]
    )
    return "\n\n".join(sections)


def _heartsteps_persona_system_variant_text(prompt_variant: str) -> str:
    if prompt_variant == "heartsteps-persona-measured":
        return (
            "Your answer is what your phone or tracker would record, not what "
            "you would remember as exercise. Tiny incidental steps still count."
        )
    if prompt_variant == "heartsteps-persona-small-steps":
        return (
            "Do not turn every mostly-still moment into exact zero. If you get "
            "up, walk around the room, leave a building, or make a short errand, "
            "that is a small nonzero value even if it is not a workout."
        )
    if prompt_variant == "heartsteps-persona-rhythm":
        return (
            "Use your own earlier rhythm: many quiet windows can coexist with "
            "occasional small or large bursts. Do not let a run of quiet moments "
            "erase the possibility of a later active window."
        )
    if prompt_variant == "heartsteps-persona-balanced":
        return (
            "Balance the current situation with your past records. The phone "
            "action is only one cue; no_suggestion does not mean no walking, "
            "and a suggestion does not mean you definitely walk."
        )
    return (
        "Keep the answer natural and concrete. Exact zero is possible, but it "
        "should mean no recorded post-decision movement."
    )


def _heartsteps_persona_prompt_variant_text(prompt_variant: str) -> str:
    if prompt_variant == "heartsteps-persona-measured":
        return (
            "Important: this is a tracker-measured 30-minute step response. "
            "Count incidental walking too: moving around at home, walking at "
            "work, going to another room, leaving a store, or taking a short "
            "errand can all be nonzero."
        )
    if prompt_variant == "heartsteps-persona-small-steps":
        return (
            "Use small values when they fit. 0.001-0.03 can mean only a few "
            "recorded steps; 0.03-0.15 can mean a small amount of movement. "
            "Do not use exact 0 unless you expect basically no recorded steps."
        )
    if prompt_variant == "heartsteps-persona-rhythm":
        return (
            "Look at your earlier records as a personal rhythm, not as a rule "
            "that the next answer must copy. Quiet periods, tiny movement, and "
            "occasional bursts are all possible for the same person."
        )
    if prompt_variant == "heartsteps-persona-balanced":
        return (
            "Make a balanced call: current context matters most, but the tracker "
            "can record nonzero steps even when you are mostly still. Do not "
            "treat no_suggestion as no activity."
        )
    return (
        "Answer naturally from the situation and your earlier records."
    )


def _heartsteps_persona_scale_text(prompt_variant: str) -> str:
    if prompt_variant == "heartsteps-persona-simple":
        return (
            "Use this scale: 0 = no meaningful post-decision steps; "
            "1 = about 1500 or more post-decision steps. Exact 0 is allowed."
        )
    return (
        "Use this scale: 0 = literally no recorded post-decision steps; "
        "0.001-0.03 = a few incidental steps; 0.03-0.15 = small movement; "
        "0.15-0.7 = meaningful walking; 1 = about 1500 or more steps."
    )


def _build_persona_profile_section(
    profile: DatasetPromptProfile,
    patient: Any,
) -> str:
    vignette = getattr(patient, "vignette", {}) or {}
    lines = ["Your profile:"]
    record_id = getattr(patient, "record_id", None) or vignette.get("record_id")
    if record_id:
        lines.append(f"- Participant ID: {record_id}")
    for key, label in profile.vignette_field_labels:
        if key == "record_id":
            continue
        value = vignette.get(key)
        if value in (None, ""):
            continue
        lines.append(f"- {label}: {_format_persona_value(value)}")
    return "\n".join(lines)


def _build_persona_current_context_section(
    profile: DatasetPromptProfile,
    obs: Any,
) -> str:
    lines = ["Your situation at this phone decision:"]
    for key, label in profile.context_field_labels:
        value = obs.context.get(key)
        if value in (None, ""):
            continue
        lines.append(f"- {label}: {_format_persona_value(value)}")
    return "\n".join(lines)


def _render_persona_history_lines(
    profile: DatasetPromptProfile,
    outcomes: list[float],
    actions: list[str] | None,
    contexts: list[dict[str, Any]] | None,
    *,
    start_index: int = 1,
) -> str:
    lines: list[str] = []
    for offset, outcome in enumerate(outcomes):
        step_no = start_index + offset
        action = actions[offset] if actions and offset < len(actions) else "unknown"
        context = contexts[offset] if contexts and offset < len(contexts) else {}
        details = _format_persona_history_context(context)
        detail_suffix = f"; {details}" if details else ""
        lines.append(
            f"- moment {step_no}: action={action}; activity={float(outcome):.3f}"
            f"{detail_suffix}"
        )
    return "\n".join(lines)


def _format_persona_history_context(context: dict[str, Any]) -> str:
    if not context:
        return ""
    compact_keys = [
        ("study_day", "study_day"),
        ("same_day_decision_number", "same_day_point"),
        ("local_decision_time", "time"),
        ("slot", "slot"),
        ("in_transit", "in_transit"),
        ("recognized_activity", "phone_activity"),
        ("location_category", "location"),
        ("weather", "weather"),
        ("temperature", "temp_c"),
        ("pre_steps", "pre_steps"),
    ]
    parts: list[str] = []
    for key, label in compact_keys:
        value = context.get(key)
        if value in (None, ""):
            continue
        parts.append(f"{label}={_format_persona_value(value)}")
    return ", ".join(parts)


def _format_persona_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if isinstance(value, list):
        return ", ".join(_format_persona_value(v) for v in value) if value else "none"
    return str(value)


# ---------------------------------------------------------------------------
# Population anchor-window prior (ablation knob)
#
# Aggregates the cohort's first ``anchor_days`` observed outcomes into
# descriptive statistics (population mean, per-patient spread, distribution,
# per-action mean and lift versus the baseline action). The result is rendered
# as a base-rate reference block in the step prompt -- it does NOT reveal the
# current patient's outcome and is constant within a run.
# ---------------------------------------------------------------------------

def compute_dataset_anchor_stats(
    patients: list,  # list[ObservedPatient]
    anchor_days: int,
    profile: DatasetPromptProfile,
) -> dict | None:
    """Aggregate first-``anchor_days`` observed outcomes across the cohort.

    Returns ``None`` if there is no data to summarize or ``anchor_days`` <= 0.
    """
    if not patients or not anchor_days or anchor_days <= 0:
        return None

    all_vals: list[float] = []
    per_patient_means: list[float] = []
    per_action: dict[str, list[float]] = {}

    for pat in patients:
        steps = pat.steps[:anchor_days]
        if not steps:
            continue
        vals = [float(s.actual) for s in steps if s.actual is not None]
        if not vals:
            continue
        all_vals.extend(vals)
        per_patient_means.append(sum(vals) / len(vals))
        for s in steps:
            if s.actual is None:
                continue
            per_action.setdefault(s.action, []).append(float(s.actual))

    if not all_vals:
        return None

    n_obs = len(all_vals)
    pop_mean = sum(all_vals) / n_obs
    low = sum(1 for v in all_vals if v <= 0.10)
    mid = sum(1 for v in all_vals if 0.10 < v < 0.70)
    high = sum(1 for v in all_vals if v >= 0.70)

    per_patient_std = 0.0
    if len(per_patient_means) >= 2:
        m = sum(per_patient_means) / len(per_patient_means)
        var = sum((x - m) ** 2 for x in per_patient_means) / (len(per_patient_means) - 1)
        per_patient_std = var ** 0.5

    baseline = profile.baseline_action
    base_vals = per_action.get(baseline, [])
    base_mean = sum(base_vals) / len(base_vals) if base_vals else None

    action_breakdown: dict[str, dict[str, float | int | None]] = {}
    for act_id, _ in profile.action_descriptions:
        vals = per_action.get(act_id, [])
        if len(vals) < 5:
            continue
        mean_v = sum(vals) / len(vals)
        lift = (mean_v - base_mean) if (base_mean is not None and act_id != baseline) else None
        action_breakdown[act_id] = {
            "mean": mean_v,
            "n": len(vals),
            "lift_vs_baseline": lift,
        }

    return {
        "n_patients": len(per_patient_means),
        "n_obs": n_obs,
        "anchor_days": int(anchor_days),
        "pop_mean": pop_mean,
        "per_patient_std": per_patient_std,
        "low_pct": low / n_obs,
        "mid_pct": mid / n_obs,
        "high_pct": high / n_obs,
        "action_breakdown": action_breakdown,
        "baseline_action": baseline,
    }


def build_dataset_anchor_prior_section(
    stats: dict | None, profile: DatasetPromptProfile
) -> str | None:
    """Render the population anchor-window prior block."""
    if not stats:
        return None
    lines = [
        f"Population anchor-window prior (descriptive cohort stats over the "
        f"first {stats['anchor_days']} step(s) of every participant in this "
        "dataset; use as a base-rate reference, not as evidence for any "
        "single participant):",
        f"- Cohort size: {stats['n_patients']} participant(s), "
        f"{stats['n_obs']} step-observation(s).",
        f"- Population mean {profile.outcome_name}: {stats['pop_mean']:.3f} "
        f"(between-participant std of patient means: "
        f"{stats['per_patient_std']:.3f}).",
        f"- Step-level distribution: {stats['low_pct']:.0%} near-zero, "
        f"{stats['mid_pct']:.0%} mid-range, "
        f"{stats['high_pct']:.0%} high-outcome step(s).",
    ]
    ab = stats.get("action_breakdown") or {}
    if ab:
        lines.append(
            "- Per-action mean outcome in the anchor window (cohort pooled; "
            "treat as weak directional hints, not individual treatment effects):"
        )
        baseline = stats.get("baseline_action", "")
        for act_id, info in ab.items():
            lift = info.get("lift_vs_baseline")
            if lift is not None:
                sign = "+" if lift >= 0 else ""
                lift_str = f" [{sign}{lift * 100:.1f} pp vs {baseline}]"
            else:
                lift_str = " [baseline]"
            lines.append(
                f"  - {act_id}: mean {info['mean']:.3f} (n={info['n']}){lift_str}"
            )
    lines.append(
        "Use this prior to calibrate the magnitude and direction of action "
        "effects. Do not assume every participant matches the cohort mean: "
        "the per-step prior history and current context dominate; this block "
        "only provides the population scale."
    )
    return "\n".join(lines)


def _build_history_guidance_section(
    profile: DatasetPromptProfile,
    shown_history: list[float],
    prompt_variant: str = DATASET_PROMPT_VARIANT_DEFAULT,
) -> str:
    """Anchoring guidance shared across datasets.

    Reminds the model to anchor on the prior observed outcomes, treat
    low-outcome steps as sticky, preserve intermittency, and avoid optimistic
    flips from a single positive cue.
    """
    variant = normalize_dataset_prompt_variant(profile, prompt_variant)
    variant_guidance = _build_history_variant_guidance(profile, shown_history, variant)
    if not shown_history:
        base = (
            "Anchoring guidance:\n"
            "- No prior outcomes for this participant are visible yet. "
            "Lean on the participant's profile and the current context; "
            "do not assume a strong response to the intervention at the "
            "first decision point."
        )
        return base + (f"\n{variant_guidance}" if variant_guidance else "")
    base = (
        "Anchoring guidance:\n"
        f"- Use the prior {profile.outcome_name} as the primary anchor for "
        "your prediction. Low-outcome steps are sticky -- improvement should "
        "be gradual and should usually require consistent recent behaviour "
        "plus a favourable current context.\n"
        "- A supportive intervention alone should rarely justify jumping to a "
        "high outcome. Concrete contextual friction (poor location, bad "
        "weather, transit, side effects, fatigue, disruption) is stronger "
        "evidence than positive-sounding traits, study participation, or the "
        "intervention itself.\n"
        "- If the visible history is mixed (some high-outcome and some "
        "low-outcome steps), preserve that intermittency rather than "
        "collapsing to a deterministic 0.0 or 1.0 prediction.\n"
        "- If only a few prior steps are visible, treat them as noisy "
        "evidence of an underlying tendency, not as a deterministic latent "
        "state. Do not snap to extremes on the basis of one or two outcomes.\n"
        "- When evidence is genuinely mixed, use intermediate values and "
        "lower adherence_certainty instead of extreme 0.0 / 1.0 outputs."
    )
    return base + (f"\n{variant_guidance}" if variant_guidance else "")


def _build_system_variant_instruction(
    profile: DatasetPromptProfile,
    prompt_variant: str,
) -> str | None:
    if profile.dataset_id != "heartsteps":
        return None
    if prompt_variant == "heartsteps-zero-inflated":
        return (
            "HeartSteps zero-inflated variant: predict one plausible realized "
            "30-minute step-count response, not a smoothed patient-average "
            "expectation. Exact 0.0 is a valid measurement when the person does "
            "not walk in the post-decision window. Preserve zero windows, zero "
            "streaks, and occasional positive bursts instead of smoothing every "
            "prediction into the 0.10-0.20 range."
        )
    if prompt_variant == "heartsteps-conservative-action":
        return (
            "HeartSteps conservative-action variant: estimate the contextual "
            "baseline first, as if no suggestion had been sent, then apply only "
            "a small action adjustment when the current context creates a real "
            "opportunity to walk. A suggestion is a weak nudge, not a direct "
            "cause of activity."
        )
    if prompt_variant == "heartsteps-soft-zero-calibrated":
        return (
            "HeartSteps soft-zero-calibrated variant: combine conservative "
            "action effects with a realistic zero-inflated activity mixture. "
            "Some windows should be exact or near zero, but do not collapse "
            "the whole trajectory to zero. Preserve the participant's mean "
            "level by making activity-compatible positive windows meaningfully "
            "positive rather than tiny."
        )
    if prompt_variant == "heartsteps-mean-preserving-mixture":
        return (
            "HeartSteps mean-preserving mixture variant: represent the "
            "zero-inflated activity distribution while preserving the "
            "participant's observed activity level. Exact zero windows are "
            "realistic, but no-suggestion windows are not automatically zero "
            "and nonzero windows must be large enough to keep the trajectory "
            "mean near the warm-start and cohort-calibrated scale."
        )
    if prompt_variant == "heartsteps-hurdle-calibrated":
        return (
            "HeartSteps hurdle-calibrated variant: forecast in two internal "
            "stages. First decide whether this 30-minute window is an inactive "
            "zero window or an active nonzero window. Then, if it is nonzero, "
            "choose a positive magnitude large enough to preserve the "
            "participant's visible mean. This avoids both failures seen in "
            "HeartSteps: smoothing every window into tiny positives, or "
            "creating realistic zeros while collapsing the mean."
        )
    if prompt_variant == "heartsteps-context-faithful":
        return (
            "HeartSteps context-faithful variant: preserve the exact proximal "
            "prediction target. Forecast only the next 30-minute post-decision "
            "step response, using the phone decision context, same-day slot, "
            "pre-decision steps, location, weather, transit status, and recent "
            "history. Do not convert this into a daily activity or general "
            "exercise-motivation prediction."
        )
    if prompt_variant == "heartsteps-human-noise":
        return (
            "HeartSteps human-noise variant: simulate ordinary human physical "
            "activity, not an optimized study participant. Real behaviour can "
            "be inconsistent, noisy, short, delayed, distracted, or absent even "
            "when a suggestion is appropriate. Avoid overly smooth trajectories "
            "and avoid making the participant more compliant merely because the "
            "prompt is helpful."
        )
    if prompt_variant == "heartsteps-structured-forecast":
        return (
            "HeartSteps structured-forecast variant: before choosing the numeric "
            "outcome internally decompose the forecast into four latent pieces: "
            "participant baseline, current-context feasibility, intervention "
            "nudge, and random short-window noise. Return only the final strict "
            "JSON object, but make the adherence value consistent with that "
            "decomposition."
        )
    if prompt_variant == "heartsteps-stochastic-realized":
        return (
            "HeartSteps stochastic-realized variant: predict a realized sample "
            "from the participant's plausible next 30-minute activity "
            "distribution, not the conditional mean. Use the simulation "
            "randomness cue in the user prompt as ordinary unobserved human "
            "variation. This cue is not real data and does not reveal the "
            "hidden outcome."
        )
    if prompt_variant == "heartsteps-stochastic-bursty":
        return (
            "HeartSteps stochastic-bursty variant: preserve the bursty tracker "
            "distribution. Many windows are quiet, but occasional positive "
            "bursts are part of normal measured activity. Use the simulation "
            "randomness cue to choose where this realized window falls inside "
            "the plausible range instead of smoothing every prediction."
        )
    return None


def _build_step_variant_section(
    profile: DatasetPromptProfile,
    prompt_variant: str,
) -> str | None:
    if profile.dataset_id != "heartsteps":
        return None
    if prompt_variant == "heartsteps-zero-inflated":
        return (
            "HeartSteps variant: zero-inflated realized outcome.\n"
            "- The value you output is the realized normalized step response "
            "for this specific 30-minute window, not just the conditional mean.\n"
            "- If the likely realised post-decision activity is no meaningful "
            "walking, output 0.00 or a very small value (0.01-0.03). Do not "
            "avoid zero just because the participant has a moderate profile.\n"
            "- Positive responses should be bursty: most windows can be near "
            "zero, while activity-compatible contexts can produce a larger "
            "realized value. Preserve that mixture."
        )
    if prompt_variant == "heartsteps-conservative-action":
        return (
            "HeartSteps variant: conservative intervention effect.\n"
            "- First infer the no-suggestion baseline for this decision point "
            "from history, pre_steps, location, activity state, weather, and "
            "time slot.\n"
            "- Then consider the action. Only add a positive suggestion effect "
            "when the context already supports walking or the recent history "
            "shows response to similar prompts.\n"
            "- If context is unfavorable, the best prediction may be identical "
            "to the no-suggestion baseline even when an active suggestion is sent."
        )
    if prompt_variant == "heartsteps-soft-zero-calibrated":
        return (
            "HeartSteps variant: soft zero-calibrated mixture.\n"
            "- Predict a realized 30-minute outcome, but keep the patient-level "
            "mean calibrated to the observed warm-start prefix and cohort prior.\n"
            "- Use exact 0.00 or very small values when the current window looks "
            "inactive. Do not replace all inactive windows with smooth 0.08-0.15 "
            "averages.\n"
            "- Avoid the opposite failure too: do not turn every uncertain window "
            "into zero. If a participant has a nonzero warm-start mean, some "
            "activity-compatible windows must be meaningfully positive.\n"
            "- Estimate the no-suggestion contextual baseline first, then apply "
            "only a small action adjustment when the action has a plausible "
            "opportunity to change behavior."
        )
    if prompt_variant == "heartsteps-mean-preserving-mixture":
        return (
            "HeartSteps variant: mean-preserving zero-inflated mixture.\n"
            "- Keep exact/near-zero outcomes for truly inactive-looking "
            "windows, but preserve the participant's recent mean outcome. "
            "If many windows are zero, the nonzero windows must be meaningfully "
            "positive rather than tiny.\n"
            "- No-suggestion means no prompt was sent; it does not mean no "
            "walking occurred. Baseline walking can still produce 0.08-0.25 "
            "normalized outcomes when pre_steps, location, slot, or activity "
            "state support it.\n"
            "- Active and sedentary-break suggestions can add a modest lift, "
            "but first estimate the context/history baseline. Do not make the "
            "action effect compensate for an unrealistically low baseline.\n"
            "- Avoid both smoothing failures: not all steps should be small "
            "positives, and not all uncertain steps should be zeros."
        )
    if prompt_variant == "heartsteps-hurdle-calibrated":
        return (
            "HeartSteps variant: calibrated hurdle mixture.\n"
            "- Think internally in two stages: (1) inactive vs active window, "
            "then (2) positive magnitude if active.\n"
            "- If the current pre-decision context looks inactive "
            "(for example transit, sedentary/unknown activity, poor location, "
            "bad weather, very low recent outcomes, or low pre_steps), output "
            "exact 0.00 or a tiny value. Do not hide inactive windows as "
            "smooth 0.06-0.10 predictions.\n"
            "- If the window is active enough to be nonzero, avoid tiny "
            "nonzero values unless the evidence is weak. To preserve a "
            "trajectory mean around 0.10-0.14 with many zeros, active windows "
            "often need values around 0.18-0.35, and occasionally higher in "
            "clearly favorable contexts.\n"
            "- No-suggestion can still have baseline walking, so nonzero "
            "values are allowed without a prompt. Suggestions are weak nudges: "
            "add only a small lift when context and history make walking "
            "plausible.\n"
            "- The target is not the best single-step MAE. The target is a "
            "trajectory distribution with realistic zeros, realistic positive "
            "bursts, and a calibrated patient-level mean."
        )
    if prompt_variant == "heartsteps-context-faithful":
        return (
            "HeartSteps variant: context-faithful proximal forecast.\n"
            "- Predict only this decision point's 30-minute post-decision steps.\n"
            "- Treat slot/time, pre-decision steps, transit status, location, "
            "weather, and phone-use recency as immediate feasibility evidence.\n"
            "- Do not infer a whole-day activity level from a single decision "
            "point, and do not let a generally active profile override a blocked "
            "current context."
        )
    if prompt_variant == "heartsteps-human-noise":
        return (
            "HeartSteps variant: realistic human short-window noise.\n"
            "- People often ignore, miss, postpone, or only partially act on "
            "suggestions; a sent suggestion is not an instruction being obeyed.\n"
            "- Preserve imperfect, noisy behaviour: similar contexts can produce "
            "different outcomes, and favorable traits do not guarantee activity.\n"
            "- The model should not smooth every inactive-looking window into "
            "the same small positive value."
        )
    if prompt_variant == "heartsteps-structured-forecast":
        return (
            "HeartSteps variant: structured internal forecast.\n"
            "- Internally estimate: baseline activity tendency, current-context "
            "feasibility, weak intervention effect, and short-window randomness.\n"
            "- The numeric output should be a realized plausible value for this "
            "specific window, not only a smooth expected mean.\n"
            "- Return only the required JSON object; do not expose the internal "
            "decomposition."
        )
    if prompt_variant == "heartsteps-stochastic-realized":
        return (
            "HeartSteps variant: stochastic realized sample.\n"
            "- First estimate this participant's plausible distribution for "
            "the current 30-minute window from profile, warm-start anchors, "
            "recent simulated history, current context, and action.\n"
            "- Then use the simulation randomness cue below to pick one "
            "realized value from that distribution. Low draws should usually "
            "select quiet/tiny outcomes; high draws should select the upper "
            "part of the plausible range when the context is not blocking.\n"
            "- Do not output only the average of possible outcomes. A simulator "
            "trajectory needs realized variation."
        )
    if prompt_variant == "heartsteps-stochastic-bursty":
        return (
            "HeartSteps variant: stochastic burst preservation.\n"
            "- Tracker activity is bursty: quiet windows and activity bursts "
            "can both happen for the same participant.\n"
            "- Use the simulation randomness cue below to decide whether this "
            "specific window is quiet, small movement, meaningful walking, or "
            "an upper-tail burst. Keep context as a gate: impossible-looking "
            "contexts should still stay low.\n"
            "- If the draw is high, the current context is plausible, and the "
            "warm-start contains positive or high windows, do not be afraid to "
            "output a meaningfully positive value such as 0.20-0.60. If the "
            "draw is extreme and the context is favorable, a higher burst is "
            "allowed."
        )
    return None


def _build_stochastic_cue_section(
    profile: DatasetPromptProfile,
    prompt_variant: str,
    patient: Any,
    obs: Any,
    action: str,
) -> str | None:
    if profile.dataset_id != "heartsteps" or prompt_variant not in HEARTSTEPS_STOCHASTIC_CUE_VARIANTS:
        return None
    record_id = getattr(patient, "record_id", None) or "unknown"
    decision = obs.context.get("decision_point_number", obs.day)
    activity_draw = _stable_unit_interval(
        profile.dataset_id, prompt_variant, record_id, str(decision), action, "activity"
    )
    magnitude_draw = _stable_unit_interval(
        profile.dataset_id, prompt_variant, record_id, str(decision), action, "magnitude"
    )
    if prompt_variant == "heartsteps-stochastic-bursty":
        guidance = (
            "- activity_draw < 0.55: favor quiet or tiny movement unless context/history are strongly active.\n"
            "- 0.55 <= activity_draw < 0.85: favor a typical small-to-moderate realized value.\n"
            "- activity_draw >= 0.85: consider an upper-tail positive window if the context is plausible.\n"
            "- magnitude_draw chooses where inside the selected range the realized value lands."
        )
    else:
        guidance = (
            "- Low activity_draw values should usually select the lower part of the plausible range.\n"
            "- Middle activity_draw values should select a typical realized value.\n"
            "- High activity_draw values should select the upper part of the plausible range when context allows.\n"
            "- magnitude_draw chooses where inside that selected part of the range the realized value lands."
        )
    return (
        "Simulation randomness cue (synthetic, not observed data; no future leakage):\n"
        f"- activity_draw: {activity_draw:.3f}\n"
        f"- magnitude_draw: {magnitude_draw:.3f}\n"
        f"{guidance}"
    )


def _stable_unit_interval(*parts: str) -> float:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    value = int(digest[:12], 16)
    return value / float(16 ** 12 - 1)


def _build_history_variant_guidance(
    profile: DatasetPromptProfile,
    shown_history: list[float],
    prompt_variant: str,
) -> str | None:
    if profile.dataset_id != "heartsteps":
        return None
    if prompt_variant == "heartsteps-zero-inflated":
        if shown_history:
            zeros = sum(1 for value in shown_history if value <= 0.03)
            zero_rate = zeros / len(shown_history)
            recent_zero_text = (
                f"The visible history has {zeros}/{len(shown_history)} "
                f"near-zero step(s) (<=0.03; {zero_rate:.0%})."
            )
        else:
            recent_zero_text = "No participant-specific zero-rate is visible yet."
        return (
            "- Zero-inflation guidance: "
            f"{recent_zero_text} Keep that zero propensity in the forecast. "
            "When recent values are near zero and the current context is not "
            "activity-compatible, a near-zero realized outcome is more realistic "
            "than a small smoothed average."
        )
    if prompt_variant == "heartsteps-conservative-action":
        return (
            "- Conservative-action guidance: do not rank active_suggestion or "
            "sedentary_break_suggestion above no_suggestion by default. Action "
            "effects are weak and conditional; context and prior trajectory are "
            "stronger evidence than the action label."
        )
    if prompt_variant == "heartsteps-soft-zero-calibrated":
        if shown_history:
            zeros = sum(1 for value in shown_history if value <= 0.03)
            positives = [value for value in shown_history if value > 0.03]
            zero_rate = zeros / len(shown_history)
            positive_mean = sum(positives) / len(positives) if positives else 0.0
            stats = (
                f"Visible near-zero rate is {zeros}/{len(shown_history)} "
                f"({zero_rate:.0%}); mean among visible positive windows is "
                f"{positive_mean:.2f}."
            )
        else:
            stats = "No participant-specific zero/positive mixture is visible yet."
        return (
            "- Soft-zero calibration guidance: "
            f"{stats} Keep both parts of the mixture: zeros should remain "
            "possible, and positive windows should stay large enough to preserve "
            "the participant's activity level. A realistic trajectory here is "
            "not all smooth small positives and not all zeros."
        )
    if prompt_variant == "heartsteps-mean-preserving-mixture":
        if shown_history:
            zeros = sum(1 for value in shown_history if value <= 0.03)
            positives = [value for value in shown_history if value > 0.03]
            zero_rate = zeros / len(shown_history)
            mean_value = sum(shown_history) / len(shown_history)
            positive_mean = sum(positives) / len(positives) if positives else 0.0
            stats = (
                f"Visible mean is {mean_value:.2f}; near-zero rate is "
                f"{zeros}/{len(shown_history)} ({zero_rate:.0%}); positive-window "
                f"mean is {positive_mean:.2f}."
            )
        else:
            stats = "No participant-specific mean/zero mixture is visible yet."
        return (
            "- Mean-preserving mixture guidance: "
            f"{stats} Match the mixture, not only the zero rate. If you output "
            "zeros at a realistic frequency, raise the positive windows enough "
            "to keep the overall forecast near the visible mean unless the "
            "current context gives strong evidence of decline."
        )
    if prompt_variant == "heartsteps-hurdle-calibrated":
        if shown_history:
            zeros = sum(1 for value in shown_history if value <= 0.03)
            positives = [value for value in shown_history if value > 0.03]
            zero_rate = zeros / len(shown_history)
            mean_value = sum(shown_history) / len(shown_history)
            positive_mean = sum(positives) / len(positives) if positives else 0.0
            implied_positive = (
                mean_value / max(1.0 - zero_rate, 0.10)
                if mean_value > 0.0
                else 0.0
            )
            stats = (
                f"Visible mean is {mean_value:.2f}; near-zero rate is "
                f"{zeros}/{len(shown_history)} ({zero_rate:.0%}); observed "
                f"positive-window mean is {positive_mean:.2f}; implied "
                f"positive mean needed to preserve the visible mean at this "
                f"zero rate is about {implied_positive:.2f}."
            )
        else:
            stats = "No participant-specific hurdle mixture is visible yet."
        return (
            "- Calibrated hurdle guidance: "
            f"{stats} Do not optimize only the zero decision. Once you choose "
            "a nonzero outcome, make its magnitude compatible with the visible "
            "mean and zero rate. A sequence of many exact zeros plus many tiny "
            "0.03-0.06 positives is under-calibrated unless the visible mean "
            "is itself near zero."
        )
    if prompt_variant == "heartsteps-context-faithful":
        return (
            "- Context-faithful guidance: compare the current decision context "
            "to the visible history by slot, timing, pre-steps, transit, "
            "location, and weather before relying on broad personality traits."
        )
    if prompt_variant == "heartsteps-human-noise":
        return (
            "- Human-noise guidance: use history as a tendency, not a script. "
            "Do not make the trajectory too smooth; short-window physical "
            "activity can fluctuate even when the same person and action repeat."
        )
    if prompt_variant == "heartsteps-structured-forecast":
        return (
            "- Structured-forecast guidance: let visible history set the "
            "participant baseline, current context set feasibility, and the "
            "current action add only a weak nudge unless the context supports it."
        )
    if prompt_variant == "heartsteps-stochastic-realized":
        if shown_history:
            positives = [value for value in shown_history if value > 0.03]
            high = sum(1 for value in shown_history if value >= 0.25)
            positive_mean = sum(positives) / len(positives) if positives else 0.0
            stats = (
                f"Visible positive-window mean is {positive_mean:.2f}; "
                f"{high}/{len(shown_history)} visible windows are >=0.25."
            )
        else:
            stats = "No participant-specific realized range is visible yet."
        return (
            "- Stochastic-realized guidance: "
            f"{stats} Use the random draw to choose a realized point within "
            "the plausible patient-specific range. Do not always output the "
            "same small value when the draw is high."
        )
    if prompt_variant == "heartsteps-stochastic-bursty":
        if shown_history:
            positives = [value for value in shown_history if value > 0.03]
            high = sum(1 for value in shown_history if value >= 0.25)
            positive_mean = sum(positives) / len(positives) if positives else 0.0
            stats = (
                f"Visible positive-window mean is {positive_mean:.2f}; "
                f"{high}/{len(shown_history)} visible windows are >=0.25."
            )
        else:
            stats = "No participant-specific burst evidence is visible yet."
        return (
            "- Stochastic-bursty guidance: "
            f"{stats} Preserve occasional upper-tail windows when the draw and "
            "context support them. A zero-inflated trajectory still needs "
            "positive bursts to match measured activity."
        )
    return None


# ---------------------------------------------------------------------------
# Section builders (internal)
# ---------------------------------------------------------------------------

def _build_framing_section(profile: DatasetPromptProfile) -> str:
    return (
        f"Dataset domain: {profile.domain_label}\n"
        f"Outcome target: {profile.outcome_name}\n"
        f"Outcome definition: {profile.outcome_definition}\n"
        f"Outcome scale: {profile.outcome_scale_description}"
    )


def _build_vignette_section(profile: DatasetPromptProfile, vignette: dict[str, Any]) -> str:
    lines = ["Participant profile:"]
    for key, label in profile.vignette_field_labels:
        val = vignette.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            val = ", ".join(str(v) for v in val) if val else "none reported"
        lines.append(f"- {label}: {val}")
    return "\n".join(lines)


def _build_capacity_section(profile: DatasetPromptProfile, vignette: dict[str, Any]) -> str | None:
    """Activity / engagement framing summary (execution-capacity analogue).

    For HeartSteps: distills selfeff_intake and conscientiousness into
    qualitative levels. Returns None if no relevant fields are present.
    """
    lines: list[str] = []

    if profile.dataset_id == "heartsteps":
        selfeff = vignette.get("selfeff_intake")
        consc = vignette.get("conscientiousness")
        walk_intake = vignette.get("walk_intake")

        if selfeff is not None:
            try:
                v = float(selfeff)
                if v >= 4.0:
                    se_level = "high"
                elif v >= 2.5:
                    se_level = "moderate"
                else:
                    se_level = "low"
            except (TypeError, ValueError):
                se_level = "unknown"
            lines.append(f"- Exercise self-efficacy: {se_level}")

        if consc is not None:
            try:
                v = float(consc)
                if v >= 4.0:
                    consc_level = "high"
                elif v >= 2.5:
                    consc_level = "moderate"
                else:
                    consc_level = "low"
            except (TypeError, ValueError):
                consc_level = "unknown"
            lines.append(f"- Conscientiousness: {consc_level}")

        if walk_intake not in (None, "unknown", ""):
            lines.append(f"- Reported walking at intake: {walk_intake}")

    if not lines:
        return None
    return "Participant activity framing:\n" + "\n".join(lines)


def _build_patient_memory_section(patient_memory: str) -> str:
    return (
        "Stable patient memory (derived only from static profile and visible "
        "warm-start anchors; contains no future/post-anchor outcomes):\n"
        f"{patient_memory}\n"
        "Use this as a slow-moving prior for the participant. It should help "
        "maintain individual consistency, but current pre-decision context and "
        "recent visible/simulated history still override it when they disagree."
    )


def _build_literature_priors_section(profile: DatasetPromptProfile) -> str:
    lines = [
        "Literature-grounded base-rate priors (use to calibrate predictions -- "
        "do not assume effects larger than the literature supports):"
    ]
    for prior in profile.literature_priors:
        lines.append(f"- {prior}")
    return "\n".join(lines)


def _build_output_schema_section(
    profile: DatasetPromptProfile,
    *,
    prompt_variant: str = DATASET_PROMPT_VARIANT_DEFAULT,
    include_reasoning: bool = True,
) -> str:
    variant = normalize_dataset_prompt_variant(profile, prompt_variant)
    reason_codes_str = ", ".join(profile.reason_codes)
    schema = {
        "adherence": f"<float 0.0-1.0, predicted {profile.outcome_name}>",
        "adherence_certainty": "<float 0.0-1.0, confidence in this prediction>",
        "state_update": "<object with any latent-state changes, or empty {}>",
    }
    if include_reasoning:
        schema["reason_code"] = f"<one of: {reason_codes_str}>"
    rules = [
        f"Predict the {profile.outcome_name} as it would be measured externally.",
        "Do not default to the clinically ideal or socially desirable value.",
        "Use intermediate values and lower certainty when evidence is mixed.",
        "Reserve high certainty for cases where context, history, and intervention "
        "all point in the same direction.",
        "The current step's actual outcome is hidden; infer only from pre-decision context.",
    ]
    if profile.dataset_id == "heartsteps":
        rules.append(
            "Predict this decision point's 30-minute post-decision step response, "
            "not total steps today, not daily activity, and not whether the user "
            "liked or answered the suggestion."
        )
    if include_reasoning:
        rules.append(f"Use reason_code from this dataset-specific set: {reason_codes_str}.")
    else:
        rules.append("Do not include reason_code in the JSON response.")
    if profile.dataset_id == "heartsteps" and variant == "heartsteps-zero-inflated":
        rules.extend(
            [
                "For this variant, adherence is a realized 30-minute outcome, not only a smoothed expected value.",
                "Exact 0.0 and very small values are valid when no meaningful post-decision walking occurs.",
            ]
        )
    elif profile.dataset_id == "heartsteps" and variant == "heartsteps-conservative-action":
        rules.extend(
            [
                "For this variant, estimate the context/history baseline before considering the suggestion action.",
                "Do not assign a positive intervention effect unless the current context makes activity plausible.",
            ]
        )
    elif profile.dataset_id == "heartsteps" and variant == "heartsteps-soft-zero-calibrated":
        rules.extend(
            [
                "For this variant, output a calibrated realized value: exact zeros are allowed, but the positive windows must preserve the participant's mean activity level.",
                "Do not smooth every inactive-looking window into a small positive value, and do not collapse all mixed evidence to zero.",
                "Estimate the no-suggestion baseline first; apply only a small context-supported action adjustment.",
            ]
        )
    elif profile.dataset_id == "heartsteps" and variant == "heartsteps-mean-preserving-mixture":
        rules.extend(
            [
                "For this variant, preserve both the zero/near-zero frequency and the overall activity mean implied by the warm-start and current context.",
                "If you output zeros frequently, nonzero windows should usually be meaningfully positive, not only 0.03-0.06.",
                "No-suggestion is a baseline condition, not a no-activity condition; baseline walking can still be nonzero.",
            ]
        )
    elif profile.dataset_id == "heartsteps" and variant == "heartsteps-hurdle-calibrated":
        rules.extend(
            [
                "For this variant, use a calibrated hurdle forecast: first decide whether the realized window is inactive/zero, then choose the nonzero magnitude if active.",
                "Exact 0.0 is valid for inactive windows, but nonzero windows should often be meaningfully positive so the patient-level mean does not collapse.",
                "Preserve both mixture components: realistic zero frequency and realistic positive bursts.",
            ]
        )
    elif profile.dataset_id == "heartsteps" and variant == "heartsteps-context-faithful":
        rules.extend(
            [
                "For this variant, keep the unit of prediction fixed: one decision point, one 30-minute post-decision step response.",
                "Current slot, pre_steps, transit, location, weather, and recognized activity are stronger evidence than broad profile traits.",
            ]
        )
    elif profile.dataset_id == "heartsteps" and variant == "heartsteps-human-noise":
        rules.extend(
            [
                "For this variant, model ordinary noisy human behavior; helpful suggestions are often ignored, delayed, or only partially acted on.",
                "Avoid overly smooth trajectories and avoid treating the user as optimized for the study objective.",
            ]
        )
    elif profile.dataset_id == "heartsteps" and variant == "heartsteps-structured-forecast":
        rules.extend(
            [
                "For this variant, decompose the forecast internally into baseline tendency, context feasibility, intervention nudge, and short-window noise.",
                "Return only the required JSON object; do not add fields for the internal decomposition.",
            ]
        )
    elif profile.dataset_id == "heartsteps" and variant == "heartsteps-stochastic-realized":
        rules.extend(
            [
                "For this variant, adherence is one realized simulated sample from the plausible next-window distribution, not only the conditional mean.",
                "Use the simulation randomness cue to choose lower, typical, or upper-tail plausible outcomes while respecting context.",
            ]
        )
    elif profile.dataset_id == "heartsteps" and variant == "heartsteps-stochastic-bursty":
        rules.extend(
            [
                "For this variant, preserve bursty tracker behavior: quiet windows are common, but occasional meaningful positive windows must remain possible.",
                "Use the simulation randomness cue to decide when an upper-tail burst is plausible; do not smooth every active-looking window into a small value.",
            ]
        )
    rules_str = "\n".join(f"- {r}" for r in rules)
    target_phrase = _prediction_target_phrase(profile)
    return (
        f"Predict {target_phrase}.\n\n"
        f"Return this JSON:\n{json.dumps(schema, indent=2)}\n\n"
        f"{rules_str}"
    )


def _build_context_section(profile: DatasetPromptProfile, obs: Any) -> str:
    """Render ONLY whitelisted pre-decision context fields."""
    whitelist_keys = {key for key, _ in profile.context_field_labels}
    lines = [_context_header(profile)]
    for key, label in profile.context_field_labels:
        val = obs.context.get(key)
        if val is None:
            continue
        lines.append(f"- {label}: {val}")
    # Warn if any context key is not in the whitelist (silent -- filtered out)
    # This is deliberate: loaders may over-collect; the whitelist is the guard.
    _ = whitelist_keys  # used for guard documentation
    return "\n".join(lines)


def _prediction_target_phrase(profile: DatasetPromptProfile) -> str:
    if profile.dataset_id == "heartsteps":
        return f"the current decision-point window's {profile.outcome_name}"
    if profile.dataset_id == "hptn067":
        return f"the current interview interval's {profile.outcome_name}"
    return f"the current step's {profile.outcome_name}"


def _context_header(profile: DatasetPromptProfile) -> str:
    if profile.dataset_id == "heartsteps":
        return "Current decision-point context (safe pre-outcome fields only):"
    if profile.dataset_id == "hptn067":
        return "Current interview interval context (safe pre-outcome fields only):"
    return "Current step context (safe pre-outcome fields only):"


def _build_observation_window_section(
    profile: DatasetPromptProfile,
    outcomes: list[float],
    prefix_format: str,
    actions: list[str] | None = None,
    contexts: list[dict[str, Any]] | None = None,
    *,
    anchor_days: int = 0,
    source_label: str = "observed",
) -> str:
    """Render the warm-start observation window.

    When ``anchor_days`` > 0 this section represents the K real measured
    outcomes that anchor the simulation. The block is immutable and the
    LLM must not revise it. When ``anchor_days`` == 0 (unanchored single-section
    mode) the same renderer is used to display whichever history was
    selected by ``source_label`` ("observed" or "simulated").
    """
    n = len(outcomes)
    if anchor_days and anchor_days > 0:
        if not outcomes:
            return (
                f"Observed warm-start prefix ({profile.outcome_name}): "
                "anchored mode is configured but the warm-start window is "
                "empty -- treat this prediction as if no prior history is "
                "available."
            )
        header = (
            f"Observed warm-start prefix for {profile.outcome_name} "
            f"(REAL measured outcomes for the first {anchor_days} step(s) of "
            f"this participant -- {n} step(s) shown).\n"
            "IMPORTANT: these values are real observations, NOT predictions. "
            "Do not predict or revise them; use them as the immutable "
            "anchor for the simulated steps that follow."
        )
        block = _build_prefix_block(outcomes, prefix_format, actions, contexts)
        return f"{header}\n{block}"

    # Unanchored single-section mode.
    label = source_label if source_label in {"observed", "simulated"} else "observed"
    if not outcomes:
        return (
            f"Prior {label} history ({profile.outcome_name}): "
            "No prior steps -- this is the first decision point."
        )
    if label == "observed":
        provenance = (
            "IMPORTANT: These values are real measured outcomes from prior "
            "steps. Do not predict or revise them; use them only to condition "
            "your prediction for the current step."
        )
        header_title = "Prior observed history"
    else:
        provenance = (
            "IMPORTANT: These values are your own prior simulated predictions, "
            "not measured outcomes. Use them only to condition the current "
            "step; do not revise them."
        )
        header_title = "Prior simulated history"
    header = (
        f"{header_title} for {profile.outcome_name} "
        f"({n} step(s) shown, most recent last).\n"
        f"{provenance} "
        "The CURRENT step's outcome is NOT shown and must be inferred."
    )
    block = _build_prefix_block(outcomes, prefix_format, actions, contexts)
    return f"{header}\n{block}"


def _build_simulated_history_section(
    profile: DatasetPromptProfile,
    outcomes: list[float],
    actions: list[str] | None,
    contexts: list[dict[str, Any]] | None,
    *,
    anchor_days: int,
) -> str:
    """Render the post-anchor autoregressive history block.

    In anchored mode this is the LLM's own simulated trajectory from step
    ``anchor_days + 1`` onward. It is rendered day-by-day, with the
    intervention shown when actions are available (raw-actions style).
    """
    if not outcomes:
        return (
            f"Recent simulated history for {profile.outcome_name} "
            f"(post-anchor steps, i.e. step {anchor_days + 1} onward):\n"
            "No simulated steps yet -- this is the first prediction after "
            "the anchored warm-start window."
        )
    n = len(outcomes)
    header = (
        f"Recent simulated history for {profile.outcome_name} "
        f"(post-anchor predictions from step {anchor_days + 1} onward; "
        f"{n} step(s) shown, most recent last).\n"
        "IMPORTANT: these values are your own prior simulated predictions, "
        "NOT measured outcomes. Do not revise them; use them only to "
        "condition the current step's prediction. The CURRENT step's "
        "outcome is NOT shown and must be inferred."
    )
    # Always render simulated history day-by-day (with actions when given) so
    # the LLM sees its own past predictions explicitly rather than collapsed
    # into a summary. Step numbering picks up where the warm-start window
    # left off so the LLM sees a continuous step-index timeline.
    fmt = "raw-actions" if actions else "raw"
    block = _build_prefix_block(
        outcomes, fmt, actions, contexts, start_index=anchor_days + 1,
    )
    return f"{header}\n{block}"


def _build_intervention_section(profile: DatasetPromptProfile, action: str) -> str:
    action_desc_map = dict(profile.action_descriptions)
    desc = action_desc_map.get(action, action)
    is_baseline = action == profile.baseline_action
    baseline_note = " (baseline / no active intervention)" if is_baseline else ""
    return (
        f"Current intervention / action:\n"
        f"- Action: {action}{baseline_note}\n"
        f"- Description: {desc}\n"
        "Note: intervention effects at any single decision point are typically "
        "small and context-dependent. Do not assume the intervention caused a "
        "large outcome shift."
    )
