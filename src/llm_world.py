"""The language model as the environment: one model call per simulated day.

A synthetic patient is described by a short persona. Each day the model is given
the persona and the realised history so far and decides the patient's activity
tendency, whether they have now disengaged, and the next day's context. Every
latent tendency is passed through the two-part emission layer, self-anchored on
the patient's own realised outcomes, so the history fed back to the model is the
emitted outcome rather than the raw tendency. No real data and no StepCountJITAI
environment are involved in generation -- that is the point of the testbed.

Two rollout modes share this machinery:

  generate_trajectory -- a behaviour policy supplies the actions and the model
                         also invents the daily context; used to build the
                         offline training corpus.
  rollout_policy      -- an external policy chooses each action from the current
                         context; used to score that policy inside the LLM world.
"""
import json
import random
import re
import sys

import numpy as np

from .llm_client import chat, reasoning_extra_body
from .paths import REPO_ROOT

# The emission layer is shared with the HeartSteps realism testbed; import the one
# canonical implementation from the vendored simulator package rather than copying it.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from simulator.adherence_simulator.outcome_models import (  # noqa: E402
    OutcomeModelState,
    TwoPartLognormalOutcomeModel,
)

ACTIONS = ["none", "generic", "match_A", "match_B"]
EPISODE_LENGTH = 40

MODEL = "SURF.Qwen3.5 122B A10B NVFP4"

_BASELINE = ["very sedentary", "low-activity", "moderately active", "quite active", "very active"]
_REACTIVITY = ["barely reacts to app prompts", "somewhat reacts to prompts",
               "reacts to prompts", "reacts strongly to prompts"]
_HABITUATION = ["tunes out repeated messages very slowly", "tunes out slowly",
                "average tolerance", "tunes out quickly"]

GENERATION_SYSTEM = (
    "You simulate, ONE DAY AT A TIME, how a specific person responds to a physical-activity "
    "coaching app. Each day the person is in a situational context (A or B) and the app takes an "
    "action: none / generic / match_A (message tailored to context A) / match_B (tailored to B). "
    "You are given THIS person and the realised history of all previous days. Using your understanding "
    "of human behaviour, decide for TODAY ONLY: the context they are in, their activity response "
    "tendency on 0.0-1.0 (0=no activity in the window, 1=strong activity), and whether they have now "
    "DISENGAGED and quit the app for good (quit=true). Account for the person traits, for HABITUATION "
    "(repeated messaging tends to wear off over the days you can see in the history), and for "
    "DISENGAGEMENT (irrelevant or excessive messaging can make them stop). Decide from the person and "
    "the history -- do not assume any action is best. "
    'Output ONLY JSON for the single day: {"context":"A"|"B","activity":0.0-1.0,"quit":true|false}.'
)

# Step prompt used for online training and for the policy-scoring benchmark.
SCORING_SYSTEM = (
    "You simulate ONE specific person in a physical-activity coaching app, one day at a time, like the "
    "step function of an environment. Given the person, the realised history, today's context, and the "
    "app action today (none / generic / match_A / match_B), output: (1) the activity tendency TODAY "
    "0.0-1.0; (2) whether they have now DISENGAGED and quit; (3) the context TOMORROW (A or B). "
    "Account for habituation (repeated messaging wears off) and disengagement (irrelevant/excessive "
    "messaging makes them stop). A matched message helps more than generic, but do not assume any "
    "action is best. "
    'Output JSON {"activity":0.0-1.0,"quit":true|false,"next_context":"A"|"B"}.'
)

def persona(rng):
    """Sample a templated patient description (age and three behaviour traits)."""
    return ("age %d; baseline %s; %s; %s" % (
        int(rng.integers(20, 80)),
        _BASELINE[rng.integers(len(_BASELINE))],
        _REACTIVITY[rng.integers(len(_REACTIVITY))],
        _HABITUATION[rng.integers(len(_HABITUATION))],
    ))


def make_emitter():
    return TwoPartLognormalOutcomeModel(sigma=0.90, anchor_weight=0.50)


def _parse_json(text):
    """Parse the model's JSON, salvaging the first {...} block if needed."""
    try:
        return json.loads(text or "{}")
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text or "", re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
        return {}


def _response_debug(response):
    choice = response.choices[0]
    message = getattr(choice, "message", None)
    content = getattr(message, "content", "") or ""
    usage = getattr(response, "usage", None)
    reasoning_tokens = None
    if usage is not None:
        details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(details, "reasoning_tokens", None)
    return {
        "finish_reason": getattr(choice, "finish_reason", None),
        "content_preview": content[:120],
        "reasoning_tokens": reasoning_tokens,
    }


def _require_json(response, required_keys):
    content = response.choices[0].message.content or ""
    parsed = _parse_json(content)
    missing = [key for key in required_keys if key not in parsed]
    if missing:
        raise ValueError(
            "LLM response missing required JSON keys %s; debug=%s"
            % (missing, _response_debug(response))
        )
    return parsed


def _bounded_activity(value):
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid activity value from LLM: {value!r}") from error


def _chat(client, *, backend="nebula", **kwargs):
    if backend.lower() == "openai":
        if "max_tokens" in kwargs:
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        if str(kwargs.get("model", "")).startswith("gpt-5"):
            kwargs.pop("temperature", None)
            kwargs.setdefault("reasoning_effort", "minimal")
    extra_body = reasoning_extra_body(backend=backend)
    if extra_body is not None:
        kwargs["extra_body"] = extra_body
    return chat(client, **kwargs)


def _emit(emitter, latent, action, history, rng):
    state = OutcomeModelState(
        dataset_id="llmworld", patient_id="synthetic",
        anchor_values=tuple(history), population_anchor_values=tuple(history),
    )
    result = emitter.decode(
        latent=latent, state=state, action=action, context={},
        simulated_history=history, rng=rng, deterministic=False, certainty=0.5,
    )
    return float(result.value)


def _generation_history(rows):
    if not rows:
        return "No previous days yet -- this is day 0."
    lines = [
        "day %d: context %s, app action %s, activity %.2f%s"
        % (t, r["context"], ACTIONS[r["action"]], r["outcome"], ", then QUIT" if r["quit"] else "")
        for t, r in enumerate(rows)
    ]
    return "History of previous days:\n" + "\n".join(lines)


def generate_trajectory(client, person, actions, seed, model=MODEL, backend="nebula",
                        temperature=0.7, max_tokens=200):
    """Autoregressively generate one patient's trajectory (one model call per day)."""
    emitter = make_emitter()
    emit_rng = random.Random(seed)
    history = []   # realised outcomes -> emission self-anchor
    rows = []
    for t, action in enumerate(actions):
        prompt = (
            "This person: %s.\n%s\nToday is day %d. The app takes action: %s. "
            "Decide TODAY only: the context (A or B), the activity tendency 0.0-1.0, and whether the "
            "person has now disengaged and quit for good. "
            'Output JSON {"context":"A"|"B","activity":0.0-1.0,"quit":true|false}.'
            % (person, _generation_history(rows), t, ACTIONS[action])
        )
        response = _chat(
            client, model=model,
            backend=backend,
            messages=[{"role": "system", "content": GENERATION_SYSTEM},
                      {"role": "user", "content": prompt}],
            temperature=temperature, max_tokens=max_tokens, response_format={"type": "json_object"},
            seed=seed + t,
        )
        parsed = _require_json(response, required_keys=("activity", "context", "quit"))
        latent = _bounded_activity(parsed["activity"])
        context = "A" if str(parsed.get("context", "A")).upper().startswith("A") else "B"
        quit_ = bool(parsed.get("quit", False))
        outcome = _emit(emitter, latent, ACTIONS[action], history, emit_rng)
        history.append(outcome)
        rows.append({"context": context, "action": action, "latent": round(latent, 3),
                     "outcome": round(outcome, 3), "quit": quit_})
        if quit_:
            break
    return rows


def _scoring_history(rows):
    if not rows:
        return "No previous days yet."
    lines = [
        "day %d: ctx %s, action %s, activity %.2f%s"
        % (t, r["context"], ACTIONS[r["action"]], r["outcome"], ", QUIT" if r["quit"] else "")
        for t, r in enumerate(rows)
    ]
    return "History:\n" + "\n".join(lines)


def llm_day_step(client, person, rows, context, action_name, model=MODEL,
                 system=SCORING_SYSTEM, backend="nebula", temperature=0.7,
                 max_tokens=120):
    """One environment step: given the action, the model returns today's latent
    activity, whether the patient has quit, and tomorrow's context."""
    prompt = (
        "This person: %s.\n%s\nToday is day %d. Today's context is %s. The app takes action: %s. "
        'Output JSON {"activity":0.0-1.0,"quit":true|false,"next_context":"A"|"B"}.'
        % (person, _scoring_history(rows), len(rows), context, action_name)
    )
    response = _chat(
        client, model=model,
        backend=backend,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=temperature, max_tokens=max_tokens, response_format={"type": "json_object"},
    )
    parsed = _require_json(response, required_keys=("activity", "quit", "next_context"))
    latent = _bounded_activity(parsed["activity"])
    quit_ = bool(parsed.get("quit", False))
    next_context = "A" if str(parsed.get("next_context", context)).upper().startswith("A") else "B"
    return latent, quit_, next_context


def emit_outcome(emitter, latent, action_name, history, rng):
    """Pass a latent tendency through the emission layer, self-anchored on ``history``."""
    return _emit(emitter, latent, action_name, history, rng)


def rollout_policy(client, policy, seed, model=MODEL, system=SCORING_SYSTEM, backend="nebula",
                   episode_length=EPISODE_LENGTH, temperature=0.7, max_tokens=120):
    """Score one policy on one synthetic patient inside the LLM world.

    The policy chooses each action from the current context; the model returns the
    activity, whether the patient quit, and the next context. Returns the patient's
    total realised activity over the episode.
    """
    emitter = make_emitter()
    rng = np.random.default_rng(seed)
    emit_rng = random.Random(seed + 7)
    policy_rng = np.random.default_rng(seed + 13)
    person = persona(rng)
    context = "A" if rng.random() < 0.5 else "B"

    streak = 0
    rows = []
    total = 0.0
    for _ in range(episode_length):
        context_index = 0 if context == "A" else 1
        action = int(policy({"c_infer": context_index, "c_true": context_index}, streak, policy_rng))
        latent, quit_, next_context = llm_day_step(
            client, person, rows, context, ACTIONS[action], model=model, system=system,
            backend=backend, temperature=temperature, max_tokens=max_tokens
        )
        outcome = _emit(emitter, latent, ACTIONS[action], [r["outcome"] for r in rows], emit_rng)
        rows.append({"context": context, "action": action, "outcome": round(outcome, 3), "quit": quit_})
        streak = 0 if action == 0 else streak + 1
        total += outcome
        if quit_:
            break
        context = next_context
    return total
