"""Counterfactual action-response probes for LLM-world transfer diagnostics.

For a fixed persona, realised history, and current context, the script asks the
same model to score all four candidate actions. This isolates the latent
action-response signal that online Q-learning needs before the emission layer
and long rollouts amplify small model differences.

Examples:
  python experiments/action_response_probe.py --quick
  python experiments/action_response_probe.py --models qwen=nebula:"SURF.Qwen3.5 122B A10B NVFP4",gptoss=nebula:"FAST.gpt-oss:120b"
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.llm_client import build_client
from src.llm_world import ACTIONS, SCORING_SYSTEM, _chat, _require_json, _scoring_history, persona
from src.paths import DATA_DIR


DEFAULT_MODELS = (
    "qwen=nebula:SURF.Qwen3.5 122B A10B NVFP4,"
    "gptoss=nebula:FAST.gpt-oss:120b,"
    "gemma=nebula:FAST.gemma4:31b,"
    "gpt5mini=openai:gpt-5-mini"
)


def parse_models(spec):
    models = []
    for item in spec.split(","):
        label, target = item.split("=", 1)
        backend, model = target.split(":", 1)
        models.append({"label": label.strip(), "backend": backend.strip(), "model": model.strip()})
    return models


def synthetic_states(n_personas, seed):
    rng = np.random.default_rng(seed)
    histories = [
        ("cold_start", []),
        ("good_match_history", [
            {"context": "A", "action": 2, "outcome": 0.62, "quit": False},
            {"context": "B", "action": 3, "outcome": 0.58, "quit": False},
        ]),
        ("message_fatigue", [
            {"context": "A", "action": 2, "outcome": 0.24, "quit": False},
            {"context": "A", "action": 2, "outcome": 0.18, "quit": False},
            {"context": "A", "action": 2, "outcome": 0.10, "quit": False},
        ]),
        ("mismatch_history", [
            {"context": "A", "action": 3, "outcome": 0.08, "quit": False},
            {"context": "B", "action": 2, "outcome": 0.07, "quit": False},
        ]),
    ]
    states = []
    for person_id in range(n_personas):
        person = persona(rng)
        for history_label, rows in histories:
            for context in ("A", "B"):
                states.append({
                    "person_id": person_id,
                    "persona": person,
                    "history_label": history_label,
                    "rows": rows,
                    "context": context,
                })
    return states


def action_prompt(state, action_name):
    return (
        "This person: %s.\n%s\nToday is day %d. Today's context is %s. The app takes action: %s. "
        'Output JSON {"activity":0.0-1.0,"quit":true|false,"next_context":"A"|"B"}.'
        % (
            state["persona"],
            _scoring_history(state["rows"]),
            len(state["rows"]),
            state["context"],
            action_name,
        )
    )


def chat_kwargs(backend, model, messages, temperature, seed, max_tokens):
    kwargs = {
        "model": model,
        "backend": backend,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if seed is not None and backend.lower() != "openai":
        kwargs["seed"] = seed
    return kwargs


def query_action(client, model_spec, state, action_name, temperature, seed, max_tokens):
    response = _chat(
        client,
        **chat_kwargs(
            model_spec["backend"],
            model_spec["model"],
            [{"role": "system", "content": SCORING_SYSTEM},
             {"role": "user", "content": action_prompt(state, action_name)}],
            temperature,
            seed,
            max_tokens,
        )
    )
    raw_content = response.choices[0].message.content or ""
    parsed = _require_json(response, required_keys=("activity", "quit", "next_context"))
    latent = float(parsed["activity"])
    latent = max(0.0, min(1.0, latent))
    next_context = "A" if str(parsed.get("next_context", state["context"])).upper().startswith("A") else "B"
    return {
        "latent": latent,
        "quit": bool(parsed.get("quit", False)),
        "next_context": next_context,
        "parse_ok": "activity" in parsed,
        "raw_content": raw_content[:500].replace("\r", " ").replace("\n", " "),
    }


def summarize_state(rows):
    by_action = {}
    for action in ACTIONS:
        vals = [r["latent"] for r in rows if r["action_name"] == action]
        quits = [1.0 if r["quit"] else 0.0 for r in rows if r["action_name"] == action]
        by_action[action] = {
            "mean": statistics.fmean(vals),
            "sd": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "quit_rate": statistics.fmean(quits),
        }
    context = rows[0]["context"]
    matched = "match_A" if context == "A" else "match_B"
    mismatched = "match_B" if context == "A" else "match_A"
    means = {a: by_action[a]["mean"] for a in ACTIONS}
    top_action = max(means, key=means.get)
    active_actions = ["generic", "match_A", "match_B"]
    top_active = max(active_actions, key=lambda a: means[a])
    return {
        "model_label": rows[0]["model_label"],
        "person_id": rows[0]["person_id"],
        "history_label": rows[0]["history_label"],
        "context": context,
        "mean_none": means["none"],
        "mean_generic": means["generic"],
        "mean_match_A": means["match_A"],
        "mean_match_B": means["match_B"],
        "matched_margin": means[matched] - means[mismatched],
        "matched_vs_none": means[matched] - means["none"],
        "generic_vs_none": means["generic"] - means["none"],
        "top_action": top_action,
        "top_active": top_active,
        "matched_top_active": top_active == matched,
        "matched_top_overall": top_action == matched,
        "latent_repeat_sd": statistics.fmean(by_action[a]["sd"] for a in ACTIONS),
        "quit_rate": statistics.fmean(by_action[a]["quit_rate"] for a in ACTIONS),
    }


def aggregate(label, state_summaries):
    def mean(key):
        return statistics.fmean(float(s[key]) for s in state_summaries)

    fatigue = [s for s in state_summaries if s["history_label"] == "message_fatigue"]
    nonfatigue = [s for s in state_summaries if s["history_label"] != "message_fatigue"]
    return {
        "model_label": label,
        "n_states": len(state_summaries),
        "matched_top_active_rate": mean("matched_top_active"),
        "matched_top_overall_rate": mean("matched_top_overall"),
        "mean_matched_margin": mean("matched_margin"),
        "mean_matched_vs_none": mean("matched_vs_none"),
        "mean_generic_vs_none": mean("generic_vs_none"),
        "mean_repeat_sd": mean("latent_repeat_sd"),
        "mean_quit_rate": mean("quit_rate"),
        "fatigue_matched_vs_none": statistics.fmean(float(s["matched_vs_none"]) for s in fatigue),
        "nonfatigue_matched_vs_none": statistics.fmean(float(s["matched_vs_none"]) for s in nonfatigue),
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--n-personas", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--out", default=str(DATA_DIR / "results" / "action_response_probe.json"))
    parser.add_argument("--quick", action="store_true", help="two personas and two repeats")
    args = parser.parse_args()

    if args.quick:
        args.n_personas = 2
        args.repeats = 2

    models = parse_models(args.models)
    states = synthetic_states(args.n_personas, args.seed)
    all_rows = []
    all_state_summaries = []
    aggregate_rows = []

    clients = {}
    for model_spec in models:
        backend = model_spec["backend"]
        clients.setdefault(backend, build_client(backend))
        client = clients[backend]
        label = model_spec["label"]
        print("probing %s (%s)" % (label, model_spec["model"]), flush=True)
        model_rows = []
        for state_id, state in enumerate(states):
            for repeat in range(args.repeats):
                for action_index, action_name in enumerate(ACTIONS):
                    call_seed = args.seed + 100_000 * state_id + 100 * repeat + action_index
                    result = query_action(
                        client, model_spec, state, action_name, args.temperature,
                        call_seed, args.max_tokens
                    )
                    row = {
                        "model_label": label,
                        "backend": backend,
                        "model": model_spec["model"],
                        "state_id": state_id,
                        "person_id": state["person_id"],
                        "history_label": state["history_label"],
                        "context": state["context"],
                        "repeat": repeat,
                        "action_name": action_name,
                        **result,
                    }
                    model_rows.append(row)
                    all_rows.append(row)
        grouped = {}
        for row in model_rows:
            key = (row["person_id"], row["history_label"], row["context"])
            grouped.setdefault(key, []).append(row)
        state_summaries = [summarize_state(rows) for rows in grouped.values()]
        all_state_summaries.extend(state_summaries)
        agg = aggregate(label, state_summaries)
        aggregate_rows.append(agg)
        print(
            "  matched-top-active %.2f | margin %.3f | repeat-sd %.3f | fatigue lift %.3f"
            % (
                agg["matched_top_active_rate"],
                agg["mean_matched_margin"],
                agg["mean_repeat_sd"],
                agg["fatigue_matched_vs_none"],
            ),
            flush=True,
        )

    output = {
        "config": {
            "models": models,
            "n_personas": args.n_personas,
            "repeats": args.repeats,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
        "aggregate": aggregate_rows,
        "state_summaries": all_state_summaries,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    write_csv(out_path.with_suffix(".raw.csv"), all_rows)
    write_csv(out_path.with_suffix(".states.csv"), all_state_summaries)
    print("saved -> %s" % out_path, flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
