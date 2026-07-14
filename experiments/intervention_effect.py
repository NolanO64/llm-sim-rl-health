"""Interventional effect of each app action on HeartSteps (Table: intervention).

For the real data and for the LLM simulation of the same patients, this estimates
the average change in next-step adherence when the app takes an action rather than
sending no suggestion:

    effect(action) = E[outcome | action] - E[outcome | no_suggestion]

with a bootstrap confidence interval over patients. A nonzero, correctly-signed
effect is exactly the property a benchmark needs and that distributional realism
alone does not guarantee, which is why this test sits alongside the C2ST.

Runs on the committed real-vs-simulated trajectory file; no LLM calls.
"""
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import REALISM_DIR

RUNS = [
    ("heartsteps_emission_two_part.json", "two-part emission (Figure 5)"),
    ("heartsteps_emission_stateful_hint.json", "stateful + action-noise hint"),
]

REAL, SIM = 1, 2  # tuple positions of the real and simulated outcome


def load_patients(path):
    """Return per-patient lists of (action, real_outcome, sim_outcome) and an action vocabulary."""
    with open(path) as f:
        data = json.load(f)
    patients = []
    vocab = {}
    for patient in data.get("patients") or []:
        rows = []
        for step in patient.get("steps") or []:
            if step.get("gt_override"):
                continue
            real = step.get("gt_adherence")
            if real is None:
                continue
            action = step.get("action_flat")
            sim = step.get("adherence")
            rows.append((action, float(real), float(sim) if sim is not None else None))
            vocab[action] = vocab.get(action, 0) + 1
        if rows:
            patients.append(rows)
    return patients, vocab


def effect(patients, action_key, none_key, field):
    action_values, none_values = [], []
    for rows in patients:
        for record in rows:
            value = record[field]
            if value is None:
                continue
            if record[0] == action_key:
                action_values.append(value)
            elif record[0] == none_key:
                none_values.append(value)
    if not action_values or not none_values:
        return None
    return sum(action_values) / len(action_values) - sum(none_values) / len(none_values)


def bootstrap_effect(patients, action_key, none_key, field, n_boot=2000, seed=1):
    point = effect(patients, action_key, none_key, field)
    rng = random.Random(seed)
    n = len(patients)
    samples = []
    for _ in range(n_boot):
        resample = [patients[rng.randrange(n)] for _ in range(n)]
        value = effect(resample, action_key, none_key, field)
        if value is not None:
            samples.append(value)
    sd = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    samples.sort()
    lo = samples[int(0.025 * len(samples))]
    hi = samples[int(0.975 * len(samples))]
    return point, sd, lo, hi


def main():
    for filename, label in RUNS:
        patients, vocab = load_patients(REALISM_DIR / filename)
        print(f"\n=== {label} | {filename} ===")
        print("action vocab (counts):", vocab)
        explicit = [k for k in vocab if str(k).lower() in ("no_suggestion", "none", "0", "no_message")]
        none_key = explicit[0] if explicit else min(vocab, key=lambda k: vocab[k])
        print("no-suggestion key: %s | %d patients" % (none_key, len(patients)))
        for action_key in [k for k in vocab if k != none_key]:
            for field, name in ((REAL, "real"), (SIM, "sim")):
                point, sd, lo, hi = bootstrap_effect(patients, action_key, none_key, field)
                print("  %-22s %-5s effect=%+.4f  sd=%.4f  95%% CI [%+.4f, %+.4f]"
                      % (str(action_key), name, point, sd, lo, hi))


if __name__ == "__main__":
    main()
