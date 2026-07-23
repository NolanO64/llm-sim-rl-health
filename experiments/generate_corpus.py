"""Generate an LLM-world training corpus.

For each synthetic patient an exploratory behaviour policy samples a sequence of
actions, and the language model autoregressively produces the trajectory -- the
context, activity tendency and disengagement each day -- with the activity passed
through the emission layer (self-anchored on the patient's own realised history).
The resulting corpus is what the offline learners train on.

Requires a Nebula API key (NEBULA_API_KEY). The committed corpora under
data/corpora were produced this way; a full run of several hundred patients takes
a while because it is one model call per simulated day.

  python experiments/generate_corpus.py smoke                 # 2 patients, printed
  python experiments/generate_corpus.py 400 data/corpora/llmworld_corpus_new.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm_client import build_client
from src.llm_world import EPISODE_LENGTH, generate_trajectory, persona


def sample_actions(rng):
    """Exploratory behaviour policy: varied actions so the corpus covers all of them."""
    return [int(rng.integers(4)) for _ in range(EPISODE_LENGTH)]


def build_patient(client, index, base_seed=1000):
    rng = np.random.default_rng(base_seed + index)
    person = persona(rng)
    actions = sample_actions(rng)
    trajectory = generate_trajectory(client, person, actions, seed=base_seed + index)
    return {"persona": person, "trajectory": trajectory}


def main():
    client = build_client()
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"

    if mode == "smoke":
        for i in range(2):
            patient = build_patient(client, i)
            outcomes = [s["outcome"] for s in patient["trajectory"]]
            print("patient %d: %s" % (i, patient["persona"]))
            print("  %d days | mean activity %.3f | zero-rate %.2f"
                  % (len(outcomes), sum(outcomes) / max(len(outcomes), 1),
                     sum(1 for y in outcomes if y == 0) / max(len(outcomes), 1)))
        return

    n_patients = int(mode)
    out = sys.argv[2] if len(sys.argv) > 2 else "data/corpora/llmworld_corpus_new.json"
    corpus = []
    failures = []
    for i in range(n_patients):
        try:
            corpus.append(build_patient(client, i))
            if i % 10 == 0:
                print("patient %d done" % i, flush=True)
        except Exception as error:
            failures.append({"patient_index": i, "error": str(error)})
            print("patient %d failed: %s" % (i, str(error)[:90]), flush=True)
    with open(out, "w") as f:
        json.dump(corpus, f)
    print("saved %d patients -> %s" % (len(corpus), out))
    if failures:
        failure_path = str(Path(out).with_suffix(".failures.json"))
        with open(failure_path, "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2)
        raise RuntimeError(
            "corpus generation had %d failed patients; partial corpus saved to %s and failures to %s"
            % (len(failures), out, failure_path)
        )


if __name__ == "__main__":
    main()
