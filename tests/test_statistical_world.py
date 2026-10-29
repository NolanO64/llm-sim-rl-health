import hashlib
import json
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.statistical_world_benchmark import train_seed, write_payload
from src.statistical_world import DEFAULT_CONFIG, StatisticalConfig, StatisticalWorld, collect_episode, rollout_policy
from src.synthetic_patient import sample_patient
from src.synthetic_q import refit_q


class StatisticalWorldTests(unittest.TestCase):
    def test_matched_generic_and_no_message_have_ordered_tendencies(self):
        for context in (0, 1):
            world = StatisticalWorld(42)
            world.context = context
            self.assertGreater(world.tendency(2 + context), world.tendency(1))
            self.assertGreater(world.tendency(1), world.tendency(0))
            self.assertEqual(world.tendency(3 - context), world.tendency(0))

    def test_repetition_reduces_response_and_no_message_allows_recovery(self):
        config = replace(DEFAULT_CONFIG, episode_length=80, quit_intercept=0,
                         quit_burden_scale=0, context_switch_probability=0)
        world = StatisticalWorld(42, config)
        action = 2 + world.context
        initial = world.tendency(action)
        for _ in range(8):
            world.step(action)
        fatigued = world.tendency(action)
        fatigue = world.fatigue
        self.assertLess(fatigued, initial)
        for _ in range(5):
            world.step(0)
        self.assertLess(world.fatigue, fatigue)
        self.assertGreater(world.tendency(action), fatigued)

    def test_fixed_seed_reproduces_complete_transition_sequence(self):
        Q = np.zeros((2, 4, 4))
        a = collect_episode(Q, 0.7, 17)
        self.assertEqual(a, collect_episode(Q, 0.7, 17))
        self.assertNotEqual(a, collect_episode(Q, 0.7, 18))
        self.assertTrue(all(0 <= t[3] <= 1 for t in a))
        self.assertFalse(any(t[-1] for t in a[:-1]))
        self.assertTrue(a[-1][-1])

    def test_quitting_is_terminal_and_has_no_bootstrap_successor(self):
        config = replace(DEFAULT_CONFIG, quit_intercept=1, quit_burden_scale=0)
        transitions = collect_episode(np.zeros((2, 4, 4)), 1.0, 3, config)
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0][-3:], (0, 0, True))
        world = StatisticalWorld(3, config)
        world.step(0)
        with self.assertRaises(RuntimeError):
            world.step(0)

    def test_policy_only_receives_deployable_observation(self):
        calls = []

        def policy(obs, streak, rng):
            self.assertEqual(set(obs), {"c_infer"})
            calls.append(streak)
            return 2 + obs["c_infer"]

        score = rollout_policy(policy, 18)
        self.assertGreater(len(calls), 0)
        self.assertTrue(np.isfinite(score))

    def test_no_reference_or_api_modules_imported_by_statistical_world(self):
        code = (
            "import sys; from src.statistical_world import StatisticalWorld; "
            "StatisticalWorld(2).step(0); "
            "assert not any(n.split('.')[0] in ('StepCountJITAI','openai') for n in sys.modules); "
            "assert 'src.llm_world' not in sys.modules"
        )
        root = Path(__file__).resolve().parents[1]
        subprocess.run([sys.executable, "-c", code], cwd=root, check=True, timeout=60)

    def test_shared_persona_sampler_preserves_original_draw_order(self):
        rng = np.random.default_rng(42)
        self.assertEqual([sample_patient(rng).description for _ in range(3)], [
            "age 25; baseline quite active; reacts to prompts; tunes out slowly",
            "age 45; baseline very active; barely reacts to app prompts; average tolerance",
            "age 32; baseline very sedentary; reacts to prompts; tunes out quickly",
        ])

    def test_shared_refit_preserves_historical_numerics(self):
        rng = np.random.default_rng(321)
        buffer = [(int(rng.integers(2)), int(rng.integers(4)), int(rng.integers(4)),
                   float(rng.random()), int(rng.integers(2)), int(rng.integers(4)),
                   bool(rng.integers(2))) for _ in range(100)]
        digest = hashlib.sha256(json.dumps(refit_q(buffer).round(12).tolist()).encode()).hexdigest()
        # Captured from the original online LLM learner before extracting it.
        self.assertEqual(digest, "bc7bef4a9ce507c4841bac71c2285ccf50d438b1701cbe6176726bb110b1c2b0")

    def test_terminal_reward_is_not_bootstrapped(self):
        Q = refit_q([(0, 0, 1, 0.4, 0, 0, True)], passes=1, alpha=1)
        self.assertEqual(Q[0, 0, 1], 0.4)

    def test_training_stops_after_full_round_and_is_reproducible(self):
        config = replace(DEFAULT_CONFIG, episode_length=8, quit_intercept=0, quit_burden_scale=0)
        result = train_seed(2, 25, config, episodes_per_round=3)
        self.assertEqual(result["timesteps"], 48)
        self.assertEqual(result["episodes"], 6)
        self.assertEqual(result, train_seed(2, 25, config, episodes_per_round=3))

    def test_training_fails_if_reference_is_already_imported(self):
        with patch.dict(sys.modules, {"StepCountJITAI": object()}):
            with self.assertRaises(RuntimeError):
                train_seed(0, 1, DEFAULT_CONFIG, 1)

    def test_invalid_config_is_rejected(self):
        for kwargs in ({"context_switch_probability": 1.1}, {"response_multiplier": float("nan")},
                       {"episode_length": 0}, {"baseline_levels": (0.1,)}, {"emission_sigma": -1}):
            with self.assertRaises(ValueError):
                StatisticalConfig(**kwargs)

    def test_cli_saves_protocol_and_refuses_silent_overwrite(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "smoke.json"
            command = [sys.executable, "experiments/statistical_world_benchmark.py", "--smoke",
                       "--variants", "default", "--out", str(output)]
            subprocess.run(command, cwd=root, check=True, capture_output=True, timeout=60)
            payload = json.loads(output.read_text())
            self.assertEqual(payload["stage"], "trained")
            self.assertEqual(payload["protocol"]["primary_variant"], "default")
            self.assertTrue(output.with_suffix(".protocol.json").exists())
            rerun = subprocess.run(command, cwd=root, capture_output=True, timeout=60)
            self.assertNotEqual(rerun.returncode, 0)
            self.assertEqual(payload, json.loads(output.read_text()))
            subprocess.run(command + ["--resume"], cwd=root, check=True, capture_output=True, timeout=60)
            self.assertEqual(payload["training"], json.loads(output.read_text())["training"])
            changed = [sys.executable, "experiments/statistical_world_benchmark.py", "--resume",
                       "--variants", "default", "--out", str(output)]
            mismatch = subprocess.run(changed, cwd=root, capture_output=True, timeout=60)
            self.assertNotEqual(mismatch.returncode, 0)
            self.assertIn(b"protocol/source differs", mismatch.stderr)

    def test_invalid_ranking_config_fails_before_training_or_writes(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "invalid.json"
            command = [sys.executable, "experiments/statistical_world_benchmark.py", "--smoke",
                       "--rank", "--config", "clean", "--out", str(output)]
            run = subprocess.run(command, cwd=root, capture_output=True, timeout=60)
            self.assertNotEqual(run.returncode, 0)
            self.assertFalse(output.exists())

    def test_atomic_checkpoint_retries_a_transient_destination_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "checkpoint.json"
            original_replace = Path.replace
            attempts = []

            def briefly_locked(path, target):
                attempts.append(target)
                if len(attempts) < 3:
                    raise PermissionError("temporary destination lock")
                return original_replace(path, target)

            with patch.object(Path, "replace", briefly_locked), patch("time.sleep"):
                write_payload(output, {"value": 1})
            self.assertEqual(len(attempts), 3)
            self.assertEqual(json.loads(output.read_text()), {"value": 1})


if __name__ == "__main__":
    unittest.main()
