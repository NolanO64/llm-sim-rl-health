"""Run reference-dependent tests in subprocesses to preserve training isolation."""
import subprocess
import sys
import unittest


class DynamicsTests(unittest.TestCase):
    def check_code(self, code):
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_grid_is_one_factor_at_a_time(self):
        self.check_code("""
from experiments.reference_dynamics_sweep import settings_grid, make_reference
grid = settings_grid()
assert len(grid) == 7
for label, settings in grid.items():
    differences = [key for key in settings if settings[key] != grid['baseline'][key]]
    assert len(differences) == (0 if label == 'baseline' else 1)
    env = make_reference(0, settings)
    assert env.\u03c11 == 50 * settings['effect_multiplier']
    assert env.\u03c12 == 200 * settings['effect_multiplier']
    assert env.\u03b5h == settings['habituation_increment']
    assert env.\u03b4h == settings['habituation_recovery']
    assert env.\u03c3s == 20 and env.sigma == 2 and env.chosen_a == .3
""")

    def test_baseline_matches_original_episode_logger(self):
        self.check_code("""
from experiments.reference_dynamics_sweep import settings_grid, evaluate
from experiments.ranking_reference_uncertainty import evaluate_episodes
from src.policies import random_action, matched_heuristic
for policy in (random_action, matched_heuristic):
    new = evaluate(policy, settings_grid()['baseline'], range(2), 4)
    old = evaluate_episodes(policy, range(2), 4)
    assert [e for row in new['seed_rows'] for e in row['returns']] == [e['return'] for e in old]
    assert [e for row in new['seed_rows'] for e in row['lengths']] == [e['length'] for e in old]
""")

    def test_cached_baseline_rejects_changed_input(self):
        self.check_code("""
import json
from experiments.reference_dynamics_sweep import cached_baseline
from src.paths import RESULTS_DIR
cache = json.loads((RESULTS_DIR/'ranking_reference_uncertainty.json').read_text())
saved = [r for r in json.loads((RESULTS_DIR/'benchmark_robust.json').read_text())['rows'] if r['policy'] != 'observable-ceiling']
rows = cached_baseline(cache, saved, cache['source_sha256'], list(range(8)), 80)
assert len(rows) == 49
assert all(round(a['mean']) == b['real_new'] for a,b in zip(rows,saved))
hashes = {**cache['source_sha256'], 'src/policies.py': 'changed'}
try:
    cached_baseline(cache, saved, hashes, list(range(8)), 80)
except ValueError:
    pass
else:
    raise AssertionError('Changed policies accepted')
""")


if __name__ == '__main__':
    unittest.main()
