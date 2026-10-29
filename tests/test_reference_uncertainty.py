import subprocess
import gzip
from pathlib import Path
import sys
import tempfile
import unittest


class ReferenceUncertaintyTests(unittest.TestCase):
    def test_restore_preserves_bytes_and_refuses_overwrite(self):
        from experiments.restore_evidence import restore
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'data').mkdir()
            raw = b'{"value": 1}\n'
            target = root / 'data/example.json'
            target.with_suffix('.json.gz').write_bytes(gzip.compress(raw, mtime=0))
            restore(root)
            self.assertEqual(target.read_bytes(), raw)
            restore(root)
            target.write_bytes(b'different')
            with self.assertRaises(ValueError):
                restore(root)
            self.assertEqual(target.read_bytes(), b'different')

    def check_code(self, code):
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_episode_logger_matches_canonical(self):
        self.check_code("""
import numpy as np
from experiments.ranking_reference_uncertainty import evaluate_episodes
from src.policies import random_action, matched_heuristic
from src.stepcount import evaluate_real
for policy in (random_action, matched_heuristic):
    records = evaluate_episodes(policy, range(2), n_episodes=3)
    assert abs(np.mean([r['return'] for r in records]) - evaluate_real(policy, seeds=range(2), n_episodes=3)) < 1e-12
    assert len(records) == 6
""")

    def test_paired_bootstrap_retains_fixed_gaps(self):
        self.check_code("""
import numpy as np
from experiments.ranking_reference_uncertainty import summarize
a = np.arange(8)[None,:] + np.array([10.,20.,30.])[:,None]
r = summarize(a, [3.,2.,1.], ['a','b','c'], n_boot=100)
assert r['point']['regret'] == 20.
assert r['conditional_percentile_intervals95']['regret'] == [20.,20.]
""")

    def test_exact_ties_use_original_candidate_order(self):
        self.check_code("from experiments.ranking_reference_uncertainty import ranking; assert ranking([2.,3.],[1.,1.])['selected_index'] == 0")


if __name__ == '__main__':
    unittest.main()
