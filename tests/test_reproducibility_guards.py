import math
import unittest
from types import SimpleNamespace

from src.llm_world import _bounded_activity, _require_json
from src.metrics import spearman


class ReproducibilityGuardTests(unittest.TestCase):
    def test_llm_json_response_must_have_required_keys(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"activity": 0.4}'),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )
        with self.assertRaises(ValueError):
            _require_json(response, required_keys=("activity", "quit", "next_context"))

    def test_activity_is_clipped_but_not_silently_defaulted(self):
        self.assertEqual(_bounded_activity(-0.2), 0.0)
        self.assertEqual(_bounded_activity(1.7), 1.0)
        self.assertEqual(_bounded_activity("0.25"), 0.25)
        with self.assertRaises(ValueError):
            _bounded_activity("not-a-number")

    def test_spearman_uses_average_ranks_for_ties(self):
        # Standard Spearman with average ranks for [1, 1, 2, 3] vs [1, 2, 2, 4].
        self.assertTrue(math.isclose(spearman([1, 1, 2, 3], [1, 2, 2, 4]), 5 / 6))


if __name__ == "__main__":
    unittest.main()
