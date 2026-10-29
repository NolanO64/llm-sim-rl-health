import unittest

from experiments.heartsteps_source_audit import action, match_compact, number, outcome
from experiments.heartsteps_effect_audit import audit_heartsteps, contrast


class HeartStepsAuditTests(unittest.TestCase):
    def test_outcome_priority_does_not_use_zero_imputation(self):
        row = {'gfsteps30': 'NA', 'jbsteps30': '5', 'gfsteps60': '100', 'jbsteps30.zero': '0'}
        self.assertEqual(outcome(row), (5., 'jbsteps30'))
        self.assertIsNone(number('nan'))

    def test_randomization_draw_is_not_eligibility(self):
        self.assertEqual(action({'send': 'False', 'is.randomized': 'False'}), 0)
        self.assertEqual(action({'send': 'True', 'send.active': 'True'}), 1)

    def test_compact_match_checks_actions(self):
        raw = [[{'send': 'False', 'gfsteps30': '150'}]]
        compact = {'patients': [{'steps': [{'gt_adherence': .1, 'action_flat': 0}]}]}
        self.assertTrue(match_compact(raw, compact)['exact_match'])
        compact['patients'][0]['steps'][0]['action_flat'] = 1
        self.assertFalse(match_compact(raw, compact)['exact_match'])

    def test_patient_balancing_is_not_observation_weighting(self):
        people = [[('0',0.,0.),('1',1.,1.)], [('0',0.,0.)]+[('1',0.,0.)]*3]
        self.assertEqual(contrast(people, '1', 1), .25)
        self.assertEqual(contrast(people, '1', 1, balanced=True), .5)

    def test_paired_error_zero_for_identical_trajectories(self):
        data = {'patients': [{'steps': [{'action_flat': a, 'gt_adherence': y, 'adherence': y}
                                       for a,y in ((0,.1),(1,.2),(2,.3))]} for _ in range(3)]}
        result = audit_heartsteps(data, n_boot=20)
        for row in result['contrasts'].values():
            self.assertEqual(row['sim_minus_real'], 0.)
            self.assertEqual(row['paired_patient_bootstrap_bias_ci95'], [0.,0.])

    def test_invalid_action_rejected(self):
        with self.assertRaises(ValueError):
            audit_heartsteps({'patients': [{'steps': [{'action_flat': 9, 'gt_adherence': .1}]}]}, n_boot=2)


if __name__ == '__main__':
    unittest.main()
