import unittest


class TrajectoryAlignmentTests(unittest.TestCase):
    def test_saved_accuracies_and_aggregate_only_output(self):
        from experiments.trajectory_alignment_audit import analyze
        result = analyze()
        expected = ((.973, .973, .986), (.743, .743, .757), (.716, .716, .730))
        for row, values in zip(result['variants'], expected):
            self.assertEqual(row['n_patient_pairs'], 37)
            for key, value in zip(('mean_aligned', 'unaligned_same_features', 'unaligned_including_mean'), values):
                self.assertEqual(row[key]['pooled'], value)
                self.assertNotIn('p_binom', row[key])
                self.assertNotIn('patients', row[key])


class RankingRobustnessTests(unittest.TestCase):
    def test_batch_and_family_results(self):
        from experiments.ranking_robustness_audit import analyze
        result = analyze()
        self.assertEqual([r['regret'] for r in result['per_batch']], [202., 149., 196.])
        self.assertAlmostEqual(result['within_family']['offline_rl']['relative_regret_percent'], 100*6/913)
        self.assertEqual(result['exploratory_batch_resampling']['ordered_resamples'], 27)
        self.assertEqual(len(result['leave_one_family_out']), 6)

    def test_ties_keep_candidate_order(self):
        from experiments.ranking_robustness_audit import ranking
        rows = [{'policy': 'a', 'real_new': 2}, {'policy': 'b', 'real_new': 3}]
        self.assertEqual(ranking(rows, [1,1])['selected'], 'a')


class DirectSummaryTests(unittest.TestCase):
    def test_direct_mean_best_and_mini_sd(self):
        from experiments.direct_training_summary import analyze
        r = analyze()
        self.assertAlmostEqual(r['direct_observable_training']['mean'], 1005.3333333333334)
        self.assertEqual(r['direct_observable_training']['maximum'], 1080)
        self.assertAlmostEqual(r['gpt5mini']['mean'], 647.112, places=2)
        self.assertAlmostEqual(r['gpt5mini']['population_sd'], 201.457, places=2)

    def test_invalid_returns_rejected(self):
        from experiments.direct_training_summary import summary
        for values in ([], [1], [1,float('nan')]):
            with self.assertRaises(ValueError):
                summary(values)


if __name__ == '__main__':
    unittest.main()
