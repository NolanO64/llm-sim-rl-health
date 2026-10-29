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


if __name__ == '__main__':
    unittest.main()
