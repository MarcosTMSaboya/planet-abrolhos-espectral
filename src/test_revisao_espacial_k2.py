import unittest
import numpy as np
from revisao_espacial_k2 import align_labels, agreement


class SpatialReviewTests(unittest.TestCase):
    def test_alignment_removes_arbitrary_permutation(self):
        ref = np.tile([0, 0, 1], 100)
        np.testing.assert_array_equal(align_labels(ref, 1-ref, 2), ref)
        three = np.tile([0, 1, 2], 100)
        np.testing.assert_array_equal(align_labels(three, (three+1)%3, 3), three)

    def test_joint_selection_and_missing_groups(self):
        ref = np.tile([0, 1], 150)
        changed = ref.copy(); changed[200:] = 0
        row = agreement(ref, changed, np.arange(300)<200, 2)
        self.assertEqual(row['n_pixels'], 200)
        self.assertEqual(row['adjusted_rand'], 1.)
        missing = agreement(ref, np.zeros(300), np.ones(300,dtype=bool), 2)
        self.assertEqual(missing['status'], 'missing_groups')
        self.assertIsNone(missing['adjusted_rand'])


if __name__ == '__main__':
    unittest.main()
