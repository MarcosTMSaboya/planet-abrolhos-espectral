import unittest
import numpy as np
from subdivisoes_k2_k3 import cross_codes,explained_fraction


class CrossingTests(unittest.TestCase):
    def test_crossing_keeps_both_cluster_ids(self):
        np.testing.assert_array_equal(cross_codes([1,1,1,2,2,2],[1,2,3,1,2,3]),[11,12,13,21,22,23])
        with self.assertRaises(ValueError):cross_codes([0],[1])
        with self.assertRaises(ValueError):cross_codes([1,2],[1])

    def test_variance_fraction_limits(self):
        self.assertEqual(explained_fraction([0,0,1,1],[1,1,2,2]),1.)
        self.assertEqual(explained_fraction([0,1,0,1],[1,1,2,2]),0.)
        self.assertIsNone(explained_fraction([1,1],[1,2]))


if __name__=='__main__':unittest.main()
