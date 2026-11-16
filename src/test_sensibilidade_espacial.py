"""Contratos da comparação espacial, sem dados reais ou ajustes pesados."""
import unittest
import numpy as np
from sensibilidade_espacial import paired_agreement, SCENARIOS
from espectral_espacial import spatial_partition


class SensitivityTests(unittest.TestCase):
    def test_ari_is_invariant_to_label_permutation(self):
        first = np.tile([0, 1, 2], 50)
        self.assertEqual(paired_agreement(first, (first+1) % 3)["adjusted_rand"], 1.)

    def test_joint_holdout_uses_only_selected_pixels(self):
        first = np.tile([0, 1, 2], 100)
        second = first.copy(); second[150:] = 0
        selected = np.arange(300) < 150
        row = paired_agreement(first, second, selected)
        self.assertEqual(row["n_pixels"], 150)
        self.assertEqual(row["adjusted_rand"], 1.)
        self.assertLess(paired_agreement(first, second)["adjusted_rand"], 1.)

    def test_no_score_for_small_or_missing_groups(self):
        labels = np.tile([0, 1, 2], 50)
        self.assertEqual(paired_agreement(labels[:90], labels[:90])["status"], "insufficient_pixels")
        self.assertIsNone(paired_agreement(labels, np.zeros(150))["adjusted_rand"])
        with self.assertRaises(ValueError):
            paired_agreement(labels, labels[:-1])

    def test_all_prespecified_geometries_feasible(self):
        valid = np.ones((400, 450), dtype=bool)
        for name, block, buffer, seed in SCENARIOS:
            with self.subTest(name=name):
                split, blocks, meta = spatial_partition(valid, block, buffer, seed=seed)
                self.assertGreaterEqual(meta["counts"]["1"], 20000)
                self.assertGreaterEqual(meta["counts"]["2"], 5000)
                self.assertFalse((blocks.selected_validation & blocks.development_pixels.gt(0)).any())


if __name__ == "__main__":
    unittest.main()
