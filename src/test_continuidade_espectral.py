import unittest
import numpy as np
from continuidade_espectral import gradient_control


class GradientTests(unittest.TestCase):
    def test_validation_does_not_change_training_thresholds(self):
        rng = np.random.default_rng(42)
        train = rng.normal(size=(120, 4))
        val = rng.normal(size=(60, 4))
        first = gradient_control(train, val, 3, np.arange(60))
        second = gradient_control(train, val + 100, 3, np.arange(60))
        np.testing.assert_array_equal(first["thresholds_train"], second["thresholds_train"])
        self.assertEqual(second["status"], "missing_groups")
        self.assertIsNone(second["silhouette_common_bands"])


if __name__ == "__main__":
    unittest.main()
