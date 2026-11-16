"""Testes leves com grades sintéticas; não acessam imagens ou modelos reais."""
import unittest

import numpy as np
from scipy.ndimage import gaussian_filter

from espectral_espacial import lag_diagnostics, spatial_partition, spatial_sample


class SpatialPartitionTests(unittest.TestCase):
    def setUp(self):
        self.valid = np.ones((71, 83), dtype=bool)
        self.valid[5:9, 11:17] = False

    def partition(self, **kwargs):
        options = dict(block_pixels=10, buffer_pixels=3, seed=42)
        options.update(kwargs)
        return spatial_partition(self.valid, **options)

    def test_no_shared_blocks_buffer_distance_and_invalid_preservation(self):
        split, blocks, metadata = self.partition()
        self.assertEqual(split.dtype, np.dtype("uint8"))
        np.testing.assert_array_equal(split == 0, ~self.valid)
        for block in blocks.itertuples():
            values = split[block.row_start:block.row_stop, block.col_start:block.col_stop]
            self.assertFalse((values == 1).any() and (values == 2).any())
        train = np.argwhere(split == 1)
        validation = np.argwhere(split == 2)
        # Teste direto da distância, independente da rotina de dilatação.
        for location in train:
            self.assertGreater(np.abs(validation - location).max(axis=1).min(), 3)
        for location in np.argwhere(split == 3):
            self.assertLessEqual(np.abs(validation - location).max(axis=1).min(), 3)
        self.assertEqual(sum(metadata["counts"].values()), self.valid.size)
        self.assertEqual(int(blocks["valid_pixels"].sum()), int(self.valid.sum()))

    def test_reproducible_single_draw_and_seed_changes_selection(self):
        first, blocks, metadata = self.partition()
        second, repeated, repeated_metadata = self.partition()
        np.testing.assert_array_equal(first, second)
        self.assertTrue(blocks.equals(repeated))
        self.assertEqual(metadata, repeated_metadata)
        changed, _, _ = self.partition(seed=11)
        self.assertFalse(np.array_equal(first == 2, changed == 2))

    def test_partial_edges_and_low_valid_coverage_are_accounted(self):
        valid = np.ones((23, 24), dtype=bool)
        valid[:10, :10] = False
        valid[:4, :10] = True  # 40%: inelegível para avaliação.
        split, blocks, _ = spatial_partition(
            valid, 10, 0, min_train_blocks=2, min_validation_blocks=2)
        partial = blocks.iloc[-1]
        self.assertEqual(int(partial["area_pixels"]), 12)
        self.assertEqual(float(partial["valid_fraction"]), 1.)
        self.assertTrue(bool(partial["validation_eligible"]))
        first = blocks.iloc[0]
        self.assertFalse(bool(first["validation_eligible"]))
        self.assertFalse(bool(first["selected_validation"]))
        self.assertEqual(int((split == 3).sum()), 0)
        self.assertEqual(len(blocks), 9)

    def test_insufficient_geometry_and_too_wide_buffer_fail(self):
        for valid, block, buffer in (
            (np.ones((10, 10), dtype=bool), 10, 0),
            (self.valid, 10, 100),
            (np.zeros((100, 100), dtype=bool), 10, 0),
        ):
            with self.subTest(block=block, buffer=buffer), self.assertRaises(ValueError):
                spatial_partition(valid, block, buffer)

    def test_invalid_partition_parameters_fail(self):
        cases = (
            {"block_pixels": 0}, {"block_pixels": 4.5}, {"buffer_pixels": -1},
            {"seed": -1}, {"seed": True}, {"validation_fraction": 0},
            {"validation_fraction": 1}, {"validation_fraction": np.nan},
            {"min_train_blocks": 0}, {"min_validation_blocks": False},
        )
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.partition(**case)
        for invalid in (np.ones((10, 10)), np.ones(10, dtype=bool),
                        np.ones((0, 10), dtype=bool)):
            with self.assertRaises(ValueError):
                spatial_partition(invalid, 10, 2)


class SpatialSampleTests(unittest.TestCase):
    def setUp(self):
        rows, cols = np.indices((80, 90))
        self.cube = np.stack((rows * 90 + cols, -1. - rows / 100., cols / 100.), axis=-1)
        self.split, _, _ = spatial_partition(np.ones((80, 90), dtype=bool), 10, 2)

    def test_exact_counts_unique_values_and_buffer_exclusion(self):
        result = spatial_sample(self.cube, self.split, train_size=123, validation_size=57)
        for key, label, size in (("train", 1, 123), ("validation", 2, 57)):
            values, positions = result[key]["values"], result[key]["positions"]
            self.assertEqual(values.shape, (size, 3))
            self.assertEqual(len(np.unique(positions, axis=0)), size)
            rr, cc = positions.T
            self.assertTrue((self.split[rr, cc] == label).all())
            np.testing.assert_array_equal(values, self.cube[rr, cc])
            self.assertTrue((values[:, 1] < 0).all())
        self.assertEqual(len(set(map(tuple, result["train"]["positions"]))
                             & set(map(tuple, result["validation"]["positions"]))), 0)

    def test_reproducibility_offset_invariance_and_population_cap(self):
        first = spatial_sample(self.cube, self.split, 123, 57)
        again = spatial_sample(self.cube, self.split, 123, 57, row_offset=90, col_offset=130)
        for key in first:
            np.testing.assert_array_equal(first[key]["values"], again[key]["values"])
            np.testing.assert_array_equal(first[key]["positions"] + [90, 130],
                                          again[key]["positions"])
        capped = spatial_sample(self.cube, self.split, 100_000, 100_000)
        self.assertEqual(len(capped["train"]["values"]), int((self.split == 1).sum()))
        self.assertEqual(len(capped["validation"]["values"]), int((self.split == 2).sum()))

    def test_invalid_inputs_and_insufficient_populations_fail(self):
        cases = (
            {"train_size": 19}, {"validation_size": 9}, {"seed": -1},
            {"row_offset": -1}, {"col_offset": .5},
        )
        for options in cases:
            with self.subTest(options=options), self.assertRaises(ValueError):
                spatial_sample(self.cube, self.split, **options)
        for label, retained in ((1, 19), (2, 9)):
            split = self.split.copy()
            indices = np.flatnonzero(split == label)
            split.flat[indices[retained:]] = 0
            with self.assertRaises(ValueError):
                spatial_sample(self.cube, split)
        invalid_split = self.split.copy()
        invalid_split[0, 0] = 4
        with self.assertRaises(ValueError):
            spatial_sample(self.cube, invalid_split)
        with self.assertRaises(ValueError):
            spatial_sample(self.cube[:-1], self.split)
        with self.assertRaises(ValueError):
            spatial_sample(self.cube, self.split.astype(float))

    def test_nonfinite_sample_population_rejected_rejected_areas_ignored(self):
        cube = self.cube.copy()
        selected = tuple(np.argwhere(self.split == 1)[0])
        cube[selected] = np.nan
        with self.assertRaises(ValueError):
            spatial_sample(cube, self.split, 20, 10)
        cube = self.cube.copy()
        cube[self.split == 3] = np.nan
        result = spatial_sample(cube, self.split, 20, 10)
        self.assertTrue(np.isfinite(result["train"]["values"]).all())


class LagDiagnosticTests(unittest.TestCase):
    def test_plane_plus_noise_reports_trend_without_claiming_independence(self):
        rng = np.random.default_rng(19)
        rr, cc = np.indices((160, 180))
        band = .7 * rr + .9 * cc + rng.normal(size=rr.shape)
        result = lag_diagnostics(band[:, :, None], np.ones(rr.shape, dtype=bool),
                                 lags=(8, 16), minimum_pairs=100)
        self.assertTrue((result["correlation_raw"] > .99).all())
        self.assertTrue((result["correlation_detrended"].abs() < .06).all())

    def test_short_range_smooth_signal_decays_with_distance(self):
        rng = np.random.default_rng(44)
        signal = gaussian_filter(rng.normal(size=(160, 180)), 2.)
        result = lag_diagnostics(signal[:, :, None], np.ones(signal.shape, dtype=bool),
                                 lags=(1, 32), minimum_pairs=100)
        short = result.loc[result["lag_pixels"] == 1, "correlation_detrended"]
        far = result.loc[result["lag_pixels"] == 32, "correlation_detrended"]
        self.assertTrue((short > .8).all())
        self.assertTrue((far.abs() < .2).all())

    def test_pair_cap_finite_filter_constant_and_oversized_lag(self):
        cube = np.zeros((120, 160, 2))
        cube[:, :, 0] = np.random.default_rng(25).normal(size=(120, 160)) - 2.
        cube[0, 0, 0] = np.nan
        valid = np.ones((120, 160), dtype=bool)
        valid[0, 1] = False
        result = lag_diagnostics(cube, valid, lags=(1, 200))
        self.assertTrue((result["n_pairs"] <= 10_000).all())
        self.assertTrue((result["n_pairs_available"] >= result["n_pairs"]).all())
        constant = result[result["band_index"] == 1]
        self.assertTrue(constant["correlation_raw"].isna().all())
        oversized = result[result["lag_pixels"] == 200]
        self.assertTrue((oversized["n_pairs"] == 0).all())
        self.assertTrue(oversized["correlation_raw"].isna().all())
        self.assertEqual(int(result.iloc[0]["n_pairs_available"]), 120 * 159 - 2)
        self.assertEqual(float(result.iloc[0]["lag_m"]), 3.)

    def test_insufficient_pairs_and_determinism(self):
        cube = np.arange(2 * 8 * 9).reshape((8, 9, 2)).astype(float)
        valid = np.ones((8, 9), dtype=bool)
        first = lag_diagnostics(cube, valid, lags=(1, 4))
        second = lag_diagnostics(cube, valid, lags=(1, 4))
        self.assertTrue(first.equals(second))
        self.assertTrue(first["correlation_raw"].isna().all())

    def test_invalid_lag_inputs_fail(self):
        cube = np.zeros((10, 12, 1))
        valid = np.ones((10, 12), dtype=bool)
        for options in ({"lags": ()}, {"lags": (1, 1)}, {"lags": (0,)},
                        {"lags": (1.5,)}, {"lags": None}, {"pixel_size": 0},
                        {"pixel_size": np.nan}, {"minimum_pairs": 2},
                        {"minimum_pairs": 10_001}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                lag_diagnostics(cube, valid, **options)
        with self.assertRaises(ValueError):
            lag_diagnostics(cube[:, :, 0], valid)
        with self.assertRaises(ValueError):
            lag_diagnostics(cube.astype(complex), valid)


if __name__ == "__main__":
    unittest.main()
