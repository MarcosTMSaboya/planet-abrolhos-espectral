"""Testes leves exclusivamente sintéticos da avaliação espacial PCA/K-means.

Nenhum raster real é acessado. Ajustes usam 120 linhas de desenvolvimento e
60 de holdout, K=3/4, duas sementes e duas inicializações no máximo.
"""
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
import pandas as pd

from espectral_avaliacao import compare_spatial_models, spectral_separation


class SpatialEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import joblib
            from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
            from threadpoolctl import threadpool_limits
        except ImportError as error:
            raise unittest.SkipTest(f"Dependência opcional ausente: {error}") from error
        cls.joblib = joblib
        cls.adjusted_rand_score = staticmethod(adjusted_rand_score)
        cls.davies_bouldin_score = staticmethod(davies_bouldin_score)
        cls.silhouette_score = staticmethod(silhouette_score)
        cls.threadpool_limits = staticmethod(threadpool_limits)
        rng = np.random.default_rng(713)
        centers = np.array([[-.06, -.03, -.02, -.01],
                            [.02, .06, -.02, -.04],
                            [.07, -.06, .04, .02]])
        cls.train = np.concatenate([center + rng.normal(0, .002, (40, 4)) for center in centers])
        cls.validation = np.concatenate([center + rng.normal(0, .002, (20, 4)) for center in centers])
        cls.reference_groups = np.repeat(np.arange(3), 20)
        cls.result = cls.compare(cls.train, cls.validation)

    @staticmethod
    def compare(train, validation, **overrides):
        options = {"k_values": (3, 4), "seed_values": (11, 42), "silhouette_size": 45, "n_init": 2}
        options.update(overrides)
        return compare_spatial_models(train, validation, **options)

    def test_scaler_and_pca_are_fit_only_on_development(self):
        extreme = self.validation * 100 + 1234
        second = self.compare(self.train, extreme, seed_values=(11,), k_values=(3,))
        np.testing.assert_allclose(self.result["scaler"].mean_, self.train.mean(axis=0))
        np.testing.assert_allclose(self.result["scaler"].scale_, self.train.std(axis=0))
        self.assertEqual(int(self.result["scaler"].n_samples_seen_), len(self.train))
        np.testing.assert_array_equal(second["scaler"].mean_, self.result["scaler"].mean_)
        np.testing.assert_array_equal(second["pca"].components_, self.result["pca"].components_)
        self.assertFalse(self.result["pca"].whiten)
        self.assertGreaterEqual(self.result["pca_metadata"]["retained_variance"], .95)
        self.assertEqual(self.result["pca_metadata"]["fit_partition"], "train")
        for representation in ("bands", "pca95"):
            a = self.result["models"][(representation, 3, 11)].named_steps["kmeans"]
            b = second["models"][(representation, 3, 11)].named_steps["kmeans"]
            np.testing.assert_array_equal(a.cluster_centers_, b.cluster_centers_)

    def test_holdout_metrics_match_direct_scores_on_same_indices(self):
        indices = self.result["evaluation_indices"]["validation"]
        self.assertEqual(len(indices), 45)
        with self.threadpool_limits(limits=1):
            for representation in ("bands", "pca95"):
                for seed in (11, 42):
                    key = (representation, 3, seed)
                    model = self.result["models"][key]
                    transformed = model[:-1].transform(self.validation)
                    predicted = model.predict(self.validation)
                    rows = self.result["metrics"]
                    row = rows.loc[(rows.representation == representation) & (rows.k == 3)
                                   & (rows.seed == seed) & (rows.split == "validation")].iloc[0]
                    self.assertEqual(row.silhouette_status, "ok")
                    self.assertEqual(row.davies_bouldin_status, "ok")
                    self.assertAlmostEqual(row.silhouette_subsample,
                                           self.silhouette_score(transformed[indices], predicted[indices]))
                    self.assertAlmostEqual(row.davies_bouldin,
                                           self.davies_bouldin_score(transformed, predicted))
                    np.testing.assert_array_equal(predicted, self.result["predictions"][key]["validation"])

    def test_missing_validation_groups_return_null_metrics_with_status(self):
        validation = np.repeat(self.train[:1], 12, axis=0)
        result = self.compare(self.train, validation, k_values=(3,), seed_values=(11,))
        rows = result["metrics"].loc[lambda frame: frame.split == "validation"]
        self.assertTrue((rows.n_groups_present == 1).all())
        self.assertTrue(rows.silhouette_subsample.isna().all())
        self.assertTrue(rows.davies_bouldin.isna().all())
        self.assertTrue((rows.silhouette_status == "missing_groups").all())
        self.assertTrue((rows.davies_bouldin_status == "missing_groups").all())
        self.assertTrue(result["stability"].empty)
        # Até uma única linha em holdout tem resultado explícito, sem ValueError.
        one = self.compare(self.train, validation[:1], k_values=(3,), seed_values=(11,))
        self.assertTrue(one["metrics"].loc[lambda frame: frame.split == "validation",
                                          "silhouette_subsample"].isna().all())

    def test_fixed_small_subsample_reports_missing_groups(self):
        result = self.compare(self.train, self.validation, k_values=(3,), seed_values=(11,),
                              silhouette_size=2)
        self.assertTrue(result["metrics"].silhouette_subsample.isna().all())
        self.assertTrue((result["metrics"].silhouette_status == "missing_groups_subsample").all())
        self.assertTrue((result["metrics"].davies_bouldin_status == "ok").all())

    def test_proportions_include_zero_counts_and_keep_statistical_names(self):
        expected = {(representation, k, seed) for representation in ("bands", "pca95")
                    for k in (3, 4) for seed in (11, 42)}
        self.assertEqual(set(self.result["models"]), expected)
        self.assertEqual(len(self.result["metrics"]), len(expected) * 2)
        for (_, k, _, split), rows in self.result["proportions"].groupby(
                ["representation", "k", "seed", "split"]):
            self.assertEqual(len(rows), k)
            self.assertEqual(set(rows.group), set(range(1, k + 1)))
            self.assertEqual(int(rows.sample_count.sum()), len(self.train) if split == "train" else len(self.validation))
            self.assertAlmostEqual(float(rows.sample_fraction.sum()), 1)
            self.assertTrue(rows.group_name.str.fullmatch(r"Grupo Espectral \d+").all())
        self.assertFalse(any("probab" in column.lower() for column in self.result["proportions"].columns))

    def test_seed_stability_uses_identical_holdout_and_is_number_invariant(self):
        for row in self.result["stability"].itertuples():
            a = self.result["predictions"][(row.representation, row.k, row.seed_a)]["validation"]
            b = self.result["predictions"][(row.representation, row.k, row.seed_b)]["validation"]
            self.assertEqual(row.n_samples, len(self.validation))
            self.assertEqual(row.split, "validation")
            self.assertAlmostEqual(row.adjusted_rand, self.adjusted_rand_score(a, b))
            self.assertAlmostEqual(row.adjusted_rand, self.adjusted_rand_score(a, (b + 1) % row.k))
            if row.k == 3:
                self.assertAlmostEqual(row.adjusted_rand, 1)
                self.assertAlmostEqual(self.adjusted_rand_score(self.reference_groups, a), 1)

    def test_reproducible_without_mutating_inputs_or_clipping_negatives(self):
        train_before, validation_before = self.train.copy(), self.validation.copy()
        repeated = self.compare(self.train, self.validation)
        for key in ("metrics", "proportions", "stability"):
            pd.testing.assert_frame_equal(self.result[key], repeated[key])
        for split in ("train", "validation"):
            np.testing.assert_array_equal(self.result["evaluation_indices"][split], repeated["evaluation_indices"][split])
        np.testing.assert_array_equal(self.train, train_before)
        np.testing.assert_array_equal(self.validation, validation_before)
        self.assertTrue((self.train < 0).any())
        scaler = self.result["scaler"]
        np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(self.train)), self.train,
                                   atol=1e-14)

    def test_trusted_local_pickle_and_joblib_roundtrip_preserve_predictions(self):
        with tempfile.TemporaryDirectory(prefix="abrolhos_trusted_synthetic_") as folder:
            for representation in ("bands", "pca95"):
                original = self.result["models"][(representation, 3, 11)]
                path = Path(folder) / f"{representation}.joblib"
                self.joblib.dump(original, path)
                restored = self.joblib.load(path)
                in_memory = pickle.loads(pickle.dumps(original))
                with self.threadpool_limits(limits=1):
                    expected = original.predict(self.validation)
                    np.testing.assert_array_equal(restored.predict(self.validation), expected)
                    np.testing.assert_array_equal(in_memory.predict(self.validation), expected)

    def test_persisted_pipelines_accept_float32_raster_values(self):
        # Regressão: StandardScaler sozinho mantém float32 do raster, enquanto
        # os centros K-means ajustados nesta análise estão em float64.
        raster_values = self.validation.astype(np.float32)
        equivalent_float64 = raster_values.astype(np.float64)
        with tempfile.TemporaryDirectory(prefix="abrolhos_trusted_float32_") as folder:
            for representation in ("bands", "pca95"):
                model = self.result["models"][(representation, 3, 11)]
                path = Path(folder) / f"{representation}.joblib"
                self.joblib.dump(model, path)
                restored = self.joblib.load(path)
                with self.threadpool_limits(limits=1):
                    float32_labels = restored.predict(raster_values)
                    float64_labels = restored.predict(equivalent_float64)
                    np.testing.assert_array_equal(float32_labels, float64_labels)
                    np.testing.assert_array_equal(float32_labels, model.predict(self.validation))
                    np.testing.assert_allclose(restored.transform(raster_values),
                                               restored.transform(equivalent_float64), rtol=0, atol=0)
                self.assertEqual(restored.named_steps["kmeans"].cluster_centers_.dtype, np.float64)
                self.assertEqual(restored.named_steps["to_float64"].transform(raster_values).dtype,
                                 np.float64)

    def test_invalid_inputs_fail_before_any_fit(self):
        invalid = [
            {"train": np.ones((10, 4))}, {"train": [[1., 2.]]},
            {"train": np.full((20, 4), np.nan)}, {"validation": np.empty((0, 4))},
            {"validation": np.ones((3, 3))}, {"validation": [[np.inf] * 4]},
            {"k_values": (True,)}, {"k_values": (3, 3)}, {"k_values": (999,)},
            {"k_values": ()}, {"seed_values": ()}, {"seed_values": (-1,)},
            {"seed_values": (11, 11)}, {"seed_values": (False,)},
            {"n_init": 0}, {"n_init": 1.5}, {"silhouette_size": 1},
            {"representations": ()}, {"representations": ("pca2",)},
            {"representations": ("bands", "bands")},
        ]
        for change in invalid:
            with self.subTest(parameters=change):
                options = {"train": self.train, "validation": self.validation,
                           "k_values": (3,), "seed_values": (11,), "n_init": 2}
                options.update(change)
                with self.assertRaises(ValueError):
                    compare_spatial_models(**options)


class SpectralSeparationTests(unittest.TestCase):
    def test_signed_sam_and_explicit_zero_norm_policy(self):
        values = np.array([[1., 0.], [1., 0.], [-1., 0.], [-1., 0.], [0., 0.], [0., 0.]])
        labels = np.repeat(np.arange(3), 2)
        result = spectral_separation(values, labels, np.array([[1., 0.], [-1., 0.], [0., 0.]]), min_norm=0)
        pair = result.loc[(result.group_a == 1) & (result.group_b == 2)].iloc[0]
        self.assertAlmostEqual(pair.spectral_angle_rad, np.pi)
        self.assertAlmostEqual(pair.spectral_angle_degrees, 180)
        self.assertEqual(pair.standardized_euclidean, 2)
        self.assertEqual(pair.within_scatter_a, 0)
        zero_pairs = result.loc[result.group_b == 3]
        self.assertTrue(zero_pairs.spectral_angle_rad.isna().all())
        self.assertTrue((zero_pairs.spectral_angle_status == "norm_below_or_equal_minimum").all())
        eta = result.attrs["eta_squared"]
        self.assertAlmostEqual(eta.eta_squared.iloc[0], 1)
        self.assertTrue(np.isnan(eta.eta_squared.iloc[1]))
        self.assertEqual(eta.status.iloc[1], "constant_feature")

    def test_threshold_is_respected_without_automatic_epsilon(self):
        tiny = np.array([[1e-10, 0.], [1e-10, 0.], [0., 1e-10], [0., 1e-10]])
        labels = np.array([0, 0, 1, 1])
        centers = np.array([[1., 0.], [0., 1.]])
        retained = spectral_separation(tiny, labels, centers, min_norm=0).iloc[0]
        self.assertAlmostEqual(retained.spectral_angle_degrees, 90)
        rejected = spectral_separation(tiny, labels, centers, min_norm=1e-10).iloc[0]
        self.assertTrue(pd.isna(rejected.spectral_angle_rad))

    def test_within_scatter_and_eta_squared_match_direct_sums(self):
        values = np.array([[-2., 1.], [0., 3.], [2., 3.], [4., 5.]])
        labels = np.array([0, 0, 1, 1])
        centers = np.array([[-1., -1.], [1., 1.]])
        result = spectral_separation(values, labels, centers, min_norm=0)
        self.assertEqual(result.iloc[0].within_scatter_a, 2)
        self.assertEqual(result.iloc[0].within_scatter_b, 2)
        self.assertEqual(result.iloc[0].n_a, 2)
        means = np.array([values[labels == i].mean(axis=0) for i in (0, 1)])
        between = (2 * (means - values.mean(axis=0))**2).sum(axis=0)
        total = ((values - values.mean(axis=0))**2).sum(axis=0)
        np.testing.assert_allclose(result.attrs["eta_squared"].eta_squared, between / total)

    def test_invalid_separation_inputs(self):
        values = np.array([[1., 0.], [1., 0.], [-1., 0.], [-1., 0.]])
        labels = np.array([0, 0, 1, 1])
        centers = np.array([[1., 0.], [-1., 0.]])
        invalid = [
            {"train_labels": [0, 0, 0, 0]}, {"train_labels": [1, 1, 2, 2]},
            {"train_labels": [0., 0., 1., 1.]}, {"train_labels": [False] * 4},
            {"train_labels": [0, 1]}, {"train_original": np.full((4, 2), np.inf)},
            {"standardized_centers": np.ones((2, 3))}, {"min_norm": -1},
            {"min_norm": np.nan}, {"min_norm": True},
        ]
        for change in invalid:
            with self.subTest(parameters=change):
                options = {"train_original": values, "train_labels": labels,
                           "standardized_centers": centers, "min_norm": 0}
                options.update(change)
                with self.assertRaises(ValueError):
                    spectral_separation(**options)


if __name__ == "__main__":
    unittest.main()
