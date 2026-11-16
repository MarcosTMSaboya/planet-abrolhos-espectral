"""Integração opcional PCA/K-means/gráficos com até 180 amostras sintéticas.

Não acessa imagens reais nem produz modelos científicos. Todos os arquivos são
temporários. Execute na raiz do projeto com a .venv:
    python -m unittest discover -s src -p test_dependencias_espectrais.py -v
"""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import agrupamento_espectral as clustering
from espectral_io import WAVELENGTHS


class OptionalSpectralIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            from sklearn.metrics import adjusted_rand_score
            from sklearn.preprocessing import StandardScaler
            from threadpoolctl import threadpool_limits
            import matplotlib
        except ImportError as error:
            raise unittest.SkipTest(f"Dependência espectral opcional ausente: {error}") from error
        cls.KMeans = KMeans
        cls.PCA = PCA
        cls.adjusted_rand_score = staticmethod(adjusted_rand_score)
        cls.StandardScaler = StandardScaler
        cls.threadpool_limits = staticmethod(threadpool_limits)
        _, _, _, _, cls.plt = clustering.dependencies()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="abrolhos_optional_synthetic_")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.thread_limit = self.threadpool_limits(limits=1)
        self.thread_limit.__enter__()
        self.addCleanup(self.thread_limit.__exit__, None, None, None)
        # Os grupos são apenas artificiais: não representam classes bentônicas.
        rng = np.random.default_rng(2026)
        centers = np.array([
            [-.04, -.03, -.02, -.01, .00, .01, .02, .03],
            [.03, .02, .01, .00, -.01, -.02, -.03, -.04],
            [.04, .04, .03, .03, .04, .03, .04, .03],
        ])
        self.x = np.concatenate([
            center + rng.normal(0, .0015, size=(60, 8)) for center in centers
        ])
        self.synthetic_groups = np.repeat(np.arange(3), 60)

    def test_scaling_and_pca_agree_with_sklearn_without_clipping_negatives(self):
        original = self.x.copy()
        reference_scaler = self.StandardScaler().fit(self.x)
        pca = clustering.pca_numpy(self.x)
        np.testing.assert_allclose(pca["mean"], reference_scaler.mean_, atol=1e-15)
        np.testing.assert_allclose(pca["scale"], reference_scaler.scale_, atol=1e-15)
        z = reference_scaler.transform(self.x)
        reference_pca = self.PCA(svd_solver="full").fit(z)
        np.testing.assert_allclose(
            pca["explained_ratio"], reference_pca.explained_variance_ratio_, atol=1e-12
        )
        # O sinal do autovetor é arbitrário; os subespaços devem coincidir.
        np.testing.assert_allclose(
            np.abs(pca["components"] @ reference_pca.components_.T), np.eye(8), atol=1e-10
        )
        reconstructed = reference_scaler.inverse_transform(pca["scores"] @ pca["components"])
        np.testing.assert_allclose(reconstructed, self.x, atol=1e-12)
        self.assertTrue((self.x < 0).any())
        np.testing.assert_array_equal(self.x, original)

    def test_npz_roundtrip_preserves_holdout_predictions_and_pca(self):
        # Separa exemplos artificiais para verificar inferência fora do ajuste.
        train, holdout = self.x[np.arange(len(self.x)) % 3 != 0], self.x[::3]
        pca = clustering.pca_numpy(train)
        scaler = self.StandardScaler().fit(train)
        model = self.KMeans(n_clusters=3, n_init=10, random_state=42).fit(scaler.transform(train))
        path = self.folder / "modelo_sintetico.npz"
        np.savez_compressed(
            path, mean=pca["mean"], scale=pca["scale"], centers=model.cluster_centers_,
            wavelengths=WAVELENGTHS, pca_components=pca["components"],
        )
        with np.load(path, allow_pickle=False) as stored:
            z = (holdout - stored["mean"]) / stored["scale"]
            distances = ((z[:, None, :] - stored["centers"][None, :, :]) ** 2).sum(axis=2)
            restored_predictions = np.argmin(distances, axis=1)
            np.testing.assert_array_equal(restored_predictions, model.predict(scaler.transform(holdout)))
            np.testing.assert_allclose(
                (z @ stored["pca_components"].T) @ stored["pca_components"], z, atol=1e-12
            )
            np.testing.assert_array_equal(stored["wavelengths"], WAVELENGTHS)
            for key in stored.files:
                self.assertTrue(np.isfinite(stored[key]).all(), key)
        self.assertEqual(
            self.adjusted_rand_score(self.synthetic_groups[::3], restored_predictions), 1.0
        )

    def test_compare_k_metrics_proportions_and_seed_stability(self):
        z = self.StandardScaler().fit_transform(self.x)
        metrics, proportions, stability, models = clustering.compare_k(
            z, seed_values=(11, 42), silhouette_size=120
        )
        self.assertEqual(set(models), {(k, seed) for k in range(3, 9) for seed in (11, 42)})
        self.assertEqual(len(metrics), 12)
        self.assertEqual(len(stability), 6)
        self.assertEqual(set(metrics.k), set(range(3, 9)))
        self.assertTrue(np.isfinite(metrics.select_dtypes(include="number").to_numpy()).all())
        self.assertTrue(metrics.silhouette_subsample.between(-1, 1).all())
        self.assertTrue((metrics.davies_bouldin_sample >= 0).all())
        self.assertTrue((metrics.inertia_train > 0).all())
        self.assertTrue((metrics.silhouette_n == 120).all())
        self.assertTrue((metrics.n_train == len(self.x)).all())
        for (k, seed), rows in proportions.groupby(["k", "seed"]):
            self.assertEqual(len(rows), k)
            self.assertEqual(int(rows.sample_count.sum()), len(self.x))
            self.assertAlmostEqual(float(rows.sample_fraction.sum()), 1)
            self.assertTrue((rows.sample_count > 0).all())
        self.assertTrue(np.isfinite(stability.adjusted_rand).all())
        self.assertTrue(stability.adjusted_rand.between(-1, 1).all())
        self.assertAlmostEqual(float(stability.loc[stability.k == 3, "adjusted_rand"].iloc[0]), 1)
        for seed in (11, 42):
            labels = models[(3, seed)].labels_
            self.assertEqual(self.adjusted_rand_score(self.synthetic_groups, labels), 1)
            # Renomear os identificadores de grupo não muda a estabilidade ARI.
            permuted = np.array([2, 0, 1])[labels]
            self.assertEqual(self.adjusted_rand_score(labels, permuted), 1)

    def test_diagnostic_and_pca_artifacts_are_readable_and_numerically_valid(self):
        pca = clustering.pca_numpy(self.x)
        clustering.sample_diagnostics(self.x, self.folder, self.plt)
        clustering.save_pca(pca, WAVELENGTHS, self.folder, self.plt)
        for filename in (
            "distribuicoes.png", "correlacao.png", "espectros_amostra.png",
            "pca_pc1_pc2.png", "variancia_pca.png",
        ):
            with self.subTest(image=filename):
                path = self.folder / filename
                self.assertGreater(path.stat().st_size, 1000)
                pixels = self.plt.imread(path)
                self.assertGreater(pixels.shape[0], 100)
                self.assertGreater(pixels.shape[1], 100)
                self.assertTrue(np.isfinite(pixels).all())
                self.assertGreater(float(pixels[..., :3].std()), .01)
        variance = pd.read_csv(self.folder / "variancia_pca.csv")
        self.assertAlmostEqual(float(variance.explained_ratio.sum()), 1)
        self.assertTrue((variance.explained_ratio >= 0).all())
        self.assertTrue((np.diff(variance.cumulative) >= 0).all())
        negatives = pd.read_csv(self.folder / "negativos_amostra.csv")
        np.testing.assert_allclose(negatives.negative_fraction, (self.x < 0).mean(axis=0))
        correlation = pd.read_csv(self.folder / "correlacao_amostra.csv", index_col=0).to_numpy()
        self.assertTrue(np.isfinite(correlation).all())
        np.testing.assert_allclose(correlation, correlation.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(correlation), np.ones(8), atol=1e-12)
        loadings = pd.read_csv(self.folder / "loadings_pca.csv", index_col=0).to_numpy()
        self.assertEqual(loadings.shape, (8, 8))
        self.assertTrue(np.isfinite(loadings).all())
        self.assertEqual(self.plt.get_fignums(), [])


if __name__ == "__main__":
    unittest.main()
