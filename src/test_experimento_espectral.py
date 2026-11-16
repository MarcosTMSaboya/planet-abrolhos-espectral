"""Integração leve: somente GeoTIFFs e amostras sintéticos temporários.

Não usa cenas do projeto, não instala bibliotecas e não executa o experimento
científico real. O único ajuste permitido aqui limita-se a 180/90 espectros,
K=3/4, duas sementes e n_init=2, para verificar a integração do executor.
"""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

import experimento_espectral as experiment
import espectral_avaliacao as evaluation
from espectral_io import FEATURE_SETS, WAVELENGTHS


class SyntheticExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="abrolhos_experiment_synthetic_")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.transform = from_origin(500000, 8050000, 3, 3)
        cls.scene_id = "synthetic_unlabeled_scene"
        cls.asset = {"scene_id": cls.scene_id, "bands": []}
        cls.cube = np.random.default_rng(785).normal(.006, .008, (44, 56, 8)).astype("float32")
        cls.cube[5, 6, 0] = -9999
        for band, wavelength in enumerate(WAVELENGTHS):
            path = cls.root / f"rhos_{wavelength}.tif"
            cls.write_raster(path, cls.cube[:, :, band], nodata=-9999)
            cls.asset["bands"].append(path)
        quality = np.zeros((8, 44, 56), dtype="uint8")
        quality[0] = 1
        quality[6] = 95
        quality[5, 10, 11] = 1
        cls.asset["udm2"] = cls.root / "udm2.tif"
        cls.write_raster(cls.asset["udm2"], quality, nodata=0)
        water = np.ones((44, 56), dtype="uint8")
        water[7, 9] = 0
        cls.mask = cls.root / "aquatic_mask.tif"
        cls.write_raster(cls.mask, water, nodata=0)
        cls.review_meta = {
            "scene_id": cls.scene_id, "reviewed": True,
            "review_scope": "visual_apparent_water_only",
            "benthic_visibility": "not_confirmed",
            "approved_for_benthic_interpretation": False,
            "reviewer": "Teste sintético, sem revisão de cena real",
            "reviewed_utc": "2026-10-20T00:00:00Z",
            "source": "Dados sintéticos criados exclusivamente para integração",
            "water_criteria": "Máscara binária sintética; não comprova água real",
            "limitations": "Sem observação ou identidade bentônica",
            "values": {"0": "excluir", "1": "incluir"},
            "mask_path": cls.mask.name, "sha256": experiment.sha256(cls.mask),
            "window": {"col": 3, "row": 2, "width": 50, "height": 40},
        }
        cls.review = cls.root / "aquatic_review.json"
        experiment.write_json(cls.review, cls.review_meta)
        cls.source_hashes = {
            str(path): experiment.sha256(path)
            for path in cls.asset["bands"] + [cls.asset["udm2"], cls.mask, cls.review]
        }
        # Falhar imediatamente se a preparação começar qualquer ajuste de modelo.
        with (patch.object(experiment, "find_assets", return_value={cls.scene_id: cls.asset}),
              patch.object(experiment, "pca_numpy", side_effect=AssertionError("PCA durante preparação")),
              patch.object(evaluation, "compare_spatial_models", side_effect=AssertionError("Modelos durante preparação")),
              patch("sklearn.cluster.KMeans.fit", side_effect=AssertionError("KMeans durante preparação")),
              patch("sklearn.decomposition.PCA.fit", side_effect=AssertionError("PCA durante preparação")),
              patch("sklearn.preprocessing.StandardScaler.fit", side_effect=AssertionError("Scaler durante preparação"))):
            cls.prepared = experiment.prepare(cls.review, cls.root / "prepared_outputs",
                                              block_pixels=8, buffer_pixels=1)
        cls.prepared_path = cls.prepared / "preparacao.json"

    @classmethod
    def write_raster(cls, path, values, nodata):
        values = np.asarray(values)
        if values.ndim == 2:
            values = values[None]
        with rasterio.Env(GDAL_PAM_ENABLED="NO"):
            with rasterio.open(
                path, "w", driver="GTiff", width=values.shape[2], height=values.shape[1],
                count=len(values), dtype=values.dtype, crs="EPSG:32724",
                transform=cls.transform, nodata=nodata,
            ) as destination:
                destination.write(values)

    def copy_preparation(self, label):
        destination = self.root / label
        shutil.copytree(self.prepared, destination)
        return destination

    def test_prepare_preserves_sources_and_has_no_training(self):
        metadata = json.loads(self.prepared_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "prepared")
        self.assertFalse(metadata["training_executed"])
        self.assertFalse(metadata["autocorrelation"]["independence_proven"])
        self.assertEqual(metadata["benthic_visibility"], "not_confirmed")
        self.assertTrue(metadata["requires_processing_authorization"])
        self.assertEqual(metadata["counts"], {
            "roi": 2000, "finite_all_bands": 1999, "clear": 1999, "eligible": 1997,
        })
        for path, expected in self.source_hashes.items():
            self.assertEqual(experiment.sha256(path), expected)
        for name, expected in metadata["artifacts_sha256"].items():
            self.assertEqual(experiment.sha256(self.prepared / name), expected)
        self.assertFalse(list(self.prepared.glob("*.joblib")))
        proposal = (self.prepared / "COMANDO_PARA_APROVACAO.txt").read_text(encoding="utf-8")
        self.assertIn("--acknowledge-aquatic-scope", proposal)
        self.assertIn("--acknowledge-spatial-limitations", proposal)

    def test_georeferenced_partition_counts_and_nonoverlap(self):
        metadata = json.loads(self.prepared_path.read_text(encoding="utf-8"))
        with rasterio.open(self.prepared / "divisao_espacial.tif") as raster:
            split = raster.read(1)
            self.assertEqual(raster.crs.to_epsg(), 32724)
            self.assertEqual((raster.height, raster.width), (40, 50))
            self.assertEqual(raster.nodata, 0)
            self.assertEqual(raster.transform,
                             rasterio.windows.transform(Window(3, 2, 50, 40), self.transform))
        with np.load(self.prepared / "amostras.npz", allow_pickle=False) as samples:
            train = samples["train_positions"]
            validation = samples["validation_positions"]
            self.assertEqual(len(train), metadata["n_train"])
            self.assertEqual(len(validation), metadata["n_validation"])
            self.assertFalse(set(map(tuple, train)) & set(map(tuple, validation)))
            self.assertFalse(set(map(tuple, (train - [2, 3]) // 8))
                             & set(map(tuple, (validation - [2, 3]) // 8)))
            for key, positions, label in (("train_values", train, 1),
                                           ("validation_values", validation, 2)):
                rr, cc = positions.T
                np.testing.assert_array_equal(samples[key], self.cube[rr, cc])
                self.assertTrue((split[rr - 2, cc - 3] == label).all())
                self.assertTrue(np.isfinite(samples[key]).all())
                self.assertTrue((samples[key] < 0).any())
            for location in train:
                self.assertGreater(np.abs(validation - location).max(axis=1).min(), 1)
        counts = metadata["partition"]["metadata"]["counts"]
        self.assertEqual(counts["0"], 3)
        for label in range(4):
            self.assertEqual(counts[str(label)], int((split == label).sum()))
        blocks = pd.read_csv(self.prepared / "blocos.csv")
        self.assertEqual(int(blocks["valid_pixels"].sum()), 1997)

    def test_preparation_plots_and_tables_are_readable(self):
        for name in ("divisao_espacial.png", "diagnostico_espacial.png"):
            with Image.open(self.prepared / name) as image:
                image.verify()
            with Image.open(self.prepared / name) as image:
                self.assertGreater(image.width, 100)
                self.assertGreater(image.height, 100)
        diagnostic = pd.read_csv(self.prepared / "correlacao_por_distancia.csv")
        self.assertEqual(set(diagnostic["wavelength_nm"]), set(WAVELENGTHS))
        self.assertTrue((diagnostic["n_pairs"] <= 10000).all())
        for role in ("train", "validation"):
            frame = pd.read_csv(self.prepared / f"amostra_{role}.csv")
            self.assertEqual(frame.shape[1], 10)
            self.assertTrue({"row", "col", "rhos_444", "rhos_866"}.issubset(frame.columns))

    def test_review_rejects_candidate_benthic_claim_checksum_and_large_window(self):
        mutations = (
            {"reviewed": False}, {"review_scope": "candidate"},
            {"benthic_visibility": "confirmed"},
            {"approved_for_benthic_interpretation": True},
            {"sha256": "wrong_checksum"}, {"reviewer": ""},
            {"window": {"row": 0, "col": 0, "width": 501, "height": 500}},
            {"window": {"row": -1, "col": 0, "width": 40, "height": 50}},
            {"window": {"row": 0, "col": 0, "width": 40., "height": 50}},
        )
        path = self.root / "invalid_review.json"
        for changes in mutations:
            with self.subTest(changes=changes):
                experiment.write_json(path, dict(self.review_meta, **changes))
                with self.assertRaises(ValueError):
                    experiment.load_aquatic_review(path)
        mask, meta = experiment.load_aquatic_review(self.review)
        self.assertEqual(mask, self.mask)
        self.assertEqual(meta, self.review_meta)

    def test_execute_scope_and_threshold_gates_precede_fitting(self):
        cases = (
            {}, {"aquatic_ack": True, "spatial_ack": False, "sam_min_norm": 0},
            {"aquatic_ack": False, "spatial_ack": True, "sam_min_norm": 0},
            {"aquatic_ack": True, "spatial_ack": True},
            {"aquatic_ack": True, "spatial_ack": True, "sam_min_norm": -1},
            {"aquatic_ack": True, "spatial_ack": True, "sam_min_norm": np.nan},
        )
        with (patch.object(experiment, "dependencies", side_effect=AssertionError("Antes das aprovações")),
              patch.object(evaluation, "compare_spatial_models") as fit):
            for options in cases:
                with self.subTest(options=options), self.assertRaises(ValueError):
                    experiment.execute_sample(self.prepared_path, self.root / "gated_outputs", **options)
            fit.assert_not_called()
        self.assertFalse((self.root / "gated_outputs").exists())

    def test_execute_rejects_changed_artifact_or_review_before_fitting(self):
        for artifact in ("amostras.npz", "blocos.csv", "divisao_espacial.tif"):
            folder = self.copy_preparation(f"tampered_{artifact.replace('.', '_')}")
            with (folder / artifact).open("ab") as destination:
                destination.write(b"tampered")
            with (patch.object(experiment, "dependencies", side_effect=AssertionError("Antes do hash")),
                  patch.object(evaluation, "compare_spatial_models") as fit):
                with self.assertRaisesRegex(ValueError, "Artefato de preparação alterado"):
                    experiment.execute_sample(folder / "preparacao.json", self.root / "tampered_outputs",
                                              aquatic_ack=True, spatial_ack=True, sam_min_norm=0)
                fit.assert_not_called()
        folder = self.copy_preparation("tampered_review")
        changed_review = self.root / "changed_review.json"
        experiment.write_json(changed_review, dict(self.review_meta, reviewer="Alterado"))
        metadata = json.loads((folder / "preparacao.json").read_text(encoding="utf-8"))
        metadata["review_manifest"] = str(changed_review)
        experiment.write_json(folder / "preparacao.json", metadata)
        with (patch.object(experiment, "dependencies", side_effect=AssertionError("Antes do hash")),
              patch.object(evaluation, "compare_spatial_models") as fit):
            with self.assertRaisesRegex(ValueError, "Revisão alterada"):
                experiment.execute_sample(folder / "preparacao.json", self.root / "tampered_outputs",
                                          aquatic_ack=True, spatial_ack=True, sam_min_norm=0)
            fit.assert_not_called()

    def test_small_synthetic_execution_writes_models_metrics_and_neutral_names(self):
        import joblib

        folder = self.copy_preparation("small_execution_preparation")
        sample_path = folder / "amostras.npz"
        with np.load(sample_path, allow_pickle=False) as stored:
            samples = {key: stored[key].copy() for key in stored.files}
        for key in ("train_values", "train_positions"):
            samples[key] = samples[key][:180]
        for key in ("validation_values", "validation_positions"):
            samples[key] = samples[key][:90]
        np.savez_compressed(sample_path, **samples)
        metadata = json.loads((folder / "preparacao.json").read_text(encoding="utf-8"))
        metadata["artifacts_sha256"]["amostras.npz"] = experiment.sha256(sample_path)
        metadata.update(n_train=180, n_validation=90)
        experiment.write_json(folder / "preparacao.json", metadata)
        real_compare = evaluation.compare_spatial_models

        def small_compare(train, validation):
            self.assertEqual((len(train), len(validation)), (180, 90))
            return real_compare(train, validation, k_values=(3, 4), seed_values=(11, 42),
                                silhouette_size=60, n_init=2)

        with patch.object(evaluation, "compare_spatial_models", side_effect=small_compare) as compare:
            output = experiment.execute_sample(folder / "preparacao.json", self.root / "synthetic_runs",
                                               aquatic_ack=True, spatial_ack=True, sam_min_norm=0)
        self.assertEqual(compare.call_count, 2)
        run = json.loads((output / "parametros.json").read_text(encoding="utf-8"))
        self.assertEqual(run["status"], "complete")
        self.assertEqual((run["n_train"], run["n_validation"]), (180, 90))
        self.assertFalse(run["raster_classification_executed"])
        self.assertFalse(run["spatial_independence_proven"])
        self.assertEqual(run["benthic_visibility"], "not_confirmed")
        self.assertFalse(list(output.rglob("*.tif")))
        for name, wavelengths in FEATURE_SETS.items():
            destination = output / name
            metrics = pd.read_csv(destination / "metrics.csv")
            self.assertEqual(set(metrics["k"]), {3, 4})
            self.assertEqual(set(metrics["split"]), {"train", "validation"})
            self.assertEqual(set(metrics["representation"]), {"bands", "pca95"})
            self.assertEqual(len(metrics), 16)
            self.assertTrue(np.isfinite(metrics["davies_bouldin"].dropna()).all())
            proportions = pd.read_csv(destination / "proportions.csv")
            self.assertTrue(proportions["group_name"].str.match(r"^Grupo Espectral [1-4]$").all())
            curves = pd.read_csv(destination / "assinaturas_bands_k3_seed11.csv")
            self.assertEqual(set(curves["wavelength_nm"]), set(wavelengths))
            self.assertTrue(curves["name"].str.match(r"^Grupo Espectral [1-3]$").all())
            models = list(destination.glob("*.joblib"))
            self.assertEqual(len(models), 8)
            model = joblib.load(destination / "modelo_bands_k3_seed11.joblib")
            indices = [WAVELENGTHS.index(w) for w in wavelengths]
            self.assertEqual(model.predict(samples["validation_values"][:, indices]).shape, (90,))
            with Image.open(destination / "pca_pc1_pc2.png") as image:
                image.verify()


if __name__ == "__main__":
    unittest.main()
