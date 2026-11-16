"""Testes pequenos, apenas com GeoTIFFs sintéticos e centros fixos.

Não acessam cenas científicas, não chamam sklearn e não treinam modelos.
Execute na raiz do projeto: .venv/Scripts/python -m unittest discover
    -s src -p test_espectral_piloto.py -v
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

import agrupamento_espectral as clustering
from espectral_io import WAVELENGTHS, grid_offset, open_scene, read_block, windows


class SyntheticSceneTests(unittest.TestCase):
    """Cada teste usa uma cena de somente 7 × 9 pixels criada em temporário."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="abrolhos_spectral_test_")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.environment = rasterio.Env(GDAL_PAM_ENABLED="NO", GDAL_CACHEMAX=1024 * 1024)
        self.environment.__enter__()
        self.addCleanup(self.environment.__exit__, None, None, None)
        self.transform = from_origin(500000, 8050000, 3, 3)
        rows, cols = np.indices((7, 9))
        self.cube = np.stack([
            (rows * .007 + cols * .003 + index * .002 - .025).astype("float32")
            for index in range(8)
        ], axis=-1)
        self.asset = {"scene_id": "sintetica_sem_rotulo", "bands": []}
        for index, wavelength in enumerate(WAVELENGTHS):
            path = self.folder / f"rhos_{wavelength}.tif"
            self.write_raster(path, self.cube[:, :, index], nodata=-9999)
            self.asset["bands"].append(path)
        self.quality = np.zeros((8, 7, 9), dtype="uint8")
        self.quality[0] = 1
        self.quality[6] = 95
        self.asset["udm2"] = self.folder / "udm2.tif"
        self.write_raster(self.asset["udm2"], self.quality, nodata=0)
        self.mask_values = np.ones((7, 9), dtype="uint8")
        self.mask_path = self.folder / "water_mask.tif"
        self.write_raster(self.mask_path, self.mask_values, nodata=0)

    def write_raster(self, path, values, *, transform=None, crs="EPSG:32724", nodata=None):
        values = np.asarray(values)
        if values.ndim == 2:
            values = values[None]
        with rasterio.open(
            path, "w", driver="GTiff", count=len(values), width=values.shape[2],
            height=values.shape[1], dtype=values.dtype, crs=crs,
            transform=self.transform if transform is None else transform, nodata=nodata,
        ) as destination:
            destination.write(values)

    def update_band_pixel(self, index, row, col, value):
        with rasterio.open(self.asset["bands"][index], "r+") as band:
            band.write(np.array([[value]], dtype="float32"), 1, window=Window(col, row, 1, 1))

    @contextmanager
    def scene(self, mask=True):
        with open_scene(self.asset, self.mask_path if mask else None) as opened:
            yield opened

    def test_nodata_nonfinite_and_internal_masks_excluded_negative_preserved(self):
        self.update_band_pixel(0, 0, 1, -9999)
        self.update_band_pixel(1, 0, 2, np.nan)
        self.update_band_pixel(2, 0, 3, np.inf)
        internal = np.full((7, 9), 255, dtype="uint8")
        internal[0, 4] = 0
        with rasterio.open(self.asset["bands"][3], "r+") as band:
            band.write_mask(internal)
        with self.scene() as (bands, quality, offset, mask):
            data, valid, finite, clear = read_block(bands, quality, offset, Window(0, 0, 9, 7), mask)
        self.assertTrue(valid[0, 0])
        np.testing.assert_array_equal(data[0, 0], self.cube[0, 0])
        self.assertLess(data[0, 0, 0], 0)
        self.assertFalse(valid[0, 1:5].any())
        self.assertEqual(int(finite.sum()), 59)
        self.assertTrue(clear.all())

    def test_udm2_zero_classes_are_raw_values_and_contaminants_rejected(self):
        # UDM2 declara nodata=0, mas zero significa ausência da classe binária.
        self.quality[0, 0, 0] = 0
        self.quality[2, 0, 1] = 1
        self.quality[7, 0, 2] = 1
        self.quality[6, 0, 3] = 101
        self.quality[6, 0, 4] = 0  # Não há corte adicional de confiança.
        self.write_raster(self.asset["udm2"], self.quality, nodata=0)
        with self.scene() as (bands, quality, offset, mask):
            _, valid, _, clear = read_block(bands, quality, offset, Window(0, 0, 9, 7), mask)
        self.assertFalse(clear[0, :4].any())
        self.assertTrue(clear[0, 4:].all())
        self.assertEqual(int(valid.sum()), 59)

    def test_integer_offset_aligns_udm2_pixels(self):
        expanded = np.zeros((8, 9, 12), dtype="uint8")
        expanded[:, 1:8, 2:11] = self.quality
        expanded[3, 4, 6] = 1
        self.write_raster(self.asset["udm2"], expanded,
                          transform=from_origin(499994, 8050003, 3, 3), nodata=0)
        with self.scene() as (bands, quality, offset, mask):
            self.assertEqual(offset, (1, 2))
            _, valid, _, _ = read_block(bands, quality, offset, Window(0, 0, 9, 7), mask)
        expected = np.ones((7, 9), dtype=bool)
        expected[3, 4] = False
        np.testing.assert_array_equal(valid, expected)

    def test_udm2_missing_border_is_not_clear(self):
        inset = self.quality[:, 1:-1, 1:-1]
        self.write_raster(self.asset["udm2"], inset,
                          transform=from_origin(500003, 8049997, 3, 3), nodata=0)
        with self.scene() as (bands, quality, offset, mask):
            self.assertEqual(offset, (-1, -1))
            _, valid, _, _ = read_block(bands, quality, offset, Window(0, 0, 9, 7), mask)
        expected = np.zeros((7, 9), dtype=bool)
        expected[1:-1, 1:-1] = True
        np.testing.assert_array_equal(valid, expected)

    def test_incompatible_grids_are_rejected(self):
        incompatible = (
            {"transform": from_origin(500001.5, 8050000, 3, 3)},
            {"transform": from_origin(500000, 8050000, 6, 6)},
            {"crs": "EPSG:4326"},
            {"crs": None},
        )
        for options in incompatible:
            with self.subTest(options=options):
                self.write_raster(self.asset["udm2"], self.quality, nodata=0, **options)
                with self.assertRaises(ValueError):
                    with self.scene():
                        pass
        # Também rejeita bandas rhos alinhadas entre si só por uma translação.
        self.write_raster(self.asset["bands"][1], self.cube[:, :, 1], nodata=-9999,
                          transform=from_origin(500003, 8050000, 3, 3))
        with self.assertRaises(ValueError):
            with self.scene():
                pass

    def test_udm2_requires_eight_uint8_bands(self):
        for values in (self.quality[:7], self.quality.astype("uint16")):
            with self.subTest(shape=values.shape, dtype=values.dtype):
                self.write_raster(self.asset["udm2"], values, nodata=0)
                with self.assertRaises(ValueError):
                    with self.scene():
                        pass

    def test_windows_cover_each_pixel_once_including_partial_edges(self):
        with rasterio.open(self.asset["bands"][0]) as reference:
            coverage = np.zeros((7, 9), dtype="uint8")
            result = list(windows(reference, 4))
            self.assertEqual(len(result), 6)
            self.assertEqual(result[-1], Window(8, 4, 1, 3))
            for window in result:
                r, c = int(window.row_off), int(window.col_off)
                coverage[r:r + int(window.height), c:c + int(window.width)] += 1
            for invalid in (0, -1, 1025):
                with self.subTest(size=invalid), self.assertRaises(ValueError):
                    list(windows(reference, invalid))
        np.testing.assert_array_equal(coverage, np.ones((7, 9), dtype="uint8"))

    def test_water_mask_excludes_zero_and_rejects_nonbinary_or_misaligned(self):
        self.mask_values[2, 3] = 0
        self.write_raster(self.mask_path, self.mask_values, nodata=0)
        with self.scene() as (bands, quality, offset, mask):
            _, valid, _, _ = read_block(bands, quality, offset, Window(0, 0, 9, 7), mask)
        self.assertEqual(int(valid.sum()), 62)
        self.assertFalse(valid[2, 3])
        self.mask_values[2, 4] = 2
        self.write_raster(self.mask_path, self.mask_values, nodata=0)
        with self.scene() as (bands, quality, offset, mask):
            with self.assertRaises(ValueError):
                read_block(bands, quality, offset, Window(0, 0, 9, 7), mask)
        self.write_raster(self.mask_path, self.mask_values, nodata=0,
                          transform=from_origin(500003, 8050000, 3, 3))
        with self.assertRaises(ValueError):
            with self.scene():
                pass

    def manifest(self, **changes):
        metadata = {
            "scene_id": self.asset["scene_id"], "reviewed": True,
            "reviewer": "teste sintético, sem revisão científica",
            "reviewed_utc": "2026-10-20T00:00:00Z", "source": "raster sintético",
            "water_criteria": "inclusão artificial para teste de engenharia",
            "exclusions": "zeros artificiais", "limitations": "sem dados científicos",
            "mask_path": self.mask_path.name, "sha256": clustering.sha256(self.mask_path),
            "values": {"0": "excluir", "1": "incluir"},
        }
        metadata.update(changes)
        path = self.folder / "mask_manifest.json"
        path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
        return path

    def test_mask_provenance_hash_and_scene_are_enforced(self):
        path, metadata = clustering.validated_mask(self.manifest(), self.asset["scene_id"])
        self.assertEqual(path, self.mask_path.resolve())
        self.assertTrue(metadata["reviewed"])
        for changes in ({"reviewed": False}, {"scene_id": "outra_cena"},
                        {"reviewer": " "}, {"sha256": "0" * 64},
                        {"values": {"0": "incluir", "1": "excluir"}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                clustering.validated_mask(self.manifest(**changes), self.asset["scene_id"])
        manifest = self.manifest()
        self.mask_values[1, 1] = 0
        self.write_raster(self.mask_path, self.mask_values, nodata=0)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            clustering.validated_mask(manifest, self.asset["scene_id"])

    def test_reservoir_is_reproducible_unique_bounded_and_tracks_counts(self):
        self.mask_values[0, 0] = 0
        self.write_raster(self.mask_path, self.mask_values, nodata=0)
        with patch.object(clustering, "dependencies", side_effect=AssertionError("Sem sklearn")):
            first = clustering.sample_scene(self.asset, self.mask_path, size=20, seed=42, block_size=3)
            again = clustering.sample_scene(self.asset, self.mask_path, size=20, seed=42, block_size=3)
            another = clustering.sample_scene(self.asset, self.mask_path, size=20, seed=97, block_size=3)
        np.testing.assert_array_equal(first[0], again[0])
        np.testing.assert_array_equal(first[1], again[1])
        self.assertFalse(np.array_equal(first[1], another[1]))
        self.assertEqual(first[0].shape, (20, 8))
        self.assertEqual(len(np.unique(first[1], axis=0)), 20)
        self.assertFalse((first[1] == [0, 0]).all(axis=1).any())
        np.testing.assert_array_equal(first[0], self.cube[first[1][:, 0], first[1][:, 1]])
        self.assertEqual(first[2], {"grid": 63, "finite_all_bands": 63, "clear": 63, "eligible": 62})
        all_values, positions, _ = clustering.sample_scene(self.asset, self.mask_path, size=100, block_size=4)
        self.assertEqual(len(all_values), 62)
        self.assertTrue((all_values < 0).any())
        self.assertEqual(len(np.unique(positions, axis=0)), 62)

    def test_insufficient_sample_is_rejected(self):
        self.mask_values[:] = 0
        self.mask_values[:2] = 1  # Apenas 18 pixels.
        self.write_raster(self.mask_path, self.mask_values, nodata=0)
        with self.assertRaisesRegex(ValueError, "Menos de 20"):
            clustering.sample_scene(self.asset, self.mask_path, size=20, block_size=3)
        for invalid in (19, 100001):
            with self.subTest(size=invalid), self.assertRaises(ValueError):
                clustering.sample_scene(self.asset, self.mask_path, size=invalid)

    def test_block_inference_matches_direct_labels_stats_and_neighbors(self):
        self.mask_values[1, 2] = 0
        self.mask_values[3, :] = 0
        self.write_raster(self.mask_path, self.mask_values, nodata=0)
        self.quality[4, 4, 4] = 1
        self.write_raster(self.asset["udm2"], self.quality, nodata=0)
        self.update_band_pixel(7, 0, 1, -9999)
        eligible = self.mask_values == 1
        eligible[4, 4] = False
        eligible[0, 1] = False
        indices = [0, 1, 2, 3]
        mean = np.array([.01, .02, .025, .03])
        scale = np.array([.02, .025, .03, .035])
        centers = np.array([[-1, -1, -1, -1], [0, 0, 0, 0], [1, 1, 1, 1]])
        z = (self.cube[eligible][:, indices] - mean) / scale
        predictions = np.argmin(np.sum((z[:, None] - centers[None]) ** 2, axis=2), axis=1)
        labels = np.zeros((7, 9), dtype="uint8")
        labels[eligible] = predictions + 1
        expected_counts = np.bincount(predictions, minlength=3)
        expected_sums = np.array([self.cube[eligible][predictions == group].astype("float64").sum(axis=0)
                                  for group in range(3)])
        expected_sums2 = np.array([(self.cube[eligible][predictions == group].astype("float64") ** 2).sum(axis=0)
                                   for group in range(3)])
        total_pairs = equal_pairs = 0
        for a, b in ((labels[:, :-1], labels[:, 1:]), (labels[:-1], labels[1:])):
            valid_pair = (a > 0) & (b > 0)
            total_pairs += int(valid_pair.sum())
            equal_pairs += int(((a == b) & valid_pair).sum())
        for block_size in (1, 3, 4, 32):
            with self.subTest(block_size=block_size):
                output = self.folder / f"groups_{block_size}.tif"
                with patch.object(clustering, "dependencies", side_effect=AssertionError("Sem sklearn")):
                    counts, sums, sums2, coherence = clustering.apply_model(
                        self.asset, self.mask_path, indices, mean, scale, centers, output, block_size)
                with rasterio.open(output) as result:
                    np.testing.assert_array_equal(result.read(1), labels)
                    self.assertEqual(result.crs, rasterio.crs.CRS.from_epsg(32724))
                    self.assertEqual(result.transform, self.transform)
                    self.assertEqual(result.nodata, 0)
                    self.assertEqual(result.dtypes, ("uint8",))
                np.testing.assert_array_equal(counts, expected_counts)
                np.testing.assert_allclose(sums, expected_sums, rtol=0, atol=1e-12)
                np.testing.assert_allclose(sums2, expected_sums2, rtol=0, atol=1e-12)
                self.assertEqual(coherence["valid_neighbor_pairs"], total_pairs)
                self.assertEqual(coherence["same_group_neighbor_pairs"], equal_pairs)
                self.assertAlmostEqual(coherence["same_group_neighbor_fraction"], equal_pairs / total_pairs)
                original_hash = clustering.sha256(output)
                with self.assertRaises(FileExistsError):
                    clustering.apply_model(self.asset, self.mask_path, indices, mean, scale, centers, output)
                self.assertEqual(clustering.sha256(output), original_hash)

    def test_empty_inference_is_unclassified_without_neighbor_ratio(self):
        self.mask_values[:] = 0
        self.write_raster(self.mask_path, self.mask_values, nodata=0)
        output = self.folder / "empty_groups.tif"
        counts, sums, sums2, coherence = clustering.apply_model(
            self.asset, self.mask_path, [0, 1], np.zeros(2), np.ones(2), np.zeros((3, 2)), output, 4)
        with rasterio.open(output) as result:
            self.assertFalse(result.read(1).any())
        self.assertFalse(counts.any())
        self.assertFalse(sums.any())
        self.assertFalse(sums2.any())
        self.assertEqual(coherence["valid_neighbor_pairs"], 0)
        self.assertIsNone(coherence["same_group_neighbor_fraction"])


class NumericalTests(unittest.TestCase):
    def test_pca_reconstructs_standardized_input_and_covariance(self):
        rng = np.random.default_rng(2026)
        data = rng.normal(loc=-.01, scale=.02, size=(47, 4))
        data[:, 1] += .6 * data[:, 0]
        original = data.copy()
        with patch.object(clustering, "dependencies", side_effect=AssertionError("Sem sklearn")):
            pca = clustering.pca_numpy(data)
        standardized = (data - pca["mean"]) / pca["scale"]
        reconstructed = pca["scores"] @ pca["components"]
        np.testing.assert_allclose(reconstructed, standardized, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(pca["components"] @ pca["components"].T, np.eye(4), atol=1e-12)
        expected_eigenvalues = np.linalg.eigvalsh(np.cov(standardized, rowvar=False))[::-1]
        np.testing.assert_allclose(pca["eigenvalues"], expected_eigenvalues, atol=1e-12)
        np.testing.assert_allclose(np.cov(pca["scores"], rowvar=False), np.diag(expected_eigenvalues), atol=1e-12)
        np.testing.assert_allclose(pca["loadings"] @ pca["loadings"].T, np.cov(standardized, rowvar=False), atol=1e-12)
        self.assertAlmostEqual(float(pca["explained_ratio"].sum()), 1)
        self.assertTrue((pca["explained_ratio"] >= 0).all())
        np.testing.assert_array_equal(data, original)
        again = clustering.pca_numpy(data)
        np.testing.assert_array_equal(pca["components"], again["components"])

    def test_pca_rejects_nonfinite_insufficient_and_constant_input(self):
        for invalid in (np.ones((2, 4)), np.ones((8, 4)),
                        np.array([[1, 2], [3, np.nan], [4, 5]]),
                        np.array([[1, 2], [3, np.inf], [4, 5]]), np.arange(8)):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                clustering.pca_numpy(invalid)

    def test_sha256_known_content(self):
        with tempfile.TemporaryDirectory(prefix="abrolhos_hash_test_") as folder:
            path = Path(folder) / "synthetic.bin"
            path.write_bytes(b"abc")
            self.assertEqual(clustering.sha256(path), hashlib.sha256(b"abc").hexdigest())


if __name__ == "__main__":
    unittest.main()
