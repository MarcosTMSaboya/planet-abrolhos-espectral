from pathlib import Path
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin

from produtos_acolite import product_folder, validate_bands


class ProductTests(unittest.TestCase):
    def test_legacy_collision_requires_full_ids(self):
        ids = ["20260213_125207_88_24d8", "20260405_125207_88_24d8"]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "ambígua"):
                product_folder(tmp, ids[0], ids)
            (Path(tmp) / ids[0]).mkdir()
            self.assertEqual(product_folder(tmp, ids[0], ids)[1], ids[0])

    def test_missing_and_incompatible_grids(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(validate_bands(tmp, "cena", (444, 492)))
            for wave, origin in ((444, 500000), (492, 500003)):
                with rasterio.open(Path(tmp) / f"cena_rhos_{wave}.tif", "w", driver="GTiff",
                        width=3, height=3, count=1, dtype="float32", crs="EPSG:32724",
                        transform=from_origin(origin, 8050000, 3, 3)) as dst:
                    dst.write(np.ones((3, 3), dtype="float32"), 1)
            with self.assertRaisesRegex(ValueError, "incompatível"):
                validate_bands(tmp, "cena", (444, 492), full_read=True)

    def test_truncated_product_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cena_rhos_444.tif"
            path.write_bytes(b"arquivo interrompido")
            with self.assertRaises(rasterio.errors.RasterioIOError):
                validate_bands(tmp, "cena", (444,), full_read=True)


if __name__ == "__main__":
    unittest.main()
