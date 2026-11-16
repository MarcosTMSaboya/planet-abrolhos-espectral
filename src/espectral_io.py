"""Infraestrutura por blocos; referências espaciais nunca são rótulos."""
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
try:
    from .produtos_acolite import product_folder
except ImportError:
    from produtos_acolite import product_folder

WAVELENGTHS = (444, 492, 533, 566, 612, 666, 707, 866)
FEATURE_SETS = {"azul_verde": (444, 492, 533, 566), "oito_bandas": WAVELENGTHS}
ROOT = Path(__file__).resolve().parents[1]


def grid_offset(reference, other):
    """Offset inteiro: não reamostra classes nem reflectâncias silenciosamente."""
    if reference.crs is None or reference.crs != other.crs:
        raise ValueError("CRS ausente ou incompatível")
    a, b = reference.transform, other.transform
    if a.b != 0 or a.d != 0 or a.a <= 0 or a.e >= 0:
        raise ValueError("Somente grades norte-acima são suportadas")
    if not np.allclose([a.a, a.b, a.d, a.e], [b.a, b.b, b.d, b.e], rtol=0, atol=1e-9):
        raise ValueError("Resolução/orientação incompatível")
    col, row = (~b) @ (a.c, a.f)
    if not np.allclose([col, row], np.round([col, row]), rtol=0, atol=1e-6):
        raise ValueError("Deslocamento fracionário de pixel")
    return int(round(row)), int(round(col))


def find_assets(root=ROOT):
    assets = {}
    for folder in (root / "PSScene", root.parent / "A5_Abrolhos_QGIS"):
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*_3B_AnalyticMS_8b_clip.tif")):
            sid = path.name.split("_3B_")[0]
            if sid in assets:
                if assets[sid]["planet"].resolve() != path.resolve():
                    raise ValueError(f"Cena duplicada/ambígua: {sid}")
                continue
            key = sid.split("_")[1]
            assets[sid] = {"scene_id": sid, "scene_key": key, "planet": path,
                           "udm2": path.with_name(f"{sid}_3B_udm2_clip.tif"),
                           "metadata": path.with_name(f"{sid}_metadata.json"),
                           "bands": [root / "data/acolite/geotiff" / key / f"{key}_rhos_{w}.tif" for w in WAVELENGTHS]}
    for sid, asset in assets.items():
        folder, key = product_folder(root / "data/acolite/geotiff", sid, assets)
        asset["bands"] = [folder / f"{key}_rhos_{w}.tif" for w in WAVELENGTHS]
    return assets


@contextmanager
def open_scene(asset, water_mask=None):
    with ExitStack() as stack:
        bands = [stack.enter_context(rasterio.open(p)) for p in asset["bands"]]
        ref = bands[0]
        for band in bands:
            if band.count != 1 or grid_offset(ref, band) != (0, 0) or (band.width, band.height) != (ref.width, ref.height):
                raise ValueError("Bandas rhos não possuem exatamente a mesma grade")
        quality = stack.enter_context(rasterio.open(asset["udm2"]))
        if quality.count != 8 or any(d != "uint8" for d in quality.dtypes):
            raise ValueError("UDM2 deve conter oito bandas uint8")
        offset = grid_offset(ref, quality)
        mask = stack.enter_context(rasterio.open(water_mask)) if water_mask else None
        if mask and (mask.count != 1 or grid_offset(ref, mask) != (0, 0)
                     or (mask.width, mask.height) != (ref.width, ref.height)):
            raise ValueError("Máscara aquática deve ter a grade e extensão exatas de rhos")
        yield bands, quality, offset, mask


def windows(src, size=256):
    if size < 1 or size > 1024:
        raise ValueError("Blocos devem ter entre 1 e 1024 pixels por lado")
    for row in range(0, src.height, size):
        for col in range(0, src.width, size):
            yield Window(col, row, min(size, src.width-col), min(size, src.height-row))


def read_block(bands, quality, offset, window, water_mask=None):
    # Converte para float antes de preencher NaN; preserva negativos válidos.
    data = np.stack([b.read(1, window=window, masked=True).astype("float32").filled(np.nan) for b in bands])
    finite = (np.isfinite(data) & (np.abs(data) < 1e20)).all(axis=0)
    dr, dc = offset
    qw = Window(window.col_off+dc, window.row_off+dr, window.width, window.height)
    q = quality.read(window=qw, boundless=True, fill_value=0)
    clear = ((q[0] == 1) & (q[1:6] == 0).all(axis=0)
             & (q[7] == 0) & (q[6] <= 100))
    eligible = finite & clear
    if water_mask is not None:
        water = water_mask.read(1, window=window, masked=True).filled(0)
        if not np.isin(water, [0, 1]).all():
            raise ValueError("Máscara aquática deve ser binária: 1 incluir, 0 excluir")
        eligible &= water == 1
    return np.moveaxis(data, 0, -1), eligible, finite, clear
