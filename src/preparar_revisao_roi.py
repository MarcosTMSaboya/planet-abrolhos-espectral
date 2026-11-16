"""Prévias para revisão de ROI; nunca infere classe ecológica ou visibilidade."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.enums import Resampling
from rasterio.windows import Window

from espectral_io import ROOT, WAVELENGTHS, find_assets


def display_channel(data):
    values = data[np.isfinite(data)]
    if not len(values):
        return np.zeros(data.shape, dtype="uint8"), [None, None]
    low, high = np.percentile(values, [2, 98])
    stretched = np.zeros(data.shape, dtype="float32")
    if high > low:
        np.subtract(data, low, out=stretched, where=np.isfinite(data))
        stretched = np.clip(stretched/(high-low), 0, 1)
    return (stretched*255).astype("uint8"), [float(low), float(high)]


def preview(asset, output, window=None, maximum=900):
    arrays, scaling, source_info = {}, {}, {}
    for wavelength in (492, 566, 666, 866):
        path = asset["bands"][WAVELENGTHS.index(wavelength)]
        with rasterio.open(path) as src:
            win = window or Window(0, 0, src.width, src.height)
            if win.col_off < 0 or win.row_off < 0 or win.col_off+win.width > src.width or win.row_off+win.height > src.height:
                raise ValueError("Janela de revisão fora da grade")
            factor = min(1, maximum/max(win.width, win.height))
            height, width = max(1, int(win.height*factor)), max(1, int(win.width*factor))
            data = src.read(1, window=win, out_shape=(height, width), masked=True,
                            resampling=Resampling.nearest).astype("float32").filled(np.nan)
            data[~np.isfinite(data) | (np.abs(data) >= 1e20)] = np.nan
            arrays[wavelength], scaling[str(wavelength)] = display_channel(data)
            if wavelength == 492:
                valid = np.isfinite(data)
                bounds = rasterio.windows.bounds(win, src.transform)
                source_info = {"crs": str(src.crs), "source_transform": list(src.transform),
                               "window": {"col_off": int(win.col_off), "row_off": int(win.row_off),
                                          "width": int(win.width), "height": int(win.height)},
                               "preview_size": [width, height], "bounds_utm": list(bounds)}
            else:
                valid &= np.isfinite(data)
            source_info.setdefault("radiometry", {})[str(wavelength)] = {
                "scale": src.scales[0], "offset": src.offsets[0], "unit": src.units[0],
                "overviews": src.overviews(1), "nodata": str(src.nodata)}
    rgb = np.dstack([arrays[w] for w in (666, 566, 492)])
    rgb[~valid] = (235, 235, 235)
    nir = np.repeat(arrays[866][:, :, None], 3, axis=2)
    nir[~valid] = (235, 235, 235)
    for label, image in (("rgb", rgb), ("nir_diagnostico", nir)):
        canvas = Image.new("RGB", (image.shape[1], image.shape[0]+42), "white")
        canvas.paste(Image.fromarray(image), (0, 42))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 4), f"{asset['scene_id']} | {label} | N para cima", fill="black")
        draw.text((8, 21), "Stretch P2-P98; escala visual por banda; sem rotulos de fundo", fill="black")
        canvas.save(output / f"{asset['scene_key']}_{label}.png")
    source_info.update(scene_id=asset["scene_id"], percentiles=scaling,
                       notice="Prévia visual sem avaliação bentônica, sem treinamento e sem máscara aquática automática.")
    (output / f"{asset['scene_key']}_preview.json").write_text(json.dumps(source_info, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-key", action="append", choices=("125207", "125210"))
    parser.add_argument("--window", nargs=4, type=int, metavar=("COL", "ROW", "WIDTH", "HEIGHT"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "outputs/agrupamento_piloto" / ("revisao_roi_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    with rasterio.Env(GDAL_CACHEMAX=16*1024*1024, GDAL_PAM_ENABLED="NO"):
        for asset in find_assets().values():
            if asset["scene_key"] in (args.scene_key or ["125207", "125210"]):
                print("Prévia", asset["scene_key"], flush=True)
                preview(asset, output, Window(*args.window) if args.window else None)
    print(output, flush=True)


if __name__ == "__main__":
    main()
