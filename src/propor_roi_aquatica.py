"""Registra polígono candidato escolhido por revisão visual, sem validá-lo como fundo."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import rasterio
from rasterio.warp import transform_geom
from rasterio.windows import Window

from espectral_io import ROOT, WAVELENGTHS, find_assets, open_scene, read_block
from preparar_revisao_roi import display_channel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", nargs=4, type=int, required=True, metavar=("COL", "ROW", "WIDTH", "HEIGHT"))
    parser.add_argument("--source-review", type=Path, required=True)
    args = parser.parse_args()
    col, row, width, height = args.window
    if min(col, row) < 0 or min(width, height) < 1 or width*height > 600*600:
        parser.error("Revisão leve limitada a 360000 pixels em janela válida")
    if not args.source_review.is_file():
        parser.error("Informe a imagem de contexto revisada")
    win = Window(col, row, width, height)
    asset = find_assets()["20260213_125207_88_24d8"]
    output = ROOT / "outputs/agrupamento_piloto" / ("roi_candidata_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    with rasterio.Env(GDAL_CACHEMAX=16*1024*1024, GDAL_PAM_ENABLED="NO"):
        with open_scene(asset) as (bands, quality, offset, _):
            ref = bands[0]
            if col+width > ref.width or row+height > ref.height:
                raise ValueError("ROI fora da grade")
            data, eligible, finite, clear = read_block(bands, quality, offset, win)
            left, bottom, right, top = rasterio.windows.bounds(win, ref.transform)
            geom = {"type": "Polygon", "coordinates": [[[left, bottom], [right, bottom], [right, top], [left, top], [left, bottom]]]}
            wgs84 = transform_geom(ref.crs, "EPSG:4326", geom)
            feature = {"type": "Feature", "properties": {"scene_id": asset["scene_id"], "status": "candidata",
                        "water_review": "aparência aquática na prévia; sem confirmação independente",
                        "benthic_visibility": "nao_confirmada", "area_km2": width*height*9/1e6}, "geometry": wgs84}
            (output / "roi_candidata.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}, indent=2, ensure_ascii=False), encoding="utf-8")
            profile = dict(driver="GTiff", width=ref.width, height=ref.height, count=1, dtype="uint8",
                           crs=ref.crs, transform=ref.transform, nodata=0, tiled=True,
                           blockxsize=256, blockysize=256, compress="DEFLATE", BIGTIFF="IF_SAFER")
            mask_path = output / "mascara_candidata.tif"
            # Arquivo novo: área não escrita recebe NoData=0. Somente a ROI é escrita.
            with rasterio.open(mask_path, "w", **profile) as dst:
                dst.write(eligible.astype("uint8"), 1, window=win)
                dst.set_band_description(1, "Candidata aquática + rhos/UDM2; NÃO revisada para classificação")
                dst.update_tags(status="candidate_not_approved", benthic_visibility="unknown")
            crop_profile = {**profile, "width": width, "height": height,
                            "transform": rasterio.windows.transform(win, ref.transform)}
            reasons = np.ones(eligible.shape, dtype="uint8")
            reasons[~clear] = 3; reasons[~finite] = 2
            with rasterio.open(output / "motivos_roi.tif", "w", **crop_profile) as dst:
                dst.write(reasons, 1)
                dst.update_tags(values="1=candidato numericamente elegível;2=rhos inválido;3=UDM2 rejeitado. Água/fundo não confirmados.")
            radiometry, rows = [], []
            for i, wavelength in enumerate(WAVELENGTHS):
                x = data[eligible, i]
                rows.append({"wavelength_nm": wavelength, "n": len(x), "mean": float(x.mean()) if len(x) else None,
                             "std": float(x.std()) if len(x) else None, "negative_fraction": float((x < 0).mean()) if len(x) else None,
                             **{name: float(value) for name, value in zip(("min", "q1", "median", "q3", "max"),
                                np.quantile(x, [0, .25, .5, .75, 1]) if len(x) else [np.nan]*5)}})
                radiometry.append({"wavelength_nm": wavelength, "scale": bands[i].scales[0],
                                   "offset": bands[i].offsets[0], "unit": bands[i].units[0]})
            pd.DataFrame(rows).to_csv(output / "estatisticas_roi.csv", index=False, encoding="utf-8-sig")
            rgb = np.dstack([display_channel(data[:, :, WAVELENGTHS.index(w)])[0] for w in (666, 566, 492)])
            rgb[~finite] = (235, 235, 235)
            rgb[finite & ~clear] = (190, 0, 180)
            canvas = Image.new("RGB", (width, height+45), "white"); canvas.paste(Image.fromarray(rgb), (0, 45))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 5), f"ROI candidata | {width*3/1000:g} x {height*3/1000:g} km | N para cima", fill="black")
            draw.text((8, 23), "Agua aparente; fundo nao confirmado; magenta=UDM2 rejeitado", fill="black")
            canvas.save(output / "roi_rgb.png")
            # Contexto georreferenciado da imagem já revisada; coordenadas registradas no JSON adjacente.
            source_json = args.source_review.with_name("125207_preview.json")
            context_meta = json.loads(source_json.read_text(encoding="utf-8"))
            cw = context_meta["window"]
            pw, ph = context_meta["preview_size"]
            context = Image.open(args.source_review).convert("RGB")
            painter = ImageDraw.Draw(context)
            coords = ((col-cw["col_off"])*pw/cw["width"], 42+(row-cw["row_off"])*ph/cw["height"],
                      (col+width-cw["col_off"])*pw/cw["width"], 42+(row+height-cw["row_off"])*ph/cw["height"])
            painter.rectangle(coords, outline="white", width=3)
            painter.text((coords[0]+5, coords[1]+5), f"ROI candidata {width*height*9/1e6:g} km2", fill="white")
            context.save(output / "contexto_roi.png")
            # Checagem localizada da máscara escrita, sem varrer a grade completa.
            with rasterio.open(mask_path) as check:
                assert check.crs == ref.crs and check.transform == ref.transform
                assert (check.width, check.height) == (ref.width, ref.height)
                np.testing.assert_array_equal(check.read(1, window=win), eligible.astype("uint8"))
                assert check.read(1, window=Window(0, 0, 1, 1))[0, 0] == 0
    metadata = {"scene_id": asset["scene_id"], "mask_path": mask_path.name,
                "sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(), "reviewed": False,
                "reviewer": "Pendente de revisão científica", "reviewed_utc": "",
                "values": {"0": "excluir", "1": "incluir"},
                "source": str(args.source_review.resolve()), "water_criteria": "Seleção visual próxima das ilhas, sem terra aparente no recorte",
                "exclusions": "Fora da ROI, NoData, não finitos, flags UDM2; superfície e profundidade não resolvidas",
                "limitations": "Não confirma água independente, visibilidade do fundo ou classe ecológica. quality_category=test; glint residual não corrigido nos settings.",
                "created_utc": datetime.now(timezone.utc).isoformat(), "window": dict(col=col, row=row, width=width, height=height),
                "crs": "EPSG:32724", "bounds_utm": [left, bottom, right, top], "area_km2": width*height*9/1e6,
                "pixel_counts": {"roi": int(finite.size), "finite_all_bands": int(finite.sum()),
                                 "clear_udm2": int(clear.sum()), "candidate_eligible": int(eligible.sum())},
                "radiometry": radiometry, "benthic_visibility": "not_confirmed", "approved_for_clustering": False}
    (output / "manifesto_mascara.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "pixel_counts": metadata["pixel_counts"], "area_km2": metadata["area_km2"]}), flush=True)


if __name__ == "__main__":
    main()
