"""Auditoria leve: cabeçalhos e nove janelas 16x16 por cena, não censo."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from espectral_io import ROOT, WAVELENGTHS, find_assets, open_scene, read_block


def audit(asset, points):
    row = {"scene_id": asset["scene_id"], "scene_key": asset["scene_key"], "status": "error"}
    details = {"paths": {k: str(v) for k, v in asset.items() if k != "bands"},
               "band_paths": list(map(str, asset["bands"]))}
    try:
        meta = json.loads(asset["metadata"].read_text(encoding="utf-8"))["properties"]
        row.update({k: meta.get(k) for k in ("acquired", "cloud_percent", "clear_percent", "sun_elevation", "view_angle")})
        with open_scene(asset) as (bands, q, offset, _):
            ref = bands[0]
            row.update(crs=str(ref.crs), width=ref.width, height=ref.height,
                       resolution_x=ref.res[0], resolution_y=ref.res[1], udm2_offset=str(offset))
            details["bands"] = [{"wavelength": w, "crs": str(b.crs), "transform": list(b.transform),
                                 "width": b.width, "height": b.height, "nodata": str(b.nodata),
                                 "dtype": b.dtypes[0], "block_shapes": b.block_shapes} for w, b in zip(WAVELENGTHS, bands)]
            details["udm2"] = {"crs": str(q.crs), "transform": list(q.transform),
                               "width": q.width, "height": q.height, "nodata": str(q.nodata)}
            chunks, valid_chunks, finite_chunks, clear_chunks = [], [], [], []
            used = set()
            for fy in (.15, .5, .85):
                for fx in (.15, .5, .85):
                    col, r = min(int(fx*ref.width), max(0, ref.width-16)), min(int(fy*ref.height), max(0, ref.height-16))
                    if (r, col) in used:
                        continue
                    used.add((r, col))
                    window = Window(col, r, min(16, ref.width), min(16, ref.height))
                    data, valid, finite, clear = read_block(bands, q, offset, window)
                    chunks.append(data.reshape(-1, 8)); valid_chunks.append(valid.ravel())
                    finite_chunks.append(finite.ravel()); clear_chunks.append(clear.ravel())
            x, valid = np.concatenate(chunks), np.concatenate(valid_chunks)
            row.update(sample_pixels=len(x), sample_valid_fraction=float(np.concatenate(finite_chunks).mean()),
                       sample_clear_fraction=float(np.concatenate(clear_chunks).mean()),
                       sample_valid_clear_fraction=float(valid.mean()),
                       footprint_km2=ref.width*ref.height*abs(ref.transform.a*ref.transform.e)/1e6)
            row["sample_valid_clear_area_proxy_km2"] = row["footprint_km2"]*row["sample_valid_clear_fraction"]
            # Área proxy só é km² com CRS métrico conhecido; sem inferir água/fundo.
            if not ref.crs.is_projected or ref.crs.linear_units != "metre":
                raise ValueError("Ranking em km² exige CRS projetado em metros")
            projected = points.to_crs(ref.crs)
            rr, cc = rasterio.transform.rowcol(ref.transform, projected.geometry.x, projected.geometry.y)
            row["reference_points_in_grid"] = int(((np.array(rr) >= 0) & (np.array(rr) < ref.height)
                                                      & (np.array(cc) >= 0) & (np.array(cc) < ref.width)).sum())
            details["sample_windows"] = [{"row": r, "col": c, "size_max": 16} for r, c in sorted(used)]
            details["sample_band_stats"] = []
            for i, w in enumerate(WAVELENGTHS):
                values = x[valid, i]
                details["sample_band_stats"].append({"wavelength_nm": w, "n": len(values),
                    "negative_fraction": float(np.mean(values < 0)) if len(values) else None,
                    "quantiles": np.quantile(values, [0, .01, .25, .5, .75, .99, 1]).tolist() if len(values) else []})
            row["status"] = "ok"
    except (OSError, ValueError, KeyError, rasterio.errors.RasterioError) as error:
        row["error"] = str(error)
    details["summary"] = row
    return row, details


def main():
    import geopandas as gpd
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/agrupamento_piloto")
    args = parser.parse_args()
    run = args.output_root / ("auditoria_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    run.mkdir(parents=True, exist_ok=False)
    points_file = ROOT / "data/campo/piloto_gpx/pontos_cobertos.csv"
    table = pd.read_csv(points_file)
    points = gpd.GeoDataFrame(table, geometry=gpd.points_from_xy(table.lon, table.lat), crs="EPSG:4326")
    rows, details = [], []
    with rasterio.Env(GDAL_CACHEMAX=32*1024*1024, GDAL_PAM_ENABLED="NO"):
        for asset in find_assets().values():
            print("Auditando", asset["scene_id"], flush=True)
            row, detail = audit(asset, points)
            rows.append(row); details.append(detail)
    df = pd.DataFrame(rows)
    candidates = df.loc[df.status.eq("ok") & df.sample_valid_clear_fraction.gt(0)].sort_values(
        ["sample_valid_clear_area_proxy_km2", "cloud_percent", "scene_id"], ascending=[False, True, True])
    selected = candidates.iloc[0].scene_id if len(candidates) else None
    df.to_csv(run / "comparacao_cenas.csv", index=False, encoding="utf-8-sig")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "selected_scene_id": selected,
                "selection_rule": "Maior proxy de área válida Clear nas 9 janelas 16x16; desempate menor cloud_percent e scene_id. Seleção provisória, sem máscara aquática.",
                "method": "Cabeçalhos das oito bandas e UDM2; 9 janelas 16x16 sistemáticas por cena. Sem leitura integral, treino ou classificação.",
                "water_mask": None, "benthic_labels_confirmed": 0,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "scenes": details, "versions": {"rasterio": rasterio.__version__, "numpy": np.__version__, "pandas": pd.__version__}}
    (run / "auditoria.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print("Cena piloto provisória:", selected, "\nResultados:", run, flush=True)


if __name__ == "__main__":
    main()
