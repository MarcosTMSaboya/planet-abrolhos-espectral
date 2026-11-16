"""Audita as observações 3x3 existentes, sem reler pixels rhos.

Referência: https://docs.planet.com/data/imagery/udm/
UDM2.1: banda 4 = haze (não necessariamente leve); banda 8 = bits UDM1.
Cada execução cria uma pasta nova. Não importa nem executa o app Streamlit.
"""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
WAVELENGTHS = (444, 492, 533, 566, 612, 666, 707, 866)
CLASSES = ("clear", "snow", "shadow", "haze", "heavy_haze", "cloud")
BITS = ("blackfill", "likely_cloud", "blue_anomalous", "green_anomalous",
        "red_anomalous", "rededge_anomalous", "nir_anomalous", "other_bands_anomalous")
POLICIES = ("strict", "allow_haze")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grid_info(src):
    return {"crs": str(src.crs), "transform": list(src.transform),
            "width": src.width, "height": src.height, "count": src.count,
            "nodata": str(src.nodata), "dtypes": list(src.dtypes)}


def grid_offset(reference, quality):
    """Permite recortes diferentes, mas exige a mesma grade de pixels."""
    if reference.crs is None or reference.crs != quality.crs:
        raise ValueError("CRS ausente ou diferente entre rhos e UDM2")
    a, b = reference.transform, quality.transform
    if abs(a.b) > 1e-12 or abs(a.d) > 1e-12 or a.a <= 0 or a.e >= 0:
        raise ValueError("Grade rhos rotacionada ou orientação não suportada")
    if not np.allclose([a.a, a.b, a.d, a.e], [b.a, b.b, b.d, b.e], rtol=0, atol=1e-9):
        raise ValueError("Resolução ou orientação UDM2 diferente de rhos")
    col, row = (~b) @ (a.c, a.f)
    if not np.allclose([col, row], np.round([col, row]), rtol=0, atol=1e-6):
        raise ValueError("UDM2 deslocado em fração de pixel; não foi reamostrado")
    return int(round(row)), int(round(col))


def assess_pixels(data, min_confidence=0):
    """Analisa valores brutos: zero nas bandas binárias é um valor válido."""
    if data.shape != (8, 3, 3):
        raise ValueError(f"Janela UDM2 incompleta: {data.shape}")
    classes = data[:6]
    flags = data[7].astype(np.uint8)
    blackfill = (flags & 1) != 0
    invalid = ((classes > 1).any(axis=0)
               | ((classes.sum(axis=0) != 1) & ~blackfill)
               | (data[6] > 100))
    result = {}
    for index, name in enumerate(CLASSES):
        result[f"udm2_{name}_central"] = int(data[index, 1, 1])
        result[f"udm2_{name}_fraction"] = float(np.mean(data[index] == 1))
    for bit, name in enumerate(BITS):
        result[f"udm2_{name}_fraction"] = float(np.mean((flags & (1 << bit)) != 0))
    result.update(udm2_unusable_central=int(flags[1, 1]),
                  udm2_unusable_fraction=float(np.mean(flags != 0)),
                  udm2_confidence_central=int(data[6, 1, 1]),
                  udm2_confidence_min=int(data[6].min()),
                  udm2_confidence_mean=float(data[6].mean()))
    if invalid.any():
        result["qc_error"] = "invalid_udm2_class_or_confidence"
        return result
    for policy in POLICIES:
        reasons = []
        for index, name in enumerate(CLASSES[1:], start=1):
            if name == "haze" and policy == "allow_haze":
                continue
            if np.any(data[index] == 1):
                reasons.append(name)
        if np.any(flags != 0):
            reasons.append("udm1_bits_set")
        if np.any(data[6] < min_confidence):
            reasons.append("low_confidence")
        usable = (data[0] == 1)
        if policy == "allow_haze":
            usable = usable | (data[3] == 1)
        if not usable.all():
            reasons.append("not_all_pixels_in_allowed_classes")
        result[f"qc_{policy}"] = "rejected" if reasons else "accepted"
        result[f"qc_{policy}_reason"] = ";".join(reasons) or "all_9_pixels_usable"
    return result


def locate_udm2(scene_id, directories):
    # Somente caminhos conhecidos; nenhuma busca recursiva no HD externo.
    filename = f"{scene_id}_3B_udm2_clip.tif"
    matches = sorted({(folder / filename).resolve()
                      for folder in directories if (folder / filename).is_file()})
    if len(matches) != 1:
        raise ValueError(f"UDM2: esperava um arquivo, encontrei {len(matches)} para {scene_id}")
    return matches[0]


def audit_scene(group, args):
    sid = str(group.iloc[0].scene_id)
    results = []
    provenance = {"scene_id": sid}
    try:
        quality_path = locate_udm2(sid, args.udm2_dir)
        key = sid.split("_")[1]
        if not (group.scene_key == key).all():
            raise ValueError("scene_key não corresponde ao scene_id completo")
        with ExitStack() as stack:
            rasters = []
            provenance["rhos"] = {}
            for wavelength in WAVELENGTHS:
                path = args.rhos_dir / key / f"{key}_rhos_{wavelength}.tif"
                src = stack.enter_context(rasterio.open(path))
                if src.count != 1:
                    raise ValueError(f"rhos_{wavelength} deve ter uma banda")
                rasters.append(src)
                provenance["rhos"][str(wavelength)] = {"path": str(path), **grid_info(src)}
            ref = rasters[0]
            for src in rasters[1:]:
                if (src.crs != ref.crs or src.width != ref.width or src.height != ref.height
                        or not src.transform.almost_equals(ref.transform, precision=1e-9)):
                    raise ValueError("As oito bandas rhos não compartilham a mesma grade")
            quality = stack.enter_context(rasterio.open(quality_path))
            provenance["udm2"] = {"path": str(quality_path), **grid_info(quality)}
            if quality.count != 8 or any(dtype != "uint8" for dtype in quality.dtypes):
                raise ValueError("UDM2 deve ter oito bandas uint8")
            dr, dc = grid_offset(ref, quality)
            provenance["udm2_offset_row_col"] = [dr, dc]
            xs, ys = transform("EPSG:4326", ref.crs,
                               group.longitude.tolist(), group.latitude.tolist())
            for (index, record), x, y in zip(group.iterrows(), xs, ys):
                result = {"_row": index, "udm2_path": str(quality_path)}
                try:
                    row, col = ref.index(x, y)
                    qr, qc = row + dr, col + dc
                    result.update(rhos_row=row, rhos_col=col, udm2_row=qr, udm2_col=qc)
                    if not (1 <= row < ref.height - 1 and 1 <= col < ref.width - 1
                            and 1 <= qr < quality.height - 1 and 1 <= qc < quality.width - 1):
                        raise ValueError("Janela 3x3 fora dos limites de rhos ou UDM2")
                    # Uma única leitura de 8 x 3 x 3 valores; nunca lê o raster completo.
                    data = quality.read(window=Window(qc - 1, qr - 1, 3, 3))
                    result.update(assess_pixels(data, args.min_confidence))
                    if "qc_error" not in result:
                        invalid_bands = [w for w in WAVELENGTHS
                                         if not np.isfinite(record[f"rhos_{w}_median"])
                                         or abs(record[f"rhos_{w}_median"]) >= 1e20
                                         or record[f"rhos_{w}_n_valid"] != 9]
                        if invalid_bands:
                            for policy in POLICIES:
                                result[f"qc_{policy}"] = "rejected"
                                result[f"qc_{policy}_reason"] += ";incomplete_or_invalid_rhos:" + ",".join(map(str, invalid_bands))
                except (OSError, ValueError, rasterio.errors.RasterioError) as error:
                    result["qc_error"] = str(error)
                results.append(result)
    except (OSError, ValueError, rasterio.errors.RasterioError) as error:
        provenance["error"] = str(error)
        results = [{"_row": index, "qc_error": str(error)} for index in group.index]
    for result in results:
        if "qc_error" in result:
            for policy in POLICIES:
                result[f"qc_{policy}"] = "indeterminate"
                result[f"qc_{policy}_reason"] = result["qc_error"]
    return results, provenance


def save_csv(df, directory, filename):
    df.to_csv(directory / filename, index=False, encoding="utf-8-sig", mode="x")


def summarize(audit, directory, policy):
    accepted = audit.loc[audit[f"qc_{policy}"] == "accepted"].copy()
    save_csv(accepted, directory, f"observacoes_limpas_{policy}.csv")
    ids = audit[["point_index", "longitude", "latitude"]].drop_duplicates("point_index").set_index("point_index")
    counts = accepted.groupby("point_index").scene_id.nunique()
    per_point = ids.copy()
    per_point["n_scenes"] = counts.reindex(ids.index, fill_value=0)
    summary, long_rows = [], []
    for w in WAVELENGTHS:
        column = f"rhos_{w}_median"
        groups = accepted.groupby("point_index")[column]
        per_point[column] = groups.median().reindex(ids.index)
        per_point[f"rhos_{w}_n_scenes"] = groups.count().reindex(ids.index, fill_value=0)
        values = per_point[column].dropna()
        q1, q3 = values.quantile(.25), values.quantile(.75)
        mean, std = values.mean(), values.std(ddof=1)
        summary.append(dict(wavelength_nm=w, n_points=len(values), mean=mean,
                            median=values.median(), std=std, min=values.min(), q1=q1,
                            q3=q3, iqr=q3-q1, max=values.max(),
                            cv=std/mean if np.isfinite(mean) and mean != 0 else np.nan))
        for point, row in per_point.iterrows():
            long_rows.append(dict(point_index=point, wavelength_nm=w,
                                  rhos_median=row[column], n_scenes=row[f"rhos_{w}_n_scenes"]))
    save_csv(per_point.reset_index(), directory, f"assinatura_algas_por_ponto_{policy}.csv")
    save_csv(pd.DataFrame(summary), directory, f"resumo_espectral_algas_{policy}.csv")
    save_csv(pd.DataFrame(long_rows), directory, f"assinatura_algas_long_{policy}.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/analise/estatistica_espectral_algas_3x3.csv")
    parser.add_argument("--rhos-dir", type=Path, default=ROOT / "data/acolite/geotiff")
    parser.add_argument("--udm2-dir", action="append", type=Path,
                        help="Pasta exata com UDM2; pode repetir. Substitui as três pastas padrão.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/analise/udm2")
    parser.add_argument("--min-confidence", type=int, default=0,
                        help="Mínimo em todos os nove pixels (0-100); padrão 0, sem corte adicional.")
    args = parser.parse_args()
    if not 0 <= args.min_confidence <= 100:
        parser.error("--min-confidence deve estar entre 0 e 100")
    args.udm2_dir = args.udm2_dir or [ROOT / "PSScene", ROOT.parent / "A5_Abrolhos_QGIS/PSScene",
                                        ROOT.parent / "A5_Abrolhos_QGIS"]
    df = pd.read_csv(args.input, dtype={"scene_id": str, "scene_key": str})
    required = {"scene_id", "scene_key", "point_index", "longitude", "latitude", "window"}
    required.update(f"rhos_{w}_{suffix}" for w in WAVELENGTHS for suffix in ("median", "n_valid"))
    if missing := required - set(df.columns):
        raise ValueError(f"Colunas ausentes: {sorted(missing)}")
    identity = ["scene_id", "scene_key", "point_index", "longitude", "latitude", "window"]
    if df.empty or df[identity].isna().any().any():
        raise ValueError("Entrada vazia ou campos obrigatórios ausentes")
    for w in WAVELENGTHS:
        for suffix in ("median", "n_valid"):
            column = f"rhos_{w}_{suffix}"
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if (df.groupby("scene_key").scene_id.nunique() > 1).any():
        raise ValueError("Mais de um scene_id usa a mesma pasta rhos; associação ambígua")
    if not (df.window == "3x3").all() or df.duplicated(["point_index", "scene_id"]).any():
        raise ValueError("Entrada deve conter observações únicas ponto-cena com janela 3x3")
    if not np.isfinite(df[["longitude", "latitude"]].to_numpy()).all():
        raise ValueError("Coordenadas não finitas")
    records, scenes = [], []
    # Limita o cache GDAL e fecha os arquivos ao concluir cada cena.
    with rasterio.Env(GDAL_CACHEMAX=16 * 1024 * 1024, GDAL_PAM_ENABLED="NO"):
        for sid, group in df.groupby("scene_id", sort=True):
            print(f"Auditando {sid}: {len(group)} janelas 3x3", flush=True)
            result, provenance = audit_scene(group, args)
            records.extend(result)
            scenes.append(provenance)
    audit = df.join(pd.DataFrame(records).set_index("_row")).sort_index()
    directory = args.output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory.mkdir(parents=True, exist_ok=False)
    save_csv(audit, directory, "auditoria_udm2.csv")
    totals = []
    for policy in POLICIES:
        summarize(audit, directory, policy)
        for unit in ("scene_id", "point_index"):
            table = (audit.groupby([unit, f"qc_{policy}"]).size()
                     .unstack(fill_value=0).reindex(columns=["accepted", "rejected", "indeterminate"], fill_value=0))
            table["total"] = table.sum(axis=1)
            save_csv(table.reset_index(), directory, f"qc_por_{unit}_{policy}.csv")
        counts = audit[f"qc_{policy}"].value_counts().to_dict()
        totals.append({"policy": policy, **counts})
        print(f"{policy}: {counts}", flush=True)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "input": str(args.input.resolve()), "input_sha256": digest(args.input),
                "script_sha256": digest(Path(__file__)), "min_confidence": args.min_confidence,
                "policy": "Todos os 9 pixels: strict=clear; allow_haze=clear ou banda 4. Banda 8 deve ser zero. Todas as bandas rhos exigem 9 valores válidos e mediana finita.",
                "haze_note": "UDM2.1 banda 4 é haze geral; allow_haze é apenas sensibilidade.",
                "source": "https://docs.planet.com/data/imagery/udm/",
                "versions": {"numpy": np.__version__, "pandas": pd.__version__, "rasterio": rasterio.__version__},
                "observations": len(audit), "points": int(audit.point_index.nunique()),
                "totals": totals, "scenes": scenes}
    (directory / "manifesto.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Resultados: {directory}", flush=True)


if __name__ == "__main__":
    main()
