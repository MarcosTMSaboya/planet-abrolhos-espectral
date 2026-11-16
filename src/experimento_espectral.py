"""Preparação espacial leve e experimento amostral com autorização explícita.

`preparar` não ajusta PCA/K-means. `executar-amostra` exige --execute e aceite
do escopo aquático; não aplica modelos à cena ou atribui habitats.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from agrupamento_espectral import dependencies, pca_numpy, save_pca, sha256
from espectral_io import FEATURE_SETS, ROOT, WAVELENGTHS, find_assets, open_scene, read_block


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load_aquatic_review(path):
    path = Path(path).resolve()
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("reviewed") is not True or meta.get("review_scope") != "visual_apparent_water_only":
        raise ValueError("Exige revisão explícita de água aparente, sem interpretação bentônica")
    if meta.get("benthic_visibility") != "not_confirmed" or meta.get("approved_for_benthic_interpretation") is not False:
        raise ValueError("Este experimento não aceita alegação de identidade/observabilidade bentônica")
    for key in ("reviewer", "reviewed_utc", "source", "water_criteria", "limitations"):
        if not isinstance(meta.get(key), str) or not meta[key].strip():
            raise ValueError(f"Campo de revisão ausente: {key}")
    if meta.get("values") != {"0": "excluir", "1": "incluir"}:
        raise ValueError("Convenção da máscara deve ser 0=excluir, 1=incluir")
    mask = (path.parent / meta["mask_path"]).resolve()
    if sha256(mask) != meta.get("sha256"):
        raise ValueError("SHA-256 da máscara não corresponde à revisão")
    win = meta["window"]
    if any(type(win.get(k)) is not int for k in ("col", "row", "width", "height")):
        raise ValueError("Janela deve conter inteiros")
    if min(win["col"], win["row"]) < 0 or min(win["width"], win["height"]) < 1:
        raise ValueError("Janela inválida")
    if win["width"]*win["height"] > 250000:
        raise ValueError("Preparação leve limitada a 250 mil pixels; propor execução separada para ROI maior")
    return mask, meta


def estimate_range(table):
    """Diagnóstico operacional do pior azul/verde; não prova independência."""
    blue_green = table.loc[table.band_index.isin([0, 1, 2, 3])]
    grouped = []
    for lag, group in blue_green.groupby("lag_pixels", sort=True):
        usable = len(group) == 8 and group.correlation_detrended.notna().all() and (group.n_pairs >= 100).all()
        grouped.append((int(lag), float(group.correlation_detrended.abs().max()) if usable else None))
    for previous, current in zip(grouped, grouped[1:]):
        if previous[1] is not None and current[1] is not None and max(previous[1], current[1]) <= .1:
            return current[0], grouped
    return None, grouped


def partition_plots(cube, valid, split, diagnostic, output):
    from matplotlib import pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    cmap = ListedColormap(["#e8e8e8", "#246fa5", "#e08214", "#bdbdbd"])
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(split, cmap=cmap, norm=BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4), interpolation="nearest")
    ax.set(xlabel="Coluna relativa à ROI", ylabel="Linha relativa à ROI", title="Partição espacial — sem classes ecológicas")
    ax.legend(handles=[Patch(color=cmap(i), label=name) for i, name in enumerate(
        ["Excluído", "Desenvolvimento", "Avaliação", "Faixa de separação"])],
        loc="upper center", bbox_to_anchor=(.5, -.1), ncol=2)
    fig.tight_layout(); fig.savefig(output / "divisao_espacial.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for band, ax in enumerate(axes.ravel()):
        for direction, group in diagnostic.loc[diagnostic.band_index.eq(band)].groupby("direction"):
            ax.plot(group.lag_m, group.correlation_detrended, marker=".", label=str(direction))
        ax.axhline(.1, color="gray", linestyle="--"); ax.axhline(-.1, color="gray", linestyle="--")
        ax.set(xlabel="Separação (m)", ylabel="Correlação após remoção do plano", title=f"{WAVELENGTHS[band]} nm", ylim=(-1, 1))
        ax.legend()
    fig.tight_layout(); fig.savefig(output / "diagnostico_espacial.png", dpi=150); plt.close(fig)


def prepare(review_manifest, output_root, seed=42, block_pixels=None, buffer_pixels=None):
    from espectral_espacial import lag_diagnostics, spatial_partition, spatial_sample
    mask_path, review = load_aquatic_review(review_manifest)
    asset = find_assets()[review["scene_id"]]
    w = review["window"]
    window = Window(w["col"], w["row"], w["width"], w["height"])
    output = Path(output_root) / ("preparacao_espacial_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "in_progress", "scene_id": review["scene_id"], "review_manifest": str(Path(review_manifest).resolve()),
                "review_manifest_sha256": sha256(review_manifest), "review_scope": review["review_scope"],
                "benthic_visibility": "not_confirmed", "training_executed": False, "window": w,
                "script_sha256": sha256(__file__), "random_seed": seed,
                "data_role": "Avaliação espacial para comparar modelos; não é teste final independente de seleção de K"}
    try:
        with rasterio.Env(GDAL_CACHEMAX=16*1024*1024, GDAL_PAM_ENABLED="NO"):
            with open_scene(asset, mask_path) as (bands, quality, offset, mask):
                if window.col_off+window.width > bands[0].width or window.row_off+window.height > bands[0].height:
                    raise ValueError("Janela ultrapassa a grade")
                cube, valid, finite, clear = read_block(bands, quality, offset, window, mask)
                transform = rasterio.windows.transform(window, bands[0].transform)
                profile = dict(driver="GTiff", width=w["width"], height=w["height"], count=1, dtype="uint8",
                               crs=bands[0].crs, transform=transform, nodata=0, compress="DEFLATE")
                pixel_size = float(bands[0].res[0])
                manifest["sources"] = [{"path": str(p.resolve()), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                                       for p in asset["bands"]+[asset["udm2"]]]
                manifest["grid"] = {"crs": str(bands[0].crs), "transform": list(transform), "pixel_size_m": pixel_size}
        manifest["counts"] = {"roi": int(valid.size), "finite_all_bands": int(finite.sum()),
                              "clear": int(clear.sum()), "eligible": int(valid.sum())}
        diagnostic = lag_diagnostics(cube, valid, pixel_size=pixel_size, lags=(1, 2, 4, 8, 16, 32, 64, 96, 128, 160))
        diagnostic["wavelength_nm"] = diagnostic.band_index.map(dict(enumerate(WAVELENGTHS)))
        diagnostic.to_csv(output / "correlacao_por_distancia.csv", index=False)
        estimated, worst = estimate_range(diagnostic)
        manifest["autocorrelation"] = {"estimated_range_pixels": estimated,
            "criterion": "Todas as quatro bandas azul/verde e duas direções com |r detrended|<=0.1 em dois lags consecutivos",
            "worst_absolute_detrended_by_lag": [{"lag_pixels": lag, "absolute_correlation": corr} for lag, corr in worst],
            "independence_proven": False, "note": "Remover tendência e observar correlação baixa não prova independência física/estatística"}
        # Ausência de alcance identificável não autoriza declarar independência.
        # Uma divisão operacional é entregue para exploração, com aceite obrigatório.
        if block_pixels is None:
            block_pixels = max(32, 2*estimated) if estimated is not None else 100
        if buffer_pixels is None:
            buffer_pixels = estimated if estimated is not None else 34
        split, blocks, partition_meta = spatial_partition(valid, block_pixels, buffer_pixels, seed=seed)
        samples = spatial_sample(cube, split, seed=seed, row_offset=w["row"], col_offset=w["col"])
        manifest["partition"] = {"block_pixels": block_pixels, "buffer_pixels": buffer_pixels,
                                 "block_m": block_pixels*pixel_size, "buffer_m": buffer_pixels*pixel_size,
                                 "metadata": partition_meta,
                                 "parameter_basis": "Diagnóstico de alcance" if estimated is not None else "300 m / 102 m operacionais; alcance não resolvido, não são distâncias validadas"}
        with rasterio.open(output / "divisao_espacial.tif", "w", **profile) as dst:
            dst.write(split, 1); dst.update_tags(values="0=excluido,1=desenvolvimento,2=avaliacao,3=buffer;sem classes ecologicas")
        blocks.to_csv(output / "blocos.csv", index=False)
        np.savez_compressed(output / "amostras.npz", train_values=samples["train"]["values"],
                            train_positions=samples["train"]["positions"], validation_values=samples["validation"]["values"],
                            validation_positions=samples["validation"]["positions"], wavelengths=WAVELENGTHS)
        for role, sample in samples.items():
            frame = pd.DataFrame(sample["values"], columns=[f"rhos_{v}" for v in WAVELENGTHS])
            frame.insert(0, "row", sample["positions"][:, 0]); frame.insert(1, "col", sample["positions"][:, 1])
            frame.to_csv(output / f"amostra_{role}.csv", index=False)
        dependencies()  # Carrega backend Agg; sem ajuste de modelos.
        partition_plots(cube, valid, split, diagnostic, output)
        manifest.update(status="prepared", n_train=len(samples["train"]["values"]),
                        n_validation=len(samples["validation"]["values"]),
                        spatial_range_unresolved=estimated is None,
                        requires_aquatic_scope_acknowledgement=True, requires_processing_authorization=True)
        manifest["artifacts_sha256"] = {name: sha256(output / name) for name in
                                       ("amostras.npz", "divisao_espacial.tif", "blocos.csv", "correlacao_por_distancia.csv")}
        command = ('.\\.venv\\Scripts\\python.exe src/experimento_espectral.py executar-amostra '
                   f'--prepared "{output / "preparacao.json"}" --execute --acknowledge-aquatic-scope '
                   '--acknowledge-spatial-limitations --sam-min-norm 0')
        (output / "COMANDO_PARA_APROVACAO.txt").write_text(
            "PROPOSTA. Não executado. 120 ajustes: 2 conjuntos x 2 representações x K3..8 x 5 sementes; n_init=10; uma thread.\n"
            "Não gera classificação raster. Não prova independência espacial ou habitats. SAM com limite matemático zero; ruído não calibrado.\n\n"
            +command+"\n", encoding="utf-8")
    except Exception as error:
        manifest.update(status="failed", error=str(error)); raise
    finally:
        write_json(output / "preparacao.json", manifest)
    return output


def execute_sample(prepared_path, output_root, aquatic_ack=False, spatial_ack=False, sam_min_norm=None):
    """Executa apenas a amostra aprovada; não percorre os rasters de origem."""
    from espectral_avaliacao import compare_spatial_models, spectral_separation
    prepared_path = Path(prepared_path).resolve()
    meta = json.loads(prepared_path.read_text(encoding="utf-8"))
    if not aquatic_ack or not spatial_ack or sam_min_norm is None or not np.isfinite(sam_min_norm) or sam_min_norm < 0:
        raise ValueError("Exige aceite dos escopos aquático/espacial e SAM min_norm explícito e não negativo")
    if meta.get("status") != "prepared" or meta.get("review_scope") != "visual_apparent_water_only":
        raise ValueError("Preparação ausente/incompatível")
    if sha256(meta["review_manifest"]) != meta["review_manifest_sha256"]:
        raise ValueError("Revisão alterada após a preparação")
    load_aquatic_review(meta["review_manifest"])
    for name, expected in meta["artifacts_sha256"].items():
        if sha256(prepared_path.parent / name) != expected:
            raise ValueError(f"Artefato de preparação alterado: {name}")
    with np.load(prepared_path.parent / "amostras.npz", allow_pickle=False) as stored:
        train, validation = stored["train_values"].copy(), stored["validation_values"].copy()
        wavelengths = stored["wavelengths"].tolist()
    if wavelengths != list(WAVELENGTHS) or len(train) > 20000 or len(validation) > 5000:
        raise ValueError("Amostra incompatível com os limites propostos")
    _, _, sklearn, mpl, plt = dependencies()
    import joblib
    output = Path(output_root) / ("experimento_amostral_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    run = {"status": "in_progress", "prepared": str(prepared_path), "prepared_sha256": sha256(prepared_path),
           "scene_id": meta["scene_id"], "scope": "aquatic_exploratory", "benthic_visibility": "not_confirmed",
           "spatial_independence_proven": False, "spatial_range_unresolved": meta["spatial_range_unresolved"],
           "raster_classification_executed": False, "n_train": len(train), "n_validation": len(validation),
           "negative_rhos": "preservados", "seeds": [11, 42, 97, 2026, 31415],
           "k_values": list(range(3, 9)), "n_init": 10, "representations": ["bands", "pca95"],
           "threads": 1, "sam_min_norm": sam_min_norm, "noise_floor_calibrated": False,
           "evaluation_role": meta["data_role"], "spatial_resampling_executed": False,
           "versions": {"numpy": np.__version__, "sklearn": sklearn.__version__, "matplotlib": mpl.__version__},
           "script_sha256": sha256(__file__), "evaluation_sha256": sha256(Path(__file__).with_name("espectral_avaliacao.py"))}
    try:
        for name, variables in FEATURE_SETS.items():
            folder = output / name; folder.mkdir()
            indices = [WAVELENGTHS.index(w) for w in variables]
            original_train, original_validation = train[:, indices], validation[:, indices]
            pca = pca_numpy(original_train)
            save_pca(pca, variables, folder, plt)
            results = compare_spatial_models(original_train, original_validation)
            for key in ("metrics", "proportions", "stability"):
                results[key].to_csv(folder / f"{key}.csv", index=False)
            np.savez_compressed(folder / "indices_silhouette.npz", **results["evaluation_indices"])
            np.savez_compressed(folder / "predicoes_amostrais.npz", **{
                f"{rep}_k{k}_seed{seed}_{role}": assigned for (rep, k, seed), partitions in results["predictions"].items()
                for role, assigned in partitions.items()})
            write_json(folder / "pca_metadata.json", results["pca_metadata"])
            np.savez_compressed(folder / "pca_train.npz", mean=pca["mean"], scale=pca["scale"], components=pca["components"], wavelengths=variables)
            for (representation, k, seed), model in results["models"].items():
                prefix = f"{representation}_k{k}_seed{seed}"
                joblib.dump(model, folder / f"modelo_{prefix}.joblib", compress=3)
                labels = results["predictions"][(representation, k, seed)]["train"]
                # Reexpressa os centros do modelo PCA no espaço das bandas padronizadas.
                centers = model.named_steps["kmeans"].cluster_centers_
                if "pca" in model.named_steps:
                    centers = model.named_steps["pca"].inverse_transform(centers)
                separation = spectral_separation(original_train, labels, centers, min_norm=sam_min_norm)
                separation.to_csv(folder / f"separabilidade_{prefix}.csv", index=False)
                eta = separation.attrs["eta_squared"].copy()
                eta["wavelength_nm"] = eta.feature_index.map(dict(enumerate(variables)))
                eta.to_csv(folder / f"efeito_por_banda_{prefix}.csv", index=False)
                rows = []
                for role, values, assigned in (("train", original_train, labels),
                        ("validation", original_validation, results["predictions"][(representation, k, seed)]["validation"])):
                    for group in range(k):
                        members = values[assigned == group]
                        for b, wavelength in enumerate(variables):
                            stats = np.quantile(members[:, b], [.25, .5, .75]) if len(members) else [np.nan]*3
                            rows.append({"split": role, "group": group+1, "name": f"Grupo Espectral {group+1}",
                                         "wavelength_nm": wavelength, "sample_count": len(members),
                                         "mean": members[:, b].mean() if len(members) else np.nan,
                                         "std": members[:, b].std() if len(members) else np.nan,
                                         "q1": stats[0], "median": stats[1], "q3": stats[2]})
                pd.DataFrame(rows).to_csv(folder / f"assinaturas_{prefix}.csv", index=False)
        run["status"] = "complete"
    except Exception as error:
        run.update(status="failed", error=str(error)); raise
    finally:
        write_json(output / "parametros.json", run)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    pre = sub.add_parser("preparar", help="Somente diagnóstico/partição/amostra; nenhum agrupamento")
    pre.add_argument("--review-manifest", type=Path, required=True)
    pre.add_argument("--block-pixels", type=int)
    pre.add_argument("--buffer-pixels", type=int)
    pre.add_argument("--seed", type=int, default=42)
    run = sub.add_parser("executar-amostra", help="Somente após autorização do processamento")
    run.add_argument("--prepared", type=Path, required=True)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--acknowledge-aquatic-scope", action="store_true")
    run.add_argument("--acknowledge-spatial-limitations", action="store_true")
    run.add_argument("--sam-min-norm", type=float, required=True)
    for command in (pre, run):
        command.add_argument("--output-root", type=Path, default=ROOT / "outputs/agrupamento_piloto")
    args = parser.parse_args()
    if args.action == "preparar":
        print(prepare(args.review_manifest, args.output_root, args.seed, args.block_pixels, args.buffer_pixels))
    else:
        if not args.execute:
            parser.error("Nenhum ajuste realizado: --execute exige autorização prévia do usuário")
        print(execute_sample(args.prepared, args.output_root, args.acknowledge_aquatic_scope,
                             args.acknowledge_spatial_limitations, args.sam_min_norm))


if __name__ == "__main__":
    main()
