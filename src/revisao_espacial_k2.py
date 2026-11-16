"""Comparação pareada K2/K3 em seis geometrias e triagem de discordância.

Usa as amostras espaciais preservadas; não aprova exclusões ou habitats.
"""
import argparse
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.warp import transform_geom
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

from agrupamento_espectral import sha256
from espectral_avaliacao import compare_spatial_models
from espectral_io import ROOT
from experimento_espectral import write_json
from sensibilidade_espacial import read_reviewed_roi, SCENARIOS, MODEL_SEEDS


def align_labels(reference, labels, k):
    """Permuta IDs para máxima concordância; não altera partições/ARI."""
    reference, labels = np.asarray(reference), np.asarray(labels)
    if reference.shape != labels.shape or reference.ndim != 1:
        raise ValueError("Rótulos devem ter as mesmas posições")
    if not (np.isin(reference, np.arange(k)).all() and np.isin(labels, np.arange(k)).all()):
        raise ValueError("Rótulos fora de 0..K-1")
    counts = np.zeros((k, k), dtype="int64")
    np.add.at(counts, (reference, labels), 1)
    rows, cols = linear_sum_assignment(-counts)
    mapping = np.empty(k, dtype="int64"); mapping[cols] = rows
    return mapping[labels]


def agreement(a, b, selection, k, minimum=100):
    a, b = np.asarray(a)[selection], np.asarray(b)[selection]
    complete = len(np.unique(a)) == k and len(np.unique(b)) == k
    status = "insufficient_pixels" if len(a) < minimum else "ok" if complete else "missing_groups"
    return {"n_pixels": len(a), "status": status,
            "adjusted_rand": float(adjusted_rand_score(a, b)) if status == "ok" else None}


def execute(prepared, previous):
    import json
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    cube, valid, meta = read_reviewed_roi(prepared)
    previous = Path(previous).resolve()
    historical = json.loads((previous / "parametros.json").read_text(encoding="utf-8"))
    if historical["status"] != "complete" or historical["prepared_sha256"] != sha256(prepared):
        raise ValueError("Rodada espacial não corresponde à preparação")
    output = ROOT / "outputs/agrupamento_piloto" / ("revisao_espacial_k2_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(exist_ok=False)
    manifest = {"status": "in_progress", "scene_id": meta["scene_id"],
        "prepared": str(Path(prepared).resolve()), "prepared_sha256": sha256(prepared),
        "previous": str(previous), "previous_manifest_sha256": sha256(previous / "parametros.json"),
        "scope": "aquatic_exploratory", "k_values": [2, 3], "seeds": list(MODEL_SEEDS),
        "new_exclusions_applied": False, "benthic_visibility": "not_confirmed",
        "independence_proven": False, "original_recommendation_replaced": False,
        "map_scope": "ROI only; descriptive predictions include training pixels"}
    try:
        for name in ["amostra_comum.npz"] + [f"{n}/{f}" for n, *_ in SCENARIOS for f in ("amostras.npz", "particao.npz")]:
            key = str(Path(name))
            if sha256(previous / name) != historical["artifacts_sha256"][key]:
                raise ValueError(f"Artefato histórico alterado: {name}")
        with np.load(previous / "amostra_comum.npz", allow_pickle=False) as saved:
            positions = saved["positions"] - [meta["window"]["row"], meta["window"]["col"]]
        rr, cc = positions.T
        probe, all_values = cube[rr, cc, :4], cube[valid, :4]
        metrics, common, stability, pairs = [], [], [], []
        labels, roles, maps = {}, {}, {}
        with threadpool_limits(limits=1):
            for name, *_ in SCENARIOS:
                folder = output / name; folder.mkdir()
                with np.load(previous / name / "amostras.npz", allow_pickle=False) as sample:
                    x, y = sample["train_values"][:, :4], sample["validation_values"][:, :4]
                    train_pos, val_pos = sample["train_positions"], sample["validation_positions"]
                with np.load(previous / name / "particao.npz", allow_pickle=False) as saved:
                    split = saved["split"]
                for pos, code in ((train_pos, 1), (val_pos, 2)):
                    local = pos - [meta["window"]["row"], meta["window"]["col"]]
                    if not np.all(split[local[:, 0], local[:, 1]] == code):
                        raise ValueError("Amostra fora da partição")
                result = compare_spatial_models(x, y, seed_values=MODEL_SEEDS,
                                               k_values=(2, 3), silhouette_size=1000)
                for key, collection in (("metrics", metrics), ("stability", stability)):
                    frame = result[key].copy(); frame.insert(0, "scenario", name); collection.append(frame)
                indices = result["evaluation_indices"]["validation"]
                z = result["scaler"].transform(y.astype("float64"))
                for (rep, k, seed), model in result["models"].items():
                    assigned = result["predictions"][(rep, k, seed)]["validation"]
                    complete = len(np.unique(assigned)) == k
                    sample_complete = len(np.unique(assigned[indices])) == k
                    common.append({"scenario": name, "representation": rep, "k": k, "seed": seed,
                        "silhouette": float(silhouette_score(z[indices], assigned[indices])) if sample_complete else None,
                        "davies_bouldin": float(davies_bouldin_score(z, assigned)) if complete else None,
                        "status": "ok" if complete and sample_complete else "missing_groups"})
                    path = folder / f"modelo_{rep}_k{k}_s{seed}.joblib"
                    joblib.dump(model, path, compress=3)
                    if not np.array_equal(joblib.load(path).predict(y[:256]), assigned[:256]):
                        raise ValueError("Modelo reaberto não reproduz previsões")
                    if seed == 42:
                        labels[(name, rep, k)] = model.predict(probe)
                        if rep == "pca95":
                            maps[(name, k)] = model.predict(all_values)
                roles[name] = split[rr, cc]
                print(f"Concluído {name}: 12 modelos", flush=True)
        for a, b in combinations(roles, 2):
            for rep in ("bands", "pca95"):
                for k in (2, 3):
                    for scope, selection in (("common_probe_descriptive", np.ones(len(probe), dtype=bool)),
                            ("joint_evaluation_only", (roles[a] == 2) & (roles[b] == 2))):
                        pairs.append({"scenario_a": a, "scenario_b": b, "representation": rep,
                            "k": k, "scope": scope, **agreement(labels[(a, rep, k)], labels[(b, rep, k)], selection, k)})
        for key, data in (("metrics", pd.concat(metrics)), ("stability", pd.concat(stability)),
                          ("metricas_espaco_comum", pd.DataFrame(common)), ("concordancia", pd.DataFrame(pairs))):
            data.to_csv(output / f"{key}.csv", index=False)
        np.savez_compressed(output / "predicoes_amostra_comum.npz",
            **{f"{n}_{rep}_k{k}": v for (n, rep, k), v in labels.items()},
            **{f"{n}_roles": v for n, v in roles.items()})
        reference = SCENARIOS[0][0]
        ref = maps[(reference, 2)]
        aligned = np.stack([align_labels(ref, maps[(name, 2)], 2) for name, *_ in SCENARIOS])
        disagreement = np.mean(aligned[1:] != aligned[0], axis=0)
        layer = np.full(valid.shape, -1., dtype="float32"); layer[valid] = disagreement
        transform = Affine(*meta["grid"]["transform"][:6])
        profile = dict(driver="GTiff", height=valid.shape[0], width=valid.shape[1], count=1,
            crs=meta["grid"]["crs"], transform=transform, compress="DEFLATE")
        with rasterio.open(output / "discordancia_k2.tif", "w", dtype="float32", nodata=-1, **profile) as dst:
            dst.write(layer, 1)
            dst.update_tags(scope="descriptive_not_habitat", meaning="fraction of five scenarios disagreeing with base after label alignment")
        # Quadrículas para inspeção, nunca polígonos de exclusão automática.
        regions, features = [], []
        for row in range(0, valid.shape[0], 50):
            for col in range(0, valid.shape[1], 50):
                keep = valid[row:row+50, col:col+50]
                if not keep.any():
                    continue
                block = layer[row:row+50, col:col+50][keep]
                record = {"region_id": f"r{row}_c{col}", "row_roi": row, "col_roi": col,
                    "n_valid": int(keep.sum()), "mean_disagreement": float(block.mean()),
                    "fraction_any_disagreement": float(np.mean(block > 0)),
                    "review_status": "pending", "decision": "", "evidence": ""}
                regions.append(record)
                corners = [transform * p for p in ((col,row),(min(col+50,valid.shape[1]),row),
                    (min(col+50,valid.shape[1]),min(row+50,valid.shape[0])),(col,min(row+50,valid.shape[0])),(col,row))]
                geom = transform_geom(meta["grid"]["crs"], "EPSG:4326", {"type":"Polygon", "coordinates":[corners]})
                features.append({"type":"Feature", "geometry":geom, "properties":record})
        pd.DataFrame(regions).sort_values("mean_disagreement", ascending=False).to_csv(output / "regioes_para_revisao.csv", index=False)
        write_json(output / "regioes_para_revisao.geojson", {"type":"FeatureCollection", "features":features})
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, k in zip(axes[:2], (2, 3)):
            img = np.full(valid.shape, np.nan); img[valid] = maps[(reference, k)] + 1
            ax.imshow(img, cmap="viridis", interpolation="nearest", vmin=1, vmax=k)
            ax.set_title(f"PCA95 / K{k} / base / semente42\nIDs locais sem identidade ecológica")
        im = axes[2].imshow(np.ma.masked_where(~valid, layer), vmin=0, vmax=1, cmap="magma")
        axes[2].set_title("K2: discordância com cenário base\n5 comparações após alinhar IDs")
        fig.colorbar(im, ax=axes[2], label="Fração dos cinco cenários")
        for ax in axes:
            ax.set(xlabel="Coluna da ROI", ylabel="Linha da ROI")
        fig.suptitle("Triagem espacial | mesmos pixels da ROI | não valida habitats ou exclusões")
        fig.tight_layout(); fig.savefig(output / "revisao_espacial.png", dpi=160); plt.close(fig)
        table = pd.DataFrame(common).groupby(["scenario", "representation", "k"])[["silhouette", "davies_bouldin"]].mean()
        rows = []
        wins = 0
        for name, *_ in SCENARIOS:
            for rep in ("bands", "pca95"):
                two, three = table.loc[(name,rep,2)], table.loc[(name,rep,3)]
                wins += int(two.silhouette > three.silhouette and two.davies_bouldin < three.davies_bouldin)
                rows.append(f"| {name} | {rep} | {two.silhouette:.4f} | {three.silhouette:.4f} | {two.davies_bouldin:.4f} | {three.davies_bouldin:.4f} |")
        report = ("# Revisão espacial K2/K3\n\n72 modelos nas seis geometrias preservadas: dois K, duas representações e três sementes. "
            "Mesmas amostras em cada comparação pareada; scaler/PCA ajustados apenas no desenvolvimento. "
            "Métricas abaixo recalculadas nas quatro bandas padronizadas, com os mesmos 1.000 pixels de avaliação por cenário.\n\n"
            "| Cenário | Representação | Silhouette K2 | K3 | DB K2 | K3 |\n|---|---|---:|---:|---:|---:|\n" + "\n".join(rows) +
            f"\n\nK2 foi favorecido pelos dois indicadores em {wins}/12 comparações. Isso não identifica dois habitats.\n\n## Estabilidade espacial\n\n")
        pair_table = pd.DataFrame(pairs)
        for k in (2,3):
            for scope in ("common_probe_descriptive", "joint_evaluation_only"):
                subset = pair_table.loc[(pair_table.k==k)&(pair_table.scope==scope)&(pair_table.representation=="pca95")]
                usable = subset.loc[subset.status.eq("ok"), "adjusted_rand"]
                report += f"- PCA95 K{k}, {scope}: {len(usable)}/15 pares completos; ARI {usable.min():.3f}–{usable.max():.3f}.\n"
        report += (f"\n## Triagem da ROI\n\n{np.mean(disagreement>0):.2%} dos pixels divergem da base em pelo menos uma das cinco comparações K2; "
            f"discordância média {disagreement.mean():.2%}. IDs foram permutados por máxima concordância na ROI, apenas para visualização. "
            "Os mapas incluem pixels de ajuste e são descritivos. ARI não depende dessa permutação.\n\n"
            "O GeoTIFF discordancia_k2.tif usa -1=fora da máscara e 0–1=fração de cenários discordantes. "
            "As quadrículas de 150 m em regioes_para_revisao.geojson (WGS84) e CSV ordenam prioridades; todas estão pendentes. "
            "Discordância pode refletir amostragem/gradiente e não é evidência suficiente para excluir pixels. "
            "Regiões de baixa discordância também precisam de revisão de superfície e observabilidade.\n\n"
            "## Limites e decisão\n\nA avaliação já participou da seleção de modelos; partições se sobrepõem e mantêm autocorrelação. "
            "Interseções de avaliação têm tamanhos diferentes; scores ausentes por falta de grupos permanecem explícitos. "
            "Manter K2 como candidato em revisão; não substituir automaticamente o mapa histórico. "
            "Abrir RGB, diagnóstico NIR e quadrículas no QGIS e registrar evidência, motivo e decisão. "
            "A máscara não foi alterada, não houve correção de glint e o fundo continua não confirmado.\n")
        (output / "RELATORIO.md").write_text(report, encoding="utf-8")
        shutil.copy2(__file__, output / Path(__file__).name)
        manifest.update(status="complete", model_count=72, models_reopened_verified=72,
            paired_comparisons_favoring_k2=wins, fraction_pixels_any_disagreement=float(np.mean(disagreement>0)))
        manifest["artifacts_sha256"] = {str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}
    except Exception as error:
        manifest.update(status="failed", error=str(error)); raise
    finally:
        write_json(output / "manifesto.json", manifest)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    args = parser.parse_args()
    print(execute(args.prepared, args.previous), flush=True)
