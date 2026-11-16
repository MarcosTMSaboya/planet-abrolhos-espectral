"""Rodada limitada de sensibilidade K3 na ROI aquática já revisada.

Seis cenários fixados antes dos ajustes, sem procurar a melhor semente.
Não gera mapa classificado nem identifica habitats. Nada executa ao importar.
"""
import argparse
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from agrupamento_espectral import dependencies, sha256
from espectral_avaliacao import compare_spatial_models
from espectral_espacial import spatial_partition, spatial_sample
from espectral_io import ROOT, WAVELENGTHS, find_assets, open_scene, read_block
from experimento_espectral import load_aquatic_review, write_json


SCENARIOS = (
    ("base_100_34_s42", 100, 34, 42),
    ("sorteio_100_34_s97", 100, 34, 97),
    ("sorteio_100_34_s2026", 100, 34, 2026),
    ("bloco_75_34_s42", 75, 34, 42),
    ("bloco_125_34_s42", 125, 34, 42),
    ("buffer_100_67_s42", 100, 67, 42),
)
MODEL_SEEDS = (11, 42, 97)


def paired_agreement(first, second, eligible=None, minimum=100):
    """ARI nos mesmos pixels; invariável à permutação dos nomes dos grupos."""
    first, second = np.asarray(first), np.asarray(second)
    if first.ndim != 1 or first.shape != second.shape:
        raise ValueError("Predições precisam corresponder aos mesmos pixels")
    keep = np.ones(first.size, dtype=bool) if eligible is None else np.asarray(eligible)
    if keep.dtype != bool or keep.shape != first.shape:
        raise ValueError("Seleção booleana incompatível")
    a, b = first[keep], second[keep]
    count = len(a)
    groups_a, groups_b = len(np.unique(a)), len(np.unique(b))
    status = ("insufficient_pixels" if count < minimum else
              "missing_groups" if groups_a != 3 or groups_b != 3 else "ok")
    return {"n_pixels": count, "groups_a": groups_a, "groups_b": groups_b,
            "status": status,
            "adjusted_rand": float(adjusted_rand_score(a, b)) if status == "ok" else None}


def read_reviewed_roi(prepared_path):
    prepared_path = Path(prepared_path).resolve()
    meta = json.loads(prepared_path.read_text(encoding="utf-8"))
    if meta.get("status") != "prepared":
        raise ValueError("Exige preparação concluída")
    if sha256(meta["review_manifest"]) != meta["review_manifest_sha256"]:
        raise ValueError("Revisão mudou desde a preparação")
    for name, expected in meta["artifacts_sha256"].items():
        if sha256(prepared_path.parent / name) != expected:
            raise ValueError(f"Artefato alterado: {name}")
    mask_path, review = load_aquatic_review(meta["review_manifest"])
    if review["scene_id"] != meta["scene_id"] or review["window"] != meta["window"]:
        raise ValueError("Cena/janela incompatível com a revisão")
    for source in meta["sources"]:
        stat = Path(source["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (source["size"], source["mtime_ns"]):
            raise ValueError("Fonte mudou desde a preparação")
    w = meta["window"]
    window = Window(w["col"], w["row"], w["width"], w["height"])
    # A preparação fixa as fontes exatas. Não redescobrir cópias adicionadas
    # posteriormente a outras pastas e não trocar silenciosamente a origem.
    sources = [Path(source["path"]) for source in meta["sources"]]
    if len(sources) != len(WAVELENGTHS) + 1 or "udm2" not in sources[-1].name.lower():
        raise ValueError("Inventário preparado deve conter oito bandas e UDM2")
    asset = {"bands": sources[:-1], "udm2": sources[-1]}
    with rasterio.Env(GDAL_CACHEMAX=16*1024*1024, GDAL_PAM_ENABLED="NO"):
        with open_scene(asset, mask_path) as (bands, quality, offset, mask):
            cube, valid, _, _ = read_block(bands, quality, offset, window, mask)
            if (str(bands[0].crs) != meta["grid"]["crs"]
                    or not np.allclose(list(rasterio.windows.transform(window, bands[0].transform)),
                                       meta["grid"]["transform"], rtol=0, atol=1e-8)):
                raise ValueError("Grade alterada")
    if int(valid.sum()) != meta["counts"]["eligible"]:
        raise ValueError("População válida alterada")
    return cube, valid, meta


def make_plots(metrics, agreement, splits, output):
    _, _, _, _, plt = dependencies()
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(2, 3, figsize=(12, 9))
    cmap = ListedColormap(["#eeeeee", "#246fa5", "#e08214", "#bdbdbd"])
    for index, (ax, (name, split)) in enumerate(zip(axes.ravel(), splits.items())):
        ax.imshow(split, cmap=cmap, norm=BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4), interpolation="nearest")
        ax.set(title=name, xlabel="Coluna da ROI" if index >= 3 else "", ylabel="Linha da ROI")
    fig.legend(handles=[Patch(color=cmap(i), label=n) for i, n in enumerate(
        ["Excluído", "Desenvolvimento", "Avaliação", "Faixa de separação"])],
        loc="lower center", ncol=4)
    fig.suptitle("Cenários fixados de sensibilidade espacial — sem classes ecológicas")
    fig.tight_layout(rect=(0, .04, 1, .96), h_pad=2.)
    fig.savefig(output / "particoes.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names = [s[0] for s in SCENARIOS]
    evaluated = metrics.loc[metrics.split.eq("validation")]
    for rep, label, shift in (("bands", "Bandas padronizadas", -.12), ("pca95", "PCA95", .12)):
        frame = evaluated.loc[evaluated.representation.eq(rep)].groupby("scenario").silhouette_subsample.agg(["mean", "min", "max"]).reindex(names)
        axes[0].errorbar(np.arange(len(names))+shift, frame["mean"],
                         # CSV/float pode deixar média-min em -1e-16 quando
                         # as três sementes têm score idêntico.
                         yerr=np.maximum(0., np.vstack((frame["mean"]-frame["min"], frame["max"]-frame["mean"]))),
                         fmt="o", capsize=3, label=label)
        matrix = np.eye(len(names))
        for row in agreement.loc[agreement.representation.eq(rep) & agreement.scope.eq("common_probe_descriptive")].itertuples():
            a, b = names.index(row.scenario_a), names.index(row.scenario_b)
            matrix[a, b] = matrix[b, a] = row.adjusted_rand
        if rep == "pca95":
            im = axes[1].imshow(matrix, vmin=0, vmax=1, cmap="viridis")
            for i in range(len(names)):
                for j in range(len(names)):
                    axes[1].text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white" if matrix[i,j] < .6 else "black")
            fig.colorbar(im, ax=axes[1], label="ARI (descritivo)")
    axes[0].set(title="K3 — avaliação espacial por cenário", ylabel="Silhouette (média e faixa de 3 sementes)")
    axes[0].legend()
    axes[1].set(title="PCA95, semente do modelo 42\nConcordância na mesma amostra da ROI")
    for ax in axes:
        ax.set_xticks(range(len(names)), names, rotation=65, ha="right", fontsize=8)
    axes[1].set_yticks(range(len(names)), names, fontsize=8)
    fig.tight_layout(); fig.savefig(output / "sensibilidade.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def execute(prepared_path, output_root):
    cube, valid, meta = read_reviewed_roi(prepared_path)
    output = Path(output_root) / ("sensibilidade_espacial_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    run = {"status": "in_progress", "prepared": str(Path(prepared_path).resolve()),
           "prepared_sha256": sha256(prepared_path), "scene_id": meta["scene_id"],
           "scope": "aquatic_exploratory", "benthic_visibility": "not_confirmed",
           "spatial_independence_proven": False, "raster_classification_executed": False,
           "k": 3, "wavelengths": list(WAVELENGTHS[:4]), "model_seeds": list(MODEL_SEEDS),
           "n_init": 10, "sample_seed": 42, "train_max": 20000, "validation_max": 5000,
           "silhouette_max": 1000, "threads": 1, "negative_rhos": "preserved",
           "scenarios_prespecified": [dict(name=n, block_pixels=b, buffer_pixels=f, partition_seed=s) for n,b,f,s in SCENARIOS],
           "limitations": ["Mesma ROI e aquisição; sem validação ecológica.",
                           "Avaliações dos cenários podem sobrepor-se e não são réplicas independentes.",
                           "Amostra comum inclui pixels de ajuste: ARI comum é descritivo, não desempenho fora da amostra.",
                           "Interseção de avaliações é fora do ajuste dos dois modelos, mas mantém autocorrelação e seleção de K anterior.",
                           "Variação da partição também varia a amostra e o scaler/PCA; efeitos não isolados causalmente.",
                           "Não avaliou/corrigiu glint nem comprovou visibilidade bentônica."]}
    write_json(output / "parametros.json", run)
    try:
        (output / "codigo").mkdir()
        for name in ("sensibilidade_espacial.py", "espectral_espacial.py", "espectral_avaliacao.py", "espectral_io.py", "experimento_espectral.py", "agrupamento_espectral.py"):
            shutil.copy2(Path(__file__).with_name(name), output / "codigo" / name)
        shutil.copy2(ROOT / "requirements-espectral.lock.txt", output / "codigo" / "requirements-espectral.lock.txt")
        indices = np.random.default_rng(7043).choice(np.flatnonzero(valid.ravel()), min(10000, int(valid.sum())), replace=False)
        rr, cc = np.unravel_index(indices, valid.shape)
        probe = cube[rr, cc, :4]
        np.savez_compressed(output / "amostra_comum.npz", values=probe, positions=np.column_stack((rr+meta["window"]["row"], cc+meta["window"]["col"])))
        metrics, proportions, stability, pca, partitions = [], [], [], [], []
        splits, probe_roles, probe_labels = {}, {}, {}
        for name, block, buffer, partition_seed in SCENARIOS:
            print(f"Preparando {name}", flush=True)
            folder = output / name; folder.mkdir()
            split, blocks, partition = spatial_partition(valid, block, buffer, seed=partition_seed)
            samples = spatial_sample(cube, split, seed=42, row_offset=meta["window"]["row"], col_offset=meta["window"]["col"])
            # Checagem espacial independente nas posições sorteadas.
            for role, code in (("train", 1), ("validation", 2)):
                positions = samples[role]["positions"] - [meta["window"]["row"], meta["window"]["col"]]
                if not np.all(split[positions[:, 0], positions[:, 1]] == code):
                    raise ValueError("Amostra fora da partição esperada")
            np.savez_compressed(folder / "amostras.npz", **{f"{role}_{key}": value for role, data in samples.items() for key,value in data.items()}, wavelengths=WAVELENGTHS)
            np.savez_compressed(folder / "particao.npz", split=split)
            blocks.to_csv(folder / "blocos.csv", index=False)
            partitions.append({"scenario": name, "n_train": len(samples["train"]["values"]), "n_validation": len(samples["validation"]["values"]), **partition})
            write_json(folder / "particao.json", partitions[-1])
            result = compare_spatial_models(samples["train"]["values"][:, :4], samples["validation"]["values"][:, :4],
                                            seed_values=MODEL_SEEDS, k_values=(3,), silhouette_size=1000)
            for key, collection in (("metrics", metrics), ("proportions", proportions), ("stability", stability)):
                frame = result[key].copy(); frame.insert(0, "scenario", name)
                frame.to_csv(folder / f"{key}.csv", index=False); collection.append(frame)
            write_json(folder / "pca_metadata.json", result["pca_metadata"])
            pca.append({"scenario": name, **result["pca_metadata"]})
            np.savez_compressed(folder / "indices_silhouette.npz", **result["evaluation_indices"])
            np.savez_compressed(folder / "predicoes_amostrais.npz", **{f"{rep}_seed{seed}_{role}": assigned for (rep,k,seed), roles in result["predictions"].items() for role,assigned in roles.items()})
            splits[name], probe_roles[name] = split, split[rr, cc]
            for (rep, k, seed), model in result["models"].items():
                path = folder / f"modelo_{rep}_k{k}_seed{seed}.joblib"
                joblib.dump(model, path, compress=3)
                if seed == 42:
                    with threadpool_limits(limits=1):
                        labels = model.predict(probe)
                        if not np.array_equal(joblib.load(path).predict(probe[:128]), labels[:128]):
                            raise ValueError("Modelo salvo mudou as predições")
                    probe_labels[(name, rep)] = labels
            print(f"Concluído {name}: 6 modelos", flush=True)
        pair_rows = []
        for a, b in combinations(splits, 2):
            for rep in ("bands", "pca95"):
                for scope, selection in (("common_probe_descriptive", None),
                         ("joint_evaluation_only", (probe_roles[a] == 2) & (probe_roles[b] == 2))):
                    pair_rows.append({"scenario_a": a, "scenario_b": b, "representation": rep,
                                      "model_seed": 42, "scope": scope,
                                      **paired_agreement(probe_labels[(a,rep)], probe_labels[(b,rep)], selection)})
        agreement = pd.DataFrame(pair_rows)
        agreement.to_csv(output / "concordancia_entre_particoes.csv", index=False)
        np.savez_compressed(output / "predicoes_amostra_comum.npz", **{f"{name}_{rep}": values for (name,rep),values in probe_labels.items()}, **{f"{name}_roles": values for name,values in probe_roles.items()})
        frames = {}
        for key, collection in (("metrics", metrics), ("proportions", proportions), ("stability", stability)):
            frames[key] = pd.concat(collection, ignore_index=True)
            frames[key].to_csv(output / f"{key}.csv", index=False)
        write_json(output / "pca_por_cenario.json", pca)
        make_plots(frames["metrics"], agreement, splits, output)
        expected = len(SCENARIOS)*2*len(MODEL_SEEDS)
        if len(list(output.glob("*/modelo_*.joblib"))) != expected or len(frames["metrics"]) != expected*2:
            raise ValueError("Contagem de resultados incompleta")
        run.update(status="complete", model_count=expected, common_probe_size=len(probe),
                   partitions=partitions, serialized_representatives_verified=12)
        run["artifacts_sha256"] = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "parametros.json"}
    except Exception as error:
        run.update(status="failed", error=str(error)); raise
    finally:
        write_json(output / "parametros.json", run)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-aquatic-scope", action="store_true")
    parser.add_argument("--acknowledge-spatial-limitations", action="store_true")
    args = parser.parse_args()
    if not (args.execute and args.acknowledge_aquatic_scope and args.acknowledge_spatial_limitations):
        parser.error("Exige execução autorizada e aceite das limitações aquáticas/espaciais")
    print(execute(args.prepared, ROOT / "outputs/agrupamento_piloto"), flush=True)


if __name__ == "__main__":
    main()
