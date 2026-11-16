"""Piloto não supervisionado. CLI exige máscara revisada e execução explícita.

Importar este módulo não lê imagens, não instala dependências e não treina.
"""
import hashlib
import importlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from espectral_io import WAVELENGTHS, open_scene, read_block, windows


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validated_mask(manifest_path, scene_id):
    """Nenhum fallback para Clear, nomes de pontos ou limiar NIR."""
    manifest_path = Path(manifest_path).resolve()
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if metadata.get("scene_id") != scene_id or metadata.get("reviewed") is not True:
        raise ValueError("Máscara deve ser revisada e específica para a cena")
    if metadata.get("review_scope") == "visual_apparent_water_only":
        raise ValueError("Revisão de água aparente exige o protocolo espacial aquático explícito")
    for key in ("reviewer", "reviewed_utc", "source", "water_criteria", "exclusions", "limitations"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"Proveniência da máscara ausente: {key}")
    path = (manifest_path.parent / metadata["mask_path"]).resolve()
    if sha256(path) != metadata.get("sha256"):
        raise ValueError("SHA-256 da máscara difere da revisão")
    if metadata.get("values") != {"0": "excluir", "1": "incluir"}:
        raise ValueError("Convenção da máscara deve ser 0=excluir, 1=incluir")
    return path, metadata


def sample_scene(asset, mask_path, size=20000, seed=42, block_size=256):
    """Reservatório uniforme por prioridades aleatórias; memória limitada.

    Percorre uma cena em blocos. Esta etapa NÃO é executada na auditoria leve.
    A mesma amostra será usada nos dois conjuntos de variáveis e sementes.
    """
    if not 20 <= size <= 100000:
        raise ValueError("Amostra deve ter de 20 a 100000 pixels")
    rng = np.random.default_rng(seed)
    values, positions, priorities = np.empty((0, 8)), np.empty((0, 2), dtype="int64"), np.empty(0)
    counts = {"grid": 0, "finite_all_bands": 0, "clear": 0, "eligible": 0}
    with open_scene(asset, mask_path) as (bands, quality, offset, mask):
        for win in windows(bands[0], block_size):
            data, valid, finite, clear = read_block(bands, quality, offset, win, mask)
            counts["grid"] += valid.size
            counts["finite_all_bands"] += int(finite.sum()); counts["clear"] += int(clear.sum())
            counts["eligible"] += int(valid.sum())
            rr, cc = np.nonzero(valid)
            if not len(rr):
                continue
            scores = rng.random(len(rr))
            # Guarda somente candidatos capazes de entrar no reservatório.
            if len(priorities) == size:
                take = scores < priorities.max()
                rr, cc, scores = rr[take], cc[take], scores[take]
            values = np.concatenate([values, data[rr, cc]])
            positions = np.concatenate([positions, np.column_stack([rr+int(win.row_off), cc+int(win.col_off)])])
            priorities = np.concatenate([priorities, scores])
            if len(priorities) > size:
                take = np.argpartition(priorities, size-1)[:size]
                values, positions, priorities = values[take], positions[take], priorities[take]
    if len(values) < 20:
        raise ValueError("Menos de 20 pixels elegíveis na ROI; não agrupar")
    order = np.argsort(priorities)
    return values[order], positions[order], counts


def pca_numpy(x):
    """PCA por SVD em variáveis padronizadas; negativos permanecem intactos."""
    x = np.asarray(x, dtype="float64")
    if x.ndim != 2 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError("PCA exige matriz finita com pelo menos três amostras")
    mean, scale = x.mean(axis=0), x.std(axis=0, ddof=0)
    if (scale <= np.finfo(float).eps).any():
        raise ValueError("Banda constante/quase constante; revisar variáveis antes do ajuste")
    z = (x-mean)/scale
    _, singular, components = np.linalg.svd(z, full_matrices=False)
    # Sinal determinístico para facilitar reprodutibilidade, sem significado ecológico.
    signs = np.sign(components[np.arange(len(components)), np.abs(components).argmax(axis=1)])
    components *= signs[:, None]
    scores = z @ components.T
    eigenvalues = singular**2/(len(x)-1)
    return {"mean": mean, "scale": scale, "components": components, "scores": scores,
            "eigenvalues": eigenvalues, "explained_ratio": eigenvalues/eigenvalues.sum(),
            "loadings": components.T*np.sqrt(eigenvalues)}


def dependencies():
    try:
        cluster = importlib.import_module("sklearn.cluster")
        metrics = importlib.import_module("sklearn.metrics")
        sklearn = importlib.import_module("sklearn")
        mpl = importlib.import_module("matplotlib")
        mpl.use("Agg")
        plt = importlib.import_module("matplotlib.pyplot")
    except ImportError as error:
        raise RuntimeError("Dependências opcionais ausentes. Solicitar autorização para requirements-espectral.txt; nada é instalado automaticamente.") from error
    return cluster, metrics, sklearn, mpl, plt


def compare_k(x, seed_values=(11, 42, 97), silhouette_size=1500):
    cluster, metrics, _, _, _ = dependencies()
    rng = np.random.default_rng(2026)
    evaluation = rng.choice(len(x), min(silhouette_size, len(x)), replace=False)
    rows, proportions, stability, models = [], [], [], {}
    for k in range(3, 9):
        predictions = []
        for seed in seed_values:
            model = cluster.KMeans(n_clusters=k, init="k-means++", n_init=10,
                                   max_iter=300, algorithm="lloyd", random_state=seed).fit(x)
            labels = model.labels_
            if len(np.unique(labels)) != k:
                raise ValueError(f"K={k}: menos de K grupos distintos; revisar diversidade espectral")
            small = labels[evaluation]
            sil = metrics.silhouette_score(x[evaluation], small) if 1 < len(np.unique(small)) < len(small) else np.nan
            rows.append({"k": k, "seed": seed, "silhouette_subsample": sil,
                         "silhouette_n": len(evaluation), "davies_bouldin_sample": metrics.davies_bouldin_score(x, labels),
                         "inertia_train": model.inertia_, "iterations": model.n_iter_, "n_train": len(x)})
            for label in range(k):
                proportions.append({"k": k, "seed": seed, "group": label+1,
                                    "sample_count": int((labels == label).sum()), "sample_fraction": float(np.mean(labels == label))})
            predictions.append(labels)
            models[(k, seed)] = model
        for (i, a), (j, b) in itertools.combinations(enumerate(seed_values), 2):
            stability.append({"k": k, "seed_a": a, "seed_b": b,
                              "adjusted_rand": metrics.adjusted_rand_score(predictions[i], predictions[j])})
    return pd.DataFrame(rows), pd.DataFrame(proportions), pd.DataFrame(stability), models


def sample_diagnostics(x, output, plt):
    table = pd.DataFrame(x, columns=[str(w) for w in WAVELENGTHS])
    table.describe(percentiles=[.01, .25, .5, .75, .99]).to_csv(output / "distribuicao_amostra.csv")
    table.corr().to_csv(output / "correlacao_amostra.csv")
    pd.DataFrame({"wavelength_nm": WAVELENGTHS, "negative_fraction": (x < 0).mean(axis=0)}).to_csv(output / "negativos_amostra.csv", index=False)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for i, ax in enumerate(axes.ravel()):
        ax.hist(x[:, i], bins=60); ax.set_title(f"{WAVELENGTHS[i]} nm"); ax.set_xlabel("rhos"); ax.set_ylabel("Pixels amostrados")
    fig.tight_layout(); fig.savefig(output / "distribuicoes.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(table.corr(), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(8), WAVELENGTHS, rotation=45); ax.set_yticks(range(8), WAVELENGTHS)
    fig.colorbar(image, ax=ax, label="Correlação Pearson"); fig.tight_layout(); fig.savefig(output / "correlacao.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(WAVELENGTHS, x[:min(100, len(x))].T, alpha=.1, color="gray")
    ax.plot(WAVELENGTHS, np.median(x, axis=0), label="Mediana amostral", color="black")
    ax.fill_between(WAVELENGTHS, *np.quantile(x, [.25, .75], axis=0), alpha=.3, label="Q1–Q3 amostral")
    ax.set(xlabel="Comprimento de onda (nm)", ylabel="rhos"); ax.legend(); fig.tight_layout()
    fig.savefig(output / "espectros_amostra.png", dpi=160); plt.close(fig)


def save_pca(pca, variables, output, plt):
    pc = [f"PC{i+1}" for i in range(len(pca["explained_ratio"]))]
    pd.DataFrame({"component": pc, "variance": pca["eigenvalues"], "explained_ratio": pca["explained_ratio"],
                  "cumulative": np.cumsum(pca["explained_ratio"])}).to_csv(output / "variancia_pca.csv", index=False)
    pd.DataFrame(pca["components"].T, index=variables, columns=pc).to_csv(output / "coeficientes_pca.csv", index_label="wavelength_nm")
    pd.DataFrame(pca["loadings"], index=variables, columns=pc).to_csv(output / "loadings_pca.csv", index_label="wavelength_nm")
    pd.DataFrame(pca["scores"], columns=pc).to_csv(output / "scores_pca_amostra.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(pca["scores"][:, 0], pca["scores"][:, 1], s=2, alpha=.2)
    ax.set(xlabel="PC1", ylabel="PC2", title="Amostra elegível — PCA padronizada")
    fig.tight_layout(); fig.savefig(output / "pca_pc1_pc2.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(pc, pca["explained_ratio"]); ax.set_ylabel("Proporção da variância explicada")
    fig.tight_layout(); fig.savefig(output / "variancia_pca.png", dpi=160); plt.close(fig)


def apply_model(asset, mask_path, feature_indices, mean, scale, centers, output, block_size=256):
    """Inferência e estatísticas exatas de contagem/média em blocos; zero=Não Classificadas."""
    if Path(output).exists():
        raise FileExistsError(output)
    k = len(centers)
    counts, sums, sums2 = np.zeros(k, dtype="int64"), np.zeros((k, 8)), np.zeros((k, 8))
    equal_pairs = total_pairs = 0
    bottom_labels = None
    stripe_bottom = None
    with open_scene(asset, mask_path) as (bands, quality, offset, mask):
        profile = bands[0].profile.copy()
        profile.update(driver="GTiff", dtype="uint8", count=1, nodata=0,
                       compress="DEFLATE", tiled=True, blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
        previous_row = None
        left_edge = None
        with rasterio.open(output, "w", **profile) as dst:
            dst.set_band_description(1, "Grupos Espectrais; 0 = Não Classificadas")
            dst.update_tags(interpretation="Agrupamento não supervisionado; sem identidade ecológica", scene_id=asset["scene_id"])
            for win in windows(bands[0], block_size):
                if previous_row != win.row_off:
                    bottom_labels, stripe_bottom = stripe_bottom, np.zeros(bands[0].width, dtype="uint8")
                    previous_row, left_edge = win.row_off, None
                data, valid, _, _ = read_block(bands, quality, offset, win, mask)
                labels = np.zeros(valid.shape, dtype="uint8")
                if valid.any():
                    z = (data[valid][:, feature_indices]-mean)/scale
                    predicted = np.argmin(((z[:, None, :]-centers[None, :, :])**2).sum(axis=2), axis=1)
                    labels[valid] = predicted+1
                    for label in range(k):
                        values = data[valid][predicted == label].astype("float64")
                        counts[label] += len(values); sums[label] += values.sum(axis=0); sums2[label] += (values**2).sum(axis=0)
                dst.write(labels, 1, window=win)
                pairs = [(labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])]
                if left_edge is not None:
                    pairs.append((left_edge, labels[:, 0]))
                if bottom_labels is not None:
                    pairs.append((bottom_labels[int(win.col_off):int(win.col_off+win.width)], labels[0]))
                for a, b in pairs:
                    ok = (a > 0) & (b > 0)
                    total_pairs += int(ok.sum()); equal_pairs += int(((a == b) & ok).sum())
                left_edge = labels[:, -1].copy()
                stripe_bottom[int(win.col_off):int(win.col_off+win.width)] = labels[-1]
    return counts, sums, sums2, {"valid_neighbor_pairs": total_pairs, "same_group_neighbor_pairs": equal_pairs,
                                "same_group_neighbor_fraction": equal_pairs/total_pairs if total_pairs else None}


def main():
    # A CLI antiga n?o tinha avalia??o espacial; delega ao protocolo expl?cito.
    from experimento_espectral import main as spatial_main
    spatial_main()


if __name__ == "__main__":
    main()
