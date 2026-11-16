"""K2/K3 e controles contínuos, nos mesmos pixels do piloto preservado.

Não cria classes ecológicas ou substitui a recomendação original.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from agrupamento_espectral import sha256
from espectral_avaliacao import compare_spatial_models
from espectral_io import ROOT, FEATURE_SETS, WAVELENGTHS
from experimento_espectral import write_json
from sensibilidade_espacial import read_reviewed_roi


def read_settings(path):
    result = {}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def gradient_control(train, validation, k, indices):
    """Discretiza PC1 por quantis de desenvolvimento; mede no espaço de bandas.

    Baseline operacional contínua, sem teste de multimodalidade ou causa física.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    scaler = StandardScaler().fit(train)
    ztrain, zval = scaler.transform(train), scaler.transform(validation)
    pca = PCA(n_components=1, svd_solver="full").fit(ztrain)
    scores = pca.transform(zval).ravel()
    thresholds = np.quantile(pca.transform(ztrain).ravel(), np.arange(1, k) / k)
    labels = np.searchsorted(thresholds, scores)
    complete = len(np.unique(labels)) == k
    sample_complete = len(np.unique(labels[indices])) == k
    return {"k": k, "pc1_variance_train": float(pca.explained_variance_ratio_[0]),
            "thresholds_train": thresholds.tolist(),
            "silhouette_common_bands": float(silhouette_score(zval[indices], labels[indices])) if sample_complete else None,
            "davies_bouldin_common_bands": float(davies_bouldin_score(zval, labels)) if complete else None,
            "status": "ok" if complete and sample_complete else "missing_groups"}


def run(prepared):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    from threadpoolctl import threadpool_limits
    import joblib

    prepared = Path(prepared).resolve()
    cube, valid, meta = read_reviewed_roi(prepared)
    output = ROOT / "outputs/agrupamento_piloto" / ("continuidade_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(exist_ok=False)
    manifest = {"status": "in_progress", "scene_id": meta["scene_id"],
                "prepared": str(prepared), "prepared_sha256": sha256(prepared),
                "scope": "aquatic_exploratory", "benthic_visibility": "not_confirmed",
                "independence_proven": False, "new_mask_review": False,
                "k_values": [2, 3], "seeds": [11, 42, 97, 2026, 31415],
                "original_recommendation_replaced": False}
    try:
        # Ler novamente a ROI acima valida fontes, máscara, amostras e hashes.
        with np.load(prepared.parent / "amostras.npz", allow_pickle=False) as stored:
            train, val = stored["train_values"].copy(), stored["validation_values"].copy()
            if stored["wavelengths"].tolist() != list(WAVELENGTHS):
                raise ValueError("Ordem de bandas incompatível")
        settings_records = []
        source_dir = ROOT / "data/acolite/output" / meta["scene_id"].split("_")[1]
        for path in sorted(source_dir.glob("*l2r_settings.txt")):
            settings = read_settings(path)
            settings_records.append({"path": str(path), "sha256": sha256(path), "settings": settings})
        write_json(output / "configuracoes_efetivas.json", settings_records)
        manifest["effective_settings_files"] = len(settings_records)
        manifest["effective_settings_note"] = "Arquivos de execução L2R; vínculo a cada NetCDF ainda requer conferência de atributos/logs."
        values = cube[valid].astype("float64")
        stats = []
        for i, wave in enumerate(WAVELENGTHS):
            q = np.quantile(values[:, i], [.01, .05, .5, .95, .99])
            stats.append(dict(wavelength_nm=wave, negative_fraction=float(np.mean(values[:, i] < 0)),
                              p01=q[0], p05=q[1], median=q[2], p95=q[3], p99=q[4]))
        pd.DataFrame(stats).to_csv(output / "estatisticas_roi.csv", index=False)
        pd.DataFrame(values, columns=WAVELENGTHS).corr(method="spearman").to_csv(output / "spearman_roi.csv")
        fig, axes = plt.subplots(2, 3, figsize=(13, 8))
        for ax, wave in zip(axes.ravel()[:4], (492, 566, 666, 866)):
            band = cube[:, :, WAVELENGTHS.index(wave)]
            low, high = np.quantile(band[valid], [.02, .98])
            im = ax.imshow(np.ma.masked_where(~valid, band), vmin=low, vmax=high, cmap="viridis")
            ax.set_title(f"{wave} nm | P2–P98 visual"); fig.colorbar(im, ax=ax, shrink=.7)
        brightness = values[:, :4].mean(axis=1)
        axes[1, 1].hist(brightness, bins=100); axes[1, 1].set_title("Magnitude média azul/verde")
        sample = np.random.default_rng(2026).choice(len(values), min(5000, len(values)), replace=False)
        axes[1, 2].scatter(values[sample, 7], brightness[sample], s=1, alpha=.2)
        axes[1, 2].set(xlabel="rhos 866 nm", ylabel="Magnitude azul/verde", title="Associação descritiva; causa não identificada")
        fig.suptitle("Diagnóstico da ROI | fundo não confirmado | eixos das imagens em pixels")
        fig.tight_layout(); fig.savefig(output / "diagnostico_roi.png", dpi=160); plt.close(fig)
        common_rows, controls, summaries = [], [], []
        with threadpool_limits(limits=1):
            for name, waves in FEATURE_SETS.items():
                folder = output / name; folder.mkdir()
                ix = [WAVELENGTHS.index(w) for w in waves]
                x, y = train[:, ix], val[:, ix]
                results = compare_spatial_models(x, y, k_values=(2, 3))
                for key in ("metrics", "proportions", "stability"):
                    results[key].to_csv(folder / f"{key}.csv", index=False)
                indices = results["evaluation_indices"]["validation"]
                z = results["scaler"].transform(y.astype("float64"))
                predictions = {}
                for (rep, k, seed), model in results["models"].items():
                    labels = results["predictions"][(rep, k, seed)]["validation"]
                    joblib.dump(model, folder / f"modelo_{rep}_k{k}_s{seed}.joblib", compress=3)
                    predictions[f"{rep}_k{k}_s{seed}"] = labels
                    common_rows.append({"feature_set": name, "representation": rep, "k": k, "seed": seed,
                        "silhouette_common_bands": float(silhouette_score(z[indices], labels[indices])),
                        "davies_bouldin_common_bands": float(davies_bouldin_score(z, labels))})
                np.savez_compressed(folder / "predicoes_avaliacao.npz", **predictions)
                np.savez_compressed(folder / "indices_metricas.npz", **results["evaluation_indices"])
                for k in (2, 3):
                    controls.append({"feature_set": name, **gradient_control(x, y, k, indices)})
        common = pd.DataFrame(common_rows)
        common.to_csv(output / "metricas_espaco_comum.csv", index=False)
        write_json(output / "controle_gradiente.json", controls)
        for (name, rep, k), group in common.groupby(["feature_set", "representation", "k"]):
            summaries.append(f"| {name} | {rep} | {k} | {group.silhouette_common_bands.mean():.4f} | {group.davies_bouldin_common_bands.mean():.4f} |")
        report = ("# Continuidade do piloto — K2/K3 e gradiente\n\n"
            "Mesmos 20.000 pixels de desenvolvimento e 5.000 de avaliação do piloto, sem nova seleção espacial. "
            "40 modelos: dois conjuntos de bandas × duas representações × dois K × cinco sementes. "
            "A avaliação já foi usada na exploração anterior; não é teste final independente.\n\n"
            "## Comparação em espaço comum\n\n"
            "Métricas recalculadas nas bandas padronizadas de cada conjunto, inclusive para modelos PCA. "
            "Mesmos 1.500 índices de avaliação; médias entre cinco sementes. Não comparar diretamente espaços de quatro e oito bandas.\n\n"
            "| Bandas | Representação | K | Silhouette | Davies–Bouldin |\n|---|---|---:|---:|---:|\n" + "\n".join(summaries) +
            "\n\n## Controle contínuo\n\n"
            "PC1 e cortes em quantis são ajustados apenas no desenvolvimento. "
            "Esse controle divide explicitamente um eixo contínuo, sem procurar categorias naturais.\n\n")
        for control in controls:
            report += f"- {control['feature_set']}, K{control['k']}: silhouette={control['silhouette_common_bands']:.4f}; DB={control['davies_bouldin_common_bands']:.4f}.\n"
        report += ("\nScores favoráveis ao K-means não rejeitam a hipótese de continuidade: ele otimiza compactação. "
            "Este controle não é teste de multimodalidade, não fornece p-valor e não demonstra habitats. "
            "Inspecionar histogramas, continuidade espacial e referências físicas independentes.\n\n"
            "## Qualidade e próxima decisão\n\n"
            "As estatísticas e associações NIR/azul-verde são diagnósticos, não detectores validados de glint ou profundidade. "
            "A máscara anterior foi preservada, sem nova aprovação de fundo visível. "
            "Conferir configuracoes_efetivas.json, logs e atributos NetCDF antes de reprocessar. "
            "Não expandir a interpretação ecológica sem observações independentes.\n")
        (output / "RELATORIO.md").write_text(report, encoding="utf-8")
        shutil.copy2(__file__, output / Path(__file__).name)
        manifest.update(status="complete", n_models=40, n_valid_roi=int(valid.sum()),
                        n_train=len(train), n_validation=len(val), source_script_sha256=sha256(__file__))
        manifest["artifacts_sha256"] = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}
    except Exception as error:
        manifest.update(status="failed", error=str(error)); raise
    finally:
        write_json(output / "manifesto.json", manifest)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.prepared))
