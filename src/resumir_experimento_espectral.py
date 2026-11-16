"""Confere e resume um experimento concluído; não ajusta modelos ou lê cenas."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from agrupamento_espectral import dependencies, sha256
from espectral_io import FEATURE_SETS, WAVELENGTHS


def verify_and_summarize(run):
    run = Path(run).resolve()
    manifest = json.loads((run / "parametros.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Experimento não concluído; não resumir arquivos parciais")
    prepared_path = Path(manifest["prepared"])
    if sha256(prepared_path) != manifest["prepared_sha256"]:
        raise ValueError("Preparação alterada após o experimento")
    with np.load(prepared_path.parent / "amostras.npz", allow_pickle=False) as stored:
        train, validation = stored["train_values"].copy(), stored["validation_values"].copy()
        validation_positions = stored["validation_positions"].copy()
    _, _, _, _, plt = dependencies()
    import joblib
    from threadpoolctl import threadpool_limits
    result = run / ("resumo_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    result.mkdir(exist_ok=False)
    rows, checks = [], []
    expected_models = len(manifest["representations"])*len(manifest["k_values"])*len(manifest["seeds"])
    for feature_set, variables in FEATURE_SETS.items():
        folder = run / feature_set
        metrics, props, stability = [pd.read_csv(folder / f"{name}.csv") for name in ("metrics", "proportions", "stability")]
        expected_keys = {(rep, k, seed, split) for rep in manifest["representations"] for k in manifest["k_values"]
                         for seed in manifest["seeds"] for split in ("train", "validation")}
        keys = set(metrics[["representation", "k", "seed", "split"]].itertuples(index=False, name=None))
        if keys != expected_keys or metrics.duplicated(["representation", "k", "seed", "split"]).any():
            raise ValueError(f"Chaves incompletas/duplicadas em {feature_set}")
        for (rep, k, seed, role), group in props.groupby(["representation", "k", "seed", "split"]):
            expected_count = len(train) if role == "train" else len(validation)
            if len(group) != k or int(group.sample_count.sum()) != expected_count or not np.isclose(group.sample_fraction.sum(), 1):
                raise ValueError(f"Contagens/proporções inconsistentes: {feature_set}/{rep}/{k}/{seed}/{role}")
        paths = list(folder.glob("modelo_*.joblib"))
        if len(paths) != expected_models:
            raise ValueError(f"Número de modelos incorreto em {feature_set}")
        model_hashes = {p.name: sha256(p) for p in paths}
        (result / f"hashes_modelos_{feature_set}.json").write_text(json.dumps(model_hashes, indent=2), encoding="utf-8")
        for (rep, k), group in metrics.groupby(["representation", "k"]):
            dev, val = group[group.split.eq("train")], group[group.split.eq("validation")]
            ari = stability[stability.representation.eq(rep) & stability.k.eq(k)]
            selected_props = props[props.representation.eq(rep) & props.k.eq(k) & props.split.eq("validation")]
            minimum_fractions = selected_props.groupby("seed").sample_fraction.min()
            row = {"feature_set": feature_set, "representation": rep, "k": int(k),
                   "silhouette_train_mean": dev.silhouette_subsample.mean(),
                   "silhouette_validation_mean": val.silhouette_subsample.mean(),
                   "silhouette_validation_std_seeds": val.silhouette_subsample.std(ddof=1),
                   "davies_bouldin_train_mean": dev.davies_bouldin.mean(),
                   "davies_bouldin_validation_mean": val.davies_bouldin.mean(),
                   "ari_validation_median": ari.adjusted_rand.median(),
                   "ari_validation_min": ari.adjusted_rand.min(),
                   "ari_validation_max": ari.adjusted_rand.max(),
                   "minimum_validation_group_fraction": minimum_fractions.min(),
                   "silhouette_validation_ok_seeds": int(val.silhouette_status.eq("ok").sum()),
                   "davies_bouldin_validation_ok_seeds": int(val.davies_bouldin_status.eq("ok").sum()),
                   "stability_ok_pairs": int(ari.status.eq("ok").sum()), "seed_pairs": len(ari)}
            row["complete_group_comparison"] = bool(
                row["silhouette_validation_ok_seeds"] == len(manifest["seeds"])
                and row["davies_bouldin_validation_ok_seeds"] == len(manifest["seeds"])
                and row["stability_ok_pairs"] == row["seed_pairs"])
            rows.append(row)
        # Reabrir representantes das duas representações e reproduzir inferência float32.
        idx = [WAVELENGTHS.index(w) for w in variables]
        with np.load(folder / "predicoes_amostrais.npz", allow_pickle=False) as predictions:
            for rep in manifest["representations"]:
                k, seed = manifest["k_values"][0], 42
                path = folder / f"modelo_{rep}_k{k}_seed{seed}.joblib"
                model = joblib.load(path)  # Somente arquivo local produzido por esta execução.
                with threadpool_limits(limits=1):
                    actual = model.predict(validation[:256, idx].astype("float32"))
                expected = predictions[f"{rep}_k{k}_seed{seed}_validation"][:256]
                np.testing.assert_array_equal(actual, expected)
                checks.append({"feature_set": feature_set, "representation": rep, "k": k,
                               "seed": seed, "n_checked": len(actual), "float32_predictions_reproduced": True})
    summary = pd.DataFrame(rows).sort_values(["feature_set", "representation", "k"])
    summary.to_csv(result / "comparacao_k.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for row, feature_set in enumerate(FEATURE_SETS):
        for rep, group in summary[summary.feature_set.eq(feature_set)].groupby("representation"):
            axes[row, 0].errorbar(group.k, group.silhouette_validation_mean,
                                  yerr=group.silhouette_validation_std_seeds, marker="o", label=rep)
            axes[row, 1].plot(group.k, group.davies_bouldin_validation_mean, marker="o", label=rep)
            complete = group.stability_ok_pairs.eq(group.seed_pairs)
            axes[row, 2].plot(group.k, group.ari_validation_median.where(complete), marker="o", label=rep)
            if not complete.all():
                axes[row, 2].text(.05, .12, "ARI omitido onde faltam grupos\nna avaliação", transform=axes[row, 2].transAxes, fontsize=9)
        for col, title in enumerate(("Silhouette de avaliação ± DP entre sementes", "Davies–Bouldin de avaliação", "ARI mediano entre sementes")):
            axes[row, col].set(xlabel="K", title=f"{feature_set}\n{title}", xticks=range(3, 9))
            axes[row, col].legend()
        axes[row, 2].set_ylim(-.05, 1.05)
    fig.suptitle("Comparação exploratória por representação — avaliação com dependência espacial residual")
    fig.tight_layout(); fig.savefig(result / "comparacao_metricas.png", dpi=160); plt.close(fig)
    verification = {"status": "verified", "scientific_models": expected_models*len(FEATURE_SETS),
                    "model_reload_checks": checks, "n_train": len(train), "n_validation": len(validation),
                    "training_run_manifest_sha256": sha256(run / "parametros.json"),
                    "source_scene_read": False, "new_models_fit": False,
                    "interpretation": "Agrupamento aquático exploratório; não valida habitats ou independência espacial"}
    (result / "verificacao_resultados.json").write_text(json.dumps(verification, indent=2, ensure_ascii=False), encoding="utf-8")
    return result, summary


def plot_details(run, result, k_values):
    """Curvas e posições somente da amostra; sem interpolar classificação raster."""
    run, result = Path(run), Path(result)
    meta = json.loads((run / "parametros.json").read_text(encoding="utf-8"))
    prepared = Path(meta["prepared"])
    with np.load(prepared.parent / "amostras.npz", allow_pickle=False) as stored:
        positions = stored["validation_positions"].copy()
    _, _, _, _, plt = dependencies()
    for feature_set in FEATURE_SETS:
        folder = run / feature_set
        with np.load(folder / "predicoes_amostrais.npz", allow_pickle=False) as predicted:
            for rep in meta["representations"]:
                for k in k_values:
                    stats = pd.read_csv(folder / f"assinaturas_{rep}_k{k}_seed42.csv")
                    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
                    for ax, role in zip(axes, ("train", "validation")):
                        for group, values in stats[stats.split.eq(role)].groupby("group"):
                            color = plt.get_cmap("tab10")(int(group)-1)
                            ax.plot(values.wavelength_nm, values["median"], color=color, marker=".", label=f"Grupo Espectral {group}")
                            ax.plot(values.wavelength_nm, values["mean"], color=color, linestyle="--")
                            ax.fill_between(values.wavelength_nm, values.q1, values.q3, color=color, alpha=.13)
                        ax.set(xlabel="Comprimento de onda (nm)", ylabel="rhos", title=role)
                    axes[1].legend(fontsize=8)
                    fig.suptitle(f"{feature_set} / {rep} / K={k} / semente 42\nContínua: mediana; tracejada: média; faixa: Q1–Q3 amostral")
                    fig.tight_layout(); fig.savefig(result / f"curvas_{feature_set}_{rep}_k{k}.png", dpi=160); plt.close(fig)
                    labels = predicted[f"{rep}_k{k}_seed42_validation"]
                    fig, ax = plt.subplots(figsize=(8, 6))
                    for label in range(k):
                        points = positions[labels == label]
                        ax.scatter(points[:, 1], points[:, 0], s=3, alpha=.65, color=plt.get_cmap("tab10")(label), label=f"Grupo Espectral {label+1}")
                    ax.invert_yaxis(); ax.set(xlabel="Coluna na cena", ylabel="Linha na cena",
                        title=f"{feature_set}/{rep}/K={k}: 5 mil pontos de avaliação\nApenas amostra; sem classificação contínua e sem identidade ecológica")
                    ax.legend(fontsize=8); fig.tight_layout()
                    fig.savefig(result / f"posicoes_amostra_{feature_set}_{rep}_k{k}.png", dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--detail-k", type=int, action="append", choices=range(3, 9))
    parser.add_argument("--existing-summary", type=Path, help="Apenas acrescenta gráficos detalhados a um resumo existente")
    args = parser.parse_args()
    if args.existing_summary:
        result = args.existing_summary
    else:
        result, summary = verify_and_summarize(args.run)
        print(summary.to_string(index=False), flush=True)
    if args.detail_k:
        plot_details(args.run, result, args.detail_k)
    print(result, flush=True)


if __name__ == "__main__":
    main()
