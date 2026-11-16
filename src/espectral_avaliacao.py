"""Avaliação espectral sem rótulos ecológicos e com holdout definido a montante.

Não acessa rasters, não instala bibliotecas e não executa ajustes ao importar.
As matrizes devem vir de blocos espaciais separados antes de chamar estas APIs.
As métricas indicam estrutura estatística, sem comprovar habitats ou fundo visível.
"""
from itertools import combinations
from numbers import Integral, Real

import numpy as np
import pandas as pd


def _matrix(values, name, minimum_rows):
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} deve ser matriz numérica finita") from error
    if (result.ndim != 2 or result.shape[0] < minimum_rows
            or result.shape[1] < 1 or not np.isfinite(result).all()):
        raise ValueError(f"{name} deve ter ao menos {minimum_rows} linhas e uma coluna, finitas")
    return result


def _integers(values, name, minimum, maximum):
    values = tuple(values)
    if not values or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral)
                         or not minimum <= v <= maximum for v in values):
        raise ValueError(f"{name}: inteiros entre {minimum} e {maximum}")
    values = tuple(int(v) for v in values)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} não pode conter duplicatas")
    return values


def _metric_row(features, labels, indices, k, silhouette_score, davies_bouldin_score):
    """Ausência de algum grupo torna métricas incompletas explícitas, sem exceção."""
    present = np.unique(labels)
    sample_labels = labels[indices]
    sample_present = np.unique(sample_labels)
    result = {
        "n_samples": len(labels), "n_groups_present": len(present),
        "missing_groups": ",".join(str(i + 1) for i in range(k) if i not in present),
        "silhouette_n": len(indices), "silhouette_subsample": None,
        "silhouette_status": "ok", "davies_bouldin": None,
        "davies_bouldin_status": "ok",
    }
    if len(present) != k:
        result["davies_bouldin_status"] = "missing_groups"
        result["silhouette_status"] = "missing_groups"
        return result
    if len(present) >= len(labels):
        result["davies_bouldin_status"] = "one_sample_per_group"
    else:
        result["davies_bouldin"] = float(davies_bouldin_score(features, labels))
    if len(sample_present) != k:
        result["silhouette_status"] = "missing_groups_subsample"
    elif len(sample_present) >= len(indices):
        result["silhouette_status"] = "one_sample_per_group"
    else:
        result["silhouette_subsample"] = float(silhouette_score(features[indices], sample_labels))
    return result


def compare_spatial_models(train, validation, seed_values=(11, 42, 97, 2026, 31415),
                           k_values=range(3, 9), silhouette_size=1500,
                           representations=("bands", "pca95"), n_init=10):
    """Ajusta StandardScaler/PCA/K-means exclusivamente em ``train``.

    Compara bandas padronizadas e PCs retendo ao menos 95% da variância de
    desenvolvimento (sem whitening). Silhouette usa uma subamostra fixa por
    partição, reutilizada em todas as representações, K e sementes. Davies–Bouldin
    usa toda a amostra da partição. Uma partição ou subamostra sem todos os K
    grupos recebe métrica nula e status explícito. ARI compara as mesmas linhas
    de validation; identifica concordância entre sementes, não acurácia ecológica.

    Retorna tabelas ``metrics``, ``proportions`` e ``stability``; ``models`` contém
    Pipelines ajustadas, indexadas por (representation, k, seed), que recebem
    bandas originais em predict(). ``predictions`` usa os mesmos índices/linhas
    de entrada. ``scaler``, ``pca``, ``pca_metadata`` e ``evaluation_indices``
    permitem auditoria. Identificadores internos sklearn são 0..K-1; tabelas usam
    Grupo Espectral 1..K. Nenhuma classificação ou probabilidade ecológica é criada.

    A função não verifica independência espacial sem coordenadas: o chamador deve
    separar os blocos antes de amostrar. Serializar Pipelines só é apropriado para
    arquivos locais confiáveis; nunca desserialize pickle/joblib não confiável.
    """
    train = _matrix(train, "train", 3)
    validation = _matrix(validation, "validation", 1)
    if validation.shape[1] != train.shape[1]:
        raise ValueError("train e validation devem conter as mesmas bandas na mesma ordem")
    seeds = _integers(seed_values, "seed_values", 0, 2**32 - 1)
    ks = _integers(k_values, "k_values", 2, len(train) - 1)
    if (isinstance(silhouette_size, (bool, np.bool_)) or not isinstance(silhouette_size, Integral)
            or silhouette_size < 2):
        raise ValueError("silhouette_size deve ser inteiro >= 2")
    if isinstance(n_init, (bool, np.bool_)) or not isinstance(n_init, Integral) or n_init < 1:
        raise ValueError("n_init deve ser inteiro >= 1")
    representations = tuple(representations)
    if (not representations or len(set(representations)) != len(representations)
            or any(value not in ("bands", "pca95") for value in representations)):
        raise ValueError("representations aceita bands e/ou pca95, sem duplicatas")
    if len(np.unique(train, axis=0)) < max(ks):
        raise ValueError("train possui menos espectros distintos que o maior K")

    # Imports tardios: preparar/inspecionar outros módulos não exige estes pacotes.
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    from threadpoolctl import threadpool_limits

    rng = np.random.default_rng(2026)
    evaluation = {
        "train": np.sort(rng.choice(len(train), min(silhouette_size, len(train)), replace=False)),
        "validation": np.sort(rng.choice(len(validation), min(silhouette_size, len(validation)), replace=False)),
    }
    rows, proportions, stability = [], [], []
    models, predictions = {}, {}
    with threadpool_limits(limits=1):
        scaler = StandardScaler().fit(train)
        z_train, z_validation = scaler.transform(train), scaler.transform(validation)
        pca = None
        pca_metadata = {
            "fit_partition": "train", "variance_target": 0.95, "whiten": False,
            "n_features": train.shape[1], "n_train": len(train),
            "constant_feature_indices": np.flatnonzero(scaler.var_ == 0).tolist(),
        }
        if "pca95" in representations:
            pca = PCA(n_components=0.95, svd_solver="full", whiten=False).fit(z_train)
            pca_metadata.update({
                "n_components": int(pca.n_components_),
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "retained_variance": float(pca.explained_variance_ratio_.sum()),
                "components": pca.components_.tolist(), "mean": pca.mean_.tolist(),
            })
        for representation in representations:
            features = {"train": z_train, "validation": z_validation}
            if representation == "pca95":
                features = {split: pca.transform(values) for split, values in features.items()}
            for k in ks:
                for seed in seeds:
                    key = (representation, k, seed)
                    model = KMeans(n_clusters=k, init="k-means++", n_init=int(n_init),
                                   max_iter=300, algorithm="lloyd", random_state=seed).fit(features["train"])
                    # Rasterio fornece float32; StandardScaler preserva esse
                    # dtype, mas K-means treinado em float64 exige buffers iguais.
                    # Persistir a conversão garante inferência consistente após
                    # joblib/pickle, inclusive em blocos vindos diretamente do TIFF.
                    to_float64 = FunctionTransformer(np.asarray, kw_args={"dtype": np.float64},
                                                     validate=False)
                    steps = [("to_float64", to_float64), ("scaler", scaler)]
                    if representation == "pca95":
                        steps.append(("pca", pca))
                    models[key] = Pipeline(steps + [("kmeans", model)])
                    predictions[key] = {"train": model.labels_.copy(),
                                        "validation": model.predict(features["validation"])}
                    for split in ("train", "validation"):
                        labels = predictions[key][split]
                        common = {"representation": representation, "k": k, "seed": seed, "split": split}
                        metrics = _metric_row(features[split], labels, evaluation[split], k,
                                              silhouette_score, davies_bouldin_score)
                        metrics.update(common)
                        metrics.update({"n_features_model": features[split].shape[1],
                                        "inertia_train": float(model.inertia_),
                                        "iterations": int(model.n_iter_)})
                        rows.append(metrics)
                        for label in range(k):
                            count = int(np.count_nonzero(labels == label))
                            proportions.append({**common, "group": label + 1,
                                                "group_name": f"Grupo Espectral {label + 1}",
                                                "sample_count": count,
                                                "sample_fraction": count / len(labels)})
                for seed_a, seed_b in combinations(seeds, 2):
                    a = predictions[(representation, k, seed_a)]["validation"]
                    b = predictions[(representation, k, seed_b)]["validation"]
                    stability.append({"representation": representation, "k": k,
                                      "seed_a": seed_a, "seed_b": seed_b,
                                      "split": "validation", "n_samples": len(validation),
                                      "adjusted_rand": float(adjusted_rand_score(a, b)),
                                      "status": "ok" if len(np.unique(a)) == k and len(np.unique(b)) == k
                                      else "missing_groups"})
    return {
        "metrics": pd.DataFrame(rows), "proportions": pd.DataFrame(proportions),
        "stability": pd.DataFrame(stability, columns=["representation", "k", "seed_a", "seed_b",
                                                       "split", "n_samples", "adjusted_rand", "status"]),
        "models": models, "predictions": predictions, "scaler": scaler, "pca": pca,
        "pca_metadata": pca_metadata, "evaluation_indices": evaluation,
    }


def spectral_separation(train_original, train_labels, standardized_centers, min_norm):
    """Mede separação entre pares de grupos, sem interpretar identidade ecológica.

    ``train_labels`` são IDs internos 0..K-1. ``standardized_centers`` deve estar
    no espaço das bandas padronizadas, inclusive quando K-means usou PCs: aplique
    pca.inverse_transform(cluster_centers_) antes de chamar. SAM é calculado entre
    médias rhos originais, preservando sinais. ``min_norm`` é um limiar explícito
    nas unidades de rhos, documentado pelo chamador: normas <= limiar produzem SAM
    nulo. Nenhum epsilon ou clipping de reflectância é aplicado.

    ``within_scatter_*`` é a média de ||x - média_do_grupo||² em rhos original.
    ``attrs['eta_squared']`` contém, por banda, a fração de soma de quadrados
    entre grupos / total (diagnóstico descritivo, sem teste inferencial).
    """
    values = _matrix(train_original, "train_original", 2)
    centers = _matrix(standardized_centers, "standardized_centers", 2)
    if centers.shape[1] != values.shape[1]:
        raise ValueError("standardized_centers deve conter todas as bandas originais")
    labels = np.asarray(train_labels)
    if (labels.ndim != 1 or len(labels) != len(values)
            or not np.issubdtype(labels.dtype, np.integer)
            or np.issubdtype(labels.dtype, np.bool_)
            or np.any(labels < 0) or np.any(labels >= len(centers))):
        raise ValueError("train_labels deve conter IDs inteiros 0..K-1, um por amostra")
    if (isinstance(min_norm, (bool, np.bool_)) or not isinstance(min_norm, Real)
            or not np.isfinite(min_norm) or min_norm < 0):
        raise ValueError("min_norm deve ser finito e >= 0, nas unidades rhos")
    if len(np.unique(labels)) != len(centers):
        raise ValueError("Todos os centros devem possuir amostras em train_labels")
    counts = np.bincount(labels, minlength=len(centers))
    means = np.array([values[labels == group].mean(axis=0) for group in range(len(centers))])
    norms = np.linalg.norm(means, axis=1)
    scatter = np.array([np.mean(np.sum((values[labels == group] - means[group])**2, axis=1))
                        for group in range(len(centers))])
    rows = []
    for a, b in combinations(range(len(centers)), 2):
        sam = None
        status = "norm_below_or_equal_minimum"
        if norms[a] > min_norm and norms[b] > min_norm:
            # Divide vetores antes do produto para reduzir overflow/underflow.
            cosine = np.dot(means[a] / norms[a], means[b] / norms[b])
            sam = float(np.arccos(np.clip(cosine, -1.0, 1.0)))
            status = "ok"
        rows.append({
            "group_a": a + 1, "group_b": b + 1,
            "standardized_euclidean": float(np.linalg.norm(centers[a] - centers[b])),
            "spectral_angle_rad": sam, "spectral_angle_degrees": None if sam is None else float(np.degrees(sam)),
            "spectral_angle_status": status, "min_norm": float(min_norm),
            "mean_norm_a": float(norms[a]), "mean_norm_b": float(norms[b]),
            "n_a": int(counts[a]), "n_b": int(counts[b]),
            "within_scatter_a": float(scatter[a]), "within_scatter_b": float(scatter[b]),
        })
    result = pd.DataFrame(rows)
    grand_mean = values.mean(axis=0)
    total = np.sum((values - grand_mean)**2, axis=0)
    between = np.sum(counts[:, None] * (means - grand_mean)**2, axis=0)
    eta = np.full(values.shape[1], np.nan)
    np.divide(between, total, out=eta, where=total > 0)
    result.attrs["eta_squared"] = pd.DataFrame({
        "feature_index": np.arange(values.shape[1]), "eta_squared": eta,
        "total_sum_squares": total, "between_sum_squares": between,
        "status": np.where(total > 0, "ok", "constant_feature"),
    })
    result.attrs["interpretation"] = "Separação estatística; sem identidade ecológica validada"
    return result
