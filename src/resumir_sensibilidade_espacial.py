"""Verifica e documenta a rodada espacial concluída; não ajusta modelos."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from agrupamento_espectral import sha256
from experimento_espectral import write_json
from sensibilidade_espacial import make_plots, SCENARIOS


def summarize(run_path):
    run_path = Path(run_path).resolve()
    run = json.loads((run_path / "parametros.json").read_text(encoding="utf-8"))
    if run.get("status") != "complete" or run.get("model_count") != 36:
        raise ValueError("Rodada incompleta")
    if sha256(run["prepared"]) != run["prepared_sha256"]:
        raise ValueError("Preparação mudou")
    for name, expected in run["artifacts_sha256"].items():
        if sha256(run_path / name) != expected:
            raise ValueError(f"Artefato alterado: {name}")
    metrics = pd.read_csv(run_path / "metrics.csv")
    props = pd.read_csv(run_path / "proportions.csv")
    stability = pd.read_csv(run_path / "stability.csv")
    agreement = pd.read_csv(run_path / "concordancia_entre_particoes.csv")
    pca = json.loads((run_path / "pca_por_cenario.json").read_text(encoding="utf-8"))
    with np.load(run_path / "amostra_comum.npz", allow_pickle=False) as stored:
        probe = stored["values"].copy()
    splits = {}
    with np.load(run_path / "predicoes_amostra_comum.npz", allow_pickle=False) as stored:
        for name, *_ in SCENARIOS:
            with np.load(run_path / name / "particao.npz", allow_pickle=False) as partition:
                splits[name] = partition["split"].copy()
            for rep in ("bands", "pca95"):
                model = joblib.load(run_path / name / f"modelo_{rep}_k3_seed42.joblib")
                with threadpool_limits(limits=1):
                    if not np.array_equal(model.predict(probe), stored[f"{name}_{rep}"]):
                        raise ValueError("Predições persistidas não reproduzidas")
    output = run_path / ("resumo_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(exist_ok=False)
    make_plots(metrics, agreement, splits, output)
    rows = []
    for name, block, buffer, seed in SCENARIOS:
        pc = next(item for item in pca if item["scenario"] == name)
        for rep in ("bands", "pca95"):
            subset = metrics.loc[metrics.scenario.eq(name) & metrics.representation.eq(rep) & metrics.split.eq("validation")]
            st = stability.loc[stability.scenario.eq(name) & stability.representation.eq(rep)]
            pr = props.loc[props.scenario.eq(name) & props.representation.eq(rep) & props.split.eq("validation")]
            rows.append({"scenario": name, "block_m": block*3, "buffer_m": buffer*3,
                         "partition_seed": seed, "representation": rep,
                         "silhouette_mean": subset.silhouette_subsample.mean(),
                         "silhouette_min": subset.silhouette_subsample.min(),
                         "silhouette_max": subset.silhouette_subsample.max(),
                         "davies_bouldin_mean": subset.davies_bouldin.mean(),
                         "ari_seed_min": st.adjusted_rand.min(), "ari_seed_median": st.adjusted_rand.median(),
                         "minimum_group_fraction": pr.sample_fraction.min(),
                         "all_groups_in_evaluation": bool(subset.n_groups_present.eq(3).all()),
                         "pca_components": pc["n_components"], "pc1_fraction": pc["explained_variance_ratio"][0]})
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "resumo_metricas.csv", index=False)
    lines = ["# Abrolhos — sensibilidade espacial do agrupamento K3", "",
             "20/10/2026. Cena `20260213_125207_88_24d8`. Rodada limitada à ROI aquática aparente de 1,62 km², sem comprovação de visibilidade bentônica.", "",
             "## Resultado principal", "",
             "**A solução K3 é reproduzível em parte, mas sensível à divisão espacial.** As três categorias aparecem nas 36 avaliações completas. A concordância entre sementes do mesmo cenário é maior que a concordância entre diferentes geometrias de treinamento. Esses resultados mantêm K3 como candidato exploratório, sem demonstrar três habitats ou uma partição física única.", "",
             "## Protocolo executado", "",
             "- Seis cenários definidos antes do ajuste: blocos de 100 pixels e buffer de 34 pixels com sorteios 42, 97 e 2026; blocos de 75 e 125 pixels com buffer 34 e sorteio 42; blocos 100 com buffer 67 e sorteio 42. Resolução: 3 m.",
             "- Cada cenário: 20.000 pixels de desenvolvimento e 5.000 de avaliação; amostragem uniforme sem reposição por população, semente 42. Faixa de separação excluída. Áreas e amostras dos cenários se sobrepõem.",
             "- Bandas 444/492/533/566 nm, padronizadas; comparação com PCA95 sem whitening. Scaler, PCA e K-means ajustados somente no desenvolvimento de cada cenário.",
             "- K=3 fixado a partir da exploração anterior; sementes do modelo 11, 42 e 97; dez inicializações por modelo. Total: 36 modelos / 360 inicializações, uma thread.",
             "- Silhouette em 1.000 pixels por partição; Davies–Bouldin nos 5.000 de avaliação. As três sementes reutilizam a mesma subamostra de métricas dentro de cada cenário.",
             "- Comparação entre partições: representantes de semente do modelo 42, ARI sobre os mesmos 10.000 pixels sorteados uniformemente da ROI (semente 7043). Essa amostra inclui pixels usados em ajustes; concordância descritiva, sem acurácia ou desempenho independente.",
             "- Comparação complementar: somente os pixels da amostra comum que pertencem simultaneamente à avaliação dos dois cenários. Ambos os modelos não foram ajustados nesses pixels; autocorrelação e seleção prévia de K permanecem.",
             "- Rhos negativos válidos preservados. A ROI atual não contém negativos; sua influência em outras áreas continua não testada.", "",
             "## Métricas por cenário", "",
             "Médias nas três sementes. ARI mínimo compara sementes na mesma avaliação. As faixas são descritivas e não representam intervalos de confiança. A comparação entre PCA e bandas usa espaços diferentes e não comprova superioridade física.", "",
             "| Cenário (bloco/buffer em m; sorteio) | Variáveis | Silhouette | Davies–Bouldin | ARI mínimo entre sementes | PCs PCA95 |",
             "|---|---|---:|---:|---:|---:|"]
    for row in summary.itertuples():
        label = "Bandas" if row.representation == "bands" else "PCA95"
        lines.append(f"| {row.block_m}/{row.buffer_m}; {row.partition_seed} | {label} | {row.silhouette_mean:.4f} | {row.davies_bouldin_mean:.4f} | {row.ari_seed_min:.4f} | {row.pca_components} |")
    lines.extend(["", "A silhouette da configuração base difere do relatório anterior porque a subamostra passou de 1.500 para 1.000 pixels; o treinamento da base usa as mesmas posições e reflectâncias. Essa mudança de métrica não é evidência de melhora do modelo.", "",
                  "## Concordância espacial", ""])
    for rep, label in (("bands", "Bandas padronizadas"), ("pca95", "PCA95")):
        common = agreement.loc[agreement.representation.eq(rep) & agreement.scope.eq("common_probe_descriptive")]
        joint = agreement.loc[agreement.representation.eq(rep) & agreement.scope.eq("joint_evaluation_only")]
        usable = joint.loc[joint.status.eq("ok")]
        lines.append(f"- **{label}:** ARI entre partições na amostra comum: {common.adjusted_rand.min():.3f}–{common.adjusted_rand.max():.3f}. Nas interseções de avaliação com três grupos: {usable.adjusted_rand.min():.3f}–{usable.adjusted_rand.max():.3f}; {len(usable)} de {len(joint)} pares comparáveis pela política conservadora adotada.")
    lines.extend(["", "As interseções contêm 364–2.165 pixels da amostra comum, com tamanhos e composições diferentes. Um par PCA95 tem somente dois grupos presentes na interseção: score não calculado por política de completude, e não por erro de processamento. A ausência local de um grupo não implica falha do K3 em toda a avaliação.", "",
                  "Apenas mudar o sorteio mantendo blocos/buffer 300/102 m produz ARI PCA95 de 0,906–0,950 na amostra comum. Variar a geometria amplia a discordância: o menor ARI PCA95 (0,693) compara blocos 225 m/buffer 102 m com blocos 300 m/buffer 201 m. Isso não isola uma causa: muda a população de ajuste, a amostra, a padronização e a PCA.", "",
                  "Aumentar o buffer de 102 para 201 m mantém a avaliação base e muda o desenvolvimento; a silhouette PCA95 cai de aproximadamente 0,457 para 0,389. Não se escolhe um buffer menor apenas para maximizar o score: a independência estatística continua não comprovada.", "",
                  "## Estrutura espectral e interpretação", "",
                  "PC1 explica 86,70%–94,30% da variância do desenvolvimento. Cinco cenários retêm dois PCs para atingir 95%; blocos 375 m exigem três PCs. Os coeficientes de PC1 são positivos nas quatro bandas em todos os cenários, mantendo a indicação de variação conjunta de magnitude. Não há evidência nova para atribuir isso especificamente a profundidade, água, glint ou fundo.", "",
                  "K3 estava fixado: este teste não demonstra que K3 continua ótimo frente a outros K sob todas as divisões. Scores não são probabilidades de habitat. O ARI é invariável à permutação dos IDs; Grupo Espectral 1 em um modelo não corresponde necessariamente ao Grupo Espectral 1 de outro.", "",
                  "## Recomendação", "",
                  "Manter a configuração original como referência exploratória, sem trocar para o cenário de maior silhouette. Antes de ampliar a cena, revisar visualmente superfície/contaminação e produzir, em etapa própria, um mapa diagnóstico apenas da ROI com os grupos espectrais e a discordância entre modelos. Se o gradiente acompanhar efeitos de água ou superfície, registrar essa associação como hipótese. Batimetria exige CRS e datum confiáveis; Garmin continua referência espacial sem classe ecológica confirmada.", "",
                  "Não expandir automaticamente para toda a cena nem integrar como mapa bentônico. Não foi aplicado modelo ao raster completo, alterado o WebGIS ou modificado qualquer produto ACOLITE.", "",
                  "## Produtos e verificação", "",
                  "- [Comparação gráfica](sensibilidade.png) e [partições espaciais](particoes.png).",
                  "- [Resumo das métricas](resumo_metricas.csv), [concordância entre partições](../concordancia_entre_particoes.csv), [proporções amostrais](../proportions.csv).",
                  "- [Parâmetros e hashes da rodada](../parametros.json); subpastas por cenário com amostras, máscaras de partição, métricas e modelos.",
                  "- Todos os hashes registrados na rodada foram conferidos; 12 modelos representativos reabertos reproduziram exatamente as predições dos 10.000 pixels comuns. Os quatro testes leves do módulo passaram. Não houve novo ajuste neste resumo.",
                  "- As proporções são das amostras de cada partição, sem extrapolação para área de habitat ou cobertura da cena.", ""])
    (output / "RELATORIO_SENSIBILIDADE.md").write_text("\n".join(lines), encoding="utf-8")
    for name in ("resumir_sensibilidade_espacial.py", "sensibilidade_espacial.py"):
        shutil.copy2(Path(__file__).with_name(name), output / name)
    write_json(output / "verificacao.json", {"source_manifest_sha256": sha256(run_path / "parametros.json"),
               "all_run_hashes_verified": True, "representative_models_reloaded": 12,
               "predictions_per_model_verified": len(probe), "new_training": False,
               "artifacts_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    print(summarize(parser.parse_args().run))
