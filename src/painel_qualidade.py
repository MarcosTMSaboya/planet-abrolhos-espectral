"""Interface das análises já produzidas; não acessa rasters."""
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

WAVELENGTHS = (444, 492, 533, 566, 612, 666, 707, 866)
STATUS = {"accepted": "Aceita", "rejected": "Rejeitada", "indeterminate": "Indeterminada"}


@st.cache_data(show_spinner=False)
def read_csv(path, signature):
    return pd.read_csv(path, dtype={"scene_id": str, "scene_key": str})


def load_table(path):
    stat = path.stat()
    return read_csv(str(path), (stat.st_mtime_ns, stat.st_size))


def download(df, label, filename, key):
    st.download_button(label, df.to_csv(index=False).encode("utf-8-sig"),
                       file_name=filename, mime="text/csv", key=key)


def render_quality(root):
    st.subheader("Qualidade UDM2 e assinaturas filtradas")
    st.caption("Resultados das janelas 3×3 já processadas. As imagens RGB e o mapa GIS continuam sendo visualizações das cenas originais.")
    base = Path(root) / "data/analise/udm2"
    runs = sorted((p for p in base.iterdir() if p.is_dir() and (p / "manifesto.json").is_file()), reverse=True) if base.exists() else []
    if not runs:
        st.info("Ainda não há resultados UDM2. Execute src/controle_qualidade_udm2.py para gerar a auditoria.")
        return
    run = st.selectbox("Execução da análise (UTC)", runs, format_func=lambda p: p.name, key="qc_run")
    labels = {"strict": "Estrita — apenas Clear", "allow_haze": "Sensibilidade — permite haze"}
    policy = st.radio("Regra de qualidade", list(labels), format_func=labels.get, horizontal=True, key="qc_policy")
    try:
        manifest = json.loads((run / "manifesto.json").read_text(encoding="utf-8"))
        audit = load_table(run / "auditoria_udm2.csv")
        points = load_table(run / f"assinatura_algas_por_ponto_{policy}.csv")
        summary = load_table(run / f"resumo_espectral_algas_{policy}.csv")
        long = load_table(run / f"assinatura_algas_long_{policy}.csv")
        status_col, reason_col = f"qc_{policy}", f"qc_{policy}_reason"
        required = {"scene_id", "point_index", status_col, reason_col}
        required.update(f"rhos_{w}_median" for w in WAVELENGTHS)
        if not required.issubset(audit.columns) or not audit[status_col].isin(STATUS).all():
            raise ValueError("Auditoria incompatível ou com decisões de qualidade desconhecidas")
        schemas = (
            (points, {"point_index", "n_scenes"}, "assinaturas por ponto"),
            (summary, {"wavelength_nm", "median", "q1", "q3"}, "resumo espectral"),
            (long, {"point_index", "wavelength_nm", "rhos_median", "n_scenes"}, "assinaturas em formato longo"),
        )
        if not isinstance(manifest, dict):
            raise ValueError("Manifesto deve conter um objeto JSON")
        for table, columns, name in schemas:
            if missing := columns - set(table.columns):
                raise ValueError(f"Colunas ausentes em {name}: {', '.join(sorted(missing))}")
        for column in ("n_scenes", "point_index"):
            if points[column].isna().any():
                raise ValueError(f"Valores ausentes nas assinaturas: {column}")
        points["n_scenes"] = pd.to_numeric(points["n_scenes"], errors="raise")
        if not np.isfinite(points.n_scenes).all() or (points.n_scenes < 0).any():
            raise ValueError("Número de cenas inválido nas assinaturas")
        if points.point_index.duplicated().any():
            raise ValueError("Pontos duplicados nas assinaturas")
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Não foi possível carregar esta execução: {error}")
        return
    if audit.empty or points.empty:
        st.info("Esta execução não contém observações ou pontos para exibir.")
        return
    counts = audit[status_col].value_counts()
    for col, label, value in zip(st.columns(5),
                                ("Observações", "Aceitas", "Rejeitadas", "Indeterminadas", "Pontos com cenas aceitas"),
                                (len(audit), counts.get("accepted", 0), counts.get("rejected", 0),
                                 counts.get("indeterminate", 0), int((points.n_scenes > 0).sum()))):
        col.metric(label, int(value))
    confidence = manifest.get("min_confidence", 0)
    st.info(f"Regra aplicada aos nove pixels: {'Clear' if policy == 'strict' else 'Clear ou Haze'}, banda 8 zerada e oito bandas rhos completas. "
            + (f"Confiança mínima: {confidence}." if confidence else "Sem corte adicional de confiança."))
    if policy == "allow_haze":
        st.warning("No UDM2.1, a banda 4 representa haze em geral. Esta opção é uma análise de sensibilidade, sem garantia de interferência leve.")
    tabs = st.tabs(["Assinaturas", "Auditoria", "Por cena e ponto", "Relatório e arquivos"])
    with tabs[0]:
        st.caption("Resumo global calculado entre pontos, após a mediana das cenas aceitas de cada ponto. Os filtros da aba Auditoria não alteram este resumo.")
        valid = summary.dropna(subset=["median"])
        if valid.empty:
            st.warning("Nenhuma assinatura válida nesta regra.")
        else:
            # A comparação sem filtro usa a mesma agregação, com valores finitos.
            before = []
            for w in WAVELENGTHS:
                values = pd.to_numeric(audit[f"rhos_{w}_median"], errors="coerce")
                values = values.where(np.isfinite(values) & (values.abs() < 1e20))
                medians = values.groupby(audit.point_index).median()
                before.append({"wavelength_nm": w, "median": medians.median(), "Série": "Antes do filtro"})
            curves = pd.concat([pd.DataFrame(before), valid[["wavelength_nm", "median"]].assign(**{"Série": "Após UDM2"})], ignore_index=True)
            band = alt.Chart(valid).mark_area(opacity=.18, color="#167d9a").encode(
                x=alt.X("wavelength_nm:Q", title="Comprimento de onda (nm)", scale=alt.Scale(zero=False)),
                y=alt.Y("q1:Q", title="Reflectância ACOLITE rhos", scale=alt.Scale(zero=False)), y2="q3:Q")
            lines = alt.Chart(curves).mark_line(point=True).encode(
                x="wavelength_nm:Q", y="median:Q", color="Série:N",
                tooltip=["Série:N", "wavelength_nm:Q", alt.Tooltip("median:Q", format=".6f")])
            st.altair_chart(band + lines, width="stretch", alt="Assinatura mediana antes e após UDM2, com intervalo interquartil após o filtro")
            st.caption("Faixa sombreada: Q1–Q3 entre pontos após o filtro. As curvas podem se sobrepor quando as medianas não mudam.")
        point = st.selectbox("Ponto para inspeção", points.point_index.tolist(), key="qc_point")
        point_row = points.loc[points.point_index == point].iloc[0]
        st.metric("Cenas aceitas neste ponto", int(point_row.n_scenes))
        if point_row.n_scenes < 2:
            st.caption("Há menos de duas cenas aceitas: não é possível avaliar a variabilidade entre cenas deste ponto.")
        observations = audit.loc[audit.point_index == point]
        accepted = observations.loc[observations[status_col] == "accepted"]
        scene_curves = []
        for _, row in accepted.iterrows():
            scene_curves.extend({"wavelength_nm": w, "rhos": row[f"rhos_{w}_median"], "Cena": row.scene_id} for w in WAVELENGTHS)
        if scene_curves:
            st.line_chart(pd.DataFrame(scene_curves), x="wavelength_nm", y="rhos", color="Cena", x_label="Comprimento de onda (nm)", y_label="Mediana rhos 3×3", alt="Assinaturas das cenas aceitas para o ponto selecionado")
        else:
            st.info("Este ponto não tem observações aceitas nesta regra.")
        st.dataframe(observations[["scene_id", status_col, reason_col]], hide_index=True, width="stretch", alt="Decisões de qualidade das cenas do ponto selecionado")
        st.dataframe(summary, hide_index=True, width="stretch", alt="Resumo espectral entre pontos após filtro UDM2")
        download(points, "Baixar assinaturas por ponto", f"assinatura_algas_por_ponto_{policy}.csv", "qc_download_points")
        download(summary, "Baixar resumo espectral", f"resumo_espectral_algas_{policy}.csv", "qc_download_summary")
        download(long, "Baixar assinaturas em formato longo", f"assinatura_algas_long_{policy}.csv", "qc_download_long")
    with tabs[1]:
        c1, c2, c3 = st.columns(3)
        scenes = c1.multiselect("Cenas", sorted(audit.scene_id.unique()), key="qc_scenes")
        selected_points = c2.multiselect("Pontos", sorted(audit.point_index.unique()), key="qc_audit_points")
        statuses = c3.multiselect("Decisões", list(STATUS), format_func=STATUS.get, key="qc_statuses")
        subset = audit.copy()
        if scenes:
            subset = subset.loc[subset.scene_id.isin(scenes)]
        if selected_points:
            subset = subset.loc[subset.point_index.isin(selected_points)]
        if statuses:
            subset = subset.loc[subset[status_col].isin(statuses)]
        st.caption(f"{len(subset)} de {len(audit)} observações. Seleções vazias incluem todos os valores.")
        leading = ["point_index", "scene_id", status_col, reason_col]
        leading += [c for c in ("udm2_cloud_fraction", "udm2_haze_fraction", "udm2_shadow_fraction", "udm2_blackfill_fraction", "udm2_confidence_min") if c in subset]
        st.dataframe(subset[leading], hide_index=True, width="stretch", alt="Auditoria das observações selecionadas")
        with st.expander("Todas as colunas e valores espectrais"):
            st.dataframe(subset, hide_index=True, width="stretch", alt="Todas as colunas da auditoria selecionada")
        download(subset, "Baixar auditoria selecionada", f"auditoria_selecionada_{policy}.csv", "qc_download_audit")
        download(subset.loc[subset[status_col] == "accepted"], "Baixar observações aceitas da seleção", f"observacoes_aceitas_selecao_{policy}.csv", "qc_download_accepted")
        with st.expander("Como interpretar os motivos"):
            st.markdown("`cloud`: nuvem; `haze`: haze; `shadow`: sombra; `udm1_bits_set`: pelo menos um bit da banda 8 acionado; "
                        "`not_all_pixels_in_allowed_classes`: nem todos os nove pixels pertencem às classes aceitas; "
                        "`incomplete_or_invalid_rhos`: extração incompleta ou mediana inválida. Frações variam de 0 a 1. "
                        "Indeterminada indica ausência de informação suficiente, erro de leitura ou incompatibilidade de grade.")
    with tabs[2]:
        for unit, title in (("scene_id", "Qualidade por cena"), ("point_index", "Qualidade por ponto")):
            st.markdown(f"#### {title}")
            table = (audit.groupby([unit, status_col]).size().unstack(fill_value=0)
                     .reindex(columns=list(STATUS), fill_value=0))
            table["total"] = table.sum(axis=1)
            st.dataframe(table.rename(columns=STATUS).reset_index(), hide_index=True, width="stretch", alt=title)
        st.markdown("#### Assinaturas e número de cenas por banda")
        st.dataframe(points, hide_index=True, width="stretch", alt="Assinaturas por ponto e número de cenas aceitas por banda")
    with tabs[3]:
        st.caption(f"Resultados carregados de: {run}")
        report = run / "RELATORIO.md"
        if report.is_file():
            st.markdown(report.read_text(encoding="utf-8"))
            st.caption("O relatório documenta a execução original; a seleção de regra acima não altera seu texto.")
        else:
            st.info("Esta execução não possui relatório em Markdown. As tabelas e o manifesto estão disponíveis.")
        with st.expander("Parâmetros e rastreabilidade da execução"):
            st.json(manifest)
        st.caption("UDM2 filtra qualidade atmosférica e dados inutilizáveis. As assinaturas rhos ainda incluem efeitos da coluna d’água e da profundidade.")
