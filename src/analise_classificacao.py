"""Visualização dos relatórios de inspeção dos grupos espectrais."""

from pathlib import Path

import streamlit as st


def render_classification_analysis(root: Path) -> None:
    st.header("Análise da classificação")
    st.caption("Grupos espectrais exploratórios. Identidade ecológica e visibilidade do fundo não confirmadas.")
    reports_root = root / "outputs" / "agrupamento_piloto"
    reports = sorted(reports_root.glob("*/INSPECAO_ROI.html"), reverse=True)
    if not reports:
        st.info("Nenhum relatório de inspeção da ROI foi encontrado.")
        return

    report = st.selectbox(
        "Relatório de análise",
        reports,
        format_func=lambda path: path.parent.name,
        key="classification_report",
    )
    st.caption(
        "Compare a imagem RGB e os grupos espectrais, ajuste a opacidade "
        "e ative ou desative cada grupo nos controles abaixo."
    )
    html = report.read_text(encoding="utf-8")
    # Os links relativos do relatório offline não resolvem dentro do iframe.
    # Os mesmos arquivos ficam disponíveis nos botões de download abaixo.
    for filename, label in (
        ("comparacao_roi.png", "Comparação lado a lado"),
        ("grupos_espectrais_roi.tif", "GeoTIFF dos grupos para QGIS"),
        ("LEIA_ME.md", "Guia e limites"),
    ):
        html = html.replace(f'<a href="{filename}">{label}</a>', label)
    st.iframe(html, height=1000)

    st.subheader("Arquivos da análise")
    for filename, label, mime in (
        ("INSPECAO_ROI.html", "Baixar relatório HTML", "text/html"),
        ("comparacao_roi.png", "Baixar comparação RGB e grupos", "image/png"),
        ("grupos_espectrais_roi.tif", "Baixar grupos para QGIS", "image/tiff"),
        ("contagem_roi.csv", "Baixar contagem e áreas da ROI", "text/csv"),
        ("LEIA_ME.md", "Baixar guia e limites", "text/markdown"),
    ):
        artifact = report.parent / filename
        if artifact.is_file():
            st.download_button(
                label, artifact.read_bytes(), file_name=filename, mime=mime,
                key=f"classification_download_{filename}",
            )
    guide = report.parent / "LEIA_ME.md"
    if guide.is_file():
        with st.expander("Metodologia e limites da análise"):
            st.markdown(guide.read_text(encoding="utf-8"))

    followups = sorted(reports_root.glob("continuidade_*/RELATORIO.md"), reverse=True)
    if followups:
        st.subheader("Comparação K2/K3 e hipótese de gradiente")
        selected = st.selectbox("Execução complementar", followups,
                               format_func=lambda path: path.parent.name,
                               key="continuity_report")
        st.markdown(selected.read_text(encoding="utf-8"))
        figure = selected.parent / "diagnostico_roi.png"
        if figure.is_file():
            st.image(str(figure), caption="Diagnóstico espectral; sem confirmação de causa física.")
        metrics = selected.parent / "metricas_espaco_comum.csv"
        if metrics.is_file():
            st.download_button("Baixar métricas K2/K3", metrics.read_bytes(),
                               file_name=metrics.name, mime="text/csv", key="continuity_metrics")

    reviews = sorted(reports_root.glob("revisao_espacial_k2_*/RELATORIO.md"), reverse=True)
    if reviews:
        st.subheader("Revisão espacial e prioridades de inspeção")
        selected_review = st.selectbox("Rodada espacial", reviews,
            format_func=lambda path: path.parent.name, key="spatial_review_report")
        st.caption("Quadrículas pendentes de revisão. Discordância entre modelos não determina exclusão de pixels.")
        st.markdown(selected_review.read_text(encoding="utf-8"))
        figure = selected_review.parent / "revisao_espacial.png"
        if figure.is_file():
            st.image(str(figure), caption="K2/K3 e discordância K2 entre geometrias; análise exploratória.")
        for filename, label, mime in (
            ("regioes_para_revisao.csv", "Baixar lista de regiões para revisar", "text/csv"),
            ("regioes_para_revisao.geojson", "Baixar quadrículas para QGIS", "application/geo+json"),
            ("discordancia_k2.tif", "Baixar raster de discordância K2", "image/tiff"),
        ):
            artifact = selected_review.parent / filename
            if artifact.is_file():
                st.download_button(label, artifact.read_bytes(), file_name=filename, mime=mime,
                                   key=f"spatial_review_{filename}")
