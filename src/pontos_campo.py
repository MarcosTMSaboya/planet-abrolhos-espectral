"""Inventário de campo nas cenas atuais; classes exigem evidência externa."""
from pathlib import Path
from html import escape
import json
import folium
import pandas as pd
import streamlit as st


def load_points(root):
    folder = Path(root) / 'data' / 'campo' / 'piloto_gpx'
    points = pd.read_csv(folder / 'pontos_cobertos.csv', keep_default_na=False)
    observations = pd.read_csv(folder / 'pontos_por_cena.csv', keep_default_na=False)
    return points, observations


def add_field_points(map_object, root, scene_ids):
    """Inclui todos os pontos cobertos pelas cenas exibidas, inclusive rejeitados."""
    try:
        points, obs = load_points(root)
    except FileNotFoundError:
        return 0
    obs = obs[obs.cena_id.isin(scene_ids)]
    points = points[points.ponto_id.isin(obs.ponto_id)]
    layer = folium.FeatureGroup(name='Pontos de campo — GPX', show=True)
    for _, point in points.iterrows():
        matches = obs[obs.ponto_id == point.ponto_id]
        approved = matches.qualidade.eq('clear_udm2').any()
        color = '#008080' if approved else '#c47810'
        popup = '<br>'.join([
            '<b>' + escape(str(point['name'])) + '</b>',
            escape(str(point.ponto_id)),
            'Classe: pendente de evidência de campo',
            'Qualidade nas cenas exibidas: ' + escape(', '.join(sorted(set(matches.qualidade)))),
            'Data GPX (a confirmar): ' + escape(str(point.time)),
        ])
        folium.CircleMarker(
            [point.lat, point.lon], radius=5, color=color, fill=True,
            fill_opacity=.85, tooltip=str(point['name']),
            popup=folium.Popup(popup, max_width=420),
        ).add_to(layer)
    layer.add_to(map_object)
    folium.LayerControl(collapsed=False).add_to(map_object)
    return len(points)


def render_field_points(root):
    st.subheader('Pontos de campo — piloto')
    st.caption('Inventário do GPX nas cenas atuais. As classes permanecem pendentes até associação com fichas, fotos ou vídeos.')
    try:
        points, obs = load_points(root)
    except FileNotFoundError:
        st.info('Execute a preparação do inventário de campo para carregar os pontos.')
        return
    c1, c2, c3 = st.columns(3)
    c1.metric('Pontos cobertos', len(points))
    c2.metric('Com pixel válido em alguma cena', obs.loc[obs.pixel_valido.eq(True), 'ponto_id'].nunique())
    c3.metric('Clear UDM2 em alguma cena', obs.loc[obs.qualidade.eq('clear_udm2'), 'ponto_id'].nunique())
    manifest = json.loads((Path(root) / 'data' / 'campo' / 'piloto_gpx' / 'manifesto.json').read_text(encoding='utf-8'))
    scenes = ['Todas as cenas'] + manifest['cenas_disponiveis']
    scene = st.selectbox('Cobertura da cena', scenes, key='campo_cena')
    current = obs if scene == scenes[0] else obs[obs.cena_id.eq(scene)]
    quality = st.selectbox('Qualidade do pixel central', ['Todos os pontos cobertos', 'Clear UDM2', 'Pixel válido', 'Revisar qualidade'], key='campo_qualidade')
    if quality == 'Clear UDM2':
        current = current[current.qualidade.eq('clear_udm2')]
    elif quality == 'Pixel válido':
        current = current[current.pixel_valido.eq(True)]
    elif quality == 'Revisar qualidade':
        current = current[~current.qualidade.eq('clear_udm2')]
    selected = points[points.ponto_id.isin(current.ponto_id)]
    st.write(f'{len(selected)} pontos distintos no filtro; {len(current)} relações ponto/cena.')
    if len(selected):
        st.map(selected, latitude='lat', longitude='lon', alt='Pontos do GPX cobertos pelas cenas filtradas')
    st.dataframe(selected, hide_index=True, width='stretch', alt='Atributos originais dos pontos e campos pendentes de validação')
    st.dataframe(current, hide_index=True, width='stretch', alt='Cobertura e qualidade de cada relação ponto e cena')
    st.download_button('Baixar pontos filtrados', selected.to_csv(index=False).encode('utf-8-sig'), 'pontos_campo_filtrados.csv', 'text/csv')
    st.download_button('Baixar relações ponto/cena', current.to_csv(index=False).encode('utf-8-sig'), 'pontos_por_cena.csv', 'text/csv')
    st.info('Próximo passo: preencher a planilha de revisão com classe observada, evidência, data real, profundidade e grupo de estação. Clear UDM2 não confirma visibilidade do fundo. O piloto supervisionado depende dessas referências.')
    folder = Path(root) / 'data' / 'campo' / 'piloto_gpx'
    st.download_button('Baixar planilha de revisão de campo', (folder / 'revisao_classes.csv').read_bytes(), 'revisao_classes.csv', 'text/csv')
