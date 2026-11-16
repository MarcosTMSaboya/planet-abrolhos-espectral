"""Consulta dos pontos candidatos à validação, sem atribuir referências."""
import json
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium


@st.cache_data(show_spinner=False,max_entries=8)
def read_points(path,signature):
    return pd.read_csv(path,keep_default_na=False)


def render(root):
    st.header('Pontos de verificação')
    st.caption('Pontos sorteados para buscar evidência independente. A classe prevista não é uma referência confirmada.')
    runs=[]
    for path in sorted((Path(root)/'outputs/validacao').glob('*/manifesto.json'),reverse=True):
        try:
            meta=json.loads(path.read_text(encoding='utf-8'))
            if meta.get('status')=='complete':runs.append((path.parent,meta))
        except (OSError,ValueError):continue
    if not runs:
        st.info('Os pontos estão sendo preparados. Atualize a página após o processamento.');return
    selected=st.selectbox('Conjunto de pontos',range(len(runs)),format_func=lambda i:runs[i][0].name,key='verification_run')
    run,meta=runs[selected]
    points=read_points(str(run/'pontos_sorteados.csv'),(run/'pontos_sorteados.csv').stat().st_mtime_ns)
    cols=st.columns(3)
    cols[0].metric('Pontos sorteados',meta['n_sampled'])
    cols[1].metric('Referências confirmadas',meta['n_reference_confirmed'])
    cols[2].metric('Pontos de campo na cena',meta['n_field_in_scene'])
    st.info('Não é preciso adivinhar a classe no RGB. Procure fotos, vídeos ou registros de campo com posição e data verificáveis; sem isso, mantenha “sem_referencia”.')
    groups=st.multiselect('Grupos previstos',[1,2,3],default=[1,2,3],key='verification_groups')
    strata=st.multiselect('Posição na classificação',['interior','transicao'],default=['interior','transicao'],key='verification_strata')
    selected_points=points.loc[points.predicted_group.isin(groups)&points.stratum.isin(strata)]
    colors={1:'#e79f00',2:'#00a8aa',3:'#8f50be'}
    m=folium.Map(location=[points.lat.mean(),points.lon.mean()],zoom_start=11,tiles='OpenStreetMap',control_scale=True)
    for point in selected_points.itertuples():
        popup=f'{point.point_id}<br>Grupo previsto: {point.predicted_group}<br>{point.stratum}<br>Referência: sem referência'
        folium.CircleMarker([point.lat,point.lon],radius=5,color=colors[point.predicted_group],fill=True,fill_opacity=.8,
            tooltip=point.point_id,popup=popup).add_to(m)
    if len(selected_points):m.fit_bounds([[selected_points.lat.min(),selected_points.lon.min()],[selected_points.lat.max(),selected_points.lon.max()]])
    st_folium(m,height=550,width=1100,key='verification_map_'+run.name,returned_objects=[])
    st.dataframe(selected_points[['point_id','predicted_group','predicted_label_provisional','stratum','reference_status','nearest_field_id','nearest_field_distance_m','field_within_30m']],hide_index=True)
    st.caption('O raio de 30 m é apenas uma busca operacional. Proximidade com Garmin e datas GPX não confirmam correspondência, classe ou data de observação de campo.')
    for filename,label,mime in (
        ('pontos_verificacao.gpkg','Baixar pontos e inventário para QGIS','application/geopackage+sqlite3'),
        ('pontos_sorteados.csv','Baixar tabela para completar referências','text/csv'),
        ('LEIA_ME.md','Baixar instruções do desenho de verificação','text/markdown'),
    ):
        artifact=run/filename
        st.download_button(label,artifact.read_bytes(),file_name=filename,mime=mime,key='verification_'+filename)
    st.code(str(run/'pontos_verificacao.gpkg'),language=None)
    with st.expander('Desenho e limitações'):
        st.markdown((run/'LEIA_ME.md').read_text(encoding='utf-8'))
