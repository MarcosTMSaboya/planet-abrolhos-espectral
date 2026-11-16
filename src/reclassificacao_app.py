"""Mapa de cena completa e revisão manual versionada no Streamlit."""
import html
import json
from pathlib import Path

import folium
from folium.plugins import Draw
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject,transform_bounds
import streamlit as st
from streamlit_folium import st_folium

from .cena_completa import BAND_OPTIONS,export_reclassification


@st.cache_data(show_spinner=False,max_entries=4)
def preview(path,signature,band,sources,correction=None,width=1200):
    """Reprojeta os previews para a grade WGS84 comum dos overlays."""
    with rasterio.open(path) as src:
        height=max(1,round(src.height*width/src.width))
        bounds=transform_bounds(src.crs,'EPSG:4326',*src.bounds,densify_pts=21)
        dst_transform=from_bounds(*bounds,width,height)
        source_transform=src.transform*src.transform.scale(src.width/width,src.height/height)
        raw=src.read(band,out_shape=(height,width),resampling=Resampling.nearest)
        classes=np.zeros((height,width),dtype='uint8')
        reproject(raw,classes,src_transform=source_transform,src_crs=src.crs,dst_transform=dst_transform,dst_crs='EPSG:4326',resampling=Resampling.nearest,src_nodata=0,dst_nodata=0)
        dst_crs=src.crs
    nir=None
    if correction is not None:
        with rasterio.open(sources[7]) as src:
            arr=src.read(1,out_shape=(height,width),resampling=Resampling.average,masked=True).astype('float32').filled(np.nan)
            arr[abs(arr)>1e20]=np.nan
            nir=np.full((height,width),np.nan,dtype='float32')
            reproject(arr,nir,src_transform=source_transform,src_crs=dst_crs,dst_transform=dst_transform,dst_crs='EPSG:4326',resampling=Resampling.bilinear,src_nodata=np.nan,dst_nodata=np.nan)
    rgb=np.zeros((height,width,4),dtype='uint8')
    for channel,index in enumerate((5,3,1)):
        with rasterio.open(sources[index]) as src:
            arr=src.read(1,out_shape=(height,width),resampling=Resampling.average,masked=True).astype('float32').filled(np.nan)
            arr[abs(arr)>1e20]=np.nan
            dest=np.full((height,width),np.nan,dtype='float32')
            reproject(arr,dest,src_transform=source_transform,src_crs=dst_crs,dst_transform=dst_transform,dst_crs='EPSG:4326',resampling=Resampling.bilinear,src_nodata=np.nan,dst_nodata=np.nan)
            vals=dest[(classes>0)&np.isfinite(dest)]
            if len(vals):
                lo,hi=np.percentile(vals,[2,98])
                if correction is not None:dest=dest-correction[0][index]*(nir-correction[1])
                if hi>lo:rgb[:,:,channel]=(np.clip(np.nan_to_num((dest-lo)/(hi-lo)),0,1)*255).astype('uint8')
    rgb[:,:,3]=np.where(classes>0,255,0)
    return classes,rgb,[[bounds[1],bounds[0]],[bounds[3],bounds[2]]]


def render(root):
    root=Path(root)
    st.header('Classificação e revisão da cena completa')
    st.caption('Grupos espectrais provisórios. Nomear uma classe ou desenhar uma correção não constitui validação independente.')
    runs=[];progress=[]
    for path in sorted((root/'outputs/cena_completa').glob('*/manifesto.json'),reverse=True):
        try:
            meta=json.loads(path.read_text(encoding='utf-8'))
            if meta['status']=='complete':runs.append((path.parent,meta))
            else:progress.append(meta)
        except (OSError,ValueError,KeyError):continue
    for meta in progress[:1]:
        if meta['status']=='in_progress':
            st.info(f"Preparando a cena {meta['scene_id']}: {meta.get('progress',0):.0%}. Atualize para acompanhar.")
        elif meta['status']=='failed':st.error('Processamento interrompido: '+meta.get('error','erro não informado'))
    if st.button('Atualizar produtos',key='refresh_full_scene'):st.rerun()
    if not runs:
        st.info('Nenhuma cena completa está pronta. O processamento inicial está sendo preparado.');return
    selected=st.selectbox('Cena processada',range(len(runs)),format_func=lambda i:runs[i][1]['scene_id']+' — '+runs[i][0].name,key='full_scene_run')
    run,meta=runs[selected]
    choice=st.selectbox('Versão e número de grupos',list(BAND_OPTIONS),index=3,key='full_scene_band')
    band=BAND_OPTIONS[choice];k=2 if band in (1,3) else 3
    if band==4:
        st.caption('Versão de trabalho escolhida: Ensaio NIR — K3. Continue a revisão mantendo os três grupos separados; retome uma versão salva para recuperar as exclusões das ilhas.')
    st.warning('A cena inteira usa validade das oito bandas e UDM2 estrita. Não há máscara de água para toda a cena: terra ou outras feições podem aparecer nos grupos.')
    if band>2:
        fraction=meta.get('outside_nir_fit_fraction')
        st.caption('Ensaio NIR transferido da ROI; correção não validada. '+(f'{fraction:.1%} dos pixels elegíveis estão fora da faixa NIR usada no ajuste.' if fraction is not None else ''))
    bucket=f"{run.name}_b{band}"
    previous_versions=[]
    for record_path in sorted(run.glob('reclassificacoes/*/revisao.json'),reverse=True):
        record=json.loads(record_path.read_text(encoding='utf-8'))
        if record.get('status')=='complete' and record.get('band')==band:previous_versions.append((record_path,record))
    if previous_versions:
        with st.expander('Retomar uma revisão salva'):
            previous=st.selectbox('Versão anterior',range(len(previous_versions)),
                format_func=lambda i:previous_versions[i][0].parent.name,key='load_choice_'+bucket)
            if st.button('Carregar esta revisão',key='load_'+bucket):
                record=previous_versions[previous][1]
                for group in range(1,4):st.session_state[f'map_{bucket}_{group}']=int(record['mapping'][str(group)])
                for c in (1,2,3):st.session_state[f'label_{bucket}_{c}']=record['labels'].get(str(c),f'Classe {c}')
                st.session_state['patches_'+bucket]=record['features']
                st.session_state['notes_'+bucket]=record['notes'];st.rerun()
    st.subheader('1. Nomear ou agrupar as classes')
    label_names={};mapping={}
    for group in range(1,4):
        if group<=k:
            cols=st.columns([1,2])
            mapping[group]=cols[0].selectbox(f'Grupo {group} → classe',[0,1,2,3],index=group,key=f'map_{bucket}_{group}',help='0 = excluir/não classificado; o mesmo código une grupos.')
        else:mapping[group]=0
    for c in (1,2,3):label_names[c]=st.text_input(f'Nome provisório da classe {c}',value=f'Classe {c}',key=f'label_{bucket}_{c}')
    opacity=st.slider('Opacidade dos grupos',0.,1.,.45,.05,key='full_opacity')
    background=st.selectbox('Imagem de fundo',['RGB original','RGB do ensaio NIR'],key='full_background')
    correction=None
    if background=='RGB do ensaio NIR':
        from .cena_completa import TRIAL
        coefficients=pd.read_csv(root/'outputs/agrupamento_piloto'/TRIAL/'coeficientes.csv')
        correction=(tuple(coefficients.slope.to_numpy()),float(meta['nir_reference']))
    path=run/'grupos_cena_completa.tif'
    with st.spinner('Carregando o mapa da cena…'):
        classes,rgb,bounds=preview(str(path),(path.stat().st_mtime_ns,path.stat().st_size),band,tuple(s['path'] for s in meta['sources']),correction)
    lut=np.array([0,mapping[1],mapping[2],mapping[3]],dtype='uint8');mapped=lut[classes]
    palette=np.array([[0,0,0,0],[231,159,0,255],[0,168,170,255],[143,80,190,255]],dtype='uint8')
    overlay=palette[mapped]
    center=[(bounds[0][0]+bounds[1][0])/2,(bounds[0][1]+bounds[1][1])/2]
    m=folium.Map(location=center,tiles='OpenStreetMap',control_scale=True)
    folium.raster_layers.ImageOverlay(rgb,bounds,name=background+' — contraste original P2–P98',opacity=1).add_to(m)
    folium.raster_layers.ImageOverlay(overlay,bounds,name='Classes provisórias',opacity=opacity,interactive=True).add_to(m)
    if st.checkbox('Mostrar pixels fora da faixa NIR de ajuste',key='show_nir_support'):
        support_path=run/'suporte_nir.tif'
        support,_,support_bounds=preview(str(support_path),(support_path.stat().st_mtime_ns,support_path.stat().st_size),1,
            tuple(s['path'] for s in meta['sources']))
        rgba=np.zeros((*support.shape,4),dtype='uint8');rgba[support==2]=[255,30,30,180]
        folium.raster_layers.ImageOverlay(rgba,support_bounds,name='NIR fora da faixa de ajuste',opacity=.7).add_to(m)
    patchkey='patches_'+bucket
    patches=st.session_state.setdefault(patchkey,[])
    for i,f in enumerate(patches):
        color=['#777777','#e79f00','#00a8aa','#8f50be'][f['properties']['class_id']]
        folium.GeoJson(f,name=f'Correção {i+1}',style_function=lambda _,color=color:{'color':color,'fillOpacity':.2},
            tooltip=html.escape(f['properties'].get('evidence','Correção provisória'))).add_to(m)
    Draw(export=False,draw_options={'polyline':False,'rectangle':True,'polygon':True,'circle':False,'circlemarker':False,'marker':False},
         edit_options={'edit':False,'remove':True}).add_to(m)
    folium.LayerControl().add_to(m);m.fit_bounds(bounds)
    st.subheader('2. Corrigir áreas no mapa (opcional)')
    st.caption('Desenhe um polígono ou retângulo e clique em “Adicionar correção”. O mapa é um preview reduzido; a exportação usa a resolução original. Exclua correções salvas pelo seletor abaixo.')
    result=st_folium(m,height=650,width=1100,key='full_map_'+bucket,returned_objects=['last_active_drawing'])
    pending_key='pending_'+bucket
    if result and result.get('last_active_drawing'):
        drawing=result['last_active_drawing']
        if drawing.get('geometry',{}).get('type') in ('Polygon','MultiPolygon'):st.session_state[pending_key]=drawing
    target=st.selectbox('Classe para a área desenhada',[0,1,2,3],format_func=lambda c:'0 — excluir/não classificado' if c==0 else f'{c} — {label_names[c]}',key='target_'+bucket)
    evidence=st.text_input('Observação / motivo da correção',key='evidence_'+bucket)
    if st.button('Adicionar correção',key='add_'+bucket):
        pending=st.session_state.get(pending_key)
        if not pending:st.error('Desenhe primeiro um polígono ou retângulo.')
        elif not evidence.strip():st.error('Registre o motivo da correção.')
        else:
            feature={'type':'Feature','geometry':pending['geometry'],'properties':{'class_id':target,'label':label_names.get(target,'Excluído'),
                'evidence':evidence,'validation':'provisional'}}
            if any(f['geometry']==feature['geometry'] and f['properties']['class_id']==target for f in patches):
                st.info('Esta geometria já foi adicionada para a mesma classe.')
            else:
                patches.append(feature);st.session_state.pop(pending_key,None);st.rerun()
    if patches:
        st.write(f'{len(patches)} correções locais nesta sessão. Áreas sobrepostas usam a última correção.')
        remove=st.selectbox('Correção a remover',range(len(patches)),format_func=lambda i:f"{i+1}: "+patches[i]['properties']['evidence'],key='remove_index_'+bucket)
        if st.button('Remover correção selecionada',key='remove_'+bucket):patches.pop(remove);st.rerun()
    st.subheader('3. Salvar uma versão para QGIS')
    notes=st.text_area('Responsável, fonte da interpretação e observações',key='notes_'+bucket)
    st.caption('“Salvar versão” exporta o GeoTIFF completo e as correções em GeoJSON. Nenhum arquivo original é sobrescrito. Correções não criam dados onde a fonte está excluída.')
    if st.button('Salvar versão reclassificada',key='export_'+bucket):
        if not notes.strip():st.error('Informe o responsável e a fonte da interpretação.')
        else:
            with st.spinner('Exportando cena completa em blocos…'):
                try:
                    exported=export_reclassification(run,band,mapping,patches,label_names,notes)
                    st.session_state['last_export_'+bucket]=str(exported)
                except Exception as error:st.error('Não foi possível exportar: '+str(error))
    last=st.session_state.get('last_export_'+bucket)
    if last:
        st.success('Versão salva: '+last)
        st.code(str(Path(last)/'reclassificacao_provisoria.tif'),language=None)
        for filename,label,mime in (
            ('reclassificacao_provisoria.tif','Baixar GeoTIFF reclassificado','image/tiff'),
            ('correcoes_locais.geojson','Baixar correções locais','application/geo+json'),
            ('revisao.json','Baixar registro da revisão','application/json'),
        ):
            artifact=Path(last)/filename
            if artifact.is_file():st.download_button(label,artifact.read_bytes(),file_name=filename,mime=mime,key='download_'+bucket+filename)
    with st.expander('Versões já salvas'):
        versions=sorted(run.glob('reclassificacoes/*/revisao.json'),reverse=True)
        for file in versions[:20]:
            record=json.loads(file.read_text(encoding='utf-8'))
            st.write(file.parent.name+' — '+record.get('validation_status','provisória'))
            st.code(str(file.parent/'reclassificacao_provisoria.tif'),language=None)
        if not versions:st.caption('Nenhuma versão salva ainda.')
    st.caption('Classes: 1 amarelo · 2 turquesa · 3 roxo. Os nomes e correções são hipóteses de revisão, não rótulos de campo confirmados.')
    st.code(str(path),language=None)
