"""Sorteio estratificado de pontos candidatos e vínculo espacial com campo.

Não converte proximidade ou nomes provisórios em referência confirmada.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.spatial import cKDTree

try:
    from .cena_completa import ROOT,digest,write_json
    from .espectral_io import windows
except ImportError:
    from cena_completa import ROOT,digest,write_json
    from espectral_io import windows


def transition_mask(halo):
    """Borda se houver vizinho válido de outra classe na vizinhança 3×3."""
    a=np.asarray(halo);center=a[1:-1,1:-1]
    result=np.zeros(center.shape,dtype=bool)
    for dr in range(3):
        for dc in range(3):
            neighbor=a[dr:dr+center.shape[0],dc:dc+center.shape[1]]
            result|=(center>0)&(neighbor>0)&(neighbor!=center)
    return result


def keep_smallest(current,priorities,positions,size):
    """Reservatório por prioridades iid uniformes: amostra SRS sem reposição."""
    oldp,oldpos=current
    p=np.concatenate((oldp,priorities));pos=np.concatenate((oldpos,positions),axis=0)
    if len(p)>size:
        selected=np.argpartition(p,size-1)[:size];p,pos=p[selected],pos[selected]
    return p,pos


def run(root=ROOT,n=20,seed=20261028):
    root=Path(root);choice=json.loads((root/'VERSAO_DE_TRABALHO.json').read_text(encoding='utf-8'))
    scene=root/choice['scene_run'];review=scene/choice['latest_saved_review_at_selection']
    record=json.loads((review/'revisao.json').read_text(encoding='utf-8'))
    raster=review/'reclassificacao_provisoria.tif'
    if record['status']!='complete' or record['band']!=4 or digest(raster)!=record['raster_sha256']:
        raise ValueError('Versão de trabalho incompatível ou alterada')
    out=root/'outputs/validacao'/('pontos_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(parents=True,exist_ok=False)
    manifest={'status':'in_progress','scene_id':choice['scene_id'],'working_review':str(review),
        'source_raster_sha256':digest(raster),'working_review_sha256':digest(review/'revisao.json'),
        'seed':seed,'requested_per_stratum':n,'reference_labels_assigned':False,
        'design':'SRS without replacement within class × 3x3 transition strata via iid random priorities',
        'boundary_definition':'A valid neighbor of a different class within one pixel (8 neighbors). Zero is not another class.',
        'association_radius_m':30,'association_note':'Operational search radius only, not validated positional accuracy or proof of correspondence',
        'water_mask_complete':False,'spatial_independence_proven':False,'accuracy_metrics_computed':False}
    try:
        strata=[(group,edge) for group in (1,2,3) for edge in (False,True)]
        reservoir={key:(np.empty(0),np.empty((0,2),dtype='int64')) for key in strata}
        counts={key:0 for key in strata};rng=np.random.default_rng(seed)
        with rasterio.open(raster) as src:
            for w in windows(src,512):
                halo=src.read(1,window=Window(w.col_off-1,w.row_off-1,w.width+2,w.height+2),boundless=True,fill_value=0)
                data=halo[1:-1,1:-1];edges=transition_mask(halo)
                for key in strata:
                    group,edge=key;rr,cc=np.nonzero((data==group)&(edges==edge))
                    counts[key]+=len(rr)
                    if len(rr):reservoir[key]=keep_smallest(reservoir[key],rng.random(len(rr)),
                        np.column_stack((rr+int(w.row_off),cc+int(w.col_off))),n)
            records=[]
            for (group,edge),(priority,positions) in reservoir.items():
                for row,col in positions[np.argsort(priority)]:
                    x,y=src.xy(int(row),int(col));total=counts[(group,edge)]
                    records.append({'point_id':f'v_{group}_{int(edge)}_{int(row)}_{int(col)}','predicted_group':group,
                        'predicted_label_provisional':record['labels'].get(str(group),f'Grupo {group}'),
                        'stratum':'transicao' if edge else 'interior','row':int(row),'col':int(col),
                        'population_pixels_stratum':total,'sample_pixels_stratum':len(positions),
                        'inclusion_probability':len(positions)/total,'reference_status':'sem_referencia',
                        'reference_class':'','reference_evidence':'','reference_date':'','reference_depth_m':None,
                        'reference_horizontal_accuracy_m':None,'reviewer':'','water_check':'pendente'})
            points=gpd.GeoDataFrame(records,geometry=gpd.points_from_xy([src.xy(r['row'],r['col'])[0] for r in records],
                [src.xy(r['row'],r['col'])[1] for r in records]),crs=src.crs)
            fieldpath=root/'data/campo/piloto_gpx/pontos_cobertos.csv'
            reviewpath=root/'data/campo/piloto_gpx/revisao_classes.csv'
            field=pd.read_csv(fieldpath,keep_default_na=False,dtype={'name':str,'ponto_id':str})
            fieldreview=pd.read_csv(reviewpath,keep_default_na=False,dtype={'name':str,'ponto_id':str})
            if field.ponto_id.duplicated().any() or fieldreview.ponto_id.duplicated().any():raise ValueError('IDs de campo duplicados')
            for col in ('classe_bentonica','status_rotulo','evidencia','data_campo_confirmada','profundidade_m','grupo_estacao'):
                if col in field:field=field.drop(columns=[col])
            field=field.merge(fieldreview.drop(columns=['name','lat','lon'],errors='ignore'),on='ponto_id',how='left',validate='one_to_one')
            field=gpd.GeoDataFrame(field,geometry=gpd.points_from_xy(field.lon,field.lat),crs='EPSG:4326').to_crs(src.crs)
            xs,ys=field.geometry.x.to_numpy(),field.geometry.y.to_numpy()
            rr,cc=rasterio.transform.rowcol(src.transform,xs,ys);rr,cc=np.asarray(rr),np.asarray(cc)
            inside=(rr>=0)&(rr<src.height)&(cc>=0)&(cc<src.width)
            field=field.loc[inside].copy();field['row']=rr[inside];field['col']=cc[inside]
            field['predicted_group']=[int(value[0]) for value in src.sample(zip(field.geometry.x,field.geometry.y))]
            field['eligible_classified_pixel']=field.predicted_group.gt(0)
            field['label_available_not_automatically_validated']=field.classe_bentonica.fillna('').str.strip().ne('')&field.evidencia.fillna('').str.strip().ne('')
            eligible_field=field.loc[field.eligible_classified_pixel].copy()
            points['nearest_field_id']='';points['nearest_field_distance_m']=np.nan;points['field_within_30m']=False
            if len(eligible_field):
                distance,index=cKDTree(np.column_stack((eligible_field.geometry.x,eligible_field.geometry.y))).query(np.column_stack((points.geometry.x,points.geometry.y)))
                points['nearest_field_id']=eligible_field.iloc[index].ponto_id.to_numpy();points['nearest_field_distance_m']=distance
                points['field_within_30m']=distance<=30
            # Candidatos existentes são mantidos separadamente; não substituem sorteados.
            manifest['field_sources']={str(fieldpath):digest(fieldpath),str(reviewpath):digest(reviewpath)}
        wgs=points.to_crs('EPSG:4326');points['lon']=wgs.geometry.x.to_numpy();points['lat']=wgs.geometry.y.to_numpy()
        if len(points)>1:
            nearest=cKDTree(np.column_stack((points.geometry.x,points.geometry.y))).query(np.column_stack((points.geometry.x,points.geometry.y)),k=2)[0][:,1]
            points['nearest_sample_distance_m']=nearest
        points.to_file(out/'pontos_verificacao.gpkg',layer='pontos_sorteados',driver='GPKG')
        if len(field):field.to_file(out/'pontos_verificacao.gpkg',layer='inventario_campo_na_cena',driver='GPKG')
        points.to_crs('EPSG:4326').to_file(out/'pontos_sorteados.geojson',driver='GeoJSON')
        points.drop(columns='geometry').to_csv(out/'pontos_sorteados.csv',index=False,encoding='utf-8-sig')
        field.drop(columns='geometry').to_csv(out/'inventario_campo_na_cena.csv',index=False,encoding='utf-8-sig')
        summary=[]
        for group,edge in strata:
            summary.append({'group':group,'stratum':'transicao' if edge else 'interior','population_pixels':counts[(group,edge)],
                'n_sampled':len(reservoir[(group,edge)][0])})
        pd.DataFrame(summary).to_csv(out/'desenho_amostral.csv',index=False)
        manifest.update(status='complete',n_sampled=len(points),strata=summary,n_field_in_scene=len(field),
            n_field_on_valid_class=len(eligible_field),n_field_with_label_and_evidence=int(field.label_available_not_automatically_validated.sum()),
            sampled_points_near_field=int(points.field_within_30m.sum()),n_reference_confirmed=0)
        text='# Pontos para verificação — Ensaio NIR K3\n\n'
        text+=f"Versão salva {review.name}, com cinco exclusões de ilhas. {len(points)} pontos sorteados, alvo de {n} por classe × interior/transição. As classes e seus nomes são previsões provisórias, não referências.\n\n"
        text+='## Desenho\n\nAmostra aleatória simples sem reposição dentro de cada um dos seis estratos, por prioridades uniformes, semente fixa. Populações e probabilidades de inclusão estão em desenho_amostral.csv e nos pontos. Transição significa vizinho válido de outra classe na janela 3×3 (um pixel); zero não define transição. Os pontos podem estar próximos: não foi imposta distância mínima para preservar o desenho probabilístico. Não há garantia de independência espacial.\n\n'
        text+=f"## Inventário de campo\n\n{len(field)} pontos existentes dentro da grade; {len(eligible_field)} sobre pixels classificados. {int(field.label_available_not_automatically_validated.sum())} possuem classe e evidência preenchidas (isso sozinho não confirma validade). {int(points.field_within_30m.sum())} dos sorteados têm ponto de campo a até 30 m. O raio é operacional; proximidade não fornece automaticamente referência para o pixel. Datas GPX não são datas de campo confirmadas. A aquisição é de 13/02/2026; conferir compatibilidade temporal das observações.\n\n"
        text+='## Abrir no QGIS e completar referências\n\nAbra pontos_verificacao.gpkg junto ao raster reclassificado. A camada pontos_sorteados contém o desenho probabilístico; inventario_campo_na_cena contém candidatos de campo separados. Não substitua os sorteados por pontos convenientes se desejar estimar acurácia por esse desenho.\n\nPreencha reference_class, reference_evidence, reference_date, reviewer, reference_depth_m e reference_horizontal_accuracy_m somente a partir de referência independente verificável. reference_status começa sem_referencia; marque confirmado apenas após verificar evidência, posição e compatibilidade temporal. water_check permite identificar terra residual ou água; a máscara de água da cena não está completa. Nomes provisórios da classificação não devem ser copiados para a referência.\n\nSe não houver evidência, mantenha sem_referencia. O inventário pode apoiar planejamento de coleta ou busca de fotos/vídeos; não é automaticamente conjunto de teste. Áreas usadas em interpretação/ajuste anterior devem ser identificadas antes de alegar validação independente. Não foi calculada matriz de confusão, acurácia ou área de habitat.\n'
        (out/'LEIA_ME.md').write_text(text,encoding='utf-8')
        shutil.copy2(__file__,out/Path(__file__).name)
        manifest['artifacts_sha256']={p.name:digest(p) for p in out.iterdir() if p.is_file()}
    except Exception as error:manifest.update(status='failed',error=str(error));raise
    finally:write_json(out/'manifesto.json',manifest)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args();print(run(args.root),flush=True)
