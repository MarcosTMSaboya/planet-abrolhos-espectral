"""Inferência e reclassificação exploratória, em blocos, sem retreinamento."""
import argparse
from contextlib import ExitStack
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from rasterio.windows import transform as window_transform

try:
    from .espectral_io import read_block,grid_offset,windows
except ImportError:
    from espectral_io import read_block,grid_offset,windows

ROOT=Path(__file__).resolve().parents[1]
PREP='preparacao_espacial_20261020T175318_984586Z/preparacao.json'
TRIAL='diagnostico_glint_usuario_20261028T143103_022403Z'
COMPARE='agrupamento_antes_depois_20261028T175559_513148Z'
BASE='revisao_espacial_k2_20261028T130952_614878Z'
BAND_OPTIONS={'Original — K2':1,'Original — K3':2,'Ensaio NIR — K2':3,'Ensaio NIR — K3':4}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write_json(path,data):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');temp.replace(path)


def infer(root=ROOT):
    import joblib
    import sqlite3
    from threadpoolctl import threadpool_limits
    root=Path(root);pilot=root/'outputs/agrupamento_piloto'
    prep=json.loads((pilot/PREP).read_text(encoding='utf-8'))
    trial=pilot/TRIAL;tm=json.loads((trial/'manifesto.json').read_text(encoding='utf-8'))
    if tm['status']!='complete' or digest(pilot/PREP)!=tm['prepared_sha256']:raise ValueError('Preparação incompatível')
    source_paths=[Path(s['path']) for s in prep['sources']]
    for p,s in zip(source_paths,prep['sources']):
        if (p.stat().st_size,p.stat().st_mtime_ns)!=(s['size'],s['mtime_ns']):raise ValueError('Fonte alterada: '+str(p))
    for filename in ('coeficientes.csv','particao_quadriculas.json','revisao_utilizada.gpkg'):
        if digest(trial/filename)!=tm['artifacts_sha256'][filename]:raise ValueError('Ensaio alterado')
    coefficients=pd.read_csv(trial/'coeficientes.csv')
    if coefficients.wavelength_nm.tolist()!=[444,492,533,566,612,666]:raise ValueError('Ordem das bandas incompatível')
    beta=coefficients.slope.to_numpy()[:4];reference=tm['nir_reference']
    models=[];model_records=[]
    for version,directory in (('original',pilot/BASE),('trial',pilot/COMPARE)):
        mm=json.loads((directory/'manifesto.json').read_text(encoding='utf-8'))
        for k in (2,3):
            relative=str(Path('base_100_34_s42')/f'modelo_pca95_k{k}_s42.joblib')
            path=directory/relative
            if digest(path)!=mm['artifacts_sha256'][relative]:raise ValueError('Modelo alterado')
            models.append(joblib.load(path));model_records.append({'version':version,'k':k,'path':str(path),'sha256':digest(path)})
    out=root/'outputs/cena_completa'/('cena_'+prep['scene_id']+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(parents=True,exist_ok=False)
    status={'status':'in_progress','scene_id':prep['scene_id'],'scope':'whole_scene_exploratory',
        'model_scope':'ROI-trained PCA95 / base / seed42; no retraining','models':model_records,
        'sources':prep['sources'],'prepared_sha256':digest(pilot/PREP),'trial_manifest_sha256':digest(trial/'manifesto.json'),
        'water_mask_available_for_whole_scene':False,'eligibility':'All 8 bands finite + strict UDM2; NOT a water or benthic visibility mask',
        'ecological_validation':False,'nir_correction_validated':False,'progress':0,'nir_reference':reference,
        'class_bands':BAND_OPTIONS,'script_sha256':digest(__file__)}
    write_json(out/'manifesto.json',status);print(str(out),flush=True)
    try:
        split=json.loads((trial/'particao_quadriculas.json').read_text(encoding='utf-8'))
        with sqlite3.connect('file:'+(trial/'revisao_utilizada.gpkg').as_posix()+'?mode=ro',uri=True) as db:
            quadrats=db.execute('select region_id,row_roi,col_roi from regioes_para_revisao').fetchall()
        fit_ranges=[]
        with rasterio.open(source_paths[7]) as nir:
            for region,row,col in quadrats:
                if region in split['fit_blue_regions']:
                    w=rasterio.windows.Window(prep['window']['col']+col,prep['window']['row']+row,50,50)
                    values=nir.read(1,window=w,masked=True).compressed();fit_ranges.extend([float(values.min()),float(values.max())])
        lo,hi=min(fit_ranges),max(fit_ranges);status['nir_fit_range']=[lo,hi]
        eligible_count=outside_count=0;class_counts=np.zeros((4,4),dtype='int64')
        with rasterio.Env(GDAL_CACHEMAX=64*1024*1024,GDAL_PAM_ENABLED='NO'),ExitStack() as stack,threadpool_limits(limits=1):
            bands=[stack.enter_context(rasterio.open(p)) for p in source_paths[:-1]]
            q=stack.enter_context(rasterio.open(source_paths[-1]));ref=bands[0]
            for band in bands:
                if grid_offset(ref,band)!=(0,0) or band.shape!=ref.shape:raise ValueError('Bandas desalinhadas')
            offset=grid_offset(ref,q)
            profile=ref.profile.copy();profile.update(count=4,dtype='uint8',nodata=0,compress='DEFLATE',tiled=True,blockxsize=256,blockysize=256,BIGTIFF='IF_SAFER')
            dst=stack.enter_context(rasterio.open(out/'grupos_cena_completa.tif','w',**profile))
            support_profile=profile.copy();support_profile['count']=1
            support=stack.enter_context(rasterio.open(out/'suporte_nir.tif','w',**support_profile))
            nblocks=((ref.height+511)//512)*((ref.width+511)//512)
            status['grid']={'width':ref.width,'height':ref.height,'crs':str(ref.crs),'transform':list(ref.transform)}
            for index,w in enumerate(windows(ref,512),1):
                data,valid,_,_=read_block(bands,q,offset,w)
                labels=np.zeros((4,int(w.height),int(w.width)),dtype='uint8');domain=np.zeros(valid.shape,dtype='uint8')
                if valid.any():
                    raw=data[valid,:4];nir_values=data[valid,7]
                    corrected=(raw.astype('float64')-(nir_values-reference)[:,None]*beta).astype('float32')
                    outside=(nir_values<lo)|(nir_values>hi)
                    domain[valid]=1+outside.astype('uint8');outside_count+=int(outside.sum())
                    eligible_count+=int(valid.sum())
                    for b,model in enumerate(models):
                        assigned=model.predict(raw if b<2 else corrected)+1
                        labels[b,valid]=assigned
                        class_counts[b]+=np.bincount(assigned,minlength=4)[:4]
                dst.write(labels,window=w);support.write(domain,1,window=w)
                if index%10==0 or index==nblocks:
                    status.update(progress=round(index/nblocks,4),eligible_pixels=eligible_count,outside_nir_fit_range=outside_count)
                    write_json(out/'manifesto.json',status)
            for name,b in BAND_OPTIONS.items():dst.set_band_description(b,name)
            dst.update_tags(scope='exploratory_not_habitat',water_mask='absent',quality='strict_UDM2_and_finite_8_bands')
            support.set_band_description(1,'0 excluded;1 NIR within fit range;2 outside fit range')
            support.update_tags(note='Numeric NIR range only; within range does not validate correction or model transfer')
        rows=[]
        for b,counts in enumerate(class_counts):
            for c in range(1,4):rows.append({'band':b+1,'group':c,'pixels':int(counts[c]),'scope':'spectral_group_not_habitat'})
        pd.DataFrame(rows).to_csv(out/'contagem_grupos.csv',index=False)
        status.update(status='complete',progress=1.,eligible_pixels=eligible_count,
            outside_nir_fit_range=outside_count,outside_nir_fit_fraction=outside_count/eligible_count if eligible_count else None)
        shutil.copy2(__file__,out/Path(__file__).name)
        status['artifacts_sha256']={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='manifesto.json'}
    except Exception as error:status.update(status='failed',error=str(error));raise
    finally:write_json(out/'manifesto.json',status)
    return out


def export_reclassification(run,band,mapping,features,labels,notes):
    """Nova versão; polígonos em WGS84 sobrescrevem só pixels elegíveis."""
    from shapely.geometry import shape
    run=Path(run);source=run/'grupos_cena_completa.tif'
    if band not in range(1,5) or set(mapping)!={1,2,3} or any(v not in (0,1,2,3) for v in mapping.values()):raise ValueError('Mapeamento inválido')
    for f in features:
        if f['geometry']['type'] not in ('Polygon','MultiPolygon') or f['properties']['class_id'] not in (0,1,2,3):raise ValueError('Polígono/classe inválido')
        geom=shape(f['geometry'])
        if geom.is_empty or not geom.is_valid:raise ValueError('Polígono vazio ou inválido; revise o desenho')
    out=run/'reclassificacoes'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True,exist_ok=False)
    write_json(out/'revisao.json',{'status':'in_progress','band':band,'mapping':mapping,'labels':labels,'notes':notes,
        'validation_status':'provisional_user_interpretation','features':features,'source':str(source)})
    lut=np.array([0,mapping[1],mapping[2],mapping[3]],dtype='uint8');counts=np.zeros(4,dtype='int64')
    try:
        with rasterio.open(source) as src:
            profile=src.profile.copy();profile.update(count=1)
            shapes=[(transform_geom('EPSG:4326',src.crs,f['geometry']),f['properties']['class_id']+1) for f in features]
            with rasterio.open(out/'reclassificacao_provisoria.tif','w',**profile) as dst:
                for w in windows(src,512):
                    original=src.read(band,window=w);data=lut[original]
                    if shapes:
                        patches=rasterize(shapes,out_shape=original.shape,transform=window_transform(w,src.transform),fill=0,dtype='uint8')
                        selected=(patches>0)&(original>0);data[selected]=patches[selected]-1
                    data[original==0]=0;dst.write(data,1,window=w);counts+=np.bincount(data.ravel(),minlength=4)
                dst.update_tags(validation='provisional_not_validated',labels=json.dumps(labels,ensure_ascii=False),zero='excluded_or_unclassified')
        write_json(out/'correcoes_locais.geojson',{'type':'FeatureCollection','features':features})
        write_json(out/'revisao.json',{'status':'complete','band':band,'mapping':mapping,'labels':labels,'notes':notes,
            'validation_status':'provisional_user_interpretation','features':features,'source':str(source),
            'source_sha256':digest(source),'counts':counts.tolist(),'raster_sha256':digest(out/'reclassificacao_provisoria.tif')})
    except Exception as error:
        write_json(out/'erro.json',{'status':'failed','error':str(error)});raise
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args();print(infer(args.root),flush=True)
