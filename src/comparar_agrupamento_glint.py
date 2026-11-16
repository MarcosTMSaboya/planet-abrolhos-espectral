"""K2/K3 antes/depois em amostras, geometrias e sementes pareadas."""
import argparse
from datetime import datetime,timezone
from itertools import combinations
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from sklearn.metrics import adjusted_rand_score,silhouette_score,davies_bouldin_score
from threadpoolctl import threadpool_limits

from agrupamento_espectral import sha256
from espectral_avaliacao import compare_spatial_models
from espectral_io import ROOT
from experimento_espectral import write_json
from revisao_espacial_k2 import align_labels,agreement
from sensibilidade_espacial import read_reviewed_roi,SCENARIOS,MODEL_SEEDS


def local_positions(positions,window,shape):
    local=np.asarray(positions,dtype='int64')-[window['row'],window['col']]
    if local.ndim!=2 or local.shape[1]!=2 or np.any(local<0) or np.any(local[:,0]>=shape[0]) or np.any(local[:,1]>=shape[1]):
        raise ValueError('Posições fora da ROI')
    return local[:,0],local[:,1]


def run(trial,baseline):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    trial,baseline=Path(trial).resolve(),Path(baseline).resolve()
    tm=json.loads((trial/'manifesto.json').read_text(encoding='utf-8'))
    bm=json.loads((baseline/'manifesto.json').read_text(encoding='utf-8'))
    spatial=Path(bm['previous'])
    sm=json.loads((spatial/'parametros.json').read_text(encoding='utf-8'))
    if any(m['status']!='complete' for m in (tm,bm,sm)) or tm['prepared_sha256']!=bm['prepared_sha256']:
        raise ValueError('Execuções não correspondem à mesma preparação')
    cube,valid,meta=read_reviewed_roi(tm['prepared'])
    def verify(folder,manifest,name):
        if sha256(folder/name)!=manifest['artifacts_sha256'][str(Path(name))]:raise ValueError('Artefato alterado: '+name)
    verify(trial,tm,'ensaio_nir_visivel_nao_validado.tif')
    verify(baseline,bm,'metricas_espaco_comum.csv')
    with rasterio.open(trial/'ensaio_nir_visivel_nao_validado.tif') as src:
        if src.shape!=valid.shape or src.count!=6 or str(src.crs)!=meta['grid']['crs'] or not np.allclose(list(src.transform),meta['grid']['transform'],rtol=0,atol=1e-8):
            raise ValueError('Grade experimental incompatível')
        after=np.moveaxis(src.read(masked=True).filled(np.nan),0,-1)[:,:,:4]
    if not np.isfinite(after[valid]).all():raise ValueError('Pixels válidos perdidos')
    out=ROOT/'outputs/agrupamento_piloto'/('agrupamento_antes_depois_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(exist_ok=False)
    manifest={'status':'in_progress','trial':str(trial),'baseline':str(baseline),
        'trial_manifest_sha256':sha256(trial/'manifesto.json'),'baseline_manifest_sha256':sha256(baseline/'manifesto.json'),
        'same_pixels_partitions_seeds':True,'new_models_expected':72,'original_models_reused':72,
        'ecological_validation':False,'independent_test':False,'original_recommendation_replaced':False,
        'limitation':'NIR preprocessing coefficients were estimated using scene regions that overlap evaluation. No independent end-to-end performance claim.'}
    try:
        verify(spatial,sm,'amostra_comum.npz')
        with np.load(spatial/'amostra_comum.npz',allow_pickle=False) as p:
            rr,cc=local_positions(p['positions'],meta['window'],valid.shape)
        probe=after[rr,cc];roles={};predictions={};maps={};rows=[];metrics=[];stability=[];new_common=[]
        with threadpool_limits(limits=1):
            for name,*_ in SCENARIOS:
                folder=out/name;folder.mkdir()
                for file in ('amostras.npz','particao.npz'):verify(spatial,sm,f'{name}/{file}')
                with np.load(spatial/name/'amostras.npz',allow_pickle=False) as p:
                    tr,tc=local_positions(p['train_positions'],meta['window'],valid.shape)
                    vr,vc=local_positions(p['validation_positions'],meta['window'],valid.shape)
                    before_train=p['train_values'][:,:4].copy();before_val=p['validation_values'][:,:4].copy()
                if not np.array_equal(cube[tr,tc,:4],before_train) or not np.array_equal(cube[vr,vc,:4],before_val):
                    raise ValueError('Pixels originais diferem das amostras históricas')
                with np.load(spatial/name/'particao.npz',allow_pickle=False) as p:split=p['split']
                if not (np.all(split[tr,tc]==1) and np.all(split[vr,vc]==2)):raise ValueError('Partição incorreta')
                train,val=after[tr,tc],after[vr,vc];roles[name]=split[rr,cc]
                result=compare_spatial_models(train,val,seed_values=MODEL_SEEDS,k_values=(2,3),silhouette_size=1000)
                for key,collection in (('metrics',metrics),('stability',stability)):
                    frame=result[key].copy();frame.insert(0,'scenario',name);collection.append(frame)
                indices=result['evaluation_indices']['validation'];z=result['scaler'].transform(val.astype('float64'))
                np.savez_compressed(folder/'indices_metricas.npz',**result['evaluation_indices'])
                saved={}
                for (rep,k,seed),model in result['models'].items():
                    oldfile=f'{name}/modelo_{rep}_k{k}_s{seed}.joblib';verify(baseline,bm,oldfile)
                    old=joblib.load(baseline/oldfile)
                    a=old.predict(before_val);b=result['predictions'][(rep,k,seed)]['validation']
                    aligned=align_labels(a,b,k)
                    rows.append({'scenario':name,'representation':rep,'k':k,'seed':seed,'scope':'same_validation_pixels',
                        'n_pixels':len(a),'ari_before_after':float(adjusted_rand_score(a,b)),
                        'changed_fraction_after_id_alignment':float(np.mean(a!=aligned))})
                    complete=len(np.unique(b))==k and len(np.unique(b[indices]))==k
                    new_common.append({'scenario':name,'representation':rep,'k':k,'seed':seed,
                        'silhouette':float(silhouette_score(z[indices],b[indices])) if complete else None,
                        'davies_bouldin':float(davies_bouldin_score(z,b)) if complete else None})
                    path=folder/f'modelo_{rep}_k{k}_s{seed}.joblib';joblib.dump(model,path,compress=3)
                    if not np.array_equal(joblib.load(path).predict(val[:256]),b[:256]):raise ValueError('Persistência divergiu')
                    saved[f'{rep}_k{k}_s{seed}_before']=a;saved[f'{rep}_k{k}_s{seed}_after']=b
                    if seed==42:
                        predictions[(name,rep,k,'before')]=old.predict(cube[rr,cc,:4])
                        predictions[(name,rep,k,'after')]=model.predict(probe)
                        if rep=='pca95' and name==SCENARIOS[0][0]:
                            ma=old.predict(cube[valid,:4]);mb=align_labels(ma,model.predict(after[valid]),k)
                            maps[k]=(ma,mb)
                np.savez_compressed(folder/'predicoes_avaliacao.npz',**saved)
                print('Concluído '+name+': 12 modelos novos e 12 comparações',flush=True)
        pairrows=[]
        for a,b in combinations(roles,2):
            for rep in ('bands','pca95'):
                for k in (2,3):
                    for version in ('before','after'):
                        select=(roles[a]==2)&(roles[b]==2)
                        pairrows.append({'scenario_a':a,'scenario_b':b,'representation':rep,'k':k,'version':version,
                            **agreement(predictions[(a,rep,k,version)],predictions[(b,rep,k,version)],select,k)})
        oldcommon=pd.read_csv(baseline/'metricas_espaco_comum.csv');oldcommon['version']='before'
        new=pd.DataFrame(new_common);new['version']='after'
        pd.concat([oldcommon,new],ignore_index=True).to_csv(out/'metricas_espaco_bandas_por_versao.csv',index=False)
        pd.concat(metrics).to_csv(out/'metricas_ensaio.csv',index=False)
        pd.concat(stability).to_csv(out/'estabilidade_sementes_ensaio.csv',index=False)
        paired=pd.DataFrame(rows);paired.to_csv(out/'concordancia_antes_depois.csv',index=False)
        pd.DataFrame(pairrows).to_csv(out/'estabilidade_espacial_antes_depois.csv',index=False)
        transform=Affine(*meta['grid']['transform'][:6])
        fig,axes=plt.subplots(2,3,figsize=(15,11),layout='constrained')
        for row,k in enumerate((2,3)):
            a,b=maps[k];stack=np.zeros((3,*valid.shape),dtype='uint8')
            stack[0,valid]=a+1;stack[1,valid]=b+1;stack[2,valid]=1+(a!=b).astype('uint8')
            with rasterio.open(out/f'comparacao_k{k}_roi.tif','w',driver='GTiff',height=valid.shape[0],width=valid.shape[1],count=3,
                    dtype='uint8',crs=meta['grid']['crs'],transform=transform,nodata=0,compress='DEFLATE') as dst:
                dst.write(stack)
                for n,title in enumerate(('original_local_group','trial_aligned_local_group','change_1_same_2_changed'),1):dst.set_band_description(n,title)
                dst.update_tags(scope='descriptive_not_habitat',alignment='maximum contingency on full ROI; includes fit pixels')
            for col in (0,1):
                axes[row,col].imshow(np.ma.masked_where(~valid,stack[col]),vmin=1,vmax=k,cmap='viridis',interpolation='nearest')
                axes[row,col].set_title(f'K{k} | '+('original' if col==0 else 'ensaio NIR, IDs alinhados'))
            axes[row,2].imshow(np.ma.masked_where(~valid,stack[2]-1),vmin=0,vmax=1,cmap='gray_r',interpolation='nearest')
            axes[row,2].set_title(f'K{k} | mudança na ROI: {np.mean(a!=b):.1%}\nPreto: mudou | branco: manteve')
        for ax in axes.ravel():ax.set(xlabel='Coluna ROI',ylabel='Linha ROI')
        fig.suptitle('PCA95 | cenário base | semente42 | mesmos pixels\nGrupos locais; mudança descritiva, sem acurácia ecológica')
        fig.savefig(out/'mapas_antes_depois.png',dpi=160);plt.close(fig)
        text='# Agrupamento K2/K3 antes e depois do ensaio NIR\n\n72 novos modelos comparados aos 72 originais preservados. Seis geometrias, bandas 444/492/533/566 nm, duas representações, K2/K3, sementes 11/42/97, n_init=10. Mesmas posições, papéis espaciais e 1.000 índices de métricas por cenário. Scaler/PCA/K-means reajustados somente no desenvolvimento de cada versão.\n\n'
        text+='## Concordância nos mesmos pixels de avaliação\n\n| Representação | K | ARI mínimo | Mediano | Máximo | Mudança mediana após alinhar IDs |\n|---|---:|---:|---:|---:|---:|\n'
        for (rep,k),g in paired.groupby(['representation','k']):
            text+=f'| {rep} | {k} | {g.ari_before_after.min():.3f} | {g.ari_before_after.median():.3f} | {g.ari_before_after.max():.3f} | {g.changed_fraction_after_id_alignment.median():.1%} |\n'
        text+='\nARI é invariável à permutação dos IDs; fração de mudanças usa máxima concordância na população comparada, não acurácia. Mapas usam alinhamento na ROI inteira e incluem pixels de ajuste; não comparar suas frações diretamente às da avaliação.\n\n## Limites\n\nA correção NIR foi estimada anteriormente em quadrículas que se sobrepõem às avaliações destes cenários. Assim, separar o ajuste do agrupamento não torna o pipeline completo independente. A análise mede sensibilidade ao pré-processamento fixo, não melhoria validada. Scores de cada versão usam sua própria padronização, portanto diferenças não comprovam superioridade física.\n\nO ensaio não é produto de glint validado; não foram atribuídos habitats, aplicadas novas exclusões ou substituída a recomendação anterior. Nas comparações espaciais, os casos sem todos os grupos ficam explícitos. Os GeoTIFFs têm três bandas: 1=grupo original, 2=grupo após ensaio com IDs alinhados, 3=1 manteve/2 mudou; zero=NoData. Somente ROI piloto.\n'
        (out/'RELATORIO.md').write_text(text,encoding='utf-8')
        for name in ('comparar_agrupamento_glint.py','espectral_avaliacao.py','revisao_espacial_k2.py'):
            shutil.copy2(Path(__file__).with_name(name),out/name)
        manifest.update(status='complete',new_models=72,reopened_models_verified=72)
        manifest['artifacts_sha256']={str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file()}
    except Exception as error:manifest.update(status='failed',error=str(error));raise
    finally:write_json(out/'manifesto.json',manifest)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial',type=Path,required=True);parser.add_argument('--baseline',type=Path,required=True)
    args=parser.parse_args();print(run(args.trial,args.baseline),flush=True)
