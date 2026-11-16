"""Compara feições visuais nos mesmos pixels antes/depois do ensaio NIR."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.features import rasterize

from agrupamento_espectral import sha256
from espectral_io import ROOT,WAVELENGTHS
from experimento_espectral import write_json
from sensibilidade_espacial import read_reviewed_roi


def summarize(values):
    values=np.asarray(values,dtype="float64")
    if values.ndim!=2 or len(values)<2 or not np.isfinite(values).all():
        raise ValueError("Amostra deve conter ao menos dois espectros finitos")
    q=np.quantile(values,[.25,.5,.75],axis=0)
    return {"mean":values.mean(axis=0),"std":values.std(axis=0),"q1":q[0],"median":q[1],"q3":q[2]}


def unit_spectra(values):
    values=np.asarray(values,dtype="float64")
    norms=np.linalg.norm(values,axis=1)
    if not np.all(norms>0):raise ValueError("Espectro de norma zero não pode ser normalizado")
    return values/norms[:,None]


def run(trial_dir,gpkg):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    trial_dir=Path(trial_dir).resolve()
    origin=json.loads((trial_dir/'manifesto.json').read_text(encoding='utf-8'))
    if origin['status']!='complete':raise ValueError('Ensaio incompleto')
    for name in ('ensaio_nir_visivel_nao_validado.tif','particao_quadriculas.json'):
        if sha256(trial_dir/name)!=origin['artifacts_sha256'][name]:raise ValueError('Ensaio alterado')
    if sha256(origin['prepared'])!=origin['prepared_sha256']:raise ValueError('Preparação alterada')
    cube,valid,meta=read_reviewed_roi(origin['prepared'])
    with rasterio.open(trial_dir/'ensaio_nir_visivel_nao_validado.tif') as src:
        if src.count!=6 or src.shape!=valid.shape or str(src.crs)!=meta['grid']['crs'] or not np.allclose(list(src.transform),meta['grid']['transform'],rtol=0,atol=1e-8):
            raise ValueError('Ensaio não coincide com a grade original')
        corrected=np.moveaxis(src.read(masked=True).filled(np.nan),0,-1)
    out=ROOT/'outputs/agrupamento_piloto'/('comparacao_feicoes_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(exist_ok=False)
    manifest={'status':'in_progress','trial':str(trial_dir),'trial_manifest_sha256':sha256(trial_dir/'manifesto.json'),
        'scope':'visual_hypotheses_only','same_pixels_before_after':True,'independent_validation':False,
        'new_model_fit':False,'original_data_changed':False}
    try:
        with sqlite3.connect('file:'+Path(gpkg).resolve().as_posix()+'?mode=ro',uri=True) as src:
            with sqlite3.connect(out/'revisao_utilizada.gpkg') as dst:src.backup(dst)
        regions=gpd.read_file(out/'revisao_utilizada.gpkg',layer='regioes_para_revisao')
        # Só descrições inequívocas; transições, forte glint e campos vazios ficam fora.
        evidence=regions.evidence.fillna('').str.strip().str.casefold()
        regions['hypothesis']=np.where(evidence.eq('areia'),'possivel_areia',
            np.where(evidence.eq('fundo elevado'),'possivel_fundo_elevado','fora_comparacao'))
        selected=regions.loc[regions.hypothesis.ne('fora_comparacao')].copy().to_crs(meta['grid']['crs'])
        if set(selected.hypothesis)!= {'possivel_areia','possivel_fundo_elevado'}:raise ValueError('Duas hipóteses explícitas não encontradas')
        if selected.region_id.duplicated().any() or not selected.geometry.is_valid.all():raise ValueError('Geometrias/IDs inválidos')
        transform=Affine(*meta['grid']['transform'][:6])
        labels=rasterize([(geom,i+1) for i,geom in enumerate(selected.geometry)],out_shape=valid.shape,transform=transform,dtype='int16')
        common=valid & np.isfinite(corrected).all(axis=2)
        if np.any(valid & ~common):raise ValueError('Ensaio perdeu pixels da população original')
        partition=json.loads((trial_dir/'particao_quadriculas.json').read_text(encoding='utf-8'))
        selected['used_to_fit_trial']=selected.region_id.isin(partition['fit_blue_regions'])
        selected.to_crs('EPSG:4326').to_file(out/'feicoes_comparadas.geojson',driver='GeoJSON')
        rows,cell_rows,summary,counts=[],[],{},{}
        for hypothesis in sorted(selected.hypothesis.unique()):
            ids=np.flatnonzero(selected.hypothesis.to_numpy()==hypothesis)+1
            mask=common & np.isin(labels,ids)
            counts[hypothesis]={'regions':len(ids),'pixels':int(mask.sum()),'area_m2':float(mask.sum()*abs(transform.a*transform.e)),
                'fit_regions':int(selected.loc[selected.hypothesis.eq(hypothesis),'used_to_fit_trial'].sum())}
            for version,data in (('original',cube[:,:,:6]),('ensaio',corrected)):
                values=data[mask]
                for representation,matrix in (('rhos',values),('forma_l2',unit_spectra(values))):
                    stats=summarize(matrix);summary[(hypothesis,version,representation)]=stats
                    for b,wave in enumerate(WAVELENGTHS[:6]):
                        rows.append({'hypothesis':hypothesis,'version':version,'representation':representation,'wavelength_nm':wave,
                            'n_pixels':len(values),**{key:float(value[b]) for key,value in stats.items()}})
            for i,region in enumerate(selected.itertuples()):
                if region.hypothesis!=hypothesis:continue
                local=common & (labels==i+1)
                for version,data in (('original',cube[:,:,:6]),('ensaio',corrected)):
                    for b,wave in enumerate(WAVELENGTHS[:6]):
                        cell_rows.append({'region_id':region.region_id,'hypothesis':hypothesis,'version':version,
                            'used_to_fit_trial':region.used_to_fit_trial,'wavelength_nm':wave,
                            'n_pixels':int(local.sum()),'median':float(np.median(data[local,b]))})
        pd.DataFrame(rows).to_csv(out/'espectros_mesmos_pixels.csv',index=False)
        pd.DataFrame(cell_rows).to_csv(out/'medianas_por_quadricula.csv',index=False)
        separation=[]
        for version in ('original','ensaio'):
            a=summary[('possivel_areia',version,'rhos')];b=summary[('possivel_fundo_elevado',version,'rhos')]
            for i,wave in enumerate(WAVELENGTHS[:6]):
                spread=np.sqrt((a['std'][i]**2+b['std'][i]**2)/2)
                separation.append({'version':version,'wavelength_nm':wave,'signed_median_difference':a['median'][i]-b['median'][i],
                    'absolute_standardized_mean_difference':float(abs(a['mean'][i]-b['mean'][i])/spread) if spread>0 else None})
        sep=pd.DataFrame(separation);sep.to_csv(out/'separacao_descritiva.csv',index=False)
        fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
        colors={'possivel_areia':'#bc7a00','possivel_fundo_elevado':'#276b9d'}
        names={'possivel_areia':'Possível areia','possivel_fundo_elevado':'Possível fundo elevado'}
        for col,version in enumerate(('original','ensaio')):
            for row,representation in enumerate(('rhos','forma_l2')):
                ax=axes[row,col]
                for hypothesis in colors:
                    s=summary[(hypothesis,version,representation)]
                    ax.plot(WAVELENGTHS[:6],s['median'],marker='o',color=colors[hypothesis],label=names[hypothesis])
                    ax.fill_between(WAVELENGTHS[:6],s['q1'],s['q3'],color=colors[hypothesis],alpha=.18)
                ax.set(title=('Original' if version=='original' else 'Ensaio NIR não validado')+(' | magnitude' if row==0 else ' | forma por pixel'),
                    ylabel='rhos' if row==0 else 'Espectro / norma L2',xlabel='Comprimento de onda (nm)')
                ax.grid(alpha=.2);ax.legend(fontsize=9)
        for row in (0,1):
            lo=min(ax.get_ylim()[0] for ax in axes[row]);hi=max(ax.get_ylim()[1] for ax in axes[row])
            for ax in axes[row]:ax.set_ylim(lo,hi)
        fig.suptitle('Mesmos pixels antes/depois | medianas e quartis descritivos\nHipóteses visuais do usuário; quartis não são intervalos de confiança')
        fig.tight_layout(rect=(0,0,1,.94));fig.savefig(out/'curvas_antes_depois.png',dpi=160);plt.close(fig)
        report='# Comparação espectral das feições visuais\n\nSeleção por descrições explícitas do GeoPackage atual: somente `areia` e `Fundo elevado`. Transições, forte glint e campos vazios excluídos desta comparação; nenhuma exclusão na máscara científica. As quadrículas inteiras de 150 m podem conter misturas.\n\n'
        for hypothesis,count in counts.items():report+=f"- {hypothesis}: {count['regions']} quadrículas, {count['pixels']} pixels; {count['fit_regions']} quadrículas participaram do ajuste NIR anterior.\n"
        report+='\n## Comparação descritiva\n\n| nm | Diferença mediana areia − elevado original | Ensaio | Diferença média padronizada original | Ensaio |\n|---|---:|---:|---:|---:|\n'
        for wave in WAVELENGTHS[:6]:
            a=sep.loc[sep.version.eq('original')&sep.wavelength_nm.eq(wave)].iloc[0];b=sep.loc[sep.version.eq('ensaio')&sep.wavelength_nm.eq(wave)].iloc[0]
            report+=f'| {wave} | {a.signed_median_difference:.5f} | {b.signed_median_difference:.5f} | {a.absolute_standardized_mean_difference:.3f} | {b.absolute_standardized_mean_difference:.3f} |\n'
        report+='\n## Como interpretar\n\nAs curvas separam magnitude de forma (normalização L2 por pixel). Faixas são quartis entre pixels, não incerteza independente. A diferença padronizada mede contraste numérico entre estes locais; não é acurácia ou probabilidade de habitat. A seleção foi visual, há autocorrelação e sobreposição com ajuste anterior. Apenas duas quadrículas de areia não caracterizam toda a variabilidade da areia.\n\nProfundidade, propriedades da água e glint podem explicar parte da diferença. Preservação de contraste não comprova preservação física do sinal bentônico. Não atribuir rótulos ecológicos nem promover o ensaio a produto validado apenas por essas curvas. O GeoJSON permite conferir os locais exatos; medianas por quadrícula ajudam a identificar heterogeneidade. Não houve novo ajuste ou classificação.\n'
        (out/'RELATORIO.md').write_text(report,encoding='utf-8')
        shutil.copy2(__file__,out/Path(__file__).name)
        manifest.update(status='complete',groups=counts,review_snapshot_sha256=sha256(out/'revisao_utilizada.gpkg'))
        manifest['artifacts_sha256']={p.name:sha256(p) for p in sorted(out.iterdir()) if p.is_file()}
    except Exception as error:manifest.update(status='failed',error=str(error));raise
    finally:write_json(out/'manifesto.json',manifest)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial-dir',type=Path,required=True);parser.add_argument('--gpkg',type=Path,required=True)
    args=parser.parse_args();print(run(args.trial_dir,args.gpkg),flush=True)
