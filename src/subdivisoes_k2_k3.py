"""Cruzamento K2×K3 e espectros dos mesmos locais antes/depois, sem ajuste."""
import argparse
from datetime import datetime,timezone
from itertools import combinations
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import rasterio

from agrupamento_espectral import sha256
from comparar_feicoes_glint import summarize,unit_spectra
from espectral_io import ROOT,WAVELENGTHS
from experimento_espectral import write_json
from sensibilidade_espacial import read_reviewed_roi


def cross_codes(k2,k3):
    a,b=np.asarray(k2),np.asarray(k3)
    if a.shape!=b.shape or not np.isin(a,[1,2]).all() or not np.isin(b,[1,2,3]).all():
        raise ValueError('Cruzamento exige IDs válidos de K2/K3 nas mesmas posições')
    return (10*a+b).astype('uint8')


def explained_fraction(values,labels):
    values=np.asarray(values,dtype='float64');labels=np.asarray(labels)
    total=np.sum((values-values.mean())**2)
    if total<=0:return None
    between=sum(np.sum(labels==group)*(values[labels==group].mean()-values.mean())**2 for group in np.unique(labels))
    return float(between/total)


def run(comparison):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.colors import ListedColormap,BoundaryNorm
    from matplotlib.patches import Patch
    comparison=Path(comparison).resolve()
    cm=json.loads((comparison/'manifesto.json').read_text(encoding='utf-8'))
    trial=Path(cm['trial']);tm=json.loads((trial/'manifesto.json').read_text(encoding='utf-8'))
    if cm['status']!='complete' or sha256(trial/'manifesto.json')!=cm['trial_manifest_sha256']:
        raise ValueError('Manifestos incompatíveis')
    cube,valid,meta=read_reviewed_roi(tm['prepared'])
    profile=None;maps={}
    for k in (2,3):
        name=f'comparacao_k{k}_roi.tif'
        if sha256(comparison/name)!=cm['artifacts_sha256'][name]:raise ValueError('Mapa alterado')
        with rasterio.open(comparison/name) as src:
            if src.shape!=valid.shape or str(src.crs)!=meta['grid']['crs'] or not np.allclose(list(src.transform),meta['grid']['transform'],rtol=0,atol=1e-8):raise ValueError('Grade divergente')
            maps[k]=src.read()[:2];profile=src.profile.copy()
    name='ensaio_nir_visivel_nao_validado.tif'
    if sha256(trial/name)!=tm['artifacts_sha256'][name]:raise ValueError('Ensaio alterado')
    with rasterio.open(trial/name) as src:after=np.moveaxis(src.read(masked=True).filled(np.nan),0,-1)
    if not np.isfinite(after[valid]).all():raise ValueError('Pixel válido ausente')
    out=ROOT/'outputs/agrupamento_piloto'/('subdivisoes_k2_k3_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(exist_ok=False)
    manifest={'status':'in_progress','comparison':str(comparison),'comparison_manifest_sha256':sha256(comparison/'manifesto.json'),
        'scope':'base_scenario_pca95_seed42_ROI_only','new_training':False,'habitat_validation':False,
        'fixed_spectral_masks':'original K2 x K3 intersections, identical pixels before/after',
        'nir_note':'866 nm is original/unmodified; association is not specific evidence of glint'}
    try:
        before_codes=cross_codes(maps[2][0,valid],maps[3][0,valid])
        after_codes=cross_codes(maps[2][1,valid],maps[3][1,valid])
        codes=(11,12,13,21,22,23);counts=[];spectra=[];summary={};nir_rows=[];conditional=[];angles=[]
        for version,cross in (('before',before_codes),('after',after_codes)):
            for code in codes:
                n=int(np.sum(cross==code));parent=int(np.sum(cross//10==code//10))
                counts.append({'version':version,'code':code,'k2':code//10,'k3':code%10,'n_pixels':n,
                    'fraction_roi':n/int(valid.sum()),'fraction_within_k2':n/parent if parent else None})
        original=cube[valid,:6];corrected=after[valid];nir=cube[valid,7]
        for code in codes:
            mask=before_codes==code
            if mask.sum()<2:continue
            a,b=mask,after_codes==code
            nir_rows.append({'original_code':code,'n_pixels':int(mask.sum()),'nir_median':float(np.median(nir[mask])),
                'nir_q1':float(np.quantile(nir[mask],.25)),'nir_q3':float(np.quantile(nir[mask],.75)),
                'same_combination_retained_fraction':float(np.mean(b[mask])),
                'spatial_jaccard_same_code':float(np.sum(a&b)/np.sum(a|b))})
            for version,values in (('before',original),('after',corrected)):
                for rep,data in (('rhos',values[mask]),('shape_l2',unit_spectra(values[mask]))):
                    s=summarize(data);summary[(code,version,rep)]=s
                    for i,wave in enumerate(WAVELENGTHS[:6]):spectra.append({'original_code':code,'version':version,'representation':rep,
                        'wavelength_nm':wave,'n_pixels':int(mask.sum()),**{key:float(value[i]) for key,value in s.items()}})
        # Associação condicional por K2: quanto K3 organiza NIR e magnitude visível?
        for version,cross,visible in (('before',before_codes,original),('after',after_codes,corrected)):
            for parent in (1,2):
                keep=cross//10==parent;labels=cross[keep]%10
                conditional.append({'version':version,'k2':parent,'n_pixels':int(keep.sum()),
                    'k3_groups_present':len(np.unique(labels)),'eta_squared_original_nir':explained_fraction(nir[keep],labels),
                    'eta_squared_blue_green_magnitude':explained_fraction(visible[keep,:4].mean(axis=1),labels)})
        for parent in (1,2):
            present=[c for c in codes if c//10==parent and (c,'before','rhos') in summary]
            for a,b in combinations(present,2):
                for version in ('before','after'):
                    x=summary[(a,version,'rhos')]['median'][:4];y=summary[(b,version,'rhos')]['median'][:4]
                    angle=np.degrees(np.arccos(np.clip(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)),-1,1)))
                    angles.append({'code_a':a,'code_b':b,'version':version,'sam_median_blue_green_degrees':float(angle)})
        for filename,data in (('contagem_cruzamentos',counts),('espectros_mesmos_locais',spectra),('nir_e_persistencia',nir_rows),
                              ('associacao_condicional',conditional),('angulos_azul_verde',angles)):
            pd.DataFrame(data).to_csv(out/f'{filename}.csv',index=False)
        transition=pd.crosstab(pd.Series(before_codes,name='before_code'),pd.Series(after_codes,name='after_code')).reindex(index=codes,columns=codes,fill_value=0)
        transition.to_csv(out/'transicoes_combinacoes.csv')
        stack=np.zeros((3,*valid.shape),dtype='uint8');stack[0,valid]=before_codes;stack[1,valid]=after_codes
        stack[2,valid]=1+(before_codes!=after_codes).astype('uint8')
        profile.update(count=3,dtype='uint8',nodata=0,compress='DEFLATE')
        with rasterio.open(out/'cruzamento_k2_k3.tif','w',**profile) as dst:
            dst.write(stack)
            for band,name in enumerate(('original_k2_times10_plus_k3','trial_k2_times10_plus_k3','1_same_2_changed'),1):dst.set_band_description(band,name)
            dst.update_tags(scope='exploratory_crossing_not_nested_hierarchy_not_habitat')
        colors=['#532a80','#9772bd','#ccb7e1','#976b00','#e3ac20','#ffe49a']
        cmap=ListedColormap(colors);norm=BoundaryNorm(np.arange(.5,7.5),6)
        fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
        for col,version in enumerate(('before','after')):
            display=np.zeros(valid.shape)
            for i,code in enumerate(codes):display[stack[col]==code]=i+1
            axes[col].imshow(np.ma.masked_where(~valid,display),cmap=cmap,norm=norm,interpolation='nearest')
            axes[col].set(title='Original' if col==0 else 'Ensaio NIR | IDs alinhados',xlabel='Coluna ROI',ylabel='Linha ROI')
        fig.legend(handles=[Patch(color=colors[i],label=f'K2={c//10}, K3={c%10}') for i,c in enumerate(codes)],loc='outside lower center',ncol=3)
        fig.suptitle('Cruzamento K2 × K3 | PCA95 / base / semente42\nCombinações locais; K3 não é necessariamente uma hierarquia de K2')
        fig.savefig(out/'mapa_subdivisoes.png',dpi=160);plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
        for row,parent in enumerate((1,2)):
            for col,version in enumerate(('before','after')):
                ax=axes[row,col]
                for i,code in enumerate(codes):
                    if code//10!=parent or (code,version,'rhos') not in summary:continue
                    s=summary[(code,version,'rhos')]
                    ax.plot(WAVELENGTHS[:6],s['median'],marker='o',color=colors[i],label=f'K2={parent} / K3={code%10}')
                    ax.fill_between(WAVELENGTHS[:6],s['q1'],s['q3'],color=colors[i],alpha=.15)
                ax.set(title=f'Locais do K2 original={parent} | '+('original' if col==0 else 'ensaio NIR'),xlabel='Comprimento de onda (nm)',ylabel='rhos')
                ax.legend(fontsize=9);ax.grid(alpha=.2)
            low=min(a.get_ylim()[0] for a in axes[row]);high=max(a.get_ylim()[1] for a in axes[row])
            for ax in axes[row]:ax.set_ylim(low,high)
        fig.suptitle('Espectros dos mesmos locais definidos no cruzamento original\nMedianas e quartis; não são intervalos de confiança')
        fig.savefig(out/'espectros_subdivisoes.png',dpi=160);plt.close(fig)
        text='# Subdivisões K2/K3 e associação com NIR\n\nAnálise descritiva do cenário base, PCA95, semente42, somente ROI. Nenhum modelo novo. IDs após o ensaio já alinhados aos originais. K2 e K3 são ajustes distintos: os cruzamentos não presumem hierarquia.\n\n## Distribuição dentro de cada K2\n\n| Versão | K2 | K3 | Pixels | % dentro de K2 |\n|---|---:|---:|---:|---:|\n'
        for r in counts:
            if r['n_pixels']:text+=f"| {r['version']} | {r['k2']} | {r['k3']} | {r['n_pixels']} | {r['fraction_within_k2']:.1%} |\n"
        text+='\n## NIR e persistência nos locais originais\n\n| Código K2/K3 | NIR mediano | Mantém combinação após ensaio | Jaccard espacial |\n|---|---:|---:|---:|\n'
        for r in nir_rows:text+=f"| {r['original_code']} | {r['nir_median']:.5f} | {r['same_combination_retained_fraction']:.1%} | {r['spatial_jaccard_same_code']:.3f} |\n"
        text+='\n## Associação dentro de cada K2\n\n| Versão | K2 | eta² NIR original | eta² magnitude azul/verde |\n|---|---:|---:|---:|\n'
        for r in conditional:text+=f"| {r['version']} | {r['k2']} | {r['eta_squared_original_nir']:.3f} | {r['eta_squared_blue_green_magnitude']:.3f} |\n"
        text+='\nEta² descreve a fração da variância que fica entre os K3 presentes em cada K2; não mede causalidade. Os grupos foram construídos usando as bandas visíveis. Associação com NIR não identifica glint: água/fundo e efeitos comuns podem participar do sinal. NIR de 866 nm não foi corrigido.\n\n## Leitura dos produtos\n\nAs curvas fixam os pixels pelos cruzamentos originais, evitando confundir mudança de membros com mudança espectral. A tabela de forma L2 e os ângulos azul/verde estão disponíveis em CSV; curvas aqui mostram magnitude e quartis, sem inferência estatística. O raster tem banda 1=código original, 2=código após ensaio, 3=1 manteve/2 mudou; código=10×K2+K3, zero=NoData. IDs não têm significado ecológico.\n\nPersistência e contraste espectral ajudam a priorizar feições para inspeção, mas não demonstram informação bentônica. Os mapas incluem pixels usados no ajuste e a correção NIR compartilha regiões com avaliação. Não houve classificação de habitats, novas exclusões ou substituição de produto.\n'
        (out/'RELATORIO.md').write_text(text,encoding='utf-8')
        shutil.copy2(__file__,out/Path(__file__).name)
        manifest.update(status='complete',n_valid=int(valid.sum()),original_crossings_present=len(nir_rows))
        manifest['artifacts_sha256']={p.name:sha256(p) for p in sorted(out.iterdir()) if p.is_file()}
    except Exception as error:manifest.update(status='failed',error=str(error));raise
    finally:write_json(out/'manifesto.json',manifest)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--comparison',type=Path,required=True)
    args=parser.parse_args();print(run(args.comparison),flush=True)
