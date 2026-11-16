"""OLS univariada com intercepto; sensibilidade às coordenadas candidatas."""
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/investigacao_coberturas/20261030T161812Z'
OUT=BASE/'r2'
OUT.mkdir(exist_ok=True)
df=pd.read_csv(BASE/'dados_integrados.csv')
scene='20260213_125207_88_24d8'
df=df[df.cena_id.eq(scene)].copy()
variables=['CCA','CYAN.','Macro','Dicsp','Hal_spp','Pad_spp','Lob_spp','SUB']
features=[str(w) for w in [444,492,533,566,612,666,707,866]]
choices=[list(g.index) for _,g in df.groupby('sitio',sort=True)]
samples=[df.loc[list(indices)] for indices in product(*choices)]

def fit(x,y):
    ok=np.isfinite(x)&np.isfinite(y)
    x=np.asarray(x)[ok];y=np.asarray(y)[ok]
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:
        return np.nan,np.nan,np.nan,len(x)
    slope,intercept=np.polyfit(x,y,1)
    r2=1-np.sum((y-(slope*x+intercept))**2)/np.sum((y-y.mean())**2)
    return r2,slope,intercept,len(x)

rows=[]
for var in variables:
    for band in features:
        for i,sample in enumerate(samples):
            r2,slope,intercept,n=fit(sample[band],sample[var])
            rows.append(dict(cobertura=var,banda_nm=band,combinacao=i,R2=r2,inclinacao=slope,intercepto=intercept,n_sitios=n))
raw=pd.DataFrame(rows)
raw.to_csv(OUT/'ajustes_por_combinacao.csv',index=False,encoding='utf-8-sig')
summary=raw.groupby(['cobertura','banda_nm']).agg(R2_min=('R2','min'),R2_mediano=('R2','median'),R2_max=('R2','max'),n_sitios=('n_sitios','min')).reset_index()
summary.to_csv(OUT/'resumo_r2.csv',index=False,encoding='utf-8-sig')
mat=summary.pivot(index='cobertura',columns='banda_nm',values='R2_mediano').reindex(index=variables,columns=features)
fig,ax=plt.subplots(figsize=(11,7))
im=ax.imshow(mat,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
ax.set_xticks(range(8),features);ax.set_yticks(range(8),variables)
ax.set(xlabel='Banda espectral (nm)',ylabel='Cobertura média histórica (%)',title='R² mediano — regressões lineares por banda\n9 sítios; 8 escolhas de coordenadas; ajuste na própria amostra')
for i in range(8):
    for j in range(8):
        v=mat.iloc[i,j]
        ax.text(j,i,f'{v:.2f}',ha='center',va='center',color='white' if v>.55 else '#152635')
fig.colorbar(im,ax=ax,label='R² (OLS com intercepto)');fig.tight_layout()
fig.savefig(OUT/'mapa_r2.png',dpi=160);plt.close(fig)

# Painéis fixos, sem selecionar apenas os melhores ajustes.
for var in ['Macro','CCA','CYAN.']:
    fig,axes=plt.subplots(2,4,figsize=(18,9))
    for ax,band in zip(axes.flat,features):
        for sample in samples:
            r2,slope,intercept,n=fit(sample[band],sample[var])
            xx=np.linspace(df[band].min(),df[band].max(),50)
            if np.isfinite(slope):
                ax.plot(xx,slope*xx+intercept,color='#198a92',alpha=.25,linewidth=1)
        for i,(site,g) in enumerate(df.groupby('sitio')):
            ax.scatter(g[band],g[var],color=plt.get_cmap('tab10')(i),s=30,zorder=3)
            for _,p in g.iterrows():
                ax.annotate(site,(p[band],p[var]),fontsize=6,xytext=(2,3),textcoords='offset points')
        s=summary[(summary.cobertura==var)&(summary.banda_nm==band)].iloc[0]
        ax.set(title=f'{band} nm | R² {s.R2_mediano:.2f} [{s.R2_min:.2f}–{s.R2_max:.2f}]',xlabel='Reflectância rhos',ylabel=f'{var} (%)')
        ax.grid(alpha=.2)
    fig.suptitle(f'{var}: cobertura histórica × reflectância — {scene}\nLinhas = escolhas alternativas de coordenadas; pontos do mesmo sítio compartilham a cobertura',fontsize=12)
    fig.tight_layout();fig.savefig(OUT/f'ajustes_{var.replace(".","")}.png',dpi=150);plt.close(fig)

report='''<!doctype html><html lang="pt-br"><meta charset="utf-8"><title>R² das coberturas</title><style>body{font:16px system-ui;max-width:1250px;margin:35px auto;padding:20px}img{width:100%}td,th{padding:6px;border:1px solid #ddd}table{border-collapse:collapse}</style>
<h1>R² entre coberturas e bandas espectrais</h1><p>Resposta: cobertura média percentual dos fotoquadrados de 2012–2026. Preditor: reflectância de uma banda da cena de 13/02/2026. Regressão linear simples com intercepto, separada para cada categoria e banda. Turf excluído.</p>
<p>São nove sítios, com TOP e ARCH conforme o ensaio anterior. Oito combinações escolhem um candidato por sítio. A cor apresenta o R² mediano dessas escolhas; nos painéis, os colchetes indicam mínimo–máximo. Não se trata de intervalo de confiança.</p>
<p>R² é a fração da variação da cobertura representada pela reta nesta amostra. Não indica o sentido da relação, causalidade, acurácia de classificação ou desempenho fora da amostra. Não é o quadrado da correlação de Spearman calculada anteriormente. Profundidade, água e diferença de período não foram controladas; 64 ajustes exploratórios podem produzir R² altos por acaso.</p><img src="mapa_r2.png">'''
for var in ['Macro','CCA','CYAN.']:
    report+=f'<h2>{var}</h2><img src="ajustes_{var.replace(".","")}.png">'
report+='<h2>Valores</h2>'+summary.to_html(index=False,float_format=lambda x:f'{x:.3f}')
report+='<p><a href="resumo_r2.csv">Resumo CSV</a> · <a href="ajustes_por_combinacao.csv">Ajustes e coeficientes CSV</a></p></html>'
(OUT/'relatorio.html').write_text(report,encoding='utf-8')
p=BASE/'relatorio.html'
t=p.read_text(encoding='utf-8')
if 'r2/relatorio.html' not in t:
    t=t.replace('<h1>Investigação dos pontos e das coberturas</h1>','<h1>Investigação dos pontos e das coberturas</h1><p><a href="r2/relatorio.html"><strong>Abrir gráficos de regressão e R²</strong></a></p>')
    p.write_text(t,encoding='utf-8')
print(summary[summary.cobertura.eq('Macro')].to_string(index=False))
print(OUT)
