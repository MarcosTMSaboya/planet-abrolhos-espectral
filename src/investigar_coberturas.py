"""Relações exploratórias com sensibilidade a candidatos; valores sem unidade confirmada."""
from pathlib import Path
from datetime import datetime, timezone
from itertools import product, combinations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/teste_sitios/20261030T153036Z'


def distance(a, b):
    la, lb = np.radians([a.lat, b.lat])
    dl = np.radians(b.lon-a.lon)
    v = np.sin((lb-la)/2)**2 + np.cos(la)*np.cos(lb)*np.sin(dl/2)**2
    return 6371000*2*np.arcsin(np.sqrt(v))


def main():
    out = ROOT / 'outputs/investigacao_coberturas' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True)
    bio = pd.read_csv(BASE / 'composicao_por_habitat.csv')
    links = pd.read_csv(BASE / 'vinculos_candidatos.csv')
    spectra = pd.read_csv(BASE / 'espectros_candidatos.csv')
    spec = spectra[spectra.n_validos_3x3.eq(9)].pivot(index=['cena_id','sitio','ponto_id'], columns='nm', values='mediana_3x3').dropna().reset_index()
    for label, a, b in [('azul_verde',492,566),('vermelho_verde',666,566),('nir_verde',866,566)]:
        spec[label] = np.where(spec[b].abs()>1e-10, spec[a]/spec[b], np.nan)
    # Valores de colunas explícitas/conhecidas como categorias; nenhuma soma taxonômica.
    variables = ['CCA','CYAN.','Macro','Dicsp','Hal_spp','Pad_spp','Lob_spp','SUB']
    wide = bio.pivot(index=['Sítio','Megahabitat','Habitat'], columns='coluna_original', values='valor_original').reset_index()
    top = wide[wide.Habitat.isin(['TOP','ARCH'])].rename(columns={'Sítio':'sitio'})
    integrated = spec.merge(top[['sitio','Habitat']+variables],on='sitio')
    integrated.to_csv(out/'dados_integrados.csv',index=False,encoding='utf-8-sig')
    pairs=[]
    for site,g in links.dropna(subset=['ponto_id']).groupby('sitio'):
        for (_,a),(_,b) in combinations(g.iterrows(),2):
            pairs.append({'sitio':site,'ponto_a':a.ponto_id,'nome_a':a.nome_ponto,'ponto_b':b.ponto_id,'nome_b':b.nome_ponto,'distancia_m':distance(a,b)})
    pairs=pd.DataFrame(pairs)
    pairs.to_csv(out/'distancias_entre_candidatos.csv',index=False,encoding='utf-8-sig')
    contrasts=[]
    for _,p in pairs.iterrows():
        a=spec[spec.ponto_id.eq(p.ponto_a)]
        b=spec[spec.ponto_id.eq(p.ponto_b)]
        for _,r in a.merge(b,on='cena_id',suffixes=('_a','_b')).iterrows():
            contrasts.append({'sitio':p.sitio,'cena_id':r.cena_id,'distancia_m':p.distancia_m,
                              'razao_verde_b_a':r['566_b']/r['566_a'],
                              'diferenca_razao_azul_verde':r.azul_verde_b-r.azul_verde_a})
    pd.DataFrame(contrasts).to_csv(out/'contrastes_candidatos.csv',index=False,encoding='utf-8-sig')
    scene=integrated.groupby('cena_id').sitio.nunique().sort_values(ascending=False).index[0]
    current=integrated[integrated.cena_id.eq(scene)]
    fig,ax=plt.subplots(figsize=(12,6))
    for i,((site,pid),g) in enumerate(spectra[spectra.cena_id.eq(scene)].groupby(['sitio','ponto_id'])):
        ax.plot(g.nm,g.mediana_3x3,marker='o',color=plt.get_cmap('tab20')(i),label=f'{site} ({pid[-6:]})')
    ax.set(xlabel='Comprimento de onda (nm)',ylabel='Reflectância rhos — mediana 3 × 3',title='Curvas com cores distintas — a maior resposta azul/verde é Siriba')
    ax.legend(bbox_to_anchor=(1.01,1),loc='upper left',fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'espectros_identificados.png',dpi=160);plt.close(fig)
    # Cada combinação escolhe apenas um candidato por sítio; não duplica a cobertura.
    choices=[list(g.index) for _,g in current.groupby('sitio',sort=True)]
    results=[]
    for selection in product(*choices):
        sample=current.loc[list(selection)]
        for var in variables:
            for feat in [492,566,666,866,'azul_verde','vermelho_verde','nir_verde']:
                pair=sample[[var,feat]].dropna()
                rho=pair[var].rank().corr(pair[feat].rank()) if pair[var].nunique()>1 and pair[feat].nunique()>1 else np.nan
                results.append({'cobertura_original':var,'feicao':str(feat),'n_sitios':len(pair),'rho':rho})
    results=pd.DataFrame(results)
    corr=results.groupby(['cobertura_original','feicao']).agg(n_sitios=('n_sitios','min'),rho_min=('rho','min'),rho_mediana=('rho','median'),rho_max=('rho','max'),n_combinacoes=('rho','count')).reset_index()
    corr['unidade_cobertura']='cobertura média (%) 2012–2026'
    corr['mesmo_sinal_em_todas']=corr.rho_min*corr.rho_max>0
    corr.to_csv(out/'correlacoes_sensibilidade.csv',index=False,encoding='utf-8-sig')
    loo=[]
    for selection in product(*choices):
        sample=current.loc[list(selection)]
        for site in sample.sitio:
            sub=sample[sample.sitio.ne(site)]
            loo.append({'sitio_retirado':site,'rho_macro_verde':sub.Macro.rank().corr(sub[566].rank()),'n_sitios':len(sub)})
    loo=pd.DataFrame(loo).groupby('sitio_retirado').agg(rho_min=('rho_macro_verde','min'),rho_mediana=('rho_macro_verde','median'),rho_max=('rho_macro_verde','max')).reset_index()
    loo.to_csv(out/'macro_verde_retirando_um_sitio.csv',index=False,encoding='utf-8-sig')
    # WALL permanece em painel próprio, sem tratá-lo como observação independente.
    panel=wide.set_index(['Sítio','Habitat'])[variables]
    fig,ax=plt.subplots(figsize=(12,7))
    z=panel.rank(pct=True)
    im=ax.imshow(z.to_numpy(),aspect='auto',cmap='YlGnBu',vmin=0,vmax=1)
    ax.set_xticks(range(len(variables)),variables,rotation=35,ha='right')
    ax.set_yticks(range(len(panel)),[f'{s} / {h}' for s,h in panel.index])
    ax.set_title('Posição relativa por coluna — valores originais, sem somar categorias')
    fig.colorbar(im,ax=ax,label='Percentil entre registros com valor (não é % de cobertura)')
    fig.tight_layout();fig.savefig(out/'painel_composicao.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    colors=plt.get_cmap('tab10')
    for k,(variable,feat) in enumerate([('CCA','azul_verde'),('Macro',566),('CYAN.','vermelho_verde')]):
        ax=axes[k]
        for i,(site,g) in enumerate(current.groupby('sitio')):
            ax.scatter(g[variable],g[feat],color=colors(i%10))
            for _,r in g.iterrows():
                ax.annotate(site,(r[variable],r[feat]),fontsize=7,xytext=(3,3),textcoords='offset points')
        ax.set(xlabel=f'{variable} — cobertura média histórica (%)',ylabel=str(feat),title=f'{variable} × {feat}')
        ax.grid(alpha=.2)
    fig.suptitle(f'{scene}: candidatos ligados ao mesmo sítio não são réplicas')
    fig.tight_layout();fig.savefig(out/'cobertura_espectro.png',dpi=160);plt.close(fig)
    ranked=corr.copy()
    ranked['ordem']=ranked.rho_mediana.abs()
    ranked=ranked.sort_values('ordem',ascending=False).head(12).drop(columns='ordem')
    text='''<!doctype html><html lang="pt-br"><meta charset="utf-8"><title>Investigação das coberturas</title>
<style>body{font:16px system-ui;max-width:1150px;margin:35px auto;padding:20px;color:#173042}table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #ccc;padding:6px}img{width:100%}</style>
<h1>Investigação dos pontos e das coberturas</h1>
<p>Ensaio descritivo com dados salvos da planilha e reflectâncias ACOLITE. As relações abaixo são hipóteses, não validação de classes.</p>
<h2>Distância entre candidatos de um sítio</h2>'''+pairs.to_html(index=False,float_format=lambda x:f'{x:.2f}')
    text+='''<p>Farol: a distância entre os pontos impede tratá-los como o mesmo pixel. RH15: pontos quase coincidentes não constituem amostras independentes. Nenhum candidato foi escolhido automaticamente.</p>
<h2>Correção da identificação das curvas</h2><p>A curva mais alta no azul/verde é de Siriba. A primeira figura repetia cores e levou a uma atribuição incorreta a Farol na explicação inicial.</p><img src="espectros_identificados.png">
<h2>Composição disponível</h2><p>TOP e WALL são mantidos separados. O painel usa posições relativas dentro de cada coluna, sem interpretar valores como porcentagem. Branco significa ausência de valor.</p><img src="painel_composicao.png">
<p>As colunas CCA, CYAN. e Macro são examinadas individualmente. Dicsp, Hal_spp, Pad_spp, Lob_spp e SUB permanecem com o rótulo original, sem assumir equivalências taxonômicas. Turf foi excluído das tabelas analíticas, correlações e gráficos; as colunas originais permanecem preservadas na fonte.</p>
<h2>Relação exploratória com a imagem</h2><p>Usa TOP para PAB1–5 e ARCH para os demais sítios, somente como hipótese operacional. Isso não confirma que esses habitats representem o pixel observado. WALL permanece fora da correlação principal, pois não há coordenada própria para esse habitat.</p><img src="cobertura_espectro.png">
<h2>Maiores associações por postos — exploração</h2><p>Spearman é calculado separadamente para cada escolha de um candidato por sítio. A tabela mostra a faixa obtida nessas escolhas. Não há teste de significância, correção de confundidores ou avaliação preditiva; explorar muitas combinações pode produzir associações altas por acaso.</p>'''
    text+=ranked.to_html(index=False,float_format=lambda x:f'{x:.3f}')
    text+='<h2>Macro × verde: retirando um sítio por vez</h2><p>Teste de influência descritivo. Cada linha retira o sítio indicado e recalcula a associação nos oito restantes, variando os candidatos disponíveis.</p>'+loo.to_html(index=False,float_format=lambda x:f'{x:.3f}')
    text+='''<p>As razões entre bandas descrevem a forma espectral; não corrigem profundidade ou coluna d’água. Apenas pixels centrais clear UDM2 e janelas com nove valores finitos nas oito bandas entram. Vizinhos não foram filtrados por UDM2.</p>
<h2>O que falta para interpretar como cobertura</h2><p>As coberturas são médias percentuais de fotoquadrados de 2012 a 2026, conforme informado pelo responsável. A relação com a imagem de 2026 representa composição histórica. Confirmar a legenda das siglas, investigar Soma de Rhodymenia Sp., confirmar os candidatos por coordenadas/ficha de campo e obter profundidade e data da coleta. A5 e RH15 não têm observações clear no inventário; por isso não há comparação entre recife e rodolito neste ensaio.</p><h2>Dados para revisão</h2><ul>'''
    text+=''.join(f'<li><a href="{p.name}">{p.name}</a></li>' for p in sorted(out.glob('*.csv')))
    text+='</ul></html>'
    (out/'relatorio.html').write_text(text,encoding='utf-8')
    print(json.dumps({'pasta':str(out),'cena':scene,'sitios':current.sitio.nunique(),'combinacoes_candidatos':len(list(product(*choices))),'associacoes':ranked.to_dict('records')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
