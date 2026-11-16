"""Diagnóstico NIR por quadrícula e ensaio exploratório de redução de glint.

As decisões do usuário orientam amostras, sem virar verdade de campo.
"""
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
from espectral_io import ROOT, WAVELENGTHS
from experimento_espectral import write_json
from sensibilidade_espacial import read_reviewed_roi


def fit_nir(values):
    """OLS visível~NIR, com intercepto; não comprova origem superficial."""
    values = np.asarray(values, dtype="float64")
    if values.ndim != 2 or values.shape[1] != 8 or len(values) < 30 or not np.isfinite(values).all():
        raise ValueError("Exige ao menos 30 espectros finitos de oito bandas")
    x, y = values[:, 7], values[:, :6]
    dx, dy = x-x.mean(), y-y.mean(axis=0)
    ss = float(dx@dx)
    if ss <= 0:
        raise ValueError("NIR constante: regressão indefinida")
    beta = dx@dy/ss
    intercept = y.mean(axis=0)-beta*x.mean()
    residual = y-(intercept+x[:, None]*beta)
    total = np.sum(dy*dy, axis=0)
    r2 = np.divide(1-np.sum(residual*residual, axis=0)/np.where(total>0,total,1), 1)
    r2[total == 0] = np.nan
    return beta, intercept, r2


def apply_trial(values, beta, nir_reference):
    """Mantém sinais e valores negativos; não corrige NIR nem Red Edge."""
    values = np.asarray(values, dtype="float64")
    return values[..., :6] - (values[..., 7]-nir_reference)[..., None]*np.asarray(beta)


def run(prepared, gpkg):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    cube, valid, meta = read_reviewed_roi(prepared)
    out = ROOT / "outputs/agrupamento_piloto" / ("diagnostico_glint_usuario_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    out.mkdir(exist_ok=False)
    manifest = {"status":"in_progress", "scene_id":meta["scene_id"],
        "prepared":str(Path(prepared).resolve()), "prepared_sha256":sha256(prepared),
        "review_source":str(Path(gpkg).resolve()), "ecological_validation":False,
        "deep_water_reference_confirmed":False, "glint_correction_validated":False,
        "original_mask_changed":False, "grouping_retrained":False,
        "method":"Exploratory visible-band OLS on NIR; minimum training NIR reference; no physical attribution guaranteed"}
    try:
        # Snapshot coerente, inclusive se o QGIS estiver com o arquivo aberto.
        with sqlite3.connect('file:'+Path(gpkg).resolve().as_posix()+'?mode=ro',uri=True) as src:
            with sqlite3.connect(out / "revisao_utilizada.gpkg") as dst:
                src.backup(dst)
        manifest["review_snapshot_sha256"] = sha256(out / "revisao_utilizada.gpkg")
        regions = gpd.read_file(out / "revisao_utilizada.gpkg", layer="regioes_para_revisao")
        if regions.region_id.duplicated().any() or regions.crs is None or not regions.geometry.is_valid.all():
            raise ValueError("Revisão sem CRS ou com geometrias/IDs inválidos")
        regions["normalized_decision"] = regions.decision.fillna("").replace({"propor_exclusão":"propor_exclusao"})
        if not regions.normalized_decision.isin(["investigar","manter","propor_exclusao"]).all():
            raise ValueError("Decisão de revisão desconhecida")
        regions = regions.to_crs(meta["grid"]["crs"]).reset_index(drop=True)
        transform = Affine(*meta["grid"]["transform"][:6])
        region_map = rasterize([(geom, i+1) for i,geom in enumerate(regions.geometry)],
            out_shape=valid.shape, transform=transform, fill=0, dtype="int16")
        if not np.all(region_map[valid]>0):
            raise ValueError("Revisão não cobre toda a ROI válida")
        # Partição por quadrículas azuis, fixada antes do ajuste.
        blue = regions.index[regions.normalized_decision.eq("investigar")].to_numpy()
        if len(blue)<6:
            raise ValueError("Poucas quadrículas investigar para separar ajuste e diagnóstico")
        shuffled = np.random.default_rng(42).permutation(blue)
        train_ids = shuffled[:int(len(blue)*2/3)]+1
        hold_ids = shuffled[int(len(blue)*2/3):]+1
        fit_mask = valid & np.isin(region_map, train_ids)
        hold_mask = valid & np.isin(region_map, hold_ids)
        beta, intercept, r2 = fit_nir(cube[fit_mask])
        rows = []
        for i, region in regions.iterrows():
            values = cube[valid & (region_map==i+1)]
            if len(values)<30: continue
            slopes, _, scores = fit_nir(values)
            for b in range(6):
                rows.append({"region_id":region.region_id,"decision":region.normalized_decision,
                    "role":"fit" if i+1 in train_ids else "blue_holdout" if i+1 in hold_ids else "comparison",
                    "wavelength_nm":WAVELENGTHS[b],"n_pixels":len(values),"slope":slopes[b],
                    "r_squared":scores[b],"nir_p05":np.quantile(values[:,7],.05),
                    "nir_p95":np.quantile(values[:,7],.95)})
        cells = pd.DataFrame(rows); cells.to_csv(out / "diagnostico_por_quadricula.csv",index=False)
        write_json(out / "particao_quadriculas.json", {
            "seed":42,"fit_blue_regions":regions.iloc[train_ids-1].region_id.tolist(),
            "holdout_blue_regions":regions.iloc[hold_ids-1].region_id.tolist(),
            "note":"Separação por quadrículas adjacentes; autocorrelação e independência não resolvidas. Verde/vermelho não usados no ajuste."})
        reference = float(cube[fit_mask,7].min())
        eligible = bool(np.isfinite(beta).all() and (beta>0).all())
        coefficients = pd.DataFrame({"wavelength_nm":WAVELENGTHS[:6],"slope":beta,
            "intercept":intercept,"fit_r_squared":r2})
        coefficients.to_csv(out / "coeficientes.csv",index=False)
        manifest.update(nir_reference=reference,fit_pixels=int(fit_mask.sum()),
            blue_holdout_pixels=int(hold_mask.sum()),trial_numerically_eligible=eligible,
            fit_regions=len(train_ids),blue_holdout_regions=len(hold_ids))
        trial = apply_trial(cube,beta,reference) if eligible else None
        stats=[]
        groups={"azul_ajuste":fit_mask,"azul_reservado":hold_mask}
        for decision in ("manter","propor_exclusao"):
            ids=regions.index[regions.normalized_decision.eq(decision)].to_numpy()+1
            groups[decision]=valid & np.isin(region_map,ids)
        if eligible:
            profile=dict(driver="GTiff",width=valid.shape[1],height=valid.shape[0],count=6,
                dtype="float32",crs=meta["grid"]["crs"],transform=transform,nodata=-9999.,compress="DEFLATE")
            with rasterio.open(out / "ensaio_nir_visivel_nao_validado.tif","w",**profile) as dst:
                for b in range(6):
                    dst.write(np.where(valid,trial[:,:,b],-9999).astype("float32"),b+1)
                    dst.set_band_description(b+1,f"rhos_{WAVELENGTHS[b]}_experimental_unvalidated")
                dst.update_tags(scope="experimental_not_validated",nir_reference=str(reference),
                    method="visible_minus_beta_times_nir_excess",reference_water_depth="unconfirmed")
            for name,mask in groups.items():
                for b in range(6):
                    before,after,nir=cube[mask,b],trial[mask,b],cube[mask,7]
                    stats.append({"group":name,"wavelength_nm":WAVELENGTHS[b],"n_pixels":len(before),
                        "corr_nir_before":np.corrcoef(before,nir)[0,1],"corr_nir_after":np.corrcoef(after,nir)[0,1],
                        "negative_before":float(np.mean(before<0)),"negative_after":float(np.mean(after<0)),
                        "median_before":float(np.median(before)),"median_after":float(np.median(after)),
                        "fraction_nir_below_reference":float(np.mean(nir<reference))})
            pd.DataFrame(stats).to_csv(out / "comparacao_grupos.csv",index=False)
        # Contraste comum, calculado exclusivamente na imagem original.
        fig,axes=plt.subplots(1,3,figsize=(15,5))
        rgb=[]
        for data in (cube,trial if eligible else cube):
            channels=[]
            for b in (5,3,1):
                low,high=np.quantile(cube[valid,b],[.02,.98])
                channels.append(np.clip((data[:,:,b]-low)/(high-low),0,1))
            image=np.stack(channels,axis=-1);image[~valid]=1;rgb.append(image)
        axes[0].imshow(rgb[0]);axes[0].set_title("Original | contraste comum")
        axes[1].imshow(rgb[1]);axes[1].set_title("Ensaio NIR | não validado" if eligible else "Ensaio não executado")
        colorcodes={"investigar":1,"manter":2,"propor_exclusao":3}
        classmap=np.zeros(valid.shape)
        for i,r in regions.iterrows(): classmap[region_map==i+1]=colorcodes[r.normalized_decision]
        from matplotlib.colors import ListedColormap
        axes[2].imshow(classmap,cmap=ListedColormap(["white","#239de0","#48ae50","#e73c39"]),vmin=0,vmax=3)
        axes[2].set_title("Interpretação do usuário preservada\nAzul: investigar | verde: manter | vermelho: propor exclusão")
        for ax in axes:ax.set(xlabel="Coluna ROI",ylabel="Linha ROI")
        fig.suptitle("Teste de sensibilidade à regressão NIR | não confirma recuperação do sinal do fundo")
        fig.tight_layout();fig.savefig(out / "comparacao_visual.png",dpi=160);plt.close(fig)
        text=("# Diagnóstico de glint orientado pela revisão do usuário\n\n"
            f"Usadas {len(regions)} quadrículas, preservando o GeoPackage original. "
            f"Ajuste em {len(train_ids)} quadrículas azuis e diagnóstico em {len(hold_ids)} azuis reservadas. "
            "Verdes e vermelhas são comparação, sem ajustar coeficientes. A partição não garante independência espacial.\n\n"
            "## Interpretação\n\nAzul: homogeneidade aparente/fundo elevado e glint; verde: fundo elevado ou areia e glint menor; "
            "vermelho: transição com glint elevado, conforme o usuário. Essas interpretações não validam profundidade, substrato ou ausência de sinal aquático NIR.\n\n"
            "## Coeficientes do ajuste\n\n| nm | Inclinação | R² ajuste |\n|---|---:|---:|\n")
        for b in range(6):text+=f"| {WAVELENGTHS[b]} | {beta[b]:.4f} | {r2[b]:.4f} |\n"
        text+=(f"\nEnsaio gerado: {eligible}. Referência NIR mínima do ajuste: {reference:.6f}. "
            "Os coeficientes são OLS sobre os azuis de ajuste; o ensaio segue a forma visível − beta × (NIR − referência). "
            "Negativos e aumentos onde NIR é menor que a referência não foram truncados nos dados científicos.\n\n"
            "## Validação e limites\n\nA redução da correlação com NIR na amostra de ajuste é consequência matemática do método e não prova remoção correta do glint. "
            "Mesmo em quadrículas reservadas, água/fundo e gradientes comuns podem explicar a associação. "
            "Examinar variação das inclinações por quadrícula e negativos em comparacao_grupos.csv. "
            "Água opticamente profunda/homogênea e a origem superficial da variação NIR continuam não confirmadas. "
            "As seis áreas vermelhas não foram excluídas, e o mapa/máscara/classificação anteriores não foram modificados.\n\n"
            "O GeoTIFF de seis bandas visíveis é apenas um ensaio exploratório para comparação no QGIS, não um produto corrigido validado. "
            "NIR e Red Edge não foram corrigidos. A comparação RGB usa o mesmo contraste original; clipping existe apenas na figura.\n")
        (out / "RELATORIO.md").write_text(text,encoding="utf-8")
        for name in ("diagnostico_glint_usuario.py","sensibilidade_espacial.py","espectral_io.py"):
            shutil.copy2(Path(__file__).with_name(name),out / name)
        manifest.update(status="complete",trial_generated=eligible)
        manifest["artifacts_sha256"]={str(p.relative_to(out)):sha256(p) for p in sorted(out.iterdir()) if p.is_file()}
    except Exception as error:
        manifest.update(status="failed",error=str(error));raise
    finally:write_json(out / "manifesto.json",manifest)
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared",type=Path,required=True)
    parser.add_argument("--gpkg",type=Path,required=True)
    args=parser.parse_args();print(run(args.prepared,args.gpkg),flush=True)
