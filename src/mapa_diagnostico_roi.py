"""Produto local de inspeção: aplica modelo existente somente à ROI revisada."""
import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from PIL import Image
from threadpoolctl import threadpool_limits

from agrupamento_espectral import dependencies, sha256
from espectral_io import ROOT, WAVELENGTHS
from experimento_espectral import write_json
from sensibilidade_espacial import read_reviewed_roi


COLORS = {0: (225, 225, 225, 0), 1: (230, 159, 0, 255),
          2: (139, 76, 192, 255), 3: (0, 191, 196, 255)}


def generate(prepared, recommendation):
    prepared, recommendation = Path(prepared).resolve(), Path(recommendation).resolve()
    rec = json.loads(recommendation.read_text(encoding="utf-8"))
    model_path = (recommendation.parent / rec["model_path"]).resolve()
    if sha256(model_path) != rec["model_sha256"]:
        raise ValueError("Modelo diferente da recomendação registrada")
    if (rec["k"], rec["representation"], rec["wavelengths_nm"]) != (3, "pca95", [444, 492, 533, 566]):
        raise ValueError("Configuração não corresponde ao piloto")
    cube, valid, meta = read_reviewed_roi(prepared)
    if rec["scene_id"] != meta["scene_id"]:
        raise ValueError("Cena incompatível com o modelo")
    output = ROOT / "outputs/agrupamento_piloto" / ("mapa_diagnostico_roi_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(exist_ok=False)
    manifest = {"status": "in_progress", "scene_id": meta["scene_id"],
                "scope": "diagnostic_map_reviewed_roi_only", "user_authorized_roi_application": True,
                "new_training": False, "whole_scene_application": False,
                "ecological_interpretation_validated": False, "benthic_visibility": "not_confirmed",
                "model_path": str(model_path), "model_sha256": sha256(model_path),
                "recommendation": str(recommendation), "recommendation_sha256": sha256(recommendation),
                "prepared": str(prepared), "prepared_sha256": sha256(prepared),
                "window": meta["window"], "grid": meta["grid"], "prediction_block_pixels": 128,
                "notice": "Grupos espectrais exploratórios; profundidade, água e superfície não separados."}
    try:
        model = joblib.load(model_path)
        labels = np.zeros(valid.shape, dtype="uint8")
        with threadpool_limits(limits=1):
            for row in range(0, valid.shape[0], 128):
                for col in range(0, valid.shape[1], 128):
                    v = valid[row:row+128, col:col+128]
                    if v.any():
                        view = labels[row:row+128, col:col+128]
                        view[v] = model.predict(cube[row:row+128, col:col+128, :4][v])+1
            indices = np.random.default_rng(42).choice(np.flatnonzero(valid.ravel()), min(512, int(valid.sum())), replace=False)
            rr, cc = np.unravel_index(indices, valid.shape)
            if not np.array_equal(labels[rr, cc], model.predict(cube[rr, cc, :4])+1):
                raise ValueError("Predição em blocos divergiu da referência")
        if not (np.isin(labels[valid], [1, 2, 3]).all() and (labels[~valid] == 0).all()):
            raise ValueError("Classes ou exclusões incorretas")
        transform = Affine(*meta["grid"]["transform"][:6])
        profile = dict(driver="GTiff", height=valid.shape[0], width=valid.shape[1],
                       crs=meta["grid"]["crs"], transform=transform, dtype="uint8", compress="DEFLATE")
        with rasterio.Env(GDAL_PAM_ENABLED="NO", GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(output / "grupos_espectrais_roi.tif", "w", count=1, nodata=0, **profile) as dst:
                dst.write(labels, 1); dst.write_colormap(1, COLORS)
                dst.update_tags(description="0=Nao Classificado;1=Grupo Espectral 1;2=Grupo Espectral 2;3=Grupo Espectral 3",
                                scope="aquatic_exploratory_not_habitat", model_sha256=manifest["model_sha256"])
            rgb = np.zeros((*valid.shape, 3), dtype="uint8")
            stretches = {}
            for channel, wave in enumerate((666, 566, 492)):
                band = cube[:, :, WAVELENGTHS.index(wave)]
                low, high = np.percentile(band[valid], [2, 98])
                rgb[:, :, channel] = (np.clip(np.nan_to_num((band-low)/(high-low)) if high > low else np.zeros_like(band), 0, 1)*255).astype("uint8")
                stretches[str(wave)] = [float(low), float(high)]
            rgb[~valid] = 225
            manifest["display_only_rgb"] = {"bands_rgb_nm": [666, 566, 492], "stretch": "P2–P98 por banda nos pixels válidos da ROI", "limits": stretches,
                                           "changes_model_input": False}
            with rasterio.open(output / "rgb_roi_visual.tif", "w", count=3, photometric="RGB", **profile) as dst:
                dst.write(np.moveaxis(rgb, -1, 0)); dst.write_mask(valid.astype("uint8")*255)
                dst.update_tags(description="RGB visual ACOLITE; stretch por banda P2-P98; nao e reflectancia original")
        lut = np.array([COLORS[i] for i in range(4)], dtype="uint8")
        rgba = lut[labels]
        colored = rgba[:, :, :3].copy(); colored[~valid] = 225
        overlay = np.rint(.55*rgb+.45*colored).astype("uint8")
        for name, values in (("rgb_roi.png", rgb), ("grupos_roi.png", colored), ("sobreposicao_roi.png", overlay), ("grupos_transparentes.png", rgba)):
            Image.fromarray(values).save(output / name)
        _, _, _, _, plt = dependencies()
        from matplotlib.patches import Patch
        extent = [0, valid.shape[1]*3, valid.shape[0]*3, 0]
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        for ax, values, title in zip(axes, (rgb, colored, overlay), ("Imagem RGB da ROI", "Grupos espectrais — K=3", "Sobreposição — 45%")):
            ax.imshow(values, extent=extent, interpolation="nearest")
            ax.set(title=title, xlabel="Distância para leste (m)", ylabel="Distância para sul (m)")
        fig.legend(handles=[Patch(color=np.array(COLORS[i][:3])/255, label=f"Grupo Espectral {i}") for i in (1,2,3)] +
                   [Patch(color="#e1e1e1", label="Não Classificado")], loc="lower center", ncol=4)
        fig.suptitle("Abrolhos | 13/02/2026 | ROI 1,62 km² | Norte para cima\nExploração espectral — identidade ecológica e visibilidade do fundo não confirmadas")
        fig.tight_layout(rect=(0, .08, 1, .91))
        fig.savefig(output / "comparacao_roi.png", dpi=180, bbox_inches="tight"); plt.close(fig)
        counts = [{"value": i, "name": "Não Classificado" if i == 0 else f"Grupo Espectral {i}",
                   "pixel_count": int((labels == i).sum()), "area_m2_roi": int((labels == i).sum())*9} for i in range(4)]
        pd.DataFrame(counts).to_csv(output / "contagem_roi.csv", index=False)
        manifest["counts"] = counts
        with rasterio.open(output / "grupos_espectrais_roi.tif") as src:
            if (src.crs.to_string() != meta["grid"]["crs"] or src.transform != transform
                    or src.nodata != 0 or not np.array_equal(src.read(1), labels)):
                raise ValueError("GeoTIFF salvo não reproduziu a grade/classes")
        rgb_uri = "data:image/png;base64,"+base64.b64encode((output / "rgb_roi.png").read_bytes()).decode()
        rgba_uri = "data:image/png;base64,"+base64.b64encode((output / "grupos_transparentes.png").read_bytes()).decode()
        html = '''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Abrolhos — inspeção da ROI</title><style>
body{font:16px system-ui;background:#f4f6f8;color:#172534;max-width:1050px;margin:24px auto;padding:0 16px}h1{font-size:26px}p{line-height:1.5}.controls{display:flex;flex-wrap:wrap;gap:16px;align-items:center;padding:16px;background:white;border-radius:8px}button{padding:8px 12px;cursor:pointer}canvas{width:100%;height:auto;display:block;background:#e1e1e1;margin-top:16px}label{white-space:nowrap}.note{font-size:14px}a{color:#175b95}</style>
<h1>Abrolhos — inspeção visual da ROI</h1><p>Cena de 13/02/2026 · 1,62 km² · Norte para cima · Pixels de 3 m</p>
<div class="controls"><button onclick="mode('rgb')">Só imagem RGB</button><button onclick="mode('groups')">Só grupos</button><button onclick="mode('overlay')">Sobrepor</button>
<label>Opacidade dos grupos <input id="alpha" type="range" min="0" max="100" value="45" oninput="draw()"><span id="percent">45%</span></label></div>
<p><label style="color:#946600"><input type="checkbox" checked id="g1" onchange="draw()">Grupo Espectral 1</label>　<label style="color:#7840a8"><input type="checkbox" checked id="g2" onchange="draw()">Grupo Espectral 2</label>　<label style="color:#007d82"><input type="checkbox" checked id="g3" onchange="draw()">Grupo Espectral 3</label></p>
<canvas id="map" width="450" height="400" aria-label="Imagem da ROI e grupos espectrais sobrepostos"></canvas>
<p id="state" role="status">Carregando imagens locais…</p><p class="note">Arraste a opacidade para comparar os limites dos grupos com a imagem. Desmarque um grupo para observar a imagem sob ele. Use “Só imagem RGB” como referência.</p>
<p><strong>As cores indicam semelhança espectral.</strong> Não representam classes confirmadas de algas, rodolitos ou outros habitats. A influência de profundidade, água e superfície permanece em investigação. RGB com contraste P2–P98 por banda apenas para visualização.</p>
<p><a href="comparacao_roi.png">Comparação lado a lado</a> · <a href="grupos_espectrais_roi.tif">GeoTIFF dos grupos para QGIS</a> · <a href="LEIA_ME.md">Guia e limites</a></p>
<script>const canvas=document.getElementById('map'),ctx=canvas.getContext('2d'),rgb=new Image(),groups=new Image();let selectedMode='overlay';
rgb.src='RGB_URI';groups.src='RGBA_URI';let loaded=0;[rgb,groups].forEach(img=>{img.onload=()=>{if(++loaded===2){document.getElementById('state').textContent='Imagens carregadas. Funciona localmente, sem conexão ou novos cálculos.';draw()}};img.onerror=()=>document.getElementById('state').textContent='Falha ao carregar imagem.'});
function mode(value){selectedMode=value;if(value==='overlay' && +document.getElementById('alpha').value===0)document.getElementById('alpha').value=45;draw()}
function draw(){if(loaded<2)return;ctx.globalAlpha=1;ctx.fillStyle='#e1e1e1';ctx.fillRect(0,0,450,400);if(selectedMode!=='groups')ctx.drawImage(rgb,0,0);document.getElementById('percent').textContent=document.getElementById('alpha').value+'%';if(selectedMode==='rgb')return;
const off=document.createElement('canvas');off.width=450;off.height=400;const oc=off.getContext('2d');oc.drawImage(groups,0,0);const data=oc.getImageData(0,0,450,400),colors=[[230,159,0],[139,76,192],[0,191,196]],enabled=[1,2,3].map(i=>document.getElementById('g'+i).checked);for(let p=0;p<data.data.length;p+=4){for(let g=0;g<3;g++)if(!enabled[g] && data.data[p]===colors[g][0] && data.data[p+1]===colors[g][1] && data.data[p+2]===colors[g][2])data.data[p+3]=0;}oc.putImageData(data,0,0);ctx.globalAlpha=selectedMode==='groups'?1:+document.getElementById('alpha').value/100;ctx.drawImage(off,0,0);ctx.globalAlpha=1;}</script></html>'''
        # O visualizador tem dimensões fixas: impedir extensão silenciosa para outra ROI.
        if valid.shape != (400, 450):
            raise ValueError("Revisar dimensões do visualizador para outra ROI")
        (output / "INSPECAO_ROI.html").write_text(html.replace("RGB_URI", rgb_uri).replace("RGBA_URI", rgba_uri), encoding="utf-8")
        (output / "LEIA_ME.md").write_text("# Inspeção visual da ROI Abrolhos\n\nAbra `INSPECAO_ROI.html` no navegador. Funciona offline. Use Só imagem RGB, Só grupos ou Sobrepor; ajuste a opacidade e ligue/desligue cada grupo. A comparação estática está em `comparacao_roi.png`.\n\nModelo reutilizado: azul/verde, PCA95, K3, semente42, do primeiro experimento. Nenhum novo treinamento. Predição em blocos de 128 pixels somente na ROI de 450 × 400 pixels. A máscara e a UDM2 da preparação foram respeitadas. Zero no GeoTIFF significa Não Classificado. As contagens/áreas em `contagem_roi.csv` descrevem esta ROI e não cobertura de habitats.\n\nPara QGIS, abra `rgb_roi_visual.tif` e `grupos_espectrais_roi.tif`; as grades coincidem em EPSG:32724, resolução 3 m. O RGB tem contraste visual P2–P98 e não substitui as bandas científicas.\n\nNa inspeção, observe se os limites acompanham mudanças graduais de brilho, textura ou possíveis efeitos de superfície. Coincidência visual não identifica sua causa. Grupo Espectral 1, 2 e 3 não são classes ecológicas; a visibilidade do fundo não foi comprovada. Nenhuma imagem original ou produto ACOLITE foi modificado.\n\nVerificações: hashes do modelo e preparação, mesma população válida, predição em blocos comparada em 512 posições, classes/exclusões e leitura de retorno do GeoTIFF com conferência de CRS, transformação e pixels.\n", encoding="utf-8")
        shutil.copy2(__file__, output / Path(__file__).name)
        manifest.update(status="complete", checks_passed=True, valid_pixels=int(valid.sum()), rejected_pixels=int((~valid).sum()))
        manifest["artifacts_sha256"] = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}
    except Exception as error:
        manifest.update(status="failed", error=str(error)); raise
    finally:
        write_json(output / "parametros.json", manifest)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--recommendation", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Exige aplicação diagnóstica à ROI autorizada")
    print(generate(args.prepared, args.recommendation), flush=True)
