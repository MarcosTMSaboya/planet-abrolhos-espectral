"""Extrai atributos NetCDF, configurações e alertas dos logs sem ler imagens."""
from datetime import datetime, timezone
import json

import rasterio

from agrupamento_espectral import sha256
from continuidade_espectral import read_settings
from espectral_io import ROOT
from experimento_espectral import write_json


def run():
    output = ROOT / "outputs/agrupamento_piloto" / ("proveniencia_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(exist_ok=False)
    records = []
    current = read_settings(ROOT / "src/acolite_settings_abrolhos.txt")
    for folder in sorted((ROOT / "data/acolite/output").iterdir()):
        if not folder.is_dir():
            continue
        record = {"folder": str(folder), "settings": [], "netcdf": [], "logs": []}
        for path in sorted(folder.glob("*l2r_settings.txt")):
            settings = read_settings(path)
            keys = ("dsf_aot_estimate", "dsf_spectrum_option", "dsf_residual_glint_correction", "inputfile", "runid")
            record["settings"].append({"path": str(path), "sha256": sha256(path),
                "selected": {key: settings.get(key) for key in keys},
                "differences_from_current": {key: {"current": value, "effective": settings.get(key)}
                                             for key, value in current.items() if settings.get(key) != value}})
        for path in sorted(folder.glob("*_L2R.nc")):
            item = {"path": str(path), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            try:
                with rasterio.open(path) as src:
                    item["tags"] = src.tags()
                    item["subdatasets"] = src.subdatasets
            except rasterio.errors.RasterioIOError as error:
                item["read_error"] = str(error)
            record["netcdf"].append(item)
        for path in sorted(folder.glob("*log_file.txt")):
            text = path.read_text(encoding="utf-8", errors="replace")
            markers = ("credentials", "No ozone", "Selected model", "Running ACOLITE", "Run ID", "Image file", "Created ")
            record["logs"].append({"path": str(path), "sha256": sha256(path),
                "evidence": [line for line in text.splitlines() if any(m in line for m in markers)]})
        records.append(record)
    write_json(output / "auditoria.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "headers_settings_logs_only", "radiometric_quality_validated": False,
        "netcdf_content_hash_computed": False, "records": records})
    report = "# Proveniência ACOLITE\n\nLeitura de atributos globais, configurações L2R e trechos de logs; sem reprocessamento.\n\n"
    for record in records:
        report += f"## Pasta {record['folder'].split(chr(92))[-1]}\n\n"
        for settings in record["settings"]:
            report += "```json\n" + json.dumps(settings["selected"], indent=2, ensure_ascii=False) + "\n```\n\n"
        for log in record["logs"]:
            report += "```text\n" + "\n".join(log["evidence"]) + "\n```\n\n"
    report += ("## Limites\n\nConfiguração registrada e presença de um NetCDF não provam qualidade radiométrica. "
        "Conferir run ID, horário, sensor e cena nos atributos em auditoria.json e nos logs. "
        "Ausência de dado auxiliar exige conferir valores efetivamente usados antes de decidir por reprocessamento. "
        "Correção residual de glint desativada não quantifica glint na ROI.\n")
    (output / "RELATORIO.md").write_text(report, encoding="utf-8")
    return output


if __name__ == "__main__":
    print(run())
