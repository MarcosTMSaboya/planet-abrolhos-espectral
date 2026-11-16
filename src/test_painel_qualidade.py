"""Verifica a integração Streamlit bloqueando qualquer acesso a rasters."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class PanelTests(unittest.TestCase):
    def test_app_metrics_policies_and_audit_filters_without_rasters(self):
        with patch("rasterio.open", side_effect=AssertionError("Painel não deve abrir rasters")), \
                patch.object(Path, "rglob", side_effect=AssertionError("Painel não deve fazer varreduras")):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            self.assertFalse(app.exception, str(app.exception))
            self.assertFalse(app.error, str(app.error))
            self.assertEqual(app.radio(key="app_module").value, "Qualidade e assinaturas")
            self.assertEqual([m.value for m in app.metric[:5]], ["70", "58", "12", "0", "15"])
            app.radio(key="qc_policy").set_value("allow_haze").run()
            self.assertFalse(app.exception, str(app.exception))
            self.assertEqual(app.metric[1].value, "58")
            app.multiselect(key="qc_statuses").set_value(["rejected"]).run()
            self.assertFalse(app.exception, str(app.exception))
            self.assertTrue(any("12 de 70 observações" in c.value for c in app.caption))
            app.multiselect(key="qc_scenes").set_value(["20260213_125207_88_24d8"]).run()
            app.multiselect(key="qc_audit_points").set_value([0]).run()
            self.assertFalse(app.exception, str(app.exception))
            self.assertTrue(any("0 de 70 observações" in c.value for c in app.caption))
            app.selectbox(key="qc_point").set_value(13).run()
            self.assertFalse(app.exception, str(app.exception))
            self.assertEqual(app.metric[5].value, "1")

    def test_missing_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = "from src.painel_qualidade import render_quality\nrender_quality(" + repr(tmp) + ")"
            app = AppTest.from_string(script, default_timeout=30).run()
            self.assertFalse(app.exception, str(app.exception))
            self.assertTrue(any("Ainda não há resultados" in i.value for i in app.info))

    def test_invalid_and_empty_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "data/analise/udm2/test"
            run.mkdir(parents=True)
            (run / "manifesto.json").write_text(json.dumps({}), encoding="utf-8")
            audit = {"scene_id": "scene", "point_index": 0, "qc_strict": "accepted",
                     "qc_strict_reason": "all_9_pixels_usable"}
            audit.update({f"rhos_{w}_median": .02 for w in (444, 492, 533, 566, 612, 666, 707, 866)})
            tables = {
                "auditoria_udm2.csv": pd.DataFrame([audit]),
                "assinatura_algas_por_ponto_strict.csv": pd.DataFrame([{"point_index": 0, "n_scenes": 1}]),
                "resumo_espectral_algas_strict.csv": pd.DataFrame([{"wavelength_nm": 444, "median": .02, "q1": .01, "q3": .03}]),
                "assinatura_algas_long_strict.csv": pd.DataFrame([{"point_index": 0, "wavelength_nm": 444, "rhos_median": .02, "n_scenes": 1}]),
            }
            script = "from src.painel_qualidade import render_quality\nrender_quality(" + repr(tmp) + ")"
            for name, table in tables.items():
                table.iloc[:0].to_csv(run / name, index=False)
            app = AppTest.from_string(script, default_timeout=30).run()
            self.assertFalse(app.exception, str(app.exception))
            self.assertTrue(any("não contém observações" in i.value for i in app.info))
            for name, table in tables.items():
                table.to_csv(run / name, index=False)
            pd.DataFrame([{"point_index": 0}]).to_csv(run / "assinatura_algas_por_ponto_strict.csv", index=False)
            app.run()
            self.assertFalse(app.exception, str(app.exception))
            self.assertTrue(any("n_scenes" in e.value for e in app.error))


if __name__ == "__main__":
    unittest.main()
