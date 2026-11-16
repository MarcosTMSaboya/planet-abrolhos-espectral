from pathlib import Path
import tempfile
import unittest
import json
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin
from src.cena_completa import export_reclassification


class ReclassificationTests(unittest.TestCase):
    def test_invalid_polygon_rejected_before_export(self):
        bowtie={'type':'Feature','geometry':{'type':'Polygon','coordinates':[[[-39,-17],[-38.98,-17.02],[-38.98,-17],[-39,-17.02],[-39,-17]]]},'properties':{'class_id':1}}
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,'inválido'):
                export_reclassification(Path(folder),1,{1:1,2:2,3:3},[bowtie],{},'test')

    def test_app_add_polygon_and_export_version(self):
        from streamlit.testing.v1 import AppTest
        from src import reclassificacao_app as ui
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/'outputs/cena_completa/fixture';run.mkdir(parents=True)
            with rasterio.open(run/'grupos_cena_completa.tif','w',driver='GTiff',width=2,height=2,count=4,
                    dtype='uint8',crs='EPSG:4326',transform=from_origin(-39,-17,.01,.01),nodata=0) as dst:
                dst.write(np.ones((4,2,2),dtype='uint8'))
            (run/'manifesto.json').write_text(json.dumps({'status':'complete','scene_id':'synthetic','sources':[{'path':'unused'}]*8,
                'outside_nir_fit_fraction':0,'nir_reference':.01}),encoding='utf-8')
            geometry={'type':'Polygon','coordinates':[[[-39,-17],[-38.98,-17],[-38.98,-17.02],[-39,-17.02],[-39,-17]]]}
            with patch.object(ui,'preview',return_value=(np.ones((2,2),dtype='uint8'),np.full((2,2,4),255,dtype='uint8'),[[-17.02,-39],[-17,-38.98]])), \
                 patch.object(ui,'st_folium',return_value={'last_active_drawing':{'type':'Feature','geometry':geometry}}):
                app=AppTest.from_string('from src.reclassificacao_app import render\nrender('+repr(str(root))+')',default_timeout=30).run()
                self.assertFalse(app.exception,str(app.exception))
                app.selectbox(key='target_fixture_b4').set_value(2)
                app.text_input(key='evidence_fixture_b4').set_value('Synthetic polygon review')
                app.button(key='add_fixture_b4').click().run()
                self.assertFalse(app.exception,str(app.exception))
                app.text_area(key='notes_fixture_b4').set_value('Synthetic tester; no field validation')
                app.button(key='export_fixture_b4').click().run()
                self.assertFalse(app.exception,str(app.exception))
                exports=list(run.glob('reclassificacoes/*/reclassificacao_provisoria.tif'))
                self.assertEqual(len(exports),1)
                with rasterio.open(exports[0]) as src:np.testing.assert_array_equal(src.read(1),np.full((2,2),2))

    def test_group_merge_polygon_override_and_nodata_preservation(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder);source=run/'grupos_cena_completa.tif'
            data=np.array([[0,1,2],[1,2,3]],dtype='uint8')
            with rasterio.open(source,'w',driver='GTiff',width=3,height=2,count=4,dtype='uint8',crs='EPSG:4326',
                    transform=from_origin(-39,-17,.01,.01),nodata=0) as dst:
                for b in range(1,5):dst.write(data,b)
            feature={'type':'Feature','geometry':{'type':'Polygon','coordinates':[[[-39,-17],[-38.98,-17],[-38.98,-17.02],[-39,-17.02],[-39,-17]]]},
                'properties':{'class_id':3,'evidence':'synthetic review'}}
            out=export_reclassification(run,2,{1:1,2:1,3:2},[feature],{1:'A',2:'B',3:'C'},'synthetic')
            with rasterio.open(out/'reclassificacao_provisoria.tif') as src:actual=src.read(1)
            np.testing.assert_array_equal(actual,[[0,3,1],[3,3,2]])
            with rasterio.open(source) as src:np.testing.assert_array_equal(src.read(2),data)
            self.assertEqual(json.loads((out/'revisao.json').read_text(encoding='utf-8'))['status'],'complete')

    def test_exclusion_polygon_uses_zero_without_creating_data(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)
            with rasterio.open(run/'grupos_cena_completa.tif','w',driver='GTiff',width=2,height=2,count=4,
                    dtype='uint8',crs='EPSG:4326',transform=from_origin(-39,-17,.01,.01),nodata=0) as dst:
                dst.write(np.ones((4,2,2),dtype='uint8'))
            f={'type':'Feature','geometry':{'type':'Polygon','coordinates':[[[-39,-17],[-38.98,-17],[-38.98,-17.02],[-39,-17.02],[-39,-17]]]},'properties':{'class_id':0}}
            out=export_reclassification(run,1,{1:1,2:2,3:3},[f],{},'test')
            with rasterio.open(out/'reclassificacao_provisoria.tif') as src:self.assertEqual(src.read(1).sum(),0)


if __name__=='__main__':unittest.main()
