import unittest
import numpy as np
from comparar_feicoes_glint import summarize,unit_spectra


class SpectralComparisonTests(unittest.TestCase):
    def test_normalization_separates_magnitude_from_shape(self):
        spectra=np.array([[1.,2.,3.],[2.,4.,6.]])
        normalized=unit_spectra(spectra)
        np.testing.assert_allclose(normalized[0],normalized[1])
        self.assertNotEqual(summarize(spectra)['median'][0],summarize(normalized)['median'][0])
        signed=np.array([[-1.,2.],[-2.,4.]])
        self.assertTrue((unit_spectra(signed)[:,0]<0).all())

    def test_undefined_spectra_rejected(self):
        with self.assertRaises(ValueError):unit_spectra(np.zeros((2,6)))
        with self.assertRaises(ValueError):summarize(np.array([[np.nan,1.],[2.,3.]]))


if __name__=='__main__':unittest.main()
