import unittest
import numpy as np
from diagnostico_glint_usuario import fit_nir,apply_trial


class GlintTests(unittest.TestCase):
    def test_known_slopes_and_signed_correction(self):
        nir=np.linspace(.01,.04,100)
        beta=np.arange(1,7)/10
        values=np.zeros((100,8));values[:,7]=nir
        values[:,:6]=.02+nir[:,None]*beta
        fit,_,r2=fit_nir(values)
        np.testing.assert_allclose(fit,beta,atol=1e-12)
        np.testing.assert_allclose(r2,1)
        np.testing.assert_allclose(apply_trial(values,fit,.01),np.tile(.02+.01*beta,(100,1)))
        low=values.copy();low[:,0]=-.02
        self.assertTrue((apply_trial(low,beta,.01)[:,0]<0).all())

    def test_constant_nir_is_rejected(self):
        with self.assertRaises(ValueError):fit_nir(np.ones((50,8)))


if __name__=='__main__':unittest.main()
