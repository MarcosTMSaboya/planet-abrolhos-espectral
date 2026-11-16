import unittest
import numpy as np
from preparar_validacao import transition_mask,keep_smallest


class ValidationDesignTests(unittest.TestCase):
    def test_transition_ignores_nodata_and_uses_other_valid_class(self):
        halo=np.array([[0,0,0,0],[0,1,1,0],[0,1,2,0],[0,0,0,0]])
        self.assertTrue(transition_mask(halo).all())
        halo[2,2]=1
        self.assertFalse(transition_mask(halo).any())

    def test_reservoir_preserves_global_lowest_priorities_across_chunks(self):
        empty=(np.empty(0),np.empty((0,2),dtype='int64'))
        priorities=np.array([.8,.1,.5,.9,.2,.3]);positions=np.column_stack((np.arange(6),np.arange(6)))
        first=keep_smallest(empty,priorities[:3],positions[:3],3)
        result=keep_smallest(first,priorities[3:],positions[3:],3)
        self.assertEqual(set(result[1][:,0]),{1,4,5})


if __name__=='__main__':unittest.main()
