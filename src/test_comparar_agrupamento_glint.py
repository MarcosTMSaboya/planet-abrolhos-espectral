import unittest
import numpy as np
from comparar_agrupamento_glint import local_positions


class PairedPositionTests(unittest.TestCase):
    def test_same_pixel_positions_and_exclusive_limits(self):
        w={'row':3900,'col':3900}
        r,c=local_positions(np.array([[3900,3900],[4299,4349]]),w,(400,450))
        np.testing.assert_array_equal(r,[0,399]);np.testing.assert_array_equal(c,[0,449])
        for bad in ([[3899,3900]],[[4300,3900]],[[3900,4350]]):
            with self.assertRaises(ValueError):local_positions(bad,w,(400,450))


if __name__=='__main__':unittest.main()
