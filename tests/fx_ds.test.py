import unittest
import random
from stocknet.envs.datasets.fx_ds import FXDataset

class TestFXEnv(unittest.TestCase):

    ds = FXDataset()

    def test_get_1_index(self):
        index = random.randint(0,len(self.ds)-1)
        i, o = self.ds[index]
        for index in range(0, len(i)):
            self.assertEqual(i[index],o[index])


if __name__ == '__main__':
    unittest.main()