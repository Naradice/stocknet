import unittest
import random
from stocknet.envs.datasets.fx_macd_ds import FXMACDDataset

class TestFXEnv(unittest.TestCase):

    def test_initialization(self):
        ds = FXMACDDataset()

if __name__ == '__main__':
    unittest.main()