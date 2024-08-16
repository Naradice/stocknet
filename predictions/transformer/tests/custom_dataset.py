import json
import math
import os
import sys
import unittest

import pandas as pd
import torch

try:
    import stocknet
except ImportError:
    stocknet_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    sys.path.append(stocknet_module_path)
    import stocknet

stocknet_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(stocknet_module_path)

from custom.dataset.cluster import ClusterIDDataset

base_path = os.path.dirname(__file__)


class TestDiffIDDataset(unittest.TestCase):
    def test_load_ds(self):
        df = pd.read_csv("L:/data/fx/OANDA-Japan MT5 Live/mt5_USDJPY_min30.csv", parse_dates=True, index_col=0)
        ds = ClusterIDDataset(df, columns=["open", "high", "low", "close"])
        src, tgt, options = ds[0:32]
        print(src.shape)


if __name__ == "__main__":
    unittest.main()
