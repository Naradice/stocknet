import os
import sys

module_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(module_path)
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(module_path)

from custom.dataset.cluster import ClusterDistDataset

src_file = "L://data/fx/OANDA-Japan MT5 Live/mt5_USDJPY_min30.csv"
columns = ["open", "high", "low", "close"]

ds = ClusterDistDataset(src_file, columns, label_num_k=30, freq=30)
src, tgt, mask = ds[0:16]
src = src
tgt = tgt

print(src.shape, tgt.shape, mask.shape)
