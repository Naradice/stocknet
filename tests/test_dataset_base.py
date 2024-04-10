import json
import os
import sys
import unittest

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(module_path)

import finance_client

from stocknet.datasets import base, factory, utils


class TestBaseDataset(unittest.TestCase):
    params_file = "./test_base_params.json"
    src_file = "L://data/fx/OANDA-Japan MT5 Live/mt5_USDJPY_min30.csv"

    def test_01_get_params(self):
        processes = [finance_client.fprocess.DiffPreProcess(), finance_client.fprocess.MinMaxPreProcess()]
        columns = ["open", "high", "low", "close"]
        dataset = base.Dataset(source=self.src_file, columns=columns, observation_length=60, prediction_length=10, processes=processes)
        params = dataset.get_params()
        with open(self.params_file, "w") as fp:
            json.dump(params, fp)

    def test_02_load(self):
        with open(self.params_file, "r") as fp:
            params = json.load(fp)
        ds = base.Dataset(**params)
        os.remove(self.params_file)

    def test_03_common_save(self):
        processes = [finance_client.fprocess.DiffPreProcess(), finance_client.fprocess.MinMaxPreProcess()]
        columns = ["open", "high", "low", "close"]
        dataset = base.Dataset(source=self.src_file, columns=columns, observation_length=60, prediction_length=10, processes=processes)
        params = utils.dataset_to_params(dataset)
        with open(self.params_file, "w") as fp:
            json.dump(params, fp)

    def test_04_common_load(self):
        with open(self.params_file, "r") as fp:
            params = json.load(fp)
        ds, batch_sizes, version_preffix = factory.load_seq2seq_datasets(params, None)
        os.remove(self.params_file)


if __name__ == "__main__":
    unittest.main()
