import json
import math
import os
import sys
import unittest

import torch

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(module_path)

from stocknet.datasets import factory, id, utils

base_path = os.path.dirname(__file__)


class TestDiffIDDataset(unittest.TestCase):
    params_file = f"{base_path}/train_params/test_diffid_params.json"
    src_file = "L://data/fx/OANDA-Japan MT5 Live/mt5_USDJPY_min30.csv"
    observation_length = 30
    prediction_length = 5

    def __default_init(self):
        device = "cpu"
        columns = "close"
        dataset = id.DiffIDDS(self.src_file, columns, self.observation_length, self.prediction_length, device)
        return dataset

    def test_00_close_id(self):
        batch_size = 16
        device = "cpu"
        ds = self.__default_init()
        self.assertGreater(ds.vocab_size, 0)
        src, tgt = ds[0:batch_size]
        self.assertEqual(src.shape, (batch_size, self.observation_length))
        self.assertEqual(tgt.shape, (batch_size, self.prediction_length))

        d_model = int(math.sqrt(ds.vocab_size))
        emb_layer = torch.nn.Embedding(ds.vocab_size, d_model, device=device)
        emb_src = emb_layer(src)
        self.assertEqual(emb_src.shape, (batch_size, self.observation_length, d_model))

    def test_10_get_params(self):
        dataset = self.__default_init()
        params = dataset.get_params()
        self.assertTrue(isinstance(params, dict))

    def test_11_common_save(self):
        dataset = self.__default_init()
        params = utils.dataset_to_params(dataset)
        with open(self.params_file, "w") as fp:
            json.dump(params, fp)

    def test_12_load(self):
        with open(self.params_file, "r") as fp:
            params = json.load(fp)
        datasets = factory.load_seq2seq_datasets(params, None)
        ds, batch_sizes, version_preffix = next(datasets)


if __name__ == "__main__":
    unittest.main()
