import json
import os
import sys
import unittest

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(module_path)

import finance_client
from torch import nn

from stocknet.nets import PositionalEncoding, Seq2SeqTransformer, factory, utils


class TestBaseDataset(unittest.TestCase):
    params_file = "./test_tf_model_params.json"

    def test_01_common_save(self):
        vocab_size = 1000
        d_model = 30
        output_layer = nn.Linear(d_model, vocab_size)
        pe = PositionalEncoding(d_model)

        model = Seq2SeqTransformer(
            num_decoder_layers=1,
            num_encoder_layers=1,
            d_model=d_model,
            positional_encoding=pe,
            output_layer=output_layer,
            dim_feedforward=10,
            nhead=6,
        )
        params = utils.model_to_params(model)
        with open(self.params_file, "w") as fp:
            json.dump(params, fp)

    def test_02_common_load(self):
        with open(self.params_file, "r") as fp:
            params = json.load(fp)
        model = factory.load_a_model(params=params)
        os.remove(self.params_file)


if __name__ == "__main__":
    unittest.main()
