import glob
import json
import os
import sys
import unittest

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(module_path)

import stocknet


class TestTransformerTrainer(unittest.TestCase):
    params_file = "./train_params/test_tf_param.json"

    def test_01_seq2seq_train(self):
        stocknet.train_from_config(self.params_file)


if __name__ == "__main__":
    unittest.main()
