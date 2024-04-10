import glob
import json
import os
import sys
import unittest

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(module_path)

import stocknet


class TestTrainer(unittest.TestCase):
    params_file = "./train_params/test_param.json"

    def test_01_seq2seq_train(self):
        stocknet.train_from_config(self.params_file)

    def test_02_reproduce_train(self):
        with open(self.params_file, "r") as fp:
            params = json.load(fp)
        model_name = params["model"]["model_name"]
        log_param_file = f"./logs/{model_name}/{model_name}*_params.json"
        params_files = glob.glob(log_param_file)
        params_file = params_files[0]
        stocknet.train_from_config(params_file)


if __name__ == "__main__":
    unittest.main()
