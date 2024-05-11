import glob
import json
import os
import sys
import unittest

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(module_path)

from stocknet.utils import generate_args

base_folder = os.path.dirname(__file__)


class TestTrainer(unittest.TestCase):
    params_file = f"{base_folder}/train_params/test_sequential_param.json"

    def test_generate_args(self):
        with open(self.params_file, "r") as fp:
            params = json.load(fp)
        ds_params = params["dataset"]
        args_list = generate_args(ds_params)
        for args in args_list:
            isinstance(args, dict)
            self.assertEqual(type(args["columns"]), list)
            self.assertEqual(type(args["processes"]), list)
            self.assertEqual(type(args["observation"]), int)
            self.assertEqual(type(args["prediction"]), int)


if __name__ == "__main__":
    unittest.main()
