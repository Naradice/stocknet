import json
import os
import sys
import unittest

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(module_path)

import torch

from stocknet.trainer import factory, utils


class TestBaseDataset(unittest.TestCase):
    params_file = "./test_train_params.json"

    def test_01_common_save(self):
        params = {}
        test_model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(test_model.parameters(), lr=0.5)
        criterion = torch.nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        params = utils.tainer_options_to_params(optimizer, criterion, scheduler, epoc=300, device="cuda", patience=2)
        with open(self.params_file, "w") as fp:
            json.dump(params, fp)

    def test_02_common_load(self):
        test_model = torch.nn.Linear(10, 1)
        with open(self.params_file, "r") as fp:
            params = json.load(fp)
        optimizer, criterion, scheduler = factory.load_trainer_options(test_model, params)
        self.assertIsNotNone(optimizer)
        self.assertIsNotNone(criterion)
        self.assertIsNotNone(scheduler)
        os.remove(self.params_file)


if __name__ == "__main__":
    unittest.main()
