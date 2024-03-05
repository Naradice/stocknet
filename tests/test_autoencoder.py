import datetime
import json
import os
import sys
import unittest

finance_client_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../finance_client"))
sys.path.append(finance_client_module_path)
import finance_client as fc

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(module_path)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import stocknet.datasets as ds
import stocknet.trainer as trainer
from stocknet.nets.ae import AELinearModel

file_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../finance_client/finance_client/data_source/mt5/OANDA-Japan MT5 Live/mt5_USDJPY_d1.csv")
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TestAELinearModel(unittest.TestCase):
    def test_training_auto_encoder(self):
        processes = [fc.utils.MinMaxPreProcess(scale=(-1, 1))]
        data_client = fc.CSVClient(
            file=file_path, frame=60 * 24, date_column="time", post_process=processes, columns=["high", "low", "open", "close"]
        )
        batch_size = 32
        observationLength = 60
        epoc_num = 2
        hidden_layer_num = 5
        middle_layer_size = 15
        version = 1

        dataset = ds.OHLCDataset(data_client=data_client, observationLength=observationLength, merge_columns=True, isTraining=False)
        train_dl = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=False, pin_memory=True)
        ds_val = ds.OHLCDataset(data_client=data_client, observationLength=observationLength, merge_columns=True, isTraining=False)
        val_loader = DataLoader(ds_val, batch_size=batch_size, drop_last=True, shuffle=False, pin_memory=True)
        i, o = dataset[0]
        input_size = i.shape[0]

        # model_name = 'bc_5min_ohlc_AE_v2'
        model_name = f"test/AE-{str(hidden_layer_num)}-{str(middle_layer_size).zfill(2)}_v{str(version)}"
        model = AELinearModel(input_size, hidden_layer_num=hidden_layer_num, middle_layer_size=middle_layer_size, device=device)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-6)
        # optimizer = optim.SGD(model.parameters(), lr=1e-6)
        loss_fn = nn.MSELoss()

        tr = trainer.Trainer(model_name, loss_fn, train_dl, val_loader, device)
        tr.training_loop(model, optimizer, epoc_num)


if __name__ == "__main__":
    unittest.main()
