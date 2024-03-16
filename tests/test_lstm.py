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
import stocknet.train.rltrainer as rltrainer
from stocknet.nets.lstm import LSTM

file_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../finance_client/finance_client/data_source/mt5/OANDA-Japan MT5 Live/mt5_USDJPY_d1.csv")
)


class TestLSTMModel(unittest.TestCase):
    def test_training_lstm(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        processes = [fc.utils.MinMaxPreProcess(scale=(-1, 1))]
        ohlc = ["high", "low", "open", "close"]
        data_client = fc.CSVClient(file=file_path, frame=60 * 24, date_column="time", post_process=processes, columns=ohlc)
        batch_size = 32
        observationLength = 60
        epoc_num = 2
        hidden_layer_num = 4
        version = 1

        dataset = ds.ShiftDataset(
            data_client=data_client, observationLength=observationLength, in_columns=ohlc, out_columns=ohlc, shift=1, isTraining=True
        )
        train_dl = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=False, pin_memory=True)
        ds_val = ds.ShiftDataset(
            data_client=data_client, observationLength=observationLength, in_columns=ohlc, out_columns=ohlc, shift=1, isTraining=False
        )
        val_loader = DataLoader(ds_val, batch_size=batch_size, drop_last=True, shuffle=False, pin_memory=True)
        i, o = dataset[0]
        input_size = i.shape[1]

        model_name = f"test/LSTM-{str(hidden_layer_num)}_v{str(version)}"
        model = LSTM(inputDim=input_size, hiddenDim=hidden_layer_num, outputDim=o.shape[0], device=device)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
        loss_fn = nn.MSELoss()

        tr = rltrainer.Trainer(model_name, loss_fn, train_dl, val_loader, device)
        tr.training_loop(model, optimizer, epoc_num)


if __name__ == "__main__":
    unittest.main()
