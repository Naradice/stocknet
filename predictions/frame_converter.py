import os
import sys

import torch
import torch.nn as nn
from trainer import training_lstm_model

finance_client_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../finance_client"))
sys.path.append(finance_client_module_path)
stocknet_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(stocknet_module_path)

import finance_client as fc
import finance_client.utils.idcprocess as indicater
import stocknet.datasets as ds

file_path = os.path.abspath("L:/data/mt5/OANDA-Japan MT5 Live/mt5_USDJPY_min5.csv")


def roll_down(epoc=-1, h_layer_sizes=[4, 8, 16], target_columns=["open", "high", "low", "close"]):
    # ema_ps_2 = indicater.EMApreProcess(key='e26', window=12, is_output=is_multi_output)
    for h_layer in h_layer_sizes:
        # processes = [fc.utils.DiffPreProcess(), fc.utils.MinMaxPreProcess(scale=(-1,1))]
        processes = [fc.utils.MinMaxPreProcess(scale=(-1, 1))]
        learning_target_columns = target_columns

        data_client = fc.CSVClient(files=file_path, frame=5, date_column="time", columns=["open", "high", "low", "close"])
        # hyper parameters
        observationDays = 60
        batch_size = 32
        hidden_layer_size = h_layer
        # epoc_num = 50
        version = 3.1
        model_name = f"roll_down/{str(observationDays)}d_LSTM{str(hidden_layer_size)}_v{str(version)}"

        dataset = ds.FrameConvertDataset(
            data_client,
            observationLength=observationDays,
            pre_processes=processes,
            in_frame=30,
            out_frame=15,
            in_columns=learning_target_columns,
            out_columns=["close"],
        )
        training_lstm_model(
            dataset=dataset, batch_size=batch_size, hidden_layer_num=hidden_layer_size, version=version, epoc_num=epoc, model_name=model_name
        )


if __name__ == "__main__":
    roll_down(epoc=-1, h_layer_sizes=[16])
