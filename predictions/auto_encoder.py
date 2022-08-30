import sys, os

finance_client_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../finance_client'))
sys.path.append(finance_client_module_path)

import finance_client as fc
import finance_client.utils.idcprocess as indicater

stocknet_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(stocknet_module_path)
import stocknet.datasets as ds
from stocknet.nets.ae import AELinearModel
import finance_client as fc
import stocknet.trainer as trainer

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from trainer import training_lenear_auto_encoder
    
def auto_encoder_macd(file, client_columns, client_date_column, frame, target_column="Close", epoc_num=-1, version=1):
    macd_ps = fc.utils.MACDpreProcess(target_column=target_column)
    processes = [fc.utils.MinMaxPreProcess(scale=(-1,1))]
    data_client = fc.CSVClient(file=file, frame=frame, columns=client_columns,date_column=client_date_column, idc_processes=[macd_ps], post_process=processes)
    columns = []
    for key, value in macd_ps.columns.items():
        columns.append(value)
        
    ##hyper parameters##
    observationLength = 7
    batch_size = 32
    hidden_layer_num = 5
    middle_layer_size = 12#should be less than data length * column num
    ####################
    
    model_name = f'macd/AE-{str(hidden_layer_num)}-{str(middle_layer_size).zfill(2)}_v{str(version)}'
    dataset = ds.Dataset(data_client=data_client, observationLength=observationLength, in_columns=columns, out_columns=columns, merge_columns=True, isTraining=True)
    training_lenear_auto_encoder(dataset=dataset, batch_size=batch_size, hidden_layer_num=hidden_layer_num, middle_layer_size=middle_layer_size, version=version, epoc_num=epoc_num, model_name=model_name)
    
def auto_encoder_ohlc(file, client_columns, client_date_column, frame, epoc_num=-1, version=1):
    processes = [fc.utils.MinMaxPreProcess(scale=(-1,1))]
    data_client = fc.CSVClient(file=file, frame=frame, columns=client_columns, date_column=client_date_column, post_process=processes)
        
    ##hyper parameters##
    observationLength = 1
    batch_size = 32
    hidden_layer_num = 5
    middle_layer_size = 4#should be less than data length * column num
    ####################
    
    model_name = f'ohlc/AE-{str(hidden_layer_num)}-{str(middle_layer_size).zfill(2)}_v{str(version)}'
    #dataset = ds.Dataset(data_client=data_client, observationLength=observationLength, in_columns=columns, out_columns=columns, merge_columns=True, isTraining=True)
    dataset = ds.OHLCDataset(data_client=data_client, observationLength=observationLength, merge_columns=True, isTraining=True)
    training_lenear_auto_encoder(dataset=dataset, batch_size=batch_size, hidden_layer_num=hidden_layer_num, middle_layer_size=middle_layer_size, version=version, epoc_num=epoc_num, model_name=model_name)
    
if __name__ == "__main__":
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../finance_client/finance_client/data_source/mt5/OANDA-Japan MT5 Live/mt5_USDJPY_d1.csv'))
    client_column = ["open", "high", "low", "close"]
    date_column = "time"
    frame=60*24
    version = 1
    #auto_encoder_macd(file_path, client_column, date_column, frame, "close", -1, version)
    auto_encoder_ohlc(file_path, client_column, date_column, frame, -1, version)
