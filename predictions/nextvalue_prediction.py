import os, sys

from trainer import training_lstm_model
import torch.nn as nn
import torch

finance_client_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../finance_client'))
sys.path.append(finance_client_module_path)
stocknet_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(stocknet_module_path)

import finance_client as fc
import finance_client.utils.idcprocess as indicater
import stocknet.datasets as ds

file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../finance_client/finance_client/data_source/mt5/OANDA-Japan MT5 Live/mt5_USDJPY_d1.csv'))
#file_path = 'L:\data\mt5\OANDA-Japan MT5 Live\mt5_USDJPY_d1.csv'

def next_ohlc_individual(epoc=-1, h_layer_sizes = [1,2,4,8], target_column = "close"):
    #ema_ps_2 = indicater.EMApreProcess(key='e26', window=12, is_output=is_multi_output)
    shift=1
    for h_layer in h_layer_sizes:
        processes = [fc.utils.MinMaxPreProcess(scale=(-1,1))]
        learning_target_columns = [target_column]
        data_client = fc.CSVClient(files=file_path, frame=60*24, date_column="time", columns=['open','high', 'low','close'])
        
        ##hyper parameters##
        observationDays = 60
        batch_size = 32
        hidden_layer_size = h_layer
        ####################
        #epoc_num = 50
        version = 3
        
        dataset = ds.ShiftDataset(data_client, observationLength=observationDays, pre_processes=processes, in_columns=learning_target_columns, out_columns=learning_target_columns, shift=shift)
        model_name = f'next_ohlc/{str(observationDays)}d_{str(shift)}_LSTM{str(hidden_layer_size)}/{target_column}_v{str(version)}'
        
        training_lstm_model(dataset=dataset, batch_size=batch_size, hidden_layer_num=hidden_layer_size, version=version, epoc_num=epoc, model_name=model_name)

def next_ohlc(epoc=-1, h_layer_sizes = [1,2,4,8, 16], target_columns = ["open", "high", "low", "close"]):
    #ema_ps_2 = indicater.EMApreProcess(key='e26', window=12, is_output=is_multi_output)
    shift = 1
    for h_layer in h_layer_sizes:
        #processes = [fc.utils.DiffPreProcess(), fc.utils.MinMaxPreProcess(scale=(-1,1))]
        processes = [fc.utils.MinMaxPreProcess(scale=(-1,1))]
        learning_target_columns = target_columns

        data_client = fc.CSVClient(files=file_path, frame=60*24, date_column="time", columns=['high', 'low','open','close'])
        ##hyper parameters##
        observationDays = 60
        batch_size = 32
        hidden_layer_size = h_layer
        ####################
        #epoc_num = 50
        version = 3.1
        model_name = f'next_ohlc/{str(observationDays)}d_diff_{str(shift)}_LSTM{str(hidden_layer_size)}_v{str(version)}'
        
        dataset = ds.ShiftDataset(data_client, observationLength=observationDays, pre_processes=processes,in_columns=learning_target_columns, out_columns=learning_target_columns, shift=shift)
        training_lstm_model(dataset=dataset, batch_size=batch_size, hidden_layer_num=hidden_layer_size, version=version, epoc_num=epoc, model_name=model_name)

def next_ema(epoc=-1, windows = [1,2,4,8,16], target_columns = ["open", "high", "low", "close"]):
    shift=2
    for window in windows:
        ids = [indicater.EMAProcess(key=column, column=column, window=window) for column in target_columns]
        processes = [fc.utils.MinMaxPreProcess(scale=(-1,1))]
        learning_target_columns = [ indicater.columns["EMA"] for indicater in ids]

        data_client = fc.CSVClient(files=file_path, frame=60*24, date_column="time", columns=['open', "high", "low",'close'])
        ##hyper parameters##
        observationDays = 60
        #processes = [process.DiffPreProcess(), process.MinMaxPreProcess(scale=(-1,1))]
        batch_size = 32
        hidden_layer_size = 5
        ####################
        #epoc_num = 50
        version = 3
        model_name = f'next_mean/{str(observationDays)}d_window{str(window)}_{str(shift)}_LSTM{str(hidden_layer_size)}_v{str(version)}'
        
        dataset = ds.ShiftDataset(data_client, observationLength=observationDays,idc_processes=ids, pre_processes=processes, in_columns=learning_target_columns, out_columns=learning_target_columns, shift=shift)
        training_lstm_model(dataset=dataset, batch_size=batch_size, hidden_layer_num=hidden_layer_size, version=version, epoc_num=epoc, model_name=model_name)

def next_nv():
    macd_ps = fc.utils.MACDProcess(target_column="close")
    idc_process = [macd_ps]
    processes = [fc.utils.DiffPreProcess(), fc.utils.MinMaxPreProcess(scale=(-1,1))]
    data_client = fc.CSVClient(files=file_path, frame=24*60, date_column="time", columns=['open','high', 'low', 'close'])
    columns = []
    for key, value in macd_ps.columns.items():
        columns.append(value)

    ##hyper parameters##
    observationLength = 12*24
    batch_size = 32
    hidden_layer_size = 5
    ####################
    epoc_num = 50
    shift = 3
    version = 3
    model_name = f'next_macd/{str(observationLength)}m_{str(shift)}_LSTM{str(hidden_layer_size)}_v{str(version)}'
    
    dataset = ds.ShiftDataset(data_client, observationLength=observationLength,idc_processes=idc_process,pre_processes=processes, in_columns=columns, out_columns=columns, shift=shift)
    training_lstm_model(dataset=dataset, batch_size=batch_size, hidden_layer_num=hidden_layer_size, version=version, epoc_num=epoc_num, model_name=model_name)

def next_high_low(epoc=-1, h_layer_sizes = [2,4,8,16], target_columns = ["open", "high", "low", "close"]):
    for h_layer in h_layer_sizes:
        #processes = [fc.utils.DiffPreProcess(), fc.utils.MinMaxPreProcess(scale=(-1,1))]
        processes = [fc.utils.MinMaxPreProcess(scale=(0,1))]
        learning_target_columns = target_columns

        data_client = fc.CSVClient(files=file_path, frame=60*24, date_column="time", columns=['high', 'low','open','close'])
        ##hyper parameters##
        observationDays = 60
        batch_size = 32
        hidden_layer_size = h_layer
        ####################
        #epoc_num = 50
        version = 3.3
        loss_fn = nn.BCELoss()
        activate_function = nn.Sigmoid()
        model_name = f'next_hl/{str(observationDays)}d_LSTM{str(hidden_layer_size)}_v{str(version)}'
        
        dataset = ds.HighLowDataset(data_client, observationLength=observationDays, pre_processes=processes, in_columns=learning_target_columns, out_columns=learning_target_columns, compare_with="close", merge_columns=False)
        training_lstm_model(dataset=dataset, batch_size=batch_size, hidden_layer_num=hidden_layer_size, 
        version=version, epoc_num=epoc, model_name=model_name,
        loss_fn=loss_fn, activation_for_output=activate_function)

def next_high_low_possibility(epoc=-1, h_layer_sizes = [2,4,8,16], target_columns = ["open", "high", "low", "close"]):
    for h_layer in h_layer_sizes:
        processes = [fc.utils.MinMaxPreProcess(scale=(0,1))]
        learning_target_columns = target_columns

        data_client = fc.CSVClient(files=file_path, frame=60*24, date_column="time", columns=['high', 'low','open','close'])
        ##hyper parameters##
        observationDays = 60
        batch_size = 32
        hidden_layer_size = h_layer
        ####################
        version = 3.3
        loss_fn = nn.MSELoss()
        activate_function = nn.Softmax(dim=0)
        model_name = f'next_hl/{str(observationDays)}d_LSTM{str(hidden_layer_size)}_v{str(version)}'
        
        dataset = ds.HighLowDataset(data_client, observationLength=observationDays, in_columns=learning_target_columns, out_columns=learning_target_columns, compare_with="close", merge_columns=False)
        training_lstm_model(dataset=dataset, batch_size=batch_size, hidden_layer_num=hidden_layer_size, 
        version=version, epoc_num=epoc, model_name=model_name,
        loss_fn=loss_fn, activation_for_output=activate_function)


if __name__ == "__main__":
    next_high_low_possibility(epoc=10, h_layer_sizes=[8])