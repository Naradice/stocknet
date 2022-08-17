import os, sys

from trainer import training_lstm_model, training_lstm_with_shift

finance_client_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../finance_client'))
sys.path.append(finance_client_module_path)

import finance_client as fc
import finance_client.utils.idcprocess as indicater

#file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../finance_client/finance_client/data_source/csv/USDJPY_forex_min30.csv'))
file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../finance_client/finance_client/data_source/mt5/OANDA-Japan MT5 Live/mt5_USDJPY_d1.csv'))

def next_ema(epoc=-1, windows = [1,2,4,8,16,32], target_columns = ["close"]):
    for window in windows:
        ids = [indicater.EMApreProcess(key=column, column=column, window=window) for column in target_columns]
        processes = [fc.utils.MinMaxPreProcess(scale=(-1,1))]
        learning_target_columns = [ indicater.columns["EMA"] for indicater in ids]

        data_client = fc.CSVClient(file=file_path, frame=60*24, idc_processes=ids,  date_column="time", post_process=processes, columns=['high', 'low','open','close'])
        ##hyper parameters##
        observationDays = 30
        #processes = [process.DiffPreProcess(), process.MinMaxPreProcess(scale=(-1,1))]
        batch_size = 32
        hidden_layer_size = 5
        middle_layer_size = 96
        ####################
        #epoc_num = 50
        version = 2
        
        training_lstm_with_shift(data_client, batch_size, observationDays, shift=1, in_column=learning_target_columns, out_column=learning_target_columns, hidden_layer_num=hidden_layer_size, middle_layer_size=middle_layer_size, version=version, epoc_num=epoc)

def next_nv():
    macd_ps = fc.utils.MACDpreProcess()
    idc_process = [macd_ps]
    data_client = fc.CSVClient(file=file_path, frame=30, idc_processes=idc_process)
    ##hyper parameters##
    observationDays = 1
    processes = [fc.utils.DiffPreProcess(), fc.utils.MinMaxPreProcess(scale=(-1,1))]
    batch_size = 32
    hidden_layer_size = 5
    middle_layer_size = 96
    ####################
    #epoc_num = 50
    shift = 3
    version = 3
    
    training_lstm_with_shift(data_client, batch_size, observationDays, processes, shift=shift, hidden_layer_num=hidden_layer_size, middle_layer_size=middle_layer_size, version=version)

if __name__ == "__main__":
    next_ema()
