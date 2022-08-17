import os, sys
from trainer import training_lstm_model

finance_client_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../finance_client'))
sys.path.append(finance_client_module_path)

import finance_client as fc
import finance_client.utils.idcprocess as indicater

def ohlc_trend():
    ohlc_column = ('Open','High','Low','Close')
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../finance_client/finance_client/data_source/csv/USDJPY_forex_min30.csv'))
    ##hyper parameters##
    observationLength = 12*24
    in_column = ohlc_column
    
    renko_ps = indicater.RenkoProcess(date_column="Time", ohlc_column=ohlc_column)
    ids = [renko_ps]
    out_column = (renko_ps.columns["TREND"],)
    #processes = [process.DiffPreProcess(), process.MinMaxPreProcess(scale=(-1,1))]
    processes = [fc.utils.MinMaxPreProcess(scale=(-1,1))]
    data_client = fc.CSVClient(file=file_path, frame=30, columns=ohlc_column, date_column="Time", idc_processes=ids, post_process=processes)
    batch_size = 32
    hidden_layer_size = 5
    middle_layer_size = 96
    ####################
    #epoc_num = 50
    version = 3
    
    
    training_lstm_model(
        data_client=data_client,
        batch_size=batch_size,
        observationLength=observationLength,
        in_column=in_column,
        out_column=out_column,
        hidden_layer_num=hidden_layer_size,
        middle_layer_size=middle_layer_size,
        version=version
    )

if __name__ == "__main__":
    ohlc_trend()