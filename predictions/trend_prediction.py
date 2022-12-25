import os
import sys

from trainer import training_lstm_model

finance_client_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../finance_client'))
sys.path.append(finance_client_module_path)

import finance_client as fc
import finance_client.utils.idcprocess as indicater


def renko_trend():
    ohlc_column = ('Open','High','Low','Close')
    file_path = os.path.abspath('L:/data/csv/USDJPY_forex_min30.csv')
    ##hyper parameters##
    observationLength = 12*24
    in_column = ohlc_column
    
    renko_ps = indicater.RenkoProcess(ohlc_column=ohlc_column)
    ids = [renko_ps]
    out_column = (renko_ps.columns["NUM"],)
    #processes = [process.DiffPreProcess(), process.MinMaxPreProcess(scale=(-1,1))]
    processes = [fc.utils.MinMaxPreProcess(scale=(-1,1))]
    batch_size = 32
    data_client = fc.CSVClient(files=file_path, frame=30, columns=ohlc_column, date_column="Time", start_index=batch_size)
    hidden_layer_size = 5
    middle_layer_size = 96
    ####################
    #epoc_num = 50
    version = 3
    
    
    training_lstm_model(
        data_client=data_client,
        idc_processes=ids,
        pre_processes=processes,
        batch_size=batch_size,
        observationLength=observationLength,
        in_column=in_column,
        out_column=out_column,
        hidden_layer_num=hidden_layer_size,
        #middle_layer_size=middle_layer_size,
        version=version
    )

if __name__ == "__main__":
    renko_trend()