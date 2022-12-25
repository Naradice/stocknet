import pandas as pd
import torch

from stocknet.datasets.finance import Dataset


class HighLowDataset(Dataset):
    """ 
    input: columns you specified of data client output 
    output: True if index value is Higher than index -1. 
    """
    
    key = "hl"

    def __init__(self, data_client, observationLength:int, idc_processes=[], pre_processes=[], in_columns=["Open", "High", "Low", "Close"] ,out_columns=["Open", "High", "Low", "Close"], compare_with="Close", merge_columns = False, seed = None, binary_mode=True, isTraining=True):
        super().__init__(data_client, observationLength,idc_processes, pre_processes, in_columns, out_columns, merge_columns, seed, isTraining)
        self.compare_with = compare_with
        list([compare_with]+out_columns)
        ## TODO: add HL column
        self.args = (data_client, observationLength, idc_processes, pre_processes, in_columns ,out_columns, compare_with, merge_columns, seed, binary_mode)
        self.init_indicies()
        if binary_mode:
            self.outputFunc = self.output_binary
        else:
            self.outputFunc = self.output_possibility
        
    def output_binary(self, batch_size):
        output = []
        columns = list(set([self.compare_with]) | set(self.out_columns))
        
        if type(batch_size) == int:
            batch_indicies = slice(batch_size, batch_size+1)
        elif type(batch_size) == slice:
            batch_indicies = batch_size
            
        symbols = self.data_client.symbols
        
        for index in self.indices[batch_indicies]:
            data = self.data_client.get_train_data(index-1, 2, columns, symbols, self.idc_processes, self.pre_processes)
            data = data.T
            last_value_to_compare = data[self.compare_with].iloc[-2]
            temp = data[self.out_columns].iloc[-1] > last_value_to_compare
            temp = temp.replace([True, False], [1.0, 0.0])
            ##TODO: convert True/False to mini/max value
            output.append(temp.tolist())
        return torch.tensor(output).reshape((len(output) * len(symbols)* len(self.out_columns) * 1))
        
        
    def output_possibility(self, batch_size):
        output = []
        columns = list(set([self.compare_with]) | set(self.out_columns))
        out_unit = pd.Series([True, False], index=['high', 'low'])
        if type(batch_size) == int:
            batch_indicies = slice(batch_size, batch_size+1)
        elif type(batch_size) == slice:
            batch_indicies = batch_size

        symbols = self.data_client.symbols
        for index in self.indices[batch_indicies]:
            data = self.data_client.get_train_data(index-1, 2, columns, symbols, self.idc_processes, self.pre_processes)
            data = data.T
            last_value_to_compare = data[self.compare_with].iloc[-2]
            temp = data[self.out_columns].iloc[-1] > last_value_to_compare
            index_ans = []
            for is_high in temp:
                if is_high == out_unit:
                    ans = 1.0
                else:
                    ans = 0.0
                index_ans.append(ans)
            output.append(index_ans)
        return torch.tensor(output).reshape((len(output) * len(symbols)* len(self.out_columns) * 1))