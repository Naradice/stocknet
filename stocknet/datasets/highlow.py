import random
from stocknet.datasets.finance import Dataset
import pandas as pd

class HighLowDataset(Dataset):
    """ 
    input: columns you specified of data client output 
    output: True if index value is Higher than index -1. 
    """
    
    key = "hl"

    def __init__(self, data_client, observationLength:int, in_columns=["Open", "High", "Low", "Close"] ,out_columns=["Open", "High", "Low", "Close"], compare_with="Close", merge_columns = False, seed = None, isTraining = True, binary_mode=True):
        super().__init__(data_client, observationLength, in_columns, out_columns, merge_columns, seed, isTraining)
        self.compare_with = compare_with
        ## TODO: add HL column
        self.args = (data_client, observationLength, in_columns ,out_columns, compare_with, merge_columns, seed)
        data = data_client.get_rate_with_indicaters()
        self.row_data = data_client.revert_postprocesses(data)
        self.init_indicies()
        if binary_mode:
            self.outputFunc = self.output_binary
        else:
            self.outputFunc = self.output_possibility
        
    def init_indicies(self):
        length = len(self.data)
        if length < self.observationLength:
            raise Exception(f"date length {length} is less than observationLength {self.observationLength}")
        
        if self.isTraining:
            self.fromIndex = self.observationLength+1
            self.toIndex = int(length*0.7)
        else:
            self.fromIndex = int(length*0.7)+1
            self.toIndex = length
        
        training_data_length = self.__len__()
        ## length to select random indices.
        k = training_data_length
        ## if allow duplication
        #self.indices = random.choices(range(self.fromIndex, self.toIndex), k=k)
        ## if disallow duplication
        self.indices = random.sample(range(self.fromIndex, self.toIndex), k=k)
        
    def output_binary(self, batch_size):
        output = []
        if type(batch_size) == int:
            batch_indicies = slice(batch_size, batch_size+1)
            out_indicies = 0
        elif type(batch_size) == slice:
            batch_indicies = batch_size
            out_indicies = slice(0,None)
        for index in self.indices[batch_indicies]:
            last_value_to_compare = self.row_data[self.compare_with].iloc[index-1]
            temp = self.row_data[self.out_columns].iloc[index] > last_value_to_compare
            ##TODO: convert True/False to mini/max value
            output.append(temp)
        return output[out_indicies]
        
        
    def output_possibility(self, batch_size):
        output = []
        out_unit = pd.Series([True, False], index=['high', 'low'])
        if type(batch_size) == int:
            batch_indicies = slice(batch_size, batch_size+1)
            out_indicies = 0
        elif type(batch_size) == slice:
            batch_indicies = batch_size
            out_indicies = slice(0,None)
        for index in self.indices[batch_indicies]:
            last_value_to_compare = self.row_data[self.compare_with].iloc[index-1]
            temp = self.row_data[self.out_columns].iloc[index] > last_value_to_compare
            index_ans = []
            for is_high in temp:
                ans = is_high == out_unit
                index_ans.append(ans)
            output.append(index_ans)
        return output[out_indicies]