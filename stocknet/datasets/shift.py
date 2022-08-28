import random
import numpy
import torch
from stocknet.datasets.finance import Dataset

class ShiftDataset(Dataset):
    
    key = "shit_ohlc"
    
    def __init__(self, data_client, observationLength=1000, in_columns=["Open", "High", "Low", "Close"], out_columns=["Open", "High", "Low", "Close"], shift=1, merge_input_columns=False, seed = None, isTraining=True):
        self.shift = shift
        super().__init__(data_client, observationLength, in_columns=in_columns, out_columns=out_columns, merge_columns=merge_input_columns, seed=seed, isTraining=isTraining)
        self.args = (observationLength,out_columns, shift, seed)
    
    def init_indicies(self):
        length = len(self.data) - self.shift
        if self.isTraining:
            self.fromIndex = self.dataLength
            self.toIndex = int(length*0.7)- self.shift
        else:
            self.fromIndex = int(length*0.7)+1
            self.toIndex = length - self.shift
        
        ##select random indices.
        k=length - self.dataLength*2 -1
        self.indices = random.choices(range(self.fromIndex, self.toIndex), k=k)
        
    def inputFunc(self, ndx):
        return self.getInputs(ndx, self.in_columns)
    
    def getInputs(self, batch_size, columns, shift=0):
        inputs = []
        if type(batch_size) == int:
            batch_indicies = slice(batch_size, batch_size+1)
            out_indicies = 0
        elif type(batch_size) == slice:
            batch_indicies = batch_size
            out_indicies = slice(0,None)
        for index in self.indices[batch_indicies]:
            temp = self.create_column_func(columns, slice(index+shift-self.dataLength, index+shift))
            inputs.append(temp)
        return inputs[out_indicies]
    
    def getNextInputs(self, batch_size, shift=1):
        inputs = []
        if type(batch_size) == int:
            indicies = slice(batch_size, batch_size+1)
            out_indicies = 0
        elif type(batch_size) == slice:
            indicies = batch_size
            out_indicies = slice(0, None)
            
        for index in self.indices[indicies]:
            temp = self.data[self.out_columns].iloc[index+shift-1]
            inputs.append(temp)
        return inputs[out_indicies]
     
    def outputFunc(self, ndx):
        return self.getNextInputs(ndx, self.shift)
    
    def __getitem__(self, ndx):
        return super().__getitem__(ndx)
