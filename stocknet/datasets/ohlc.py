import random
import numpy
import torch
from stocknet.datasets.finance import Dataset

class OHLCDataset(Dataset):
    """
    Basic OHLC dataset
    """
    
    key = "ohlc"
    
    def __init__(self, data_client, observationLength=1, seed=None, merge_columns=False, isTraining=True):
        ohlc_dict = data_client.get_ohlc_columns()
        ohlc_columns = [ohlc_dict["Open"], ohlc_dict["High"], ohlc_dict["Low"], ohlc_dict["Close"]]
        super().__init__(data_client, observationLength=observationLength, in_columns=ohlc_columns, out_columns=ohlc_columns, merge_columns=merge_columns, seed=seed, isTraining=isTraining)
        
        self.__rowdata__ = data_client.get_rate_with_indicaters()
        self.data = self.__rowdata__.copy()
        self.dataLength = observationLength
        
        self.isTraining = isTraining
        self.init_indicies()
        
    def init_indicies(self):
        length = len(self.data)
        if self.isTraining:
            self.fromIndex = self.dataLength
            self.toIndex = int(length*0.7)
        else:
            self.fromIndex = int(length*0.7)+1
            self.toIndex = length
        
        ##select random indices.
        k=length - self.dataLength*2 -1
        self.indices = random.choices(range(self.fromIndex, self.toIndex), k=k)
    
    def outputFunc(self, ndx, shift=0):
        '''
        ndx: slice type or int
        return ans value array from actual tick data. data format (rate, diff etc) is depends on data initialization. default is diff
        '''
        return self.getInputs(ndx, self.out_columns)
    
    def inputFunc(self, ndx):
        return self.getInputs(ndx, self.in_columns)
    
    def getInputs(self, batch_size, columns):
        inputs = []
        if type(batch_size) == int:
            batch_indicies = slice(batch_size, batch_size+1)
            out_indicies = 0
        elif type(batch_size) == slice:
            batch_indicies = batch_size
            out_indicies = slice(0,None)
        for index in self.indices[batch_indicies]:
            temp = self.create_column_func(columns, slice(index-self.dataLength, index))
            inputs.append(temp)
        return inputs[out_indicies]
    
    def __len__(self):
        return self.toIndex - self.fromIndex
    
    def getRowData(self, ndx):
        inputs = []
        if type(ndx) == slice:
            for index in self.indices[ndx]:
                inputs.append(self.__rowdata__[index-self.dataLength:index].values.tolist())
        else:
            index = ndx
            inputs = self.__rowdata__[index+1-self.dataLength:index+1].values.tolist()
        return inputs
    
    def getActialIndex(self,ndx):
        inputs = []
        if type(ndx) == slice:
            for index in self.indices[ndx]:
                inputs.append(index)
        else:
            inputs = self.indices[ndx]

        return inputs