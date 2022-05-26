import random
import datetime
import numpy
import pandas as pd
import torch
from stocknet.envs.market_clients.market_client_base import MarketClientBase
from stocknet.envs.utils import standalization, indicaters
from stocknet.envs.utils.preprocess import ProcessBase

class Dataset():
    """
    Basic OHLC dataset
    """
    
    key = "ohlc"

    def __init__(self, data_client: MarketClientBase, observationDays=1, data_length:int = None, out_ohlc_columns = ["Open", "High", "Low", "Close"], seed = None, isTraining = True):
        self.seed(seed)
        self.__rowdata__ = data_client.get_rates(-1)
        #self.dtype = torch.float32
        self.args = (observationDays, data_length, out_ohlc_columns, seed)
        columns_dict = data_client.get_ohlc_columns()
        self.columns = []
        __ohlc_columns = [str.lower(value) for value in out_ohlc_columns]
        if 'open' in __ohlc_columns:
            self.columns.append(columns_dict['Open'])
        if 'high' in __ohlc_columns:
            self.columns.append(columns_dict['High'])
        if 'low' in __ohlc_columns:
            self.columns.append(columns_dict['Low'])
        if 'close' in __ohlc_columns:
            self.columns.append(columns_dict['Close'])    
        
        self.out_columns = self.columns.copy()
                
        self.budget_org = 100000
        self.leverage = 25
        self.volume_point = 10000
        self.point = 0.001

        self.dims = 5
        frame = data_client.frame
        self.dataLength = int(observationDays * 24 * (60 / frame))
        
        self.__preprocesess = []
        self.isTraining = isTraining
        self.data = self.__rowdata__
        self.__init_indicies()
        
    def __init_indicies(self):
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
        
    
    def run_preprocess(self):
        """
        Ex. you can define MACD as process. The results of the process are stored as dataframe[key] = values
        """
        self.data = self.__rowdata__.copy()
        
        if len(self.__preprocesess) > 0:
            processes = self.__preprocesess
        else:
            raise Exception("Need to register preprocess before running processes")
        
        for process in processes:
            values_dict = process.run(self.data)
            for column, values in values_dict.items():
                self.data[column] = values
        self.data = self.data.dropna(how = 'any')
        self.__init_indicies()
                
    def add_indicater(self, process: ProcessBase):
        values_dict = process.run(self.__rowdata__)
        for column, values in values_dict.items():
            self.__rowdata__[column] = values
            if process.is_input:
                self.columns.append(column)
            if process.is_output:
                self.out_columns.append(column)
            
    def add_indicaters(self, processes: list):
        for process in processes:
            self.add_indicater(process)
    
    def register_preprocess(self, process: ProcessBase):
        """ register preprocess for data.

        Args:
            process (ProcessBase): any process you define
            option (_type_): option for a process
        """
        self.__preprocesess.append(process)
    
    def register_preprocesses(self, processes: list):
        """ register preprocess for data.

        Args:
            processes (list[processBase]): any process you define
            options: option for a processes[key] (key is column name of additional data)
        """
        
        for process in processes:
            self.register_preprocess(process)
    
    def outputFunc(self, ndx, shift=0):
        '''
        ndx: slice type or int
        return ans value array from actual tick data. data format (rate, diff etc) is depends on data initialization. default is diff
        '''
        return self.getInputs(ndx, self.out_columns,shift=shift)
    
    def inputFunc(self, ndx):
        return self.getInputs(ndx, self.columns)
    
    def getSymbolInfo(self, symbol='USDJPY'):
        if symbol == 'USDJPY':
             return {
                 "point": 0.001,
                 "min":0.1,
                 "rate":100000
             }

        return None
    
    def getInputs(self, ndx, columns,shift=0,):
        inputs = []
        if type(ndx) == int:
            indicies = slice(ndx, ndx+1)
            for index in self.indices[indicies]:
                temp = numpy.array([])
                for column in columns:
                    temp = numpy.append(temp, self.data[column][index+shift-self.dataLength:index+shift].values.tolist())
                inputs.append(temp)
            return inputs[0]
        elif type(ndx) == slice:
            indicies = ndx
            for index in self.indices[indicies]:
                temp = numpy.array([])
                for column in columns:
                    temp = numpy.append(temp, self.data[column][index+shift-self.dataLength:index+shift].values.tolist())
                inputs.append(temp)
            return inputs
    
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
    
    def __getitem__(self, ndx):
        inputs = numpy.array(self.inputFunc(ndx), dtype=numpy.dtype('float32'))
        outputs = numpy.array(self.outputFunc(ndx), dtype=numpy.dtype('float32'))
        return torch.tensor(inputs), torch.tensor(outputs)
        
    def render(self, mode='human', close=False):
        '''
        '''
        pass
        
    def seed(self, seed=None):
        '''
        '''
        if seed == None:
            random.seed(1017)
        else:
            random.seed(seed)
            
class ShiftDataset(Dataset):
    
    key = "shit_ohlc"
    
    def __init__(self, data_client: MarketClientBase, observationDays=1,out_ohlc__columns=["Open", "High", "Low", "Close"], floor = 1, seed = None, isTraining=True):
        super().__init__(data_client, observationDays, out_ohlc_columns=out_ohlc__columns, seed=seed, isTraining=isTraining)
        self.args = (observationDays,out_ohlc__columns, floor, seed)
        self.shift = floor
    
    def __init_indicies(self):
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
        return self.getInputs(ndx, self.columns)
    
    def getInputs(self, ndx, columns, shift=0):
        inputs = []
        if type(ndx) == int:
            indicies = slice(ndx, ndx+1)
            for index in self.indices[indicies]:
                temp = (self.data[columns].iloc[index+shift-self.dataLength:index+shift].values.tolist())
                inputs.append(temp)
            return inputs[0]
        elif type(ndx) == slice:
            indicies = ndx
            for index in self.indices[indicies]:
                temp = (self.data[columns].iloc[index+shift-self.dataLength:index+shift].values.tolist())
                inputs.append(temp)
            return inputs
    
    def getNextInputs(self, ndx, shift=1):
        inputs = []
        if type(ndx) == int:
            indicies = slice(ndx, ndx+1)
            for index in self.indices[indicies]:
                temp = self.data[self.out_columns].iloc[index+shift-1]
                inputs.append(temp)
            return inputs[0]
        elif type(ndx) == slice:
            indicies = ndx
            for index in self.indices[indicies]:
                temp = self.data[self.out_columns].iloc[index+shift-1]
                inputs.append(temp)
            return inputs
     
    def outputFunc(self, ndx):
        return self.getNextInputs(ndx, self.shift)
    
    def __getitem__(self, ndx):
        return super().__getitem__(ndx)