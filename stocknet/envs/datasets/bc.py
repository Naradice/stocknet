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

    def __init__(self, data_client: MarketClientBase,  observationDays=1, isTraining = True):
        
        self.__rowdata__ = data_client.get_rates(-1)
        #self.dtype = torch.float32
        
        self.__initialized =  False
        columns_dict = data_client.get_ohlc_columns()
        self.columns = [columns_dict['Open'], columns_dict['High'], columns_dict['Low'], columns_dict['Close']]
                
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
            self.columns.append(column)
    
    def register_preprocess(self, process: ProcessBase):
        """ register preprocess for data.

        Args:
            process (ProcessBase): any process you define
            option (_type_): option for a process
        """
        self.__preprocesess.append((process))
    
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
        return self.getInputs(ndx, shift=shift)
    
    def getSymbolInfo(self, symbol='USDJPY'):
        if symbol == 'USDJPY':
             return {
                 "point": 0.001,
                 "min":0.1,
                 "rate":100000
             }

        return None
    
    def getInputs(self, ndx, shift=0):
        inputs = []
        if type(ndx) == int:
            indicies = slice(ndx, ndx+1)
            for index in self.indices[indicies]:
                temp = numpy.array([])
                for column in self.columns:
                    temp = numpy.append(temp, self.data[column][index+shift-self.dataLength:index+shift].values.tolist())
                inputs.append(temp)
            return inputs[0]
        elif type(ndx) == slice:
            indicies = ndx
            for index in self.indices[indicies]:
                temp = numpy.array([])
                for column in self.columns:
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
        inputs = numpy.array(self.getInputs(ndx), dtype=numpy.dtype('float32'))
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
    
    def __init__(self, data_client: MarketClientBase, observationDays=1, floor = 1,isTraining=True):
        super().__init__(data_client, observationDays, isTraining)
        self.shift = floor
        
    def outputFunc(self, ndx):
        return super().outputFunc(ndx, self.shift)
    
    def __getitem__(self, ndx):
        return super().__getitem__(ndx)