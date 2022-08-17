import random
import numpy
import torch
from stocknet.envs.datasets.base import DatasetBase
from finance_client.client_base import Client

class Dataset(DatasetBase):
    """
    common dataset
    """
    
    key = "common"

    def __init__(self, data_client: Client, observationLength:int, in_columns=["Open", "High", "Low", "Close"] ,out_columns=["Open", "High", "Low", "Close"], seed = None, isTraining = True):
        super().__init__(observationLength, in_columns, out_columns, seed, isTraining)
        self.seed(seed)
        self.data = data_client.get_rate_with_indicaters(-1)
        columns_dict = data_client.get_ohlc_columns()
        self.dataLength = observationLength
        self.isTraining = isTraining
        self.init_indicies()
        
    def init_indicies(self):
        length = len(self.data)
        if length < self.observationLength:
            raise Exception(f"date length {length} is less than observationLength {self.observationLength}")
        
        if self.isTraining:
            self.fromIndex = self.dataLength
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
    
    def outputFunc(self, ndx, shift=0):
        '''
        ndx: slice type or int
        return ans value array from actual tick data. data format (rate, diff etc) is depends on data initialization. default is diff
        '''
        return self.getInputs(ndx, self.out_columns, shift=shift)
    
    def inputFunc(self, ndx):
        return self.getInputs(ndx, self.columns)
    
    def getInputs(self, batch_size: slice, columns:list, shift=0,):
        inputs = []
        if type(batch_size) == int:
            batch_indicies = slice(batch_size, batch_size+1)
            for index in self.indices[batch_indicies]:
                temp = numpy.array([])
                for column in columns:
                    temp = numpy.append(temp, self.data[column][index+shift-self.dataLength:index+shift].values.tolist())
                inputs.append(temp)
            return inputs[0]
        elif type(batch_size) == slice:
            batch_indicies = batch_size
            for index in self.indices[batch_indicies]:
                temp = numpy.array([])
                for column in columns:
                    temp = numpy.append(temp, self.data[column][index+shift-self.dataLength:index+shift].values.tolist())
                inputs.append(temp)
            return inputs
    
    def __len__(self):
        return self.toIndex - self.fromIndex
    
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

class OHLCDataset():
    """
    Basic OHLC dataset
    """
    
    key = "ohlc"

    def __init__(self, data_client: Client, observationDays=1, data_length:int = None, in_columns=["Open", "High", "Low", "Close"], out_columns=["Open", "High", "Low", "Close"], seed = None, isTraining = True):
        self.seed(seed)
        self.__rowdata__ = data_client.get_rate_with_indicaters(-1)
        self.data = self.__rowdata__.copy()
        #self.dtype = torch.float32
        self.args = (observationDays, data_length, out_columns, seed)
        columns_dict = data_client.get_ohlc_columns()
        self.columns = in_columns        
        self.out_columns = out_columns

        self.dims = 5
        frame = data_client.frame
        self.dataLength = int(observationDays * 24 * (60 / frame))
        
        self.__preprocesess = []
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
            
class ShiftDataset(OHLCDataset):
    
    key = "shit_ohlc"
    
    def __init__(self, data_client: Client, observationLength=1000,in_columns=["Open", "High", "Low", "Close"], out_columns=["Open", "High", "Low", "Close"], floor = 1, seed = None, isTraining=True):
        self.shift = floor
        super().__init__(data_client, observationLength, in_columns=in_columns, out_columns=out_columns, seed=seed, isTraining=isTraining)
        self.args = (observationLength,out_columns, floor, seed)
    
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
    
class HighLowDataset(Dataset):
    
    key = "hl_ohlc"
    
    def __init__(self, data_client: Client, observationLength=1000,in_columns=["Open", "High", "Low", "Close"], out_columns=["Open", "High", "Low", "Close"], floor = 1, seed = None, isTraining=True):
        super().__init__(data_client, observationLength, in_columns=in_columns, out_columns=out_columns, seed=seed, isTraining=isTraining)
        self.args = (observationLength,out_columns, floor, seed)
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
    
    def getFutureHighLow(self, shift=1) -> numpy.array:
        pass

    def getFutureHighLowArray(self, shift=1) -> numpy.array:
        pass
     
    def outputFunc(self, ndx):
        return self.getNextInputs(ndx, self.shift)


class RewardDataset(OHLCDataset):
    
    key = "shift_ohlc"
    
    def __init__(self, data_client: Client, observationDays=1,in_column = ["Open", "High", "Low", "Close"], column = "Close" , seed = None, isTraining=True):
        super().__init__(data_client, observationDays, out_columns=[], seed=seed, isTraining=isTraining)
        self.args = (observationDays,column, seed)
        self.column = column
    
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
        return self.getInputs(ndx, self.columns)
    
    def getInputs(self, ndx, columns):
        inputs = []
        if type(ndx) == int:
            indicies = slice(ndx, ndx+1)
            for index in self.indices[indicies]:
                temp = (self.data[columns].iloc[index-self.dataLength:index].values.tolist())
                inputs.append(temp)
            return inputs[0]
        elif type(ndx) == slice:
            indicies = ndx
            for index in self.indices[indicies]:
                temp = (self.data[columns].iloc[index-self.dataLength:index].values.tolist())
                inputs.append(temp)
            return inputs
    
    def caliculate_reward(self, index):
        window = 10
        values = self.data[self.column].iloc[index-window: index + 100]
        ewa = values.ewm(span=window, adjust=True).mean()
        sign = numpy.sign(ewa.diff())
        count = 0
        direction = sign.iloc[1]##change 1
        cp = [numpy.NaN]
        trend_count = 3
        for index in range(1, len(sign)):
            s = sign.iloc[index]
            if s != direction:
                count += 1
                if count == trend_count:
                    direction = s
                    count = 0
                    print(f"direction changed on {index}")
                    cp.append(numpy.NaN)
                    cp[-trend_count] = ewa.iloc[index - trend_count]
                else:
                    cp.append(numpy.NaN)
            else:
                count = 0
                cp.append(numpy.NaN)
        
    
    def getReward(self, ndx):
        """
        each index have 
        [
            reward if you bought coin
            reward if you sell coin
        ]
        """
        inputs = []
        if type(ndx) == int:
            indicies = slice(ndx, ndx+1)
            for index in self.indices[indicies]:
                temp = self.data[self.out_columns].iloc[index-1]
                inputs.append(temp)
            return inputs[0]
        elif type(ndx) == slice:
            indicies = ndx
            for index in self.indices[indicies]:
                temp = self.data[self.out_columns].iloc[index-1]
                inputs.append(temp)
            return inputs
     
    def outputFunc(self, ndx):
        return self.getReward(ndx)
    
    def __getitem__(self, ndx):
        return super().__getitem__(ndx)

    