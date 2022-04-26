import random
import datetime
import numpy
import pandas as pd
import torch
import matplotlib as plt

dtype = torch.float32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


## Need to add preprocess like minmax
class FXDataset:

    def __init__(self, interval_days=1, isTraining = True, data = None):
        if data == None:
            rates = pd.read_csv('/mnt/landisk/data/fx/NextBoaderPossibility/fx_USDJPY_5_2020-08-03T23-05-00_to_2021-12-04T07-50-00.csv', header=0, index_col=0, parse_dates=True)
        else:
            rates = data
        self.__rowdata__ = rates
        self.data_initialization()
        length = len(self.data)
        
        self.budget_org = 100000
        self.leverage = 25
        self.volume_point = 10000
        self.point = 0.001

        self.dataRange = datetime.timedelta(days=2)
        self.dims = 5
        interval_days = 1
        MINUTES_SPAN = 5
        
        totalMinutes = interval_days * 24 * 60
        self.span  = int(totalMinutes/MINUTES_SPAN)+1
        
        ##select random indices.
        self.indices = random.sample(range(self.span, length - self.span -1), k=length - self.span*2 -1)
        if isTraining:
            self.fromIndex = self.span
            self.toIndex = int(length*0.7)
        else:
            self.fromIndex = int(length*0.7)+1
            self.toIndex = length+1

        self.outputFunc = self.__getAns4AE__

    def data_initialization(self):
        diff_array = self.__rowdata__.diff()
        self.data = pd.DataFrame(diff_array, columns=['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume'])
        self.data = self.data.drop(columns=['time', 'real_volume'])
        self.data.tick_volume, _, _ = self.minmaxNormalization(self.data.tick_volume)
        self.data.spread, self.minSp, self.maxSp = self.minmaxNormalization(self.data.spread)
        off_set = 1
        self.data = self.data[off_set:]

    
    def __getAns4AE__(self, ndx, shift=0, columns=['open', 'high', 'low', 'close','tick_volume', 'spread']):
        '''
        ndx: slice type or int
        return ans value array from actual tick data. data format (rate, diff etc) is depends on data initialization. default is diff
        '''
        return self.getInputs(ndx, shift, columns)
    
    def EMA(self, data, interval):
        sema = [-1 for i in range(0,interval-1)]
        lastValue = data[0:interval].sum()/interval
        sema.append(lastValue)
        alpha = 2/(interval+1)
        for i in range(interval, len(data)):
            lastValue = lastValue * (1 - alpha) + data[i]*alpha
            sema.append(lastValue)
        return sema
    
    def getSymbolInfo(self, symbol='USDJPY'):
        if symbol == 'USDJPY':
             return {
                 "point": 0.001,
                 "min":0.1,
                 "rate":100000
             }

        return None
    
    '''
    def getInputs(self, ndx):
        inputs = []
        if type(ndx) == int:
            indicies = slice(ndx, ndx+1)
            for index in self.indices[indicies]:
                inputs.append(self.data[index+1-self.span:index+1].values.tolist())
            return inputs[0]
        elif type(ndx) == slice:
            indicies = ndx
            for index in self.indices[indicies]:
                inputs.append(self.data[index+1-self.span:index+1].values.tolist())
            return inputs
    '''
    
    def getInputs(self, ndx, shift=0, columns=['open', 'high', 'low', 'close','tick_volume', 'spread']):
        inputs = []
        if type(ndx) == int:
            indicies = slice(ndx, ndx+1)
            for index in self.indices[indicies]:
                temp = numpy.array([])
                for column in columns:
                    temp = numpy.append(temp, self.data[column][index+1+shift-self.span:index+1+shift].values.tolist())
                inputs.append(temp)
            return inputs[0]
        elif type(ndx) == slice:
            indicies = ndx
            for index in self.indices[indicies]:
                temp = numpy.array([])
                for column in columns:
                    temp = numpy.append(temp, self.data[column][index+1+shift-self.span:index+1+shift].values.tolist())
                inputs.append(temp)
            return inputs
    
    def __len__(self):
        return self.toIndex - self.fromIndex
    
    def __getRowData__(self, ndx):
        inputs = []
        if type(ndx) == slice:
            for index in self.indices[ndx]:
                inputs.append(self.__rowdata__[index+2-self.span:index+1].values.tolist())
        else:
            index = ndx
            inputs = self.__rowdata__[index+2-self.span:index+1].values.tolist()

        return inputs
    
    def __getActialIndex__(self,ndx):
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
        return torch.tensor(inputs, device=device).to(dtype=dtype), torch.tensor(outputs, device=device).to(dtype=dtype)
    
    def minmaxNormalization(self, data):
        if type(data) == numpy.ndarray:
            temp_data = data[~numpy.isnan(data)]
        elif type(data) == pd.core.series.Series:
            temp_data = data.dropna()
        else:
            print(f"unkown type: {type(data)}")
            temp_data = data
        X_max, X_min = max(temp_data), min(temp_data)
        data_norm = (data - X_min) / (X_max - X_min)
        return data_norm, X_min, X_max

    def denormalization(self, value, X_min, X_max):
        return value * (X_max - X_min) + X_min
        
    def render(self, mode='human', close=False):
        '''
        '''
        if mode == 'gui':
            pass
        elif mode == 'human':
            pass
        elif mode == 'console':
            pass
        
    def seed(self, seed=None):
        '''
        '''
        if seed == None:
            random.seed(1017)
        else:
            random.seed(seed)
            
class FXMACDDataset(FXDataset):
    def __init__(self):
        super().__init__()


    def MACDInit(self, short_window=12, long_window=16):
        rates = {}
        rates["short_ema"] = self.EMA(self.__rowdata__.close, short_window)
        rates["long_ema"] = self.EMA(self.__rowdata__.close, long_window)
        rates = pd.DataFrame(rates)
        rates.diff().dropna()

        self.data["short_ema"] = rates["short_ema"]
        self.data["long_ema"] = rates["long_ema"]

    def data_initialization(self):
        self.data = self.__rowdata__
        short_window = 12
        long_window = 16
        self.MACDInit(short_window, long_window)
        diff_array = self.__rowdata__.diff()
        print(diff_array)
        self.data = pd.DataFrame(diff_array, columns=['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume', "short_ema", "long_ema"])
        self.data = self.data.drop(columns=['time', 'real_volume'])
        off_set = long_window
        self.data = self.data[off_set:]
        print(self.data)

    def __getitem__(self, ndx):
        inputs = numpy.array(self.getInputs(ndx, columns=['open', 'high', 'low', 'close','tick_volume', 'spread', "short_ema", "long_ema"]), dtype=numpy.dtype('float32'))
        outputs = numpy.array(self.outputFunc(ndx), dtype=numpy.dtype('float32'))
        return torch.tensor(inputs, device=device).to(dtype=dtype), torch.tensor(outputs, device=device).to(dtype=dtype)
    
class FXNextMEANDiffDataset:
    def __init__(self,next_index=1, short_window=12, long_window=26, isTraining = True, seed=0, mode="default"):
        self.next = next_index
        random.seed(seed)
        rates = pd.read_csv('/mnt/landisk/data/fx/NextBoaderPossibility/fx_USDJPY_5_2020-08-03T23-05-00_to_2021-12-04T07-50-00.csv', header=0, index_col=0, parse_dates=True)
        self.rowdata = rates
        #rates["ema"] = rates.close.windows(10).mean()
        rates["short_ema"] = self.EMA(rates.close, short_window)
        rates["long_ema"] = self.EMA(rates.close, long_window)
        diff_array = rates.diff().dropna()
        data = pd.DataFrame(diff_array, columns=['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume', 'short_ema', "long_ema"])
        data = data.drop(columns=['time', 'real_volume'])
        data.tick_volume, _, _ = self.minmaxNormalization(rates.tick_volume)
        data.spread,_,_ = self.minmaxNormalization(rates.spread)
        self.all_data = data
        length = len(self.all_data)
        print(length)
            
        self.dataRange = datetime.timedelta(days=2)
        self.dims = 5
        self.mode = mode
        INTERVAL_DAYS = 2
        MINUTES_SPAN = 5

        totalMinutes = INTERVAL_DAYS * 24 * 60
        self.span  = int(totalMinutes/MINUTES_SPAN)+1
        
        ##select random indices.
        self.indices = random.sample(range(self.span, length - self.span -1), k=length - self.span*2 -1)
        if isTraining:
            self.fromIndex = self.span
            self.toIndex = int(length*0.7)
        else:
            self.fromIndex = int(length*0.7)+1
            self.toIndex = length+1
        self.outputFunc = self.__getAns__

    def EMA(self, data, interval):
        sema = [-1 for i in range(0,interval-1)]
        lastValue = data[0:interval].sum()/interval
        sema.append(lastValue)
        alpha = 2/(interval+1)
        for i in range(interval, len(data)):
            lastValue = lastValue * (1-alpha) + data[i]*alpha
            sema.append(lastValue)
        return sema
        
    def __getDiffArray__(self, data):
        for i in range(1,len(data)):
            data[i] - data[i-1]
        
    def __rateToArray__(self, value):
        output = [0 for i in range(0,3000)]
        i = round((value -0.85)*10000)
        if i >= 3000:
            i = 2999
        elif i < 0:
            i = 0
        output[i] = 1
        return output
        
    def __len__(self):
        return self.toIndex - self.fromIndex
    
    def __getAns__(self,ndx):
        if type(ndx) == int:
            indicies = slice(ndx, ndx+1)
            index = self.indices[ndx]
            return self.all_data[['short_ema', 'long_ema']].iloc[index+self.next].values

        elif type(ndx) == slice:
            indicies = ndx
        ans = []
        for index in self.indices[indicies]:
            ans.append(self.all_data[['short_ema', 'long_ema']].iloc[index+self.next].values)
        return ans
        
    def __getRowData__(self, ndx):
        inputs = []
        if type(ndx) == slice:
            for index in self.indices[ndx]:
                inputs.append(self.rowdata[index+1-self.span:index+1].values.tolist())
        else:
            index = ndx
            inputs = self.rowdata[index+1-self.span:index+1].values.tolist()

        return inputs
    
    def __getInputs__(self, ndx):
        inputs = []
        data = self.all_data
        if type(ndx) == int:
            indicies = slice(ndx, ndx+1)
            for index in self.indices[indicies]:
                inputs.append(data[['short_ema', 'long_ema']][index+1-self.span:index+1].values.tolist())
            return inputs[0]
        elif type(ndx) == slice:
            indicies = ndx
            for index in self.indices[indicies]:
                inputs.append(data[['short_ema', 'long_ema']][index+1-self.span:index+1].values.tolist())
            return inputs
    
    def __getActialIndex__(self,ndx):
        inputs = []
        if type(ndx) == slice:
            for index in self.indices[ndx]:
                inputs.append(index)
        else:
            inputs = self.indices[ndx]

        return inputs
    
    def __getitem__(self, ndx):
        ins = np.array(self.__getInputs__(ndx), dtype=np.dtype('float32'))
        outputs = np.array(self.outputFunc(ndx), dtype=np.dtype('float32'))
        return torch.tensor(ins, device=device).to(dtype=dtype), torch.tensor(outputs, device=device).to(dtype=dtype)
        #return ins, outputs
    
    def minmaxNormalization(self, data):
        if type(data) == np.ndarray:
            temp_data = data[~np.isnan(data)]
        elif type(data) == pd.core.series.Series:
            temp_data = data.dropna()
        else:
            print(f"unkown type: {type(data)}")
            temp_data = data
        X_max, X_min = max(temp_data), min(temp_data)
        data_norm = (data - X_min) / (X_max - X_min)
        return data_norm, X_min, X_max

    def convertAnsToBoaderValue(self,rate):
        ansValue = rate*(self.ansmax - self.ansmin) + self.ansmin
        return ansValue