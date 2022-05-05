from stocknet.envs.utils import standalization, indicaters
import pandas as pd

class ProcessBase:
   
    columns = []
   
    def __init__(self, key:str):
        self.key = key
    
    def run(self, data: pd.DataFrame) -> dict:
        """ process to apply additionally. if an existing key is specified, overwrite existing values

        Args:
            data (pd.DataFrame): row data of dataset

        """
        raise Exception("Need to implement process method")
    
    def update(self, tick:pd.Series) -> pd.Series:
        """ update data using next tick

        Args:
            tick (pd.DataFrame): new data

        Returns:
            dict: appended data
        """
        raise Exception("Need to implement")
    
    def get_minimum_required_length(self) -> int:
        raise Exception("Need to implement")
    
    def concat(self, data:pd.DataFrame, new_data: pd.Series):
        return pd.concat([data, pd.DataFrame.from_records([new_data])], ignore_index=True)
    
    def revert(self, data_set: tuple):
        """ revert processed data to row data with option value

        Args:
            data (tuple): assume each series or values or processed data is passed
            
        Returns:
            Boolean, dict: return (True, data: pd.dataFrame) if reverse_process is defined, otherwise (False, None)
        """
        return False, None
    
    def init_params(self, data: pd.DataFrame):
        pass
    
class MACDpreProcess(ProcessBase):
    
    option = {
        "column": "Close",
        "short_window": 12,
        "long_window": 26,
        "signal_window":9
    }
    
    last_data = None
    
    columns = ['ShortEMA', 'LongEMA', 'MACD', 'Signal']
    
    def __init__(self, key='macd', option=None):
        super().__init__(key)
        if option != None:
            self.option.update(option)

    def run(self, data: pd.DataFrame):
        option = self.option
        target_column = option['column']
        short_window = option['short_window']
        long_window = option['long_window']
        signal_window = option['signal_window']
        self.option = option
        
        short_ema, long_ema, MACD, Signal = indicaters.MACD_from_ohlc(data, target_column, short_window, long_window, signal_window)
        self.last_data = pd.DataFrame({'ShortEMA':short_ema, 'LongEMA': long_ema, 'MACD':MACD, 'Signal':Signal}).iloc[-self.get_minimum_required_length():]
        return {'ShortEMA':short_ema, 'LongEMA': long_ema, 'MACD':MACD, 'Signal':Signal}
    
    def update(self, tick:pd.Series):
        option = self.option
        target_column = option['column']
        short_window = option['short_window']
        long_window = option['long_window']
        signal_window = option['signal_window']
        
        
        short_ema, long_ema, MACD = indicaters.update_macd(
            new_tick=tick,
            short_ema_value=self.last_data['ShortEMA'].iloc[-1],
            long_ema_value=self.last_data['LongEMA'].iloc[-1],
            column=target_column, short_window = short_window, long_window = long_window)
        Signal = (self.last_data['MACD'].iloc[-signal_window + 1:].sum() + MACD)/signal_window
        
        new_data = pd.Series({'ShortEMA':short_ema, 'LongEMA': long_ema, 'MACD':MACD, 'Signal':Signal})
        self.last_data = self.concat(self.last_data.iloc[1:], new_data)
        return new_data
        
    def get_minimum_required_length(self):
        return self.option["long_window"] + self.option["signal_window"] - 2
    
    def revert(self, data_set:tuple):
        if type(data_set) == pd.DataFrame:
            if 'ShortEMA' in data_set:
                data_set = (data_set['ShortEMA'],)
        #assume ShortEMA is in 1st
        short_ema = data_set[0]
        short_window = self.option['short_window']
        out = indicaters.revert_EMA(short_ema, short_window)
        return True, out

class EMApreProcess(ProcessBase):
    
    option = {
        "column": "Close",
        "window": 12
    }
    
    last_data = None
    
    columns = ['EMA']
    
    def __init__(self, key='ema',window = 12, column = 'Close'):
        super().__init__(key)
        self.option['window'] = window
        self.option['column'] = column

    def run(self, data: pd.DataFrame):
        option = self.option
        target_column = option['column']
        window = option['window']
        self.option = option
        column = self.columns[0]
        
        ema = indicaters.EMA(data[target_column], window)
        self.last_data = pd.DataFrame({column:ema}).iloc[-self.get_minimum_required_length():]
        return {'EMA':ema}
    
    def update(self, tick:pd.Series):
        option = self.option
        target_column = option['column']
        window = option['window']
        column = self.columns[0]
        
        
        short_ema, long_ema, MACD = indicaters.update_ema(
            new_tick=tick,
            column=target_column,
            window = window)
        
        new_data = pd.Series({column:short_ema})
        self.last_data = self.concat(self.last_data.iloc[1:], new_data)
        return new_data
        
    def get_minimum_required_length(self):
        return self.option["window"]
    
    def revert(self, data_set:tuple):
        #assume EMA is in 1st
        ema = data_set[0]
        window = self.option['window']
        out = indicaters.revert_EMA(ema, window)
        return True, out

class DiffPreProcess(ProcessBase):
    
    last_tick:pd.DataFrame = None
    
    def __init__(self, key = "diff"):
        super().__init__(key)
        
    def run(self, data: pd.DataFrame) -> dict:
        columns = data.columns
        result = {}
        for column in columns:
            result[column] = data[column].diff()
        
        self.last_tick = data.iloc[-1]
        return result
    
    def update(self, tick:pd.Series):
        """ assuming data is previous result of run()

        Args:
            data (pd.DataFrame): previous result of run()
            tick (pd.Series): new row data
            option (Any, optional): Currently no option (Floor may be added later). Defaults to None.
        """
        new_data = tick - self.last_tick
        self.last_tick = tick
        return new_data
        
    
    def get_minimum_required_length(self):
        return 2
    
    def revert(self, data_set: tuple):
        columns = self.last_tick.columns
        result = []
        if type(data_set) == pd.DataFrame:
            data_set = tuple(data_set[column] for column in columns)
        if len(data_set) == len(columns):
            for i in range(0, len(columns)):
                last_data = self.last_tick[columns[i]]
                data = data_set[i]
                row_data = [last_data]
                for index in range(len(data)-1, -1, -1):
                    last_data = data[index] - last_data
                    row_data.append(last_data)
                row_data = reversed(row_data)
                result.append(row_data)
            return True, result
        else:
            raise Exception("number of data is different")

class MinMaxPreProcess(ProcessBase):
    
    opiton = {}
    params = {}
    
    def __init__(self, key: str = 'minmax', scale = (-1, 1)):
        self.opiton['scale'] = scale
        super().__init__(key)
    
    def run(self, data: pd.DataFrame) -> dict:
        columns = data.columns
        result = {}
        option = self.opiton
        if 'scale' in option:
            scale = option['scale']
        else:
            scale = (-1, 1)
            
        for column in columns:
            result[column], _max, _min =  standalization.mini_max_from_series(data[column], scale)
            if column not in self.params:
                self.params[column] = (_min, _max)
        self.columns = columns
        return result
    
    def update(self, tick:pd.Series):
        columns = self.columns
        scale = self.opiton['scale']
        result = {}
        for column in columns:
            _min, _max = self.params[column]
            new_value = tick[column]
            if new_value < _min:
                _min = new_value
                self.params[column] = (_min, _max)
            if new_value > _max:
                _max = new_value
                self.params[column] = (_min, _max)
            
            scaled_new_value = standalization.mini_max(new_value, _min, _max, scale)
            result[column] = scaled_new_value
            
        new_data = pd.Series(result)
        return new_data

    def get_minimum_required_length(self):
        return 1
    
    def revert(self, data_set:tuple):
        columns = self.columns
        if type(data_set) == pd.DataFrame:
            return standalization.revert_mini_max_from_data(data_set, self.params, self.opiton['scale'])
        elif len(data_set) == len(columns):
            result = []
            for i in range(0, len(columns)):
                _min, _max = self.params[columns[i]]
                data = data_set[i]
                row_data = standalization.revert_mini_max_from_data(data, (_min, _max), self.opiton['scale'])
                result.append(row_data)
            return True, result
        else:
            raise Exception("number of data is different")
            