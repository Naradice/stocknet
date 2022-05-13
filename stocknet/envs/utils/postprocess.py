import numpy
from stocknet.envs.utils import standalization, indicaters
import pandas as pd
from stocknet.envs.utils.process import ProcessBase


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
            