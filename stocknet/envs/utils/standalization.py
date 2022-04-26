from multiprocessing.sharedctypes import Value
from matplotlib.pyplot import sca
import pandas as pd

def revert_mini_max(scaled_value, _min, _max, scale=(0, 1)):
    if scale[0] >= scale[1]:
        raise ValueError("mini_max function scale should be (min, max)")
    std = (scaled_value - scale[0])/(scale[1] - scale[0])
    value = std * (_max - _min) + _min
    return value

def mini_max(value, _min, _max, scale=(0,1)):
    if scale[0] >= scale[1]:
        raise ValueError("mini_max function scale should be (min, max)")
    std = (value - _min)/(_max - _min)
    scaled = std * (scale[1] - scale[0]) + scale[0]
    return scaled

def mini_max_from_array(array,_min=None, _max = None, scale=(0,1)):
    if _min == None:
        _min = min(array)
    if _max == None:
        _max = max(array)
    return [ mini_max(x, _min, _max, scale) for x in array], _max, _min

def revert_mini_max_from_series(series: pd.Series, _min, _max, scale = (0 ,1)):
    std = (series - scale[0])/(scale[1] - scale[0])
    values = std * (_max - _min) + _min
    return values

def mini_max_from_series(series: pd.Series, scale = (0,1)):
    _max = series.max()
    _min = series.min()
    std = (series - _min)/(_max - _min)
    scaled = std * (scale[1] - scale[0]) + scale[0]
    return scaled, _max, _min

def revert_mini_max_from_data(data, opt, scale=(0,1)):
    if type(data) == list:
        data = pd.Series(data)
    if type(data) == pd.Series:
        _min = opt[0]
        _max = opt[1]
        return revert_mini_max_from_series(data, _min, _max, scale)
    elif type(data) == pd.DataFrame:
        data_ = data.copy()
        for key in data:
            _min = opt[key][0]
            _max = opt[key][1]
            data_[key] = revert_mini_max_from_series(data[key], _min, _max, scale)
        return data_
            
        
def mini_max_from_data(data, scale=(0,1)):
    if type(data) == list:
        result, _max, _min = mini_max_from_array(data, scale)
        return result, (_min, _max)
    elif type(data) == pd.Series:
        result,_max, _min = mini_max_from_series(data, scale)
        return pd.Series(result), (_min, _max)
    elif type(data) == pd.DataFrame:
        #target is columns
        opt = {}
        data_ = data.copy()
        for key in data:
            data_[key], _max, _min = mini_max_from_series(data_[key])
            opt[key] = (_min, _max)
        return data_, opt
    else:
        #todo add numpy
        raise Exception(f"{type(data)} is not supported")

    
def standalization(value, var, mean):
    pass