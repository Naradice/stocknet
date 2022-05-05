from more_itertools import last
import numpy as np
import pandas as pd

def sum(data):
    amount = 0
    if type(data) == list:
        for value in data:
            amount += value
    else:#check more
        amount = data.sum()
    return amount

def revert_EMA(data, interval:int):
    """ revert data created by EMA function to row data

    Args:
        data (DataFrame or Series or list): data created by EMA function to row data
        interval (int): window size
    """
    
    if len(data) > interval:
        alpha_r = (interval+1)/2
        if type(data) == pd.DataFrame:
            raise Exception("Not implemented")
        result = [data[0]]
        for i in range(1,  len(data)):
            row = data[i]* alpha_r + data[i-1]*(1-alpha_r)
            result.append(row)
        return True, result
    else:
        raise Exception("data length should be greater than interval")

def EMA(data, interval):
    '''
    return list of EMA. remove interval -1 length from the data
    if data length is less than interval, return EMA with length of data as interval
    '''
    if len(data) > interval:
        if type(data) == pd.DataFrame or type(data) == pd.Series:
            return data.ewm(span=interval, adjust=False).mean()
        #ema = [np.NaN for i in range(0,interval-1)]
        lastValue = data[0]
        ema = [lastValue]
        alpha = 2/(interval+1)
        for i in range(1, len(data)):
            lastValue = lastValue * (1 - alpha) + data[i]*alpha
            ema.append(lastValue)
        return ema
    else:
        raise Exception("data list has no value")
    
def SMA(data, window):
    '''
    return list of Simple Moving Average.
    if data length is less than interval, return EMA with length of data as interval
    '''
    if window < 2:
        raise Exception(f"window size should be greater than 2. specified {window}")
    if len(data) < window:
        raise Exception(f"data length should be greater than window. currently {len(data)} < {window}")
    if type(data) == pd.DataFrame or type(data) == pd.Series:
        return data.rolling(window).mean()
    sma = [np.NaN for i in range(0, window-1)]
    ## TODO: improve the loop
    for i in range(window, len(data)+1):
        start_index = i - window
        sma_value = 0
        for j in range(start_index, start_index + window):
            sma_value += data[j]
        sma.append(sma_value/window)
    return sma

def update_macd(new_tick, short_ema_value, long_ema_value, column = 'Close', short_window=12, long_window=26):
    new_data = new_tick[column]
    short_alpha = 2/(short_window+1)
    long_alpha = 2/(long_window+1)

    ##TODO replace here from hard code to function
    new_short_ema = short_ema_value* (1 - short_alpha) + new_data*short_alpha
    new_long_ema = long_ema_value * (1 - long_alpha) + new_data * long_alpha
    new_macd = new_short_ema - new_long_ema
    
    return new_short_ema, new_long_ema, new_macd

def update_ema(new_tick, ema_value, window, column='Close'):
    new_data = new_tick[column]
    alpha = 2/(window+1)
    
    new_ema = ema_value* (1 - alpha) + new_data*alpha
    return new_ema

def MACD_from_ohlc(data, column = 'Close', short_window=12, long_window=26, signal_window=9):
    '''
    caliculate MACD and Signal indicaters from OHLC. Close is used by default.
    input: data (dataframe), column (string), short_windows (int), long_window (int), signal (int)
    output: short_ema, long_ema, MACD, signal
    '''
    short_ema = EMA(data[column], short_window)
    long_ema = EMA(data[column], long_window)
    MACD, Signal = MACD_from_EMA(short_ema, long_ema, signal_window)
    return short_ema, long_ema, MACD, Signal

def MACD_from_EMA(short_ema, long_ema, signal_window):
    '''
    caliculate MACD and Signal indicaters from EMAs.
    output: macd, signal
    '''
    if type(short_ema) == pd.Series and type(long_ema) == pd.Series:
        macd = short_ema - long_ema
    else:
        macd = [x-y for (x, y) in zip(short_ema, long_ema)]
    signal = SMA(macd, signal_window)
    return macd, signal

def bolinger_from_array(data, window = 20,  alpha=2):
    if type(data) == list:
        data = pd.Series(data)
    else:
        raise Exception(f'data type {type(data)} is not supported in bolinger_from_array')
    stds = data.rolling(window).std()
    mas = data.rolling(window).mean()
    b_high = mas.iloc[-1] + stds.iloc[-1]*alpha
    b_low = mas.iloc[-1] - stds.iloc[-1]*alpha
    return b_high, b_low, mas, stds

def bolinger_from_ohlc(data, column = 'Close', window = 20,  alpha=2):
    if type(data) == pd.DataFrame or type(data) == pd.Series:
        data = data.values
    else:
        raise Exception(f'data type {type(data)} is not supported in bolinger_from_ohlc')
    stds = data[column].rolling(window).std()
    mas = data[column].rolling(window).mean()
    b_high = mas.iloc[-1] + stds.iloc[-1]*alpha
    b_low = mas.iloc[-1] - stds.iloc[-1]*alpha
    return b_high, b_low, mas, stds