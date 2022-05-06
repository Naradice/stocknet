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

def update_EMA(last_ema_value, new_value, window:int):
    """
    update Non weighted EMA with alpha= 2/(1+window)
    
    Args:
        last_ema_value (float): last EMA value caluculated by EMA function
        new_value (float): new value of data
        window (int): window size
    """
    alpha = 2/(1+window)
    return last_ema_value * (1 - alpha) + new_value*alpha
    

def EMA(data, interval):
    '''
    return list of EMA. remove interval -1 length from the data
    if data length is less than interval, return EMA with length of data as interval
    '''
    if len(data) > interval:
        if type(data) == pd.DataFrame or type(data) == pd.Series:
            data_cp = data.copy()
            return data_cp.ewm(span=interval, adjust=False).mean()
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
    
def EWA(data:pd.DataFrame, window:int, alpha=None):
    """ Caliculate Exponential Weighted Moving Average

    Args:
        data (pd.DataFrame): ohlc data
        window (int): window size
        alpha(float, optional): specify weight value. Defaults to 2/(1+window). 0 < alpha <= 1.
    """
    if len(data) > window:
        if type(data) == pd.DataFrame or type(data) == pd.Series:
            data_cp = data.copy()
            if alpha == None:
                return data_cp.ewm(span=window, adjust=True).mean()
            else:
                return data_cp.ewa(span=window, adjust=True, alpha=alpha)
        lastValue = data[0]
        ema = [lastValue]
        alpha = 2/(window+1)
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
    """ caliculate latest macd with new ohlc data and previouse ema values
    you need to caliculate Signal on yourself if need it
    Args:
        new_tick (pd.Series): latest ohlc tick. length should be 1
        short_ema_value (float): previouse ema value of short window (fast ema)
        long_ema_value (float): previouse ema value of long window (late ema)
        column (str, optional): Defaults to 'Close'.
        short_window (int, optional):  Defaults to 12.
        long_window (int, optional): Defaults to 26.

    Returns:
        tuple(float, float, float): short_ema, long_ema, macd
    """
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

def bolinger_from_series(data: pd.Series, window = 14, alpha=2):
    stds = data.rolling(window).std(ddof=0)
    mas = data.rolling(window).mean()
    b_high = mas + stds*alpha
    b_low = mas - stds*alpha
    return mas, b_high, b_low, stds*alpha*2

def bolinger_from_array(data, window = 14,  alpha=2):
    if type(data) == list:
        data = pd.Series(data)
    else:
        raise Exception(f'data type {type(data)} is not supported in bolinger_from_array')
    return bolinger_from_series(data, window=window, alpha=alpha)

def bolinger_from_ohlc(data: pd.DataFrame, column = 'Close', window = 14, alpha=2):
    return bolinger_from_series(data[column], window=window, alpha=alpha)

def ATR_from_ohlc(data: pd.DataFrame, ohlc_columns = ('Open', 'High', 'Low', 'Close'), window = 14):
    """
    function to calculate True Range and Average True Range
    
    Args:
        data (pd.DataFrame): ohlc data
        ohlc_columns (tuple, optional): Defaults to ('Open', 'High', 'Low', 'Close').
        window (int, optional): Defaults to 14.

    Returns:
        pd.Series: Name:ATR, dtype:float64. inlucdes Null till window size
    """
    high_cn = ohlc_columns[1]
    low_cn = ohlc_columns[2]
    close_cn = ohlc_columns[3]
    
    df = data.copy()
    df["H-L"] = df[high_cn] - df[low_cn]
    df["H-PC"] = abs(df[high_cn] - df[close_cn].shift(1))
    df["L-PC"] = abs(df[low_cn] - df[close_cn].shift(1))
    df["TR"] = df[["H-L","H-PC","L-PC"]].max(axis=1, skipna=False)
    #df["ATR"] = df["TR"].ewm(span=window, adjust=False).mean()#removed min_periods=window option
    df["ATR"] = EMA(df["TR"], interval=window)
    #df["ATR"] = df["TR"].rolling(window=n).mean()
    return df["ATR"]

def update_ATR(pre_data:pd.Series, new_data: pd.Series, ohlc_columns = ('Open', 'High', 'Low', 'Close'), window = 14):
    """ latest caliculate atr

    Args:
        pre_data (pd.Series): ohlc + ATR
        new_data (pd.Series): ohlc
        ohlc_columns (tuple, optional): Defaults to ('Open', 'High', 'Low', 'Close').
        window (int, optional): Defaults to 14.

    Returns:
        float: new atr value
    """
    high_cn = ohlc_columns[1]
    low_cn = ohlc_columns[2]
    close_cn = ohlc_columns[3]
    pre_tr = pre_data['ATR']
    
    hl = new_data[high_cn] - new_data[low_cn]
    hpc = abs(new_data[high_cn] - pre_data[close_cn])
    lpc = abs(new_data[low_cn] - pre_data[close_cn])
    tr = max([hl, hpc, lpc])
    atr = update_EMA(last_ema_value=pre_tr,new_value=tr, window=window)
    return atr

def RSI_from_ohlc(data:pd.DataFrame, column = 'Close', window=14):
    """
    
    RSI is a momentum oscillator which measures the speed and change od price movements

    Args:
        data (pd.DataFrame): ohlc time series data
        column (str, optional): Defaults to 'Close'.
        window (int, optional): Defaults to 14.

    Returns:
        _type_: 0 to 100
    """
    df = data.copy()
    df["change"] = df[column].diff()
    df["gain"] = np.where(df["change"]>=0, df["change"], 0)
    df["loss"] = np.where(df["change"]<0, -1*df["change"], 0)
    df["avgGain"] = df["gain"].ewm(alpha=1/window, min_periods=window).mean() ##tradeview said exponentially weighted moving average with aplpha = 1/length is used
    df["avgLoss"] = df["loss"].ewm(alpha=1/window, min_periods=window).mean()
    df["rs"] = df["avgGain"]/df["avgLoss"]
    df["rsi"] = 100 - (100/ (1 + df["rs"]))
    return df["rsi"]

