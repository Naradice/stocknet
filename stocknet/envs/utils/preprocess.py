import numpy
from stocknet.envs.utils import standalization, indicaters
import pandas as pd
from stocknet.envs.utils.process import ProcessBase

class MACDpreProcess(ProcessBase):
    
    option = {
        "column": "Close",
        "short_window": 12,
        "long_window": 26,
        "signal_window":9
    }
    
    last_data = None
    
    def __init__(self, key='macd', option=None):
        super().__init__(key)
        if option != None:
            self.option.update(option)
        self.columns = {
            'S_EMA': f'{key}_S_EMA', 'L_EMA':f'{key}_L_EMA', 'MACD':f'{key}_MACD', 'Signal':f'{key}_Signal'
        }

    def run(self, data: pd.DataFrame):
        option = self.option
        target_column = option['column']
        short_window = option['short_window']
        long_window = option['long_window']
        signal_window = option['signal_window']
        self.option = option
        
        short_ema, long_ema, MACD, Signal = indicaters.MACD_from_ohlc(data, target_column, short_window, long_window, signal_window)
        
        cs_ema = self.columns['S_EMA']
        cl_ema = self.columns['L_EMA']
        c_macd = self.columns['MACD']
        c_signal = self.columns['Signal']
        
        self.last_data = pd.DataFrame({cs_ema:short_ema, cl_ema: long_ema, c_macd:MACD, c_signal:Signal}).iloc[-self.get_minimum_required_length():]
        return {cs_ema:short_ema, cl_ema: long_ema, c_macd:MACD, c_signal:Signal}
    
    def update(self, tick:pd.Series):
        option = self.option
        target_column = option['column']
        short_window = option['short_window']
        long_window = option['long_window']
        signal_window = option['signal_window']
        
        cs_ema = self.columns['S_EMA']
        cl_ema = self.columns['L_EMA']
        c_macd = self.columns['MACD']
        c_signal = self.columns['Signal']
        
        
        short_ema, long_ema, MACD = indicaters.update_macd(
            new_tick=tick,
            short_ema_value=self.last_data[cs_ema].iloc[-1],
            long_ema_value=self.last_data[cl_ema].iloc[-1],
            column=target_column, short_window = short_window, long_window = long_window)
        Signal = (self.last_data[c_macd].iloc[-signal_window + 1:].sum() + MACD)/signal_window
        
        new_data = pd.Series({cs_ema:short_ema, cl_ema: long_ema, c_macd:MACD, c_signal:Signal})
        self.last_data = self.concat(self.last_data.iloc[1:], new_data)
        return new_data
        
    def get_minimum_required_length(self):
        return self.option["long_window"] + self.option["signal_window"] - 2
    
    def revert(self, data_set:tuple):
        cs_ema = self.columns['S_EMA']
        
        if type(data_set) == pd.DataFrame:
            if cs_ema in data_set:
                data_set = (data_set[cs_ema],)
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
    
    def __init__(self, key='ema', window = 12, column = 'Close'):
        super().__init__(key)
        self.option['window'] = window
        self.option['column'] = column
        self.columns = {
            "EMA":f'{key}_EMA'
        }

    def run(self, data: pd.DataFrame):
        option = self.option
        target_column = option['column']
        window = option['window']
        column = self.columns["EMA"]
        
        ema = indicaters.EMA(data[target_column], window)
        self.last_data = pd.DataFrame({column:ema}).iloc[-self.get_minimum_required_length():]
        return {column:ema}
    
    def update(self, tick:pd.Series):
        option = self.option
        target_column = option['column']
        window = option['window']
        column = self.columns["EMA"]
        
        
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

class BBANDpreProcess(ProcessBase):
    
    option = {
        "column": "Close",
        "window": 14,
        'alpha':2
    }
    
    last_data = None
    
    def __init__(self, key='bolinger', window = 14, alpha=2, target_column = 'Close'):
        super().__init__(key)
        self.option['column'] = target_column
        self.option['window'] = window
        self.option['alpha'] = alpha
        
        self.columns = {
            "MB": f"{key}_MB",
            "UB": f"{key}_UB",
            "LB": f"{key}_LB",
            "Width": f"{key}_Width"
        }

    def run(self, data: pd.DataFrame):
        option = self.option
        target_column = option['column']
        window = option['window']
        alpha = option['alpha']
        
        ema, ub, lb, bwidth = indicaters.bolinger_from_ohlc(data, target_column, window=window, alpha=alpha)
        
        c_ema = self.columns['MB']
        c_ub = self.columns['UB']
        c_lb = self.columns['LB']
        c_width = self.columns['Width']
        
        self.last_data = pd.DataFrame({c_ema:ema, c_ub: ub, c_lb:lb, c_width:bwidth, target_column:data[target_column] }).iloc[-self.get_minimum_required_length():]
        return {c_ema:ema, c_ub: ub, c_lb:lb, c_width:bwidth}
    
    def update(self, tick:pd.Series):
        option = self.option
        target_column = option['column']
        window = option['window']
        alpha = option['alpha']
        
        target_data = self.last_data[target_column].values
        target_data = numpy.append(target_data, tick[target_column])
        target_data = target_data[-window:]
        
        
        new_sma = target_data.mean()
        std = target_data.std(ddof=0)
        new_ub = new_sma + alpha * std
        new_lb = new_sma - alpha * std
        new_width = alpha*2*std
        
        c_ema = self.columns['MB']
        c_ub = self.columns['UB']
        c_lb = self.columns['LB']
        c_width = self.columns['Width']
        
        new_data = pd.Series({c_ema:new_sma, c_ub: new_ub, c_lb:new_lb, c_width:new_width, target_column: tick[target_column]})
        self.last_data = self.concat(self.last_data.iloc[1:], new_data)
        return new_data[[c_ema, c_ub, c_lb, c_width]]
        
    def get_minimum_required_length(self):
        return self.option['window']
    
    def revert(self, data_set:tuple):
        #pass
        return True, None

class ATRpreProcess(ProcessBase):
    
    option = {
        "ohlc_column": ('Open', 'High', 'Low', 'Close'),
        "window": 14
    }
    
    last_data = None
    
    available_columns = ["ATR"]
    columns = available_columns
    
    def __init__(self, key='atr', window = 14, alpha=2, ohlc_column_name = ('Open', 'High', 'Low', 'Close')):
        super().__init__(key)
        self.option['column'] = ohlc_column_name
        self.option['window'] = window
        self.columns = {'ATR': f'{key}_ATR'}

    def run(self, data: pd.DataFrame):
        option = self.option
        target_columns = option['ohlc_column']
        window = option['window']
        
        atr_series = indicaters.ATR_from_ohlc(data, target_columns, window=window)
        self.last_data = data.iloc[-self.get_minimum_required_length():].copy()
        last_atr = atr_series.iloc[-self.get_minimum_required_length():].values
        
        c_atr = self.columns['ATR']
        self.last_data[c_atr] = last_atr
        return {c_atr:atr_series.values}
        
    def update(self, tick:pd.Series):
        option = self.option
        target_columns = option['ohlc_column']
        window = option['window']
        
        pre_data = self.last_data.iloc[-1]
        new_atr_value = indicaters.update_ATR(pre_data, tick, target_columns, window)
        df = tick.copy()
        c_atr = self.columns['ATR']
        df[c_atr] = new_atr_value
        self.last_data = self.concat(self.last_data.iloc[1:], df)
        return df[[c_atr]]
        
    def get_minimum_required_length(self):
        return self.option['window']
    
    def revert(self, data_set:tuple):
        #pass
        return True, None

class RSIpreProcess(ProcessBase):
    
    option = {
        "ohlc_column": ('Open', 'High', 'Low', 'Close'),
        "window": 14
    }
    
    last_data = None
    
    available_columns = ["RSI"]
    columns = available_columns
    
    def __init__(self, key='rsi', window = 14, ohlc_column_name = ('Open', 'High', 'Low', 'Close')):
        super().__init__(key)
        self.option['column'] = ohlc_column_name
        self.option['window'] = window
        self.columns = {
            "RSI": f'{key}_RSI',
            "GAIN": f'{key}_AVG_GAIN',
            "LOSS": f'{key}_AVG_LOSS'
        }

    def run(self, data: pd.DataFrame):
        option = self.option
        target_column = option['ohlc_column'][0]
        window = option['window']
        c_rsi = self.columns['RSI']
        c_gain = self.columns['GAIN']
        c_loss = self.columns['LOSS']
        
        atr_series = indicaters.RSI_from_ohlc(data, target_column, window=window)
        atr_series.columns = [c_gain, c_loss, c_rsi]
        self.last_data = data.iloc[-self.get_minimum_required_length():]
        self.last_data[c_rsi] = atr_series.iloc[-self.get_minimum_required_length():]
        return atr_series[c_rsi].values
        
    def update(self, tick:pd.Series):
        option = self.option
        target_column = option['ohlc_column'][0]
        window = option['window']
        c_rsi = self.columns['RSI']
        c_gain = self.columns['GAIN']
        c_loss = self.columns['LOSS']
        columns = (c_gain, c_loss, c_rsi, target_column)
        
        pre_data = self.last_data.iloc[-1]
        new_gain_val, new_loss_val, new_rsi_value = indicaters.update_RSI(pre_data, tick, columns, window)
        tick[c_gain] = new_gain_val
        tick[c_loss] = new_loss_val
        tick[c_rsi] = new_rsi_value
        self.last_data = self.concat(self.last_data.iloc[1:], tick)
        return tick[[c_rsi]]
        
    def get_minimum_required_length(self):
        return self.option['window']
    
    def revert(self, data_set:tuple):
        #pass
        return True, None


####
# Not implemented as I can't caliculate required length
####
class RollingProcess(ProcessBase):
    last_tick:pd.DataFrame = None
    
    def __init__(self, key = "roll", frame_from:int = 5, frame_to: int = 30):
        super().__init__(key)
        self.frame_from = frame_from
        self.frame_to = frame_to
        raise NotImplemented
        
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