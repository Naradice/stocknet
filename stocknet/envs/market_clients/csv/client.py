import pandas as pd
import random
import os
import uuid
from stocknet.envs.utils import standalization
from stocknet.envs.market_clients.frames import Frame

class CSVClient():
    kinds = 'csv'

    def __read_csv__(self, columns, date_col=None):
        #super().__init__()
        if self.frame in self.files:
            file = self.files[self.frame]
            usecols = columns
            if date_col != None:
                if date_col not in usecols:
                    usecols.append(date_col)
                self.data = pd.read_csv(file,  header=0, parse_dates=[date_col], usecols = usecols)
            else:
                self.data = pd.read_csv(file,  header=0, usecols = usecols)
            self.columns = usecols
        else:
            raise Exception(f"{self.frame} is not available in CSV Client.")
        
    def __rolling_frame(self, data:pd.DataFrame, from_frame:int, to_frame: int):
        Highs = []
        Lows = []
        Opens = []
        Closes = []
        window = int(to_frame/from_frame)
        _window_ = window-1
        ## TODO: replace hard code
        for i in range(0, len(data), window):
            Highs.append(data['High'].iloc[i:i+window].max())
            Lows.append(data['Low'].iloc[i:i+window].min())
            Opens.append(data['Open'].iloc[i])
            Closes.append(data['Close'].iloc[i+_window_])
        return pd.DataFrame({'High': Highs, 'Low':Lows, 'Open':Opens, 'Close':Closes})
    
    def __get_rolling_start_index(self):
        pass

    def __init__(self, file = None, frame: int= Frame.MIN5, provider="bitflayer", out_frame:int=None, columns = ['High', 'Low','Open','Close'], date_column = "Timestamp", seed=1017):
        """CSV Client for bitcoin, etc. currently bitcoin in available only.
        Need to change codes to use settings file
        
        Args:
            file (str, optional): You can directly specify the file name. Defaults to None.
            file_frame (int, optional): You can specify the frame of data. CSV need to exist. Defaults to 5.
            provider (str, optional): Provider of data to load csv file. Defaults to "bitflayer".
            out_frame (int, optional): output frame. Ex F_5MIN can convert to F_30MIN. Defaults to None.
            columns (list, optional): ohlc columns name. Defaults to ['High', 'Low','Open','Close'].
            date_column (str, optional): If specified, time is parsed. Otherwise ignored. Defaults to Timestamp
        """
        random.seed(seed)
        self.args = (file, frame, provider, out_frame, columns, date_column, seed)
        if type(file) == str:
            file = os.path.abspath(file)
        try:
            self.frame = frame
        except Exception as e:
            print(e)
        self.ask_positions = {}
        if file == None:
            if provider == "bitflayer":
                self.files = {
                    1:'',
                    5:'/home/cow/python/torch/Stock/Data/bitcoin_5_2017T0710-2021T103022.csv',
                    'provider': provider
                }
                self.base_point = 0.01
        elif type(file) == str:
            self.files = {
                self.frame: file,
                'provider': provider
            }
            if provider == "bitflayer":
                self.base_point = 0.01
        else:
            raise Exception(f"unexpected file type is specified. {type(file)}")
        self.__read_csv__(columns, date_column)
        self.date_column = date_column
        if out_frame != None and frame != out_frame:
            try:
                to_frame = int(out_frame)
            except Exception as e:
                print(e)
            if frame < to_frame:
                self.data = self.__rolling_frame(from_frame=frame, to_frame=to_frame)
                self.frame = to_frame
            elif to_frame == frame:
                print("file_frame and frame are same value. row file_frame is used.")
            else:
                raise Exception("frame should be greater than file_frame as we can't decrease the frame.")
            self.out_frames = to_frame
        else:
            self.out_frames = self.frame
        self.__step_index = random.randint(0, len(self.data))
        ## TODO: change static string to var
        self.__high_max = self.get_min_max('High')[1]
        self.__low_min = self.get_min_max('Low')[0]

    def get_rates(self, interval=1, frame:int=None, live:bool=False):
        if frame == None or frame == self.frame:
            if interval > 1:
                rates = None
                if self.__step_index >= interval-1:
                    try:
                        #return data which have interval length
                        rates = self.data.iloc[self.__step_index - interval+1:self.__step_index+1].copy()
                        return rates
                    except Exception as e:
                        print(e)
                else:
                    self.__step_index = random.randint(0, len(self.data))
                    return self.get_rates(interval)
            elif interval == -1:
                rates = self.data.copy()
                return rates
            else:
                raise Exception("interval should be greater than 0.")
        elif frame > self.frame:
            if self.date_column == None:
                pass
            else:
                rolling_start_index = None
                if frame <= 60:
                    for i in range(0, int(60/self.frame)):
                        current_min = self.data[self.date_column].iloc[self.__step_index - i].minute/self.frame
                        if current_min % frame == 0:
                            rolling_start_index = self.__step_index - i
                            break
                elif frame <= 60*24:
                    raise NotImplemented
                else:
                    raise NotImplemented
                rates = self.data.iloc[rolling_start_index - int(interval * (frame/self.frame)) +1 :rolling_start_index+1].copy()
                rates = self.__rolling_frame(rates, self.frame, frame)
            return rates
        else:
            raise Exception("frame should be greater than data frame")

    def get_current_ask(self):
        tick = self.data.iloc[self.__step_index]
        mean = (tick.High + tick.Low)/2
        return random.uniform(mean, tick.High)
        

    def get_current_bid(self):
        tick = self.data.iloc[self.__step_index]
        mean = (tick.High + tick.Low)/2
        return random.uniform(tick.Low, mean)

    def market_buy(self, amount) -> dict:
        boughtCoinRate = self.get_current_ask()
        id = str(uuid.uuid4())
        position = {"price":boughtCoinRate, "step":self.__step_index, "amount":amount, "id":id}
        self.ask_positions[id] = position
        return position

    def market_sell(self, order) -> dict:
        print("market_sell is not implemented.")
        return None

    def sell_settlement(self, position:dict, point: int=None):
        if position["id"] in self.ask_positions:
            bid = self.get_current_bid()
            bought_rate = position["price"]
            amount = position["amount"]
            if point == None or point >= amount:##sell all amounts
                rate_diff = (bid - position["price"])
                self.ask_positions.pop(position["id"])
            elif point < amount:
                rate_diff = (bid - position["price"])
                id = position["id"]
                remaining_amount = amount - point
                position["amount"] = remaining_amount
                self.ask_positions[id] = position
            else:
                raise Exception("amount should be int")
            
            pl = rate_diff * point * self.base_point
            
            return bid, bought_rate, rate_diff, pl
        else:
            return None, None, None
        
    def buy_settlement(self, position):
        print("buy_settlement is not implemented.")
        pass

    def sell_all_settlement(self):
        results = []
        bid = self.get_current_bid()
        for key, position in self.ask_positions.items():
            rate_diff = (bid - position["price"])
            pl = rate_diff * position["amount"] * self.base_point
            results.append(
                (bid, position["price"], rate_diff, pl)
                )
        self.ask_positions = {}
        return results
        
    def close(self):
        pass

    def reset(self):
        self.ask_positions = {}#ignore if there is position
        self.__step_index = random.randint(0, len(self.data))

    def get_holding_steps(self, position="ask"):
        steps_diff = []
        for key, ask_position in self.ask_positions.items():
            steps_diff.append(self.__step_index - ask_position["step"])
        return steps_diff

    def get_next_tick(self, frame=5):
        if self.__step_index < len(self.data)-2:
            self.__step_index += 1
            tick = self.data.iloc[self.__step_index]
            return tick, False
        else:
            self.__step_index = random.randint(0, len(self.data))
            tick = self.data.iloc[self.__step_index]
            return tick, True
        
    def max(self, column):
        if column in self.data.columns:
            return self.data[column].max()
        else:
            raise ValueError(f"{column} is not defined in {self.data.columns}")
            return None
        
    def min(self, column):
        if column in self.data.columns:
            return self.data[column].min()
        else:
            raise ValueError(f"{column} is not defined in {self.data.columns}")
        
    def __getitem__(self, ndx):
        return self.data.iloc[ndx]
        
    def get_min_max(self, column, data_length = 0):
        if column in self.data.columns:
            if data_length == 0:
                return self.data[column].min(), self.data[column].max()
            else:
                if data_length > 0:
                    target_df = self.data[column].iloc[self.__step_index:self.__step_index + data_length]
                else:
                    target_df = self.data[column].iloc[self.__step_index + data_length + 1:self.__step_index +1]
                return target_df.min(), target_df.max()
                    
        else:
            raise ValueError(f"{column} is not defined in {self.data.columns}")
        
    def get_current_index(self):
        return self.__step_index
    
    def get_ohlc_columns(self):
        columns = {}
        for column in self.data.columns.values:
            column_ = str(column).lower()
            if column_ == 'open':
                columns['Open'] = column
            elif column_ == 'high':
                columns['High'] = column
            elif column_ == 'low':
                columns['Low'] = column
            elif column_ == 'close':
                columns['Close'] = column
        return columns
    
    def get_diffs(self, position='ask') -> list:
        if position == 'ask':
            if len(self.ask_positions) > 0:
                amounts = []
                current_bid = self.get_current_bid()
                for key, position in self.ask_positions.items():
                    amounts.append(current_bid - position['price'])
                return amounts
            else:
                return [0.0]
        else:
            print(f'this is not implemented in get_diff_amount with position: {position}')
        
    ##minimux should be done in this class?
    def get_diffs_with_minmax(self, position='ask')-> list:
        if position == 'ask':
            if len(self.ask_positions) > 0:
                amounts = []
                current_bid = self.get_current_bid()
                current_normalized_bid = standalization.mini_max(current_bid, self.__low_min, self.__high_max, (0, 1))
                for key, position in self.ask_positions.items():
                    normalized_price = standalization.mini_max(position['price'], self.__low_min, self.__high_max, (0, 1))
                    amounts.append((current_normalized_bid - normalized_price) * position["amount"])
                return amounts
            else:
                return [0.0]
        else:
            print(f'this is not implemented in get_diff_amount with position: {position}')
            
    def get_positions(self, kinds=None):
        positions = []
        if kinds == None:
            pass
        elif kinds == 'ask':
            #for key, position in self.ask_positions.items():
            #    positions.append(position)
            return self.ask_positions
    
    def get_params(self) -> dict:
        param = {
            'type':self.kinds,
            'provider': self.files,
            'source_frame': self.frame,
            'out_frame': self.out_frames
        }
        
        return param

class MultiFrameClient(CSVClient):
    
    kinds = "multi_csv"
    
    def __init__(self, file=None, frame: int = Frame.MIN5, provider="bitflayer", columns=['High', 'Low', 'Open', 'Close'], out_frames = [Frame.MIN30, Frame.H1], date_column="Timestamp", seed=1017):
        out_frame =None
        super().__init__(file, frame, provider, out_frame, columns, date_column, seed)
        self.args = (file, frame, provider, out_frame, columns, out_frames, date_column, seed)
        self.out_frames = out_frames
        
    def get_ohlc_columns(self):
        columns = {}
        for column in self.data.columns.values:
            column_ = str(column).lower()
            if column_ == 'open':
                columns['Open'] = [f'{str(frame)}_column' for frame in self.out_frames]
            elif column_ == 'high':
                columns['High'] = column
            elif column_ == 'low':
                columns['Low'] = column
            elif column_ == 'close':
                columns['Close'] = column
        return columns