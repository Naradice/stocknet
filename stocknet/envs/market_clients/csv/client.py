import pandas as pd
import random
from stocknet.envs.utils import standalization

class CSVClient():

    def __read_csv__(self, columns):
        #super().__init__()
        if self.frame in self.files:
            file = self.files[self.frame]
            usecols = columns
            self.data = pd.read_csv(file,  header=0, parse_dates=True, usecols = usecols)
            self.columns = usecols
        else:
            raise Exception(f"{self.frame} is not available in CSV Client.")
        
    def __rolling_frame(self, from_frame:int, to_frame: int):
        Highs = []
        Lows = []
        Opens = []
        Closes = []
        window = to_frame/from_frame
        _window_ = window-1
        ## TODO: replace hard code
        for i in range(0, len(self.data)-window, window):
            Highs.append(self.data['High'][i:i+window].max())
            Lows.append(self.data['Low'][i:i+window].min())
            Opens.append(self.data['Open'][i])
            Closes.append(self.data['Close'][i+_window_])
        self.data = pd.DataFrame({'High': Highs, 'Low':Lows, 'Open':Opens, 'Close':Closes})

    def __init__(self, file = None, file_frame=5, provider="bitflayer", frame=None, columns = ['High', 'Low','Open','Close'], normalization = None):
        self.kinds = 'bc'
        try:
            self.frame = int(file_frame)
        except Exception as e:
            print(e)
        self.index = 0
        self.ask_positions = []
        if file == None:
            if provider == "bitflayer":
                self.files = {
                    1:'',
                    5:'/home/cow/python/torch/Stock/Data/bitcoin_5_2017T0710-2021T103022.csv'
                }
        elif type(file) == str:
            self.files = {
                self.frame: file
            }
        else:
            raise Exception(f"unexpected file type is specified. {type(file)}")
        self.__read_csv__(columns)
        if frame != None and file_frame != frame:
            try:
                to_frame = int(frame)
            except Exception as e:
                print(e)
            if file_frame < to_frame:
                self.__rolling_frame(from_frame=file_frame,to_frame=to_frame)
                self.frame = to_frame
            elif to_frame == file_frame:
                print("file_frame and frame are same value. row file_frame is used.")
            else:
                raise Exception("frame should be greater than file_frame as we can't decrease the frame.")
            
        self.__step_index = random.randint(0, len(self.data))
        ## TODO: change static string to var
        self.__high_max = self.get_min_max('High')[1]
        self.__low_min = self.get_min_max('Low')[0]

    def get_rates(self, interval=1):
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
        result = {"price":boughtCoinRate, "step":self.index, "amount":amount}
        self.ask_positions.append(result)
        return result

    def market_sell(self, order) -> dict:
        print("market_sell is not implemented.")
        return None

    def sell_settlement(self, position):
        bid = self.get_current_bid()
        bought_rate = position["price"]
        rate_diff = (bid - position["price"])
        remainings = []
        for item in self.ask_positions:
            if item["step"] != position["step"]:
                remainings.append(item)
        self.ask_positions = remainings
        return bid, bought_rate, rate_diff

    def buy_settlement(self, position):
        print("buy_settlement is not implemented.")
        pass

    def sell_all_settlement(self):
        results = []
        for position in self.ask_positions:
            bid = self.get_current_bid()
            results.append(
                (bid, position["price"], (bid - position["price"]))
                )
        self.ask_positions = []
        return results
        
    def close(self):
        pass

    def reset(self):
        self.ask_positions = []#ignore if there is position
        self.__step_index = random.randint(0, len(self.data))

    def get_next_tick(self, frame=5):
        if self.__step_index < len(self.data)-2:
            self.__step_index += 1
            tick = self.data.iloc[self.__step_index]
            return tick, False
        else:
            self.__step_index = random.randint(0, len(self.data))
            tick = self.data.iloc[self.__step_index]
            return tick, True
        
    @property
    def max(self, column):
        if column in self.data.columns:
            return self.data[column].max()
        else:
            raise ValueError(f"{column} is not defined in {self.data.columns}")
            return None
        
    @property
    def min(self, column):
        if column in self.data.columns:
            return self.data[column].min()
        else:
            raise ValueError(f"{column} is not defined in {self.data.columns}")
        
    def __getitem__(self, ndx):
        return self.data.iloc[ndx]
        
    def get_min_max(self, column):
        if column in self.data.columns:
            return self.data[column].min(), self.data[column].max()
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
                for position in self.ask_positions:
                    amounts.append(current_bid - position['price'])
                return amounts
            else:
                return [0.0]
        else:
            print(f'this is not implemented in get_diff_amount with position: {position}')
            
    def get_diffs_with_minmax(self, position='ask')-> list:
        if position == 'ask':
            if len(self.ask_positions) > 0:
                amounts = []
                current_bid = self.get_current_bid()
                current_normalized_bid = standalization.mini_max(current_bid, self.__low_min, self.__high_max, (0, 1))
                for position in self.ask_positions:
                    normalized_price = standalization.mini_max(position['price'], self.__low_min, self.__high_max, (0, 1))
                    amounts.append(current_normalized_bid - normalized_price)
                return amounts
            else:
                return [0.0]
        else:
            print(f'this is not implemented in get_diff_amount with position: {position}')
            
    def get_positions(self, position=None):
        if position == None:
            #self.ask_positions
            pass