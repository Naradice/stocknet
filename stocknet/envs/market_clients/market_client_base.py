from abc import abstractmethod
import random
import pandas as pd

class MarketClientBase:
    
    def __init__(self):
        pass

    @abstractmethod
    def get_rates(self, frame, interval) -> pd.DataFrame:
        raise Exception("Need to implement get_rates")
        pass

    @abstractmethod
    def get_current_ask(self):
        raise Exception("Need to implement get_current_ask")
        pass
    
    @abstractmethod
    def get_current_bid(self):
        raise Exception("Need to implement get_current_bid")
        pass
    
    @abstractmethod
    def market_buy(self, amount):
        raise Exception("Need to implement market_buy")
        pass

    @abstractmethod
    def market_sell(self, order):
        raise Exception("Need to implement market_sell")
        pass
    
    @abstractmethod
    def buy_settlement(self, position):
        print("Need to implement buy_settlement.")
        pass
    
    @abstractmethod
    def sell_all_settlement(self):
        raise Exception("Need to implement sell_all_setlement")

    @abstractmethod
    def close(self):
        pass
    
    @abstractmethod
    def get_next_tick(self, frame=5):
        raise Exception("Need to implement get_next_tick")

    @abstractmethod
    def reset(self):
        raise Exception("Need to implement reset")
            
    @property
    def max(self):
        raise Exception("Need to implement max")
        
    @property
    def min(self):
        raise Exception("Need to implement min")
    
    @property
    def frame(self):
        raise Exception("Need to implement frame")
    
    @property
    def columns(self):
        raise Exception("Need to implement columns")
        
    @abstractmethod
    def get_ohlc_columns(self):
        raise Exception("Need to implement")    
    
    @abstractmethod
    def __getitem__(self, ndx):
        return None
    
    @abstractmethod
    def get_diffs_with_minmax(self, position='ask')-> list:
        pass
    
    @abstractmethod
    def get_positions(self):
        pass
    
    @abstractmethod
    def get_holding_steps(self, position="ask")-> list:
        pass