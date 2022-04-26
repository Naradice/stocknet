from stocknet.envs.datasets.utils import indicaters, standalization
import gym
import random
import pandas as pd
import numpy as np
from stocknet.envs.render.graph import Rendere
from stocknet.envs.market_clients.market_client_base import MarketClientBase
from stocknet.envs.utils.preprocess import ProcessBase

class TickEnv(gym.Env):

    def __init__(self, data_client:MarketClientBase, columns = ['High', 'Low','Open','Close'], observationDays=1, useBudgetColumns=True, featureFirst=True, use_diff= True):
        '''
        init 
        '''        
        self.data_client = data_client
        ## initialize render params
        self.viewer = Rendere()
        self.viewer.add_subplots(3)# 0:row candle, 1:  row macd, 2:diff candle, 3 diff macd
        
        ## initialize params
        self.__ubc__ = useBudgetColumns
        self.__ff__ = featureFirst
        self.INVALID_REWARD = -.001
        self.VALID_REWARD = 0.0005
        self.seed = 0
        self.dataLength = int((24*12)*observationDays)
        self.action_space = gym.spaces.Discrete(3)
        self.dataSet = None
        ## columns to ingest values from dataSet for observation
        self.ohlc_columns_dict = self.data_client.get_ohlc_columns()
        self.columns = []
        # initialize macd if it is specified
        columns = [str.lower(column) for column in columns]
        required_length = self.dataLength        
        ## add columns
        self.__ohlc_columns = [self.ohlc_columns_dict['Open'], self.ohlc_columns_dict['High'], self.ohlc_columns_dict['Low'], self.ohlc_columns_dict['Close']]
        if 'open' in columns:
            self.columns.append(self.ohlc_columns_dict['Open'])
        if 'close' in columns:
            self.columns.append(self.ohlc_columns_dict['Close'])
        if 'high' in columns:
            self.columns.append(self.ohlc_columns_dict['High'])
        if 'low' in columns:
            self.columns.append(self.ohlc_columns_dict['Low'])
        
        if useBudgetColumns:
            ob_shape = (self.dataLength, len(self.columns) + 2)
        else:
            ob_shape = (self.dataLength, len(self.columns))
            
        ## initialize gym parameters
        self.observation_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=ob_shape
        )
        self.reward_range = [-1., 1.]
        
        ## initialize
        self.reset()

    def __initialize_indicaters(self):
        pass
    
    def __run_processes(self, tick: pd.DataFrame):
        pass
    
    def add_indicater(self, process: ProcessBase, option=None):
        values_dict = process.run(self.__rowdata__, option)
        for column, values in values_dict.items():
            self.__rowdata__[column] = values
            self.columns.append(column)
            
    def register_preprocess(self, process: ProcessBase, option=None):
        """ register preprocess for data.

        Args:
            process (ProcessBase): any process you define
            option (_type_): option for a process
        """
        self.__preprocesess.append((process, option))
    
    def register_preprocesses(self, processes: list, options: dict):
        """ register preprocess for data.

        Args:
            processes (list[processBase]): any process you define
            options: option for a processes[key] (key is column name of additional data)
        """
        
        for process in processes:
            key = process.key
            option = None
            if key in options:
                option = options[key]
            self.register_preprocess(process, option)
            
    @classmethod
    def get_available_columns(self):
        return ['High', 'Low','Open','Close','macd', 'bolinger']
    
    @property
    def ask(self):
        return self.data_client.get_current_ask()
    
    @property
    def bid(self):
        return self.data_client.get_current_bid()
    
    def get_data_index(self):
        return self.data_client.get_current_index()
    
    def get_tick_from_index(self, index):
        return self.data_client[index]
        
    def get_obs_from_index(self, index):
        if self.__req_length <= index:
            data = self.data_client[index -self.__req_length:index]
            data["short_ema"], data["long_ema"], data["macd"], data["signal"] = self.MACD(data.Close)
            return data
        else:
            raise Exception(f"index should be greater than required length. {index} < {self.__req_length}")
    
    def __update_observation(self, tick:pd.DataFrame):
        self.dataSet = pd.concat([self.dataSet, pd.DataFrame.from_records([tick.to_dict()])])
        obs = self.run_preprocess()
        ohlc = self.dataSet[self.__ohlc_columns].iloc[-self.dataLength:]
        self.viewer.register_ohlc(ohlc, 0, 'ROW OHLC Candle', *self.__ohlc_columns)
        return obs
    
    def get_next_observation(self):
        tick, done = self.data_client.get_next_tick()
        obs = self.__update_observation(tick)
        return obs, done
        
    def reset(self):
        '''
        '''
        self.pl = 0
        self.askPositions = []
        self.bidPositions = []
        self.boughtCoinRate = 0
        self.soldCoinRate = 0
        self.budget = 1
        self.coin = 0
        self.budgets = [0 for i in range(0, self.dataLength)]
        #self.coins = [self.coin for i in range(0, self.dataLength)]
        self.rewards = 0
        
        ## reset index
        self.data_client.reset()
        req_length = self.__get_required_length()
        self.dataSet = self.data_client.get_rates(req_length)
        
        ## add inidicaters to dataSet and columns
        obs = self.__initialize_processes()
        if self.__ubc__:
            obs['budgets'] = self.budgets
        
    def evaluate(self, action):
        reward = 0
        if action == 1:
            '''
            buy coin with 10 point if possible.
            if you don't have much budget, return negative reward 
            '''
            reward = self.__buyCoin__()

        elif action == 2:
            '''
            sell coin with 100% if possible. (R_sell - R_buy)/R_buy
            if you don't have coin, return negative reward
            '''
            reward = self.__sellCoin__()

        elif action == 0:
            '''
            hold.
            '''
            reward = self.__stay__()
        else:
            raise Exception(f"The action number {action} exeeds the lengths in evaluate function.")
        return reward

    def step(self, action): # actionを実行し、結果を返す
        done = False
        reward = 0

        reward = self.evaluate(action)        
        self.rewards += reward
        self.observation, done = self.get_next_observation()
        
        if self.pl < -1:
            done = True
        if self.__ubc__:
            self.observation["budgets"] = self.budgets
            #self.observation["coins"] = self.coins
        
        if self.__ff__:
            return self.observation.T.to_numpy(), reward, done, {}
        else:
            return self.observation.to_numpy(), reward, done, {}
        
    def render(self, mode='human', close=False):
        '''
        '''
        self.viewer.plot()
    
    def close(self): # 環境を閉じて後処理をする
        '''
        '''
        self.data_client.close()
        pass
    
    def seed(self, seed=None): # ランダムシードを固定する
        '''
        '''
        if seed == None:
            seed = 0
        random.seed(seed)
        self.data_client.set_seed(seed)
        
    def __set__simple_bugets(self, value):
        self.budget = value
        self.budgets = [value for i in range(0, self.dataLength)]
        
    def __set__simple_coins(self, value):
        self.coin = value
        self.coins = [value for i in range(0, self.dataLength)]
        
    
    def __set_history_bugets(self, new_buget):
        self.budget = new_buget
        self.budgets[:-1] = self.budgets[1:]
        self.budgets[-1] = self.budget
        
    def __set_hisoty_coins(self, new_coin):
        self.coin = new_coin
        self.coins[:-1] = self.coins[1:]
        self.coins[-1] = self.coin
        
    def __set_diff_as_bugets(self):
        diffs = self.data_client.get_diffs_with_minmax()
        amount = 0
        if len(diffs) > 1:
            print(f"Warning: diffs have two or more positions. {len(diffs)}")
        for diff in diffs:
            amount += diff
        self.budgets[:-1] = self.budgets[1:len(self.budgets)]
        self.budgets[-1] = amount
        
    def __buyCoin__(self, amount=1):
        '''
        buy coin using all budget
        '''
        if self.budget > 0:
            self.budget = 0
            ### Multiple order version
            #self.budget = self.budgets[len(self.budgets)-1] - budget
            #use all buget for simplify
            #self.coin = budget/self.current_buy_rate
            
            ## Simple Version
            #self.__set__simple_bugets(0)
            #self.__set__simple_coins(1)
            self.__set_diff_as_bugets()
            self.coin = 1
            result = self.data_client.market_buy(amount=amount)
            
            return self.VALID_REWARD
        else:
            return self.INVALID_REWARD
        
    def __sellCoin__(self):
        if self.coin > 0:
            #add base reward for valid action
            reward = self.VALID_REWARD
            #self.__set__simple_bugets(1)
            #self.__set__simple_coins(0)
            self.__set__simple_bugets(0)
            results = self.data_client.sell_all_settlement()
            self.budget = 1
            self.coin = 0
            count = 0
            
            for result in results:
                count += 1
                reward += result[2]/result[1]
                self.pl += result[2]/result[1]
            if reward > 0:
                reward = reward * 10#geta
            reward = np.clip(reward, *self.reward_range)
            return reward
        else:
            return self.INVALID_REWARD
    
    def __stay__(self, S=5, T=1/2):
        reward = 0.0
        self.__set_diff_as_bugets()
        #sell_price = self.bid
        #for position in self.askPositions:
        #    ask_diff = (sell_price - position['price'])/position['price']
        #    step_diff = self.index - position['step']
        #    if step_diff < S:
        #        alpha = step_diff/S * T
        #    else:
        #        alpha = T
        #    reward += alpha * ask_diff
        return reward