from stocknet.envs.utils import indicaters, standalization
import gym
import random
import pandas as pd
import numpy as np
from stocknet.envs.render.graph import Rendere
from stocknet.envs.market_clients.market_client_base import MarketClientBase
#from render.graph import Rendere

class BC5Env(gym.Env):
    
    def MACD(self, data, short_window=12, long_window=26, signal_window = 9 ):
        s_ema = indicaters.EMA(data, short_window)
        l_ema = indicaters.EMA(data, long_window)
        macd = [x-y for (x, y) in zip(s_ema, l_ema)]
        signal = indicaters.SMA(macd, signal_window)
        return s_ema, l_ema, macd, signal
    
    def __initMACD__(self, short_window=12, long_window=26, signal_window = 9):
        self.dataSet["short_ema"], self.dataSet["long_ema"], self.dataSet["macd"], self.dataSet["signal"] = self.MACD(self.dataSet.Close, short_window, long_window, signal_window)
        
        
    def Bolinger(self, data, alpha=2, window=20):
        b_high, b_low, ema, stds = indicaters.bolinger_from_ohlc(data, column='Close', window=window, alpha=alpha)
        
    def __initBolinger__(self,alpha=2, window=20):
        b_high, b_low, ema, stds = self.Bolinger(self.dataSet, alpha, window)
        self.dataSet['b_mean'] = ema
        self.dataSet['b_high'] = ema + stds*alpha
        self.dataSet['b_low'] = ema - stds*alpha
        self.dataSet['stds'] = stds

    def __init__(self, data_client:MarketClientBase, max_step:int, columns = ['High', 'Low','Open','Close'], observationDays=1, useBudgetColumns=True, featureFirst=True, use_diff= True):
        '''
        init 
        '''        
        self.data_client = data_client
        self.max_step = max_step
        ## initialize render params
        self.viewer = Rendere()
        self.viewer.add_subplots(3)# 0:row candle, 1:  row macd, 2:diff candle, 3 diff macd
        
        ## initialize params
        self.__ubc__ = useBudgetColumns
        self.__ff__ = featureFirst
        self.__ud__ = use_diff
        self.INVALID_REWARD = -.001
        self.VALID_REWARD = 0.0005
        self.seed = 0
        #self.observationDays = observationDays
        self.dataLength = int((24*12)*observationDays)
        self.budget = 1
        self.coin = 0
        self.action_space = gym.spaces.Discrete(3)
        self.dataSet = None
        ## columns to ingest values from dataSet for observation
        self.ohlc_columns_dict = self.data_client.get_ohlc_columns()
        self.columns = []
        # initialize macd if it is specified
        columns = [str.lower(column) for column in columns]
        required_length = self.dataLength
        
        ## initialize parameters
        self.__use_macd = False
        if 'macd' in columns:
            self.short_window = 12
            self.long_window = 26
            self.signal_window = 9
            required_length += self.long_window + self.signal_window -2
            self.columns.append("short_ema")
            self.columns.append("long_ema")
            self.columns.append("macd")
            self.columns.append("signal")
            short_ema = pd.Series(indicaters.EMA(self.data_client.data.Close, self.short_window))
            long_ema = pd.Series(indicaters.EMA(self.data_client.data.Close, self.long_window))
            macd = pd.Series([x-y for (x, y) in zip(short_ema, long_ema)])
            signal = pd.Series(indicaters.SMA(macd, self.signal_window))
            #if use_diff:
            #    short_ema = short_ema.diff()
            #    long_ema = long_ema.diff()
            #    macd = macd.diff()
            #    signal = signal.diff()
            if self.__ud__:
                short_ema = short_ema.diff()[1:]
                long_ema = long_ema.diff()[1:]
                macd = macd.diff()[1:]
                signal = signal.diff()[1:]
                self.__s_ema_param = (short_ema.min(), short_ema.max())
                self.__l_ema_param = (long_ema.min(), long_ema.max())
                self.__macd_param = (macd.min(), macd.max())
                self.__signal_param = (signal.min(), signal.max())
            else:                
                self.__s_ema_param = (short_ema.min(), short_ema.max())
                self.__l_ema_param = (long_ema.min(), long_ema.max())
                self.__macd_param = (macd.min(), macd.max())
                self.__signal_param = (signal.min(), signal.max())
            
            self.__use_macd = True
        
        self.__use_bolinger = False
        if 'bolinger' in columns:
            self.columns.append("b_mean")
            self.columns.append("stds")
            self.bolinger_window = 20
            if required_length < self.bolinger_window + self.dataLength:
                required_length += self.bolinger_window
                
            ## TODO: Add param for standalization
            self.__use_bolinger = True
        if self.__ud__:
            required_length += 1
        self.__req_length = required_length
        
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
        if self.__use_macd:
            short_alpha = 2/(self.short_window+1)
            long_alpha = 2/(self.long_window+1)
            short_ema = self.dataSet["short_ema"].iloc[-1]*(1 - short_alpha) + tick.Close * short_alpha
            long_ema = self.dataSet["long_ema"].iloc[-1]*(1 - long_alpha) + tick.Close * long_alpha
            macd = short_ema - long_ema
            signal = (self.dataSet["macd"].iloc[-self.signal_window+1:].sum() + macd)/self.signal_window
            tick["short_ema"] = short_ema
            tick["long_ema"] = long_ema
            tick["macd"] = macd
            tick["signal"] = signal
        if self.__use_bolinger:
            ema = self.dataSet['b_mean'].iloc[-1].mean()
            tick['b_mean'] = None
            tick['stds'] = None
            #tick['b_high'] = ema + stds*alpha
            #tick['b_low'] = ema - stds*alpha
        
        #self.dataSet = self.dataSet.append(tick, ignore_index = True)[1:].copy()
        self.dataSet = pd.concat([self.dataSet, pd.DataFrame.from_records([tick.to_dict()])])
        ohlc = self.dataSet[self.__ohlc_columns].iloc[-self.dataLength:]
        self.viewer.register_ohlc(ohlc, 0, 'ROW OHLC Candle', *self.__ohlc_columns)
    
    def get_observation(self):
        obs = self.dataSet.copy()
        if self.__ud__:
            #obs = obs[-(self.dataLength+1):].pct_change()[1:]
            obs = obs.iloc[-(self.dataLength+1):].diff()[1:]
        else:
            # need to apply any standalization
            obs = obs.iloc[-self.dataLength:]

        if self.__use_macd:
            obs["short_ema"],_,_ = standalization.mini_max_from_array(obs["short_ema"].values, self.__s_ema_param[0], self.__s_ema_param[1], (0,1))
            obs["long_ema"],_,_ = standalization.mini_max_from_array(obs["long_ema"].values, self.__l_ema_param[0], self.__l_ema_param[1], (0,1))
            obs["macd"],_,_ = standalization.mini_max_from_array(obs["macd"].values, self.__macd_param[0], self.__macd_param[1], (0,1))
            obs["signal"],_,_ = standalization.mini_max_from_array(obs["signal"].values, self.__signal_param[0], self.__signal_param[1], (0,1))
        #if obs["short_ema"].min() ==  obs["short_ema"].max():
        #    print("strange obs")
        #if obs["short_ema"].min() == 0 and obs["short_ema"].max() == 0:
        #    print("strange obs")
        return obs[self.columns]
    
    def get_next_observation(self):
        tick, done = self.data_client.get_next_tick()
        self.__update_observation(tick)
        
        obs = self.get_observation()
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
        self.dataSet = self.data_client.get_rates(self.__req_length)
        
        ## add inidicaters to dataSet and columns
        if self.__use_macd:
            self.__initMACD__(short_window=self.short_window, long_window=self.long_window, signal_window=self.signal_window)
        if self.__use_bolinger:
            self.__initBolinger__(window=self.bolinger_window)
        
        observation = self.get_observation()
        if self.__ubc__:
            observation['budgets'] = self.budgets
            #observation['coins'] = self.coins
        if self.__ff__:
            return observation.T.to_numpy()
        else:
            return observation.to_numpy()
        
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
        self.budgets[:-1] = self.budgets[1:len(self.budgets)]
        self.budgets[-1] = self.budget
        
    def __set_hisoty_coins(self, new_coin):
        self.coin = new_coin
        self.coins[:-1] = self.coins[1:len(self.budgets)]
        self.coins[len(self.coins)-1] = self.coin
        
    def __set_diff_as_bugets(self, mono=False):
        diffs = self.data_client.get_diffs_with_minmax()
        amount = 0
        if len(diffs) > 1:
            print(f"Warning: diffs have two or more positions. {len(diffs)}")
        for diff in diffs:
            amount += diff
        if mono:
            self.budgets = [amount for i in range(0, len(self.budgets))]
        else:
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
            self.__set_diff_as_bugets(mono=True)
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
            #if reward > 0:
            #    reward = reward * 10#geta
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
    
    def get_params(self) -> dict:
        params = {}
        if self.__ud__:
            params['post process'] = {
                'function':'diff',
                'floor':1
            }
        if self.__use_bolinger:
            # TODO: Add params when bolinger is implemented
            pass
        if self.__use_macd:
            params['macd'] = {
                'short_window': self.short_window,
                'long_window': self.long_window,
                'signal_window': self.signal_window,
                'post process': {
                    'function':'minmax',
                    'short_ema':[self.__s_ema_param[0], self.__s_ema_param[1]],
                    'long_ema': [self.__l_ema_param[0], self.__l_ema_param[1]],
                    'macd': [self.__macd_param[0], self.__macd_param[1]],
                    'signal': [self.__signal_param[0], self.__signal_param[1]]
                }
            }
        return params