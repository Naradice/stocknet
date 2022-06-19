from math import remainder
from stocknet.envs.utils import indicaters, standalization
import gym
import random
import pandas as pd
import numpy as np
from stocknet.envs.render.graph import Rendere
from stocknet.envs.market_clients.market_client_base import MarketClientBase
import stocknet.envs.utils as utils
#from render.graph import Rendere

class BCEnv(gym.Env):

    def __init__(self, data_client:MarketClientBase, max_step:int, columns = ['High', 'Low','Open','Close'], observationDays=1, useBudgetColumns=True, featureFirst=True, mono=False):
        """ Bitcoin Environment of OpenGym
        Assuming the data_client provide ohlc without regular market close

        Args:
            data_client (MarketClientBase): Client to provide ohlc data
            max_step (int): max step to caliculate min max of reward
            columns (list, optional): Column names of ohlc to include an obervation. Defaults to ['High', 'Low','Open','Close']. If [] is specified, no ohlc data is provided.
            observationDays (int, optional): Decide observation length with observationDays * 24 * (60/data_client.frame) .Defaults to 1. data_client.frame > 1h is not supported yet.
            useBudgetColumns (bool, optional): If True, difference from bought rate and current rate is provided as observation. Defaults to True.
            featureFirst (bool, optional): If true (FeatreNum, ObservationLength), otherwie (ObservationLength, FeatureNum). Defaults to True.
            mono (bool, optional): Tentative Param for testing purpose. Return budget result with simple format. Defaults to False.
        """
        self.data_client = data_client
        self.max_step = max_step
        ## initialize render params
        self.indicaters = []
        self.indicaters_length = 0
        self.preprocess = []
        self.preprocess_length = 0
        self.preprocess_initialized = True
        self.viewer = Rendere()
        self.viewer.add_subplots(3)# 0:row candle, 1:  row macd, 2:diff candle, 3 diff macd
        
        ## initialize params
        self.__ubc__ = useBudgetColumns
        self.__ff__ = featureFirst
        self.INVALID_REWARD = -.001
        self.VALID_REWARD = 0#0.0005
        self.STAY_GOOD_REWARD = .001
        self.STAY_BAD_REWARD = 0#0.000001
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
        columns = [str.lower(column) for column in columns]
        self.__req_length = self.dataLength
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
            
        self.b_mono = False
        if useBudgetColumns:
            ob_shape = (self.dataLength, len(self.columns) + 1)
            self.b_mono = mono
        else:
            ob_shape = (self.dataLength, len(self.columns))
            
        ## initialize gym parameters
        self.observation_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=ob_shape
        )
        self.reward_range = [-1., 1.]
        

    @classmethod
    def get_available_columns(self):
        return ['High', 'Low','Open','Close','macd', 'bolinger']
    
    @property
    def ask(self):
        return self.data_client.get_current_ask()
    
    @property
    def bid(self):
        return self.data_client.get_current_bid()
    
    def add_indicater(self, process:utils.ProcessBase):
        self.indicaters.append(process)
        req_length = process.get_minimum_required_length() - 1
        if self.indicaters_length < req_length:
            self.indicaters_length = req_length
        for key, column in process.columns.items():
            self.columns.append(column)
        
    def add_indicaters(self, processes: list):
        for process in processes:
            self.add_indicater(process)
        
    def register_preprocess(self, process:utils.ProcessBase):
        self.preprocess_initialized = False
        self.preprocess.append(process)
        self.preprocess_length += process.get_minimum_required_length()-1
    
    def register_preprocesses(self, processes: list):
        for process in processes:
            self.register_preprocess(process)
    
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
        
        
    def __update_observation(self, tick:pd.Series):
        # update indicater of dataSet
        df = tick.copy()
        for indicater in self.indicaters:
            new_data = indicater.update(tick)
            for key,column in indicater.columns.items():
                df[column] = new_data[column]
        self.dataSet = utils.ProcessBase.concat(None, self.dataSet.iloc[1:], df)
        
        # update preprocess
        new_target_tick = df[self.columns]
        for process in self.preprocess:
            new_target_tick = process.update(new_target_tick)
        self.obs = utils.ProcessBase.concat(None, self.obs.iloc[1:], new_target_tick)
        
        #ohlc = self.dataSet[self.__ohlc_columns].iloc[-self.dataLength:]
        #self.viewer.register_ohlc(ohlc, 0, 'ROW OHLC Candle', *self.__ohlc_columns)
    
    def initialize_preprocess_params(self):
        """
        initialize parameters
        to initialize a parameter of mini max, just run preprocesses to entire data
        """
        if len(self.preprocess) > 0:
            data = self.data_client.get_rates(-1)
            for indicater in self.indicaters:
                values_dict = indicater.run(data)
                for key, values in values_dict.items():
                    data[key] = values
            data = data.dropna()
            data = data[self.columns]
            for process in self.preprocess:
                result_dict = process.run(data)
                data = pd.DataFrame(result_dict)
            self.preprocess_initialized = True
    
    def initialize_additional_obs(self):
        self.pls = [0 for i in range(0, len(self.obs))]
        if self.__ubc__:
            self.obs['pl'] = self.pls
            #self.obs['coins'] = self.coins
    
    def initialize_observation(self):
        obs = self.dataSet[self.columns].copy()
        for process in self.preprocess:
            result_dict = process.run(obs)
            obs = pd.DataFrame(result_dict)
        self.obs = obs
        self.initialize_additional_obs()
    
    def get_next_observation(self):
        tick, done = self.data_client.get_next_tick()
        self.__update_observation(tick)
        return self.obs, done
        
    def reset(self):
        '''
        '''
        self.pl = 0
        self.boughtCoinRate = 0
        self.soldCoinRate = 0
        self.budget = 1
        self.coin = 0
        self.current_pl = 0
        
        ## reset index
        self.__req_length = self.dataLength + self.indicaters_length + self.preprocess_length
        self.data_client.reset()
        column = self.__ohlc_columns[3]
        _min, _max = self.data_client.get_min_max(column, data_length = self.max_step)#When client is change, this need to be implemente d in the other way
        self.max_reward = _max - _min
        self.dataSet = self.data_client.get_rates(self.__req_length)
        
        ## add inidicaters to dataSet and columns
        for indicater in self.indicaters:
            values_dict = indicater.run(self.dataSet)
            for key, values in values_dict.items():
                self.dataSet[key] = values
        if self.preprocess_initialized == False:
            self.initialize_preprocess_params()
            
        self.initialize_observation()
        if self.__ff__:
            return self.obs.iloc[-self.dataLength:].T.to_numpy()
        else:
            return self.obs.iloc[-self.dataLength:].to_numpy()
        
    def evaluate(self, action, debug=False):
        reward = 0
        if action == 1:
            '''
            buy coin with 10 point if possible.
            if you don't have much budget, return negative reward 
            '''
            reward = self.__buyCoin__(debug=debug)

        elif action == 2:
            '''
            sell coin with 100% if possible. (R_sell - R_buy)/R_buy
            if you don't have coin, return negative reward
            '''
            reward = self.__sellCoin__(debug)

        elif action == 0:
            '''
            hold.
            '''
            reward = self.__stay__(debug)
        else:
            raise Exception(f"The action number {action} exeeds the lengths in evaluate function.")
        return reward
    
    def step(self, action): # actionを実行し、結果を返す
        done = False
        reward = 0

        reward = self.evaluate(action, False)
        self.obs, done = self.get_next_observation()
        
        if self.pl < -100000:
            done = True
        if self.__ubc__:
            self.obs["pl"] = self.pls
            #self.observation["coins"] = self.coins
        
        if self.__ff__:
            return self.obs.iloc[-self.dataLength:].T.to_numpy(), reward, done, {}
        else:
            return self.obs.iloc[-self.dataLength:].to_numpy(), reward, done, {}
        
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
        self.pls = [value for i in range(0, len(self.obs))]
        
    def __set__simple_coins(self, value):
        self.coin = value
        self.coins = [value for i in range(0, len(self.obs))]
        
    
    def __set_history_bugets(self, new_buget):
        self.pls[:-1] = self.pls[1:len(self.pls)]
        self.pls[-1] = new_buget
        
    def __set_hisoty_coins(self, new_coin):
        self.coin = new_coin
        self.coins[:-1] = self.coins[1:len(self.coins)]
        self.coins[len(self.coins)-1] = self.coin
        
    def get_current_pl(self):
        diffs = self.data_client.get_diffs_with_minmax()
        amount = sum(diffs)
        return amount
        
    def __set_diff_as_bugets(self, mono=False):
        amount = self.get_current_pl()
        amount = np.clip(amount, *self.reward_range)
        if mono:
            self.pls = [amount for i in range(0, len(self.obs))]
        else:
            self.pls[:-1] = self.pls[1:len(self.pls)]
            self.pls[-1] = amount
        return amount
        
    def __buyCoin__(self, amount=1, debug=False):
        '''
        buy coin using all budget
        '''
        if self.budget > 0:
            self.budget = 0
            result = self.data_client.market_buy(amount=amount)
            #result is {"price":boughtCoinRate, "step":self.index, "amount":amount}
            current_amount = self.__set_diff_as_bugets(mono=self.b_mono)
            if debug:
                print("bought", result["price"], "slip", current_amount)
            self.current_pl = current_amount
            
            self.coin = 1
            
            return self.VALID_REWARD
        else:
            return self.INVALID_REWARD
        
    def __sellCoin__(self, debug):
        if self.coin > 0:
            #add base reward for valid action
            reward = self.VALID_REWARD
            #self.__set__simple_bugets(1)
            #self.__set__simple_coins(0)
            self.__set__simple_bugets(0)
            #amount = self.get_current_pl()
            results = self.data_client.sell_all_settlement()
            #results are list of (bid, position["price"], (bid - position["price"]))
            self.budget = 1
            self.coin = 0
            count = 0
            self.current_pl = 0
            
            for result in results:
                count += 1
                reward += result[2]/self.max_reward
                self.pl += result[3]
            self.pls = [0 for i in range(0, len(self.obs))]
            reward = np.clip(reward, *self.reward_range)
            if debug:
                print(f"Sold {count} position","reward", reward)
            
            return reward
        else:
            return self.INVALID_REWARD
    
    def __stay__(self, debug):
        reward = 0.0
        amount = self.__set_diff_as_bugets(mono=self.b_mono)
        diff = amount - self.current_pl
        self.current_pl = amount
        if diff >= 0:
            reward += self.STAY_GOOD_REWARD
            reward += amount
            if amount > 0:
                reward += diff*5
            else:
                reward += diff
        elif diff < 0:
            reward += self.STAY_BAD_REWARD
            if amount > 0:
                reward += amount + diff
                reward = diff
            else:
                reward += diff*3
        reward = np.clip(reward, *self.reward_range)
        if debug and diff != 0:
            print(f"Stay","reward", reward)
        return reward
    
    def get_params(self) -> dict:
        params = {}
        return params
    
class BCMultiActsEnv(BCEnv):
    
    def __init__(self, data_client: MarketClientBase, max_step: int, bugets: int, columns=['High', 'Low', 'Open', 'Close'], observationDays=1, useBudgetColumns=True, featureFirst=True, mono=False):
        """ Bitcoin Environment of OpenGym
        Assuming the data_client provide ohlc without regular market close

        Args:
            data_client (MarketClientBase): Client to provide ohlc data
            max_step (int): max step to caliculate min max of reward
            bugets (int): bugets represents how how many points agent can buy bc as maximum. number of action become 2N + 1
            columns (list, optional): Column names of ohlc to include an obervation. Defaults to ['High', 'Low','Open','Close']. If [] is specified, no ohlc data is provided.
            observationDays (int, optional): Decide observation length with observationDays * 24 * (60/data_client.frame) .Defaults to 1. data_client.frame > 1h is not supported yet.
            useBudgetColumns (bool, optional): If True, difference from bought rate and current rate is provided as observation. Defaults to True.
            featureFirst (bool, optional): If true (FeatreNum, ObservationLength), otherwie (ObservationLength, FeatureNum). Defaults to True.
            mono (bool, optional): Tentative Param for testing purpose. Return budget result with simple format. Defaults to False.
        """
        assert(type(bugets)==int, "bugets should be int")
        self.num_actions = bugets
        super().__init__(data_client, max_step, columns, observationDays, useBudgetColumns, featureFirst, mono)
        self.budget = bugets
        self.action_space = gym.spaces.Discrete(self.num_actions*2 + 1)
    
    def reset(self):
        obs = super().reset()
        self.budget = self.num_actions
        return obs
        
    def initialize_additional_obs(self):
        self.pls = [0 for i in range(0, len(self.obs))]
        if self.__ubc__:
            self.obs['pl'] = self.pls
            #self.obs['coins'] = self.coins
        self.obs['budget'] = [1 for i in range(0, len(self.obs))]
    
    def __set_diff_as_bugets(self, mono=False):
        amount = self.get_current_pl()
        amount = np.clip(amount, *self.reward_range)
        if mono:
            self.pls = [amount for i in range(0, len(self.obs))]
        else:
            self.pls[:-1] = self.pls[1:len(self.pls)]
            self.pls[-1] = amount
        return amount
    
    def __buyCoin__(self, amount:int=1, debug=False):
        '''
        buy coin using all budget
        '''
        if self.budget > 0:
            remaining_budget = self.budget - amount
            if remaining_budget < 0:
                self.budget = 0
                self.coin = self.num_actions
                amount = amount + remaining_budget
            else:
                self.budget = remaining_budget
                self.coin += amount
            self.data_client.market_buy(amount=amount)
            current_amount = self.__set_diff_as_bugets(mono=self.b_mono)
            if debug:
                print("bought", result["price"], "slip", current_amount)
            
            return self.VALID_REWARD
        else:
            return self.INVALID_REWARD
    
    def __sellCoin__(self, point, debug):
        if self.coin > 0:
            #add base reward for valid action
            reward = self.VALID_REWARD
            positions_dict = self.data_client.get_positions("ask")
            
            # sort and caliculate total amount
            amounts = 0
            positions = [{"price":0} for i in range(0, len(positions_dict))]
            
            for key, position in positions_dict.items():
                amounts += position["amount"]
                bought_rate = position["price"]
                for index in range(0, len(positions)):
                    if positions[index]["price"] < bought_rate:
                        positions[index+1:] = positions[index:-1]
                        positions[index] = position
                        break
            ##
            if self.budget + amounts > 10:
                print("went wrong.")
            results = []
            if amounts <= point:
                results = self.data_client.sell_all_settlement()
                if self.coin != amounts:
                    print(f"sell coin: something went wrong: {self.coin}, {amounts}, {point}")
                self.coin = 0
                self.budget = self.num_actions
            else:
                sold_amount = 0
                for position in positions:
                    sold_amount += position["amount"]
                    if sold_amount < point:
                        __point = position["amount"]
                        result = self.data_client.sell_settlement(position=position, point=__point)
                        results.append(result)
                    else:
                        over_point = sold_amount - point
                        __point = position["amount"] - over_point
                        result = self.data_client.sell_settlement(position=position, point=__point)
                        results.append(result)
                        point = sold_amount -over_point
                        break
                self.coin = self.coin - point
                self.budget += point
                
            #caliculate pl
            pl = 0
            for result in results:
                pl += result[3]
            self.pl += pl
            ##
            
            #caliculate reward
            for result in results:
                reward += result[2]/self.max_reward
            reward = np.clip(reward, *self.reward_range)
            ##
            
            return reward
        else:
            return self.INVALID_REWARD
        
    def evaluate(self, action, debug=False):
        reward = 0
        if action == 0:
            '''
            hold.
            '''
            reward = self.__stay__(debug)

        elif action > 0 and action <= self.num_actions:
            '''
            buy coin with 10 point if possible.
            if you don't have much budget, return negative reward 
            '''
            amount = action
            reward = self.__buyCoin__(amount,debug=debug)

        elif action > self.num_actions and action <= self.num_actions*2:
            '''
            sell coin with 100% if possible. (R_sell - R_buy)/R_buy
            if you don't have coin, return negative reward
            '''
            amount = action - self.num_actions
            reward = self.__sellCoin__(amount, debug)
        else:
            raise Exception(f"The action number {action} exeeds the lengths in evaluate function.")
        return reward

    def step(self, action): # actionを実行し、結果を返す
        done = False
        reward = 0

        reward = self.evaluate(action, False)
        self.obs, done = self.get_next_observation()
        
        amount = self.get_current_pl()
        amount = np.clip(amount, *self.reward_range)
        self.pls = [amount for i in range(0, len(self.obs))]
        
        if self.pl < -100000:
            done = True
        if self.__ubc__:
            self.obs["pl"] = self.pls
            #self.observation["coins"] = self.coins
        self.obs["budget"] = [self.budget/self.num_actions for i in range(0, len(self.obs))]
        
        if self.__ff__:
            return self.obs.iloc[-self.dataLength:].T.to_numpy(), reward, done, {}
        else:
            return self.obs.iloc[-self.dataLength:].to_numpy(), reward, done, {}

    def __stay__(self, debug):
        reward = 0.0
        """
        amount = self.get_current_pl()
        amount = np.clip(amount, *self.reward_range)
        self.pls = [amount for i in range(0, len(self.obs))]
        diff = amount - self.current_pl
        self.current_pl = amount
        if diff >= 0:
            reward += self.STAY_GOOD_REWARD
            reward += amount
            if amount > 0:
                reward += diff*5
            else:
                reward += diff
        elif diff < 0:
            reward += self.STAY_BAD_REWARD
            if amount > 0:
                reward += amount + diff
                reward = diff
            else:
                reward += diff*3
        reward = np.clip(reward, *self.reward_range)
        if debug and diff != 0:
            print(f"Stay","reward", reward)
        """
        return reward

class BCStopEnv(BCEnv):
    """
    Buy coins with stop loss and profit
    reward is decided by refering a future data
    """
    
    key = "stop"
    
    def __init__(self, data_client: MarketClientBase, max_step: int, stop_loss_point: int = 0.05, stop_profit_point:int = 0.1, usebb = False, frames: list = None, columns=['High', 'Low', 'Open', 'Close'], observationDays=1, featureFirst=True):
        """ Bitcoin Environment of OpenGym
        Assuming the data_client provide ohlc without regular market close

        Args:
            data_client (MarketClientBase): Client to provide ohlc data
            max_step (int): max step to caliculate min max of reward
            stop_loss_point (int): specify max_loss_piint
            stop_profit_point (int): specify max_profit_point which should be bigger than max_loss_point
            usebb (bool): use Bolinger band instead of max loss/profit point
            columns (list, optional): Column names of ohlc to include an obervation. Defaults to ['High', 'Low','Open','Close']. If [] is specified, no ohlc data is provided.
            observationDays (int, optional): Decide observation length with observationDays * 24 * (60/data_client.frame) .Defaults to 1. data_client.frame > 1h is not supported yet.
            featureFirst (bool, optional): If true (FeatreNum, ObservationLength), otherwie (ObservationLength, FeatureNum). Defaults to True.
        """
        if type(data_client.base_point) == type(None):
            raise Exception("data client has no base point definition.")
        super().__init__(data_client, max_step, columns, observationDays, False, featureFirst, False)
        self.__use_bb = usebb
        if usebb:
            self.add_bolinger()
            self.stop_loss = -1
            self.stop_profit = -1
        else:
            self.stop_loss = stop_loss_point
            self.stop_profit = stop_profit_point
        self.action_space = gym.spaces.Discrete(2)
    
    def add_bolinger(self):
        process = utils.BBANDpreProcess(key="bstop")
        self.indicaters.append(process)
        self.b_column = process.columns["Width"]
        req_length = process.get_minimum_required_length() - 1
        
        if self.indicaters_length < req_length:
            self.indicaters_length = req_length
    
    def reset(self):
        obs = super().reset()
        self.ask_value = self.data_client.get_current_ask()
        self.update_stop_value()
        self.count = 0
        return obs
        
    def initialize_additional_obs(self):
        pass

    
    def __buyCoin__(self, amount:int=1, debug=False):
        '''
        buy coin using all budget
        '''
        stop_loss = self.ask_value - self.ask_value * self.stop_loss
        stop_profit = self.ask_value + self.ask_value * self.stop_profit
        #position = self.data_client.buy(amount=amount)
        data = self.data_client.get_future_rates(self.max_step)
        low_column = self.ohlc_columns_dict['Low']
        high_column = self.ohlc_columns_dict['High']
        
        ## caliculate reward with direction change
        #close_column = self.ohlc_columns_dict['Close']
        #ewa = data[close_column].ewm(span=10, adjust=True).mean()
        
        reward = -0.1
        if self.stop_loss != -1:
            for index, row in data.iterrows():
                low = row[low_column]
                high = row[high_column]
                if low <= stop_loss:
                    reward = -0.5
                    self.pl = self.pl - self.ask_value * self.stop_loss
                    break
                elif high >= stop_profit:
                    reward = 1
                    self.pl += self.ask_value * self.stop_profit
                    break
                
        return reward 
    
    def update_stop_value(self):
        width = self.dataSet[self.b_column].iloc[-1]
        self.stop_profit = width/self.ask_value
        self.stop_loss = self.stop_profit/2
        if self.data_client.base_point > self.stop_loss:
            self.stop_loss = -1
            self.stop_profit = -1
        
        
    def evaluate(self, action, debug=False):
        reward = 0
        if action == 0:
            '''
            hold.
            '''
            reward = self.__stay__(debug)

        elif action == 1:
            '''
            buy coin with 10 point if possible.
            if you don't have much budget, return negative reward 
            '''
            amount = action
            reward = self.__buyCoin__(amount,debug=debug)
        else:
            raise Exception(f"The action number {action} exeeds the lengths in evaluate function.")
        return reward

    def step(self, action): # actionを実行し、結果を返す
        done = False
        self.count += 1
        reward = 0
        self.ask_value = self.data_client.get_current_ask()

        if self.__use_bb:
            self.update_stop_value()
            
        reward = self.evaluate(action, False)
        self.obs, done = self.get_next_observation()
        
        if self.pl < -100000 or self.count == self.max_step:
            done = True
        
        if self.__ff__:
            return self.obs.iloc[-self.dataLength:].T.to_numpy(), reward, done, {}
        else:
            return self.obs.iloc[-self.dataLength:].to_numpy(), reward, done, {}

    def __stay__(self, debug):
        reward = 0.0
        return reward

#TODO: make this env class to observation class
class BCDateEnv(BCEnv):

    def __init__(self, data_client:MarketClientBase, columns = ['High', 'Low','Open','Close'],maxStepDays=1 ,observationLength=1, useBudgetColumns=True, featureFirst=True, mono=False):
        """ Bitcoin Environment of OpenGym
        Assuming the data_client provide ohlc without regular market close

        Args:
            data_client (MarketClientBase): Client to provide ohlc data
            max_step (int): max step to caliculate min max of reward
            columns (list, optional): Column names of ohlc to include an obervation. Defaults to ['High', 'Low','Open','Close']. If [] is specified, no ohlc data is provided.
            observationDays (int, optional): Decide observation length with observationDays * 24 * (60/data_client.frame) .Defaults to 1. data_client.frame > 1h is not supported yet.
            useBudgetColumns (bool, optional): If True, difference from bought rate and current rate is provided as observation. Defaults to True.
            featureFirst (bool, optional): If true (FeatreNum, ObservationLength), otherwie (ObservationLength, FeatureNum). Defaults to True.
            mono (bool, optional): Tentative Param for testing purpose. Return budget result with simple format. Defaults to False.
        """
        assert observationLength > 0
        max_step = int(maxStepDays * (60*24)/self.frame)
        super().__init__(data_client, max_step, columns, 1, useBudgetColumns, featureFirst, mono)
        self.dataLength = observationLength
        self.max_step_days = maxStepDays
                    
    def reset(self, retry=0):
        '''
        '''
        if retry <= 10:
            self.pl = 0
            self.boughtCoinRate = 0
            self.soldCoinRate = 0
            self.budget = 1
            self.coin = 0
            self.current_pl = 0
            self.current_index = 0
            
            ## reset index
            self.__req_length = self.dataLength + self.indicaters_length + self.preprocess_length
            suc = self.data_client.reset(mode="day")
            if suc:
                entire_data = self.data_client.get_future_rates(self.max_step)
                #check if last data have expected date
                ohlc_obs = self.data_client.get_rates(self.dataLength)
                #check if obs data have expected date
                if suc:
                    column = self.__ohlc_columns
                    _min, _max = self.data_client.get_min_max(column, data_length = self.max_step)
                    self.max_reward = _max - _min
                    self.dataSet = self.data_client.get_rates(self.__req_length)
                    
                    ## add inidicaters to dataSet and columns
                    for indicater in self.indicaters:
                        values_dict = indicater.run(self.dataSet)
                        for key, values in values_dict.items():
                            self.dataSet[key] = values
                    if self.preprocess_initialized == False:
                        self.initialize_preprocess_params()
                        
                    self.initialize_observation()
                    if self.__ff__:
                        return self.obs.iloc[-self.dataLength:].T.to_numpy()
                    else:
                        return self.obs.iloc[-self.dataLength:].to_numpy()
                else:
                    self.reset(retry=retry+1)
            else:
                self.reset(retry=retry+1)
        else:
            raise Exception("")
    
    def step(self, action): # actionを実行し、結果を返す
        done = False
        reward = 0

        reward = self.evaluate(action, False)
        self.obs, done = self.get_next_observation()
        self.current_index += 1
        
        if self.pl < -100000:
            done = True
        if self.current_index == self.max_step:
            done = True
        if self.__ubc__:
            self.obs["pl"] = self.pls
            #self.observation["coins"] = self.coins
        
        if self.__ff__:
            return self.obs.iloc[-self.dataLength:].T.to_numpy(), reward, done, {}
        else:
            return self.obs.iloc[-self.dataLength:].to_numpy(), reward, done, {}