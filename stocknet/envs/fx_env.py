import datetime
import random

import gym
import numpy
import pandas as pd
from datasets import FXDataset
from gym import error, spaces, utils
from gym.utils import seeding


class FXEnv:
    def __init__(self, isTraining=True, data=None, featureFirst=False, randomizeInitialState=True):
        self.ds = FXDataset(1, isTraining, data)
        self.FF = featureFirst
        self.RIS = randomizeInitialState
        # For Reinforce lerning
        self.action_space = gym.spaces.Discrete(3 + 2)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(self.span, 6))
        self.reward_range = [-1, 1]
        self.VALID_REWARD = 0
        self.INVALID_REWARD = 0  # -.001
        self.reset()

    def __len__(self):
        return len(self.ds)

    def reset(self):
        """ """
        self.askPositions = []
        self.bidPositions = []
        self.budget = self.budget_org
        self.coin = 0
        if self.RIS:
            self.index = random.randint(self.fromIndex, self.toIndex)
        else:
            self.index = self.ds.fromIndex
        self.stepCount = 0
        self.rewards = 0
        self.winCount = 0
        self.orderCount = 0
        self.pl = 0
        self.budget_history = [1 for i in range(0, self.span)]
        self.ask_diff_history = [0 for i in range(0, self.span)]
        self.bid_diff_history = [0 for i in range(0, self.span)]
        observations = self.__getitem__(self.index + self.stepCount)
        observations = numpy.hstack([observations, numpy.atleast_2d(self.budget_history).T])
        observations = numpy.hstack([observations, numpy.atleast_2d(self.ask_diff_history).T])
        observations = numpy.hstack([observations, numpy.atleast_2d(self.bid_diff_history).T])
        if self.FF:
            observations = observations.T
        observations = numpy.array([observations])
        return observations

    def getSymbolInfo(self, symbol="USDJPY"):
        if symbol == "USDJPY":
            return {"point": 0.001, "min": 0.1, "rate": 100000}

    def GET_CURRENT_ASK(self):
        if self.index + self.stepCount < len(self.rowdata):
            value = (
                self.rowdata.close.iloc[self.index + self.stepCount - 1] + self.rowdata["spread"].iloc[self.index + self.stepCount - 1] * self.point
            )
            # value = random.uniform(next_data["Open"].iloc[0], next_data["High"].iloc[0])
            # value = next_data["Open"].iloc[0] + next_data["spread"].iloc[0]*self.point
            return value

    def GET_CURRENT_BID(self):
        if self.index + self.stepCount < len(self.rowdata):
            value = (
                self.rowdata.close.iloc[self.index + self.stepCount - 1] - self.rowdata["spread"].iloc[self.index + self.stepCount - 1] * self.point
            )
            # value = random.uniform(next_data["Low"].iloc[0], next_data["Open"].iloc[0])
            # value = next_data["Open"].iloc[0] - next_data["spread"].iloc[0]*self.point
            return value

    def __getitem__(self, ndx):
        ins, out = self.ds[ndx]
        return ins

    def badget_in_use_and_diff(self):
        budget_in_use = 0
        ask_diff = 0
        bid_diff = 0
        sell_price = self.GET_CURRENT_BID()
        ask_price = self.GET_CURRENT_ASK()
        for position in self.askPositions:
            ask_diff += (sell_price - position["price"]) / position["price"]
            budget_in_use += position["volume"] * sell_price * self.volume_point / self.leverage
        for position in self.bidPositions:
            bid_diff += (position["price"] - ask_price) / ask_price
            budget_in_use += position["volume"] * ask_price * self.volume_point / self.leverage
        return budget_in_use, ask_diff, bid_diff

    def evaluate(self, action):
        reward = 0
        if action == 1:
            """
            buy coin with 10 point if possible.
            if you don't have much budget, return negative reward
            """
            reward = self.__buy__()

        elif action == 2:
            """
            sell coin with 10 point if possible.
            if you don't have much budget, return negative reward
            """
            reward = self.__sell__()

        elif action == 0:
            """
            hold. reward is 0
            """
            reward = self.__stay__()
        elif action == 3:
            """
            buy settlement of bid position.
            if there are no position, return negative reward
            """
            reward = self.__settlement__("buy")
        elif action == 4:
            """
            sell settlement of ask position.
            if there are no position, return negative reward
            """
            reward = self.__settlement__("sell")
        else:
            raise Exception(f"The action number {action} exeeds the lengths in evaluate function.")
        return reward

    def step(self, action):
        self.stepCount += 1
        if self.index + self.stepCount <= self.toIndex:
            observations = self.__getitem__(self.index + self.stepCount)
            done = False
            reward = 0.0
            option = None
            reward = self.evaluate(action)
            self.rewards = self.rewards + reward

            budget_in_use, ask_diff, bid_diff = self.badget_in_use_and_diff()
            option = [self.budget / self.budget_org, ask_diff, bid_diff]
            # budget
            self.budget_history[:-1] = self.budget_history[1:]
            self.budget_history[-1] = self.budget / self.budget_org
            self.ask_diff_history[:-1] = self.ask_diff_history[1:]
            self.ask_diff_history[-1] = ask_diff
            self.bid_diff_history[:-1] = self.ask_diff_history[1:]
            self.bid_diff_history[-1] = bid_diff
            observations = numpy.hstack([observations, numpy.atleast_2d(self.budget_history).T])
            observations = numpy.hstack([observations, numpy.atleast_2d(self.ask_diff_history).T])
            observations = numpy.hstack([observations, numpy.atleast_2d(self.bid_diff_history).T])
            if self.FF:
                observations = observations.T
            observations = numpy.array([observations])

            if self.orderCount > 0:
                winRate = self.winCount / self.orderCount
            else:
                winRate = -1

            # if (self.budget + budget_in_use) - self.budget_org < - self.budget_org * 0.2:
            if self.pl < -10:
                done = True
            else:
                done = False
            # if self.orderCount < 12*24*30 or self.pl > 0:
            # done = False
            return observations, reward, done, option
        else:
            observations = self.__getitem__(self.index + self.stepCount - 1)
            ask_diff = 0
            sell_price = self.GET_CURRENT_BID()
            budget_in_use, ask_diff, bid_diff = self.badget_in_use_and_diff()
            self.budget_history[:-1] = self.budget_history[1:]
            self.budget_history[-1] = self.budget / self.budget_org
            self.ask_diff_history[:-1] = self.ask_diff_history[1:]
            self.ask_diff_history[-1] = ask_diff
            self.bid_diff_history[:-1] = self.ask_diff_history[1:]
            self.bid_diff_history[-1] = bid_diff
            observations = numpy.hstack([observations, numpy.atleast_2d(self.budget_history).T])
            observations = numpy.hstack([observations, numpy.atleast_2d(self.ask_diff_history).T])
            observations = numpy.hstack([observations, numpy.atleast_2d(self.bid_diff_history).T])
            if self.FF:
                observations = observations.T
            observations = numpy.array([observations])
            return observations, 0, True, {}

    def render(self, mode="human", close=False):
        """ """
        ask_diff = 0
        sell_price = self.GET_CURRENT_BID()
        budget_in_use, _, _ = self.badget_in_use_and_diff()
        if self.orderCount > 0:
            winRate = self.winCount / self.orderCount
        else:
            winRate = -1
        print(f"budget:{self.budget} + {budget_in_use}, pl:{self.pl}, winRate:{winRate}")

    def __stay__(self, S=5, T=1 / 2):
        reward = 0.0
        sell_price = self.GET_CURRENT_BID()
        for position in self.askPositions:
            ask_diff = (sell_price - position["price"]) / position["price"]
            if position["step"] < S:
                alpha = (position["step"] - self.stepCount) / S * T
            else:
                alpha = T
            reward += alpha * ask_diff
        ask_price = self.GET_CURRENT_ASK()
        for position in self.bidPositions:
            bid_diff = (position["price"] - ask_price) / ask_price
            if position["step"] < S:
                alpha = (position["step"] - self.stepCount) / S * T
            else:
                alpha = T
            reward += alpha * bid_diff
        # raise Exception(f"Unexpected action value in __stay__ function: {action}")

        return reward

    def close(self):
        """ """
        pass

    def seed(self, seed=None):
        """ """
        if seed == None:
            random.seed(1017)
        else:
            random.seed(seed)

    def __settlement__(self, type, price=None):
        reward = 0
        # settlement bid position
        if type == "buy":
            if len(self.bidPositions) > 0 and len(self.askPositions) > 0:  # 全額売買のみ　かつ　両建て無し
                # print("buy settlement")
                reward = self.VALID_REWARD
                current_buy_rate = self.GET_CURRENT_ASK()
                for position in self.bidPositions:
                    reward += (position["price"] - current_buy_rate) / current_buy_rate
                    self.budget += (position["volume"] * self.volume_point * current_buy_rate) / self.leverage
                    # print(f"actual reward: {((position['price'] - current_buy_rate))}")
                    # print(f"BID SETTLEMENT: {position['price']} - {current_buy_rate} = {position['price'] - current_buy_rate}")
                    pl = position["price"] - current_buy_rate
                    # print(pl)
                    self.pl += pl
                    if pl > 0:
                        self.winCount += 1
                    else:
                        # twice
                        reward += (position["price"] - current_buy_rate) / current_buy_rate
                self.bidPositions = []
            else:
                reward = self.INVALID_REWARD
        elif type == "sell":  # settlement ask position
            reward = self.VALID_REWARD
            if len(self.askPositions) > 0 and len(self.bidPositions) > 0:
                # print("sell settlement")
                current_sell_rate = self.GET_CURRENT_BID()
                for position in self.askPositions:
                    reward += (current_sell_rate - position["price"]) / position["price"]
                    self.budget += (position["volume"] * self.volume_point * current_sell_rate) / self.leverage
                    # print(f"actual reward: {((current_sell_rate - position['price']))}")
                    # print(f"ASK SETTLEMENT: {current_sell_rate} - {position['price']} = {current_sell_rate - position['price']}")
                    pl = current_sell_rate - position["price"]
                    self.pl += pl
                    # print(pl)
                    if pl > 0:
                        self.winCount += 1
                    else:
                        # twice
                        reward += (current_sell_rate - position["price"]) / position["price"]
                self.askPositions = []
            else:
                reward = self.INVALID_REWARD
        # print(f"{self.budget}, {reward}")
        return reward

    def __buy__(self, volume=0.1):
        current_buy_rate = self.GET_CURRENT_ASK()
        reward = 0
        required_budget = (volume * self.volume_point * current_buy_rate) / self.leverage
        # if self.budget > required_budget:
        if len(self.askPositions) == 0:
            reward = self.VALID_REWARD
            means = self.rowdata.close.rolling(12).mean()
            mean = means.iloc[-1]
            # reward = (mean - current_buy_rate)/mean
            position = {"volume": volume, "price": current_buy_rate, "step": self.stepCount}
            self.askPositions.append(position)
            self.budget = self.budget - required_budget
            self.orderCount += 1
        else:
            reward = self.INVALID_REWARD
            # reward = 0
        return reward

    def __sell__(self, volume=0.1):
        current_sell_rate = self.GET_CURRENT_BID()
        reward = 0
        required_budget = (volume * self.volume_point * current_sell_rate) / self.leverage
        # if self.budget > required_budget:
        if len(self.bidPositions) == 0:
            reward = self.VALID_REWARD
            means = self.rowdata.close.rolling(12).mean()
            mean = means.iloc[-1]
            # eward = (current_sell_rate - mean)/mean
            position = {"volume": volume, "price": current_sell_rate, "step": self.stepCount}
            # if self.budget * self.leverage >= volume*self.volume_point * current_sell_rate:
            self.bidPositions.append(position)
            self.budget = self.budget - required_budget
            self.orderCount += 1
        else:
            reward = self.INVALID_REWARD
            # reward = 0
        return reward
