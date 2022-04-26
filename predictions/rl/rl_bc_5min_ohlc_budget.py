import pfrl
import torch.nn.functional as F
import torch.nn as nn
import torch
import random
import numpy
import datetime
import sys
from stocknet.envs.bc_env import BC5Env
from stocknet.envs.market_clients.csv.client import CSVClient
import torch
import torch.nn as nn
from torch.optim import SGD
import math
import numpy as np

class PredictorMultiple(nn.Module):
    def __init__(self, layer_num, size, inputDim, n_actions, removeHistoryData = True):
        super().__init__()
        self.size = size
        self.rhd = removeHistoryData
        self.ActionHistoryDim = 2
        if removeHistoryData:
            input_dims = inputDim - self.ActionHistoryDim
        else:
            input_dims = inputDim
        self.layerDips = size * input_dims
        self.layers = nn.ModuleList()
        for i in range(0, layer_num):
            layer = nn.Linear(self.layerDips, self.layerDips)
            self.layers.append(layer)
        
        out_in_dims = self.layerDips
        if self.rhd:
            out_in_dims += self.ActionHistoryDim
        self.output_layer = nn.Linear( out_in_dims , n_actions)

    def forward(self, inputs):
        batch_size, feature_len,seq_len = inputs.shape[0], inputs.shape[1],inputs.shape[2]
        if self.rhd:
            feature_len = feature_len - self.ActionHistoryDim
            last_actions = inputs[:, -self.ActionHistoryDim:,-1] # [1, ActionHistoryDim] (ex.torch.Size([1, 3]))
        out = inputs[:, :feature_len, :]
        layerDips = feature_len * seq_len
        #assert layerDips == self.layerDips
        out = out.view(-1, layerDips)
        for layer in self.layers:
            out = torch.tanh(layer(out))
        if self.rhd:
            out = torch.cat((out, last_actions), dim=1)
        out = torch.tanh(self.output_layer(out))
        return pfrl.action_value.DiscreteActionValue(out)

dtype = torch.float32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Lerning with device:", device)


data_client = CSVClient('data_source/bitcoin_5_2017T0710-2021T103022.csv')
env = BC5Env(data_client, columns=["macd"], useBudgetColumns=True, use_diff=True)

obs = env.reset()
inputDim, size = obs.shape
batch_size = 2
model = PredictorMultiple(30,size, inputDim, 3, removeHistoryData=False) #modelの宣言
criterion = nn.MSELoss() #評価関数の宣言
batch_size = 2

optimizer = torch.optim.Adam(model.parameters(), eps=1e-7)
gamma = 0.9
explorer = pfrl.explorers.ConstantEpsilonGreedy(epsilon=0.1, random_action_func=env.action_space.sample)