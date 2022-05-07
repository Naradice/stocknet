import pfrl
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch
import torch.nn as nn
from torch.optim import SGD
import numpy

from stocknet.nets.dense import SimpleDense
from stocknet.trainer import RlTrainer
from stocknet.envs.bc_env import BC5Env
from stocknet.envs.market_clients.csv.client import CSVClient
import stocknet.envs.utils.preprocess as process

dtype = torch.float32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Lerning with device:", device)

model_name = 'rl/bc_5min/multi/Dense8_1h_v2'
max_step = 1000
data_client = CSVClient('data_source/bitcoin_5_2017T0710-2021T103022.csv')
env = BC5Env(data_client, columns=["Close"],max_step=max_step, observationDays=1/24,useBudgetColumns=True, use_diff=True)
env.add_indicater(process.MACDpreProcess())
env.add_indicater(process.BBANDpreProcess())
env.add_indicater(process.ATRpreProcess())
processes = [process.MinMaxPreProcess(scale=(-1,1))]
env.register_preprocesses(processes)

obs = env.reset()
inputDim, size = obs.shape
print(inputDim, size, env.columns)

model = SimpleDense(8,size, inputDim, 3, removeHistoryData=False, lr=True) #modelの宣言
#model = ConvDense16(size)#.to(device=device)
criterion = nn.MSELoss() #評価関数の宣言
batch_size = 16

optimizer = torch.optim.Adam(model.parameters(), eps=1e-5)
gamma = 0.9
explorer = pfrl.explorers.ConstantEpsilonGreedy(epsilon=0.05, random_action_func=env.action_space.sample)
replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=batch_size)
phi = lambda x: x.astype(numpy.float32, copy=False)
gpu = 0
agent = pfrl.agents.DoubleDQN(
    model,
    optimizer,
    replay_buffer,
    gamma,
    explorer,
    minibatch_size=batch_size,
    replay_start_size=batch_size,
    update_interval=1,
    target_update_interval=50000,
    phi=phi,
    gpu=gpu
)

trainer = RlTrainer()
trainer.add_end_time(0,8)
trainer.training_loop(env, agent, model_name, 10000, max_step_len=max_step, render=False)