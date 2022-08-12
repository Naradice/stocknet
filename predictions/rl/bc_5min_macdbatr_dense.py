import pfrl
from stocknet.nets.dense import SimpleDense, ConvDense16
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch
import torch.nn as nn
from torch.optim import SGD
import numpy

from stocknet.nets.dense import SimpleDense
from stocknet.trainer import RlTrainer
from stocknet.envs.bc_env import BCEnv, BCMultiActsEnv, BCStopEnv
import finance_client.finance_client as fc

dtype = torch.float32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print("Lerning with device:", device)

model_name = 'rl/bc_5min/stop/Conv_2h_v3'
max_step = 1000
idc_process = [fc.utils.MACDpreProcess(), fc.utils.BBANDpreProcess(), fc.utils.ATRpreProcess()]
data_client = fc.CSVClient(file='data_source/bitcoin_5_2017T0710-2021T103022.csv', idc_processes=idc_process)
#env = BCEnv(data_client, columns=[],max_step=max_step, observationDays=1/24,useBudgetColumns=True)
#env = BCMultiActsEnv(data_client, bugets=10,columns=[],max_step=max_step, observationDays=1/48,useBudgetColumns=True)
env = BCStopEnv(data_client, usebb=True, columns=[],max_step=max_step, observationDays=1/12)
processes = [fc.utils.MinMaxPreProcess(scale=(-1,1))]
env.register_preprocesses(processes)

obs = env.reset()
inputDim, size = obs.shape
print(inputDim, size, env.columns)

#model = SimpleDense(8,size, inputDim, 3, removeHistoryData=False, lr=True) #modelの宣言
model = ConvDense16(size, channel=inputDim, out_size=2)#.to(device=device)
criterion = nn.MSELoss() #評価関数の宣言
batch_size = 1

#optimizer = torch.optim.Adam(model.parameters(), eps=1e-6)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
gamma = 0.99
explorer = pfrl.explorers.ConstantEpsilonGreedy(epsilon=0.3, random_action_func=env.action_space.sample)
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
trainer.add_end_time(0,4)
trainer.training_loop(env, agent, model_name, 20000, max_step_len=max_step, render=False)