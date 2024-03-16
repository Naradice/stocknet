import numpy
import pfrl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

import finance_client.finance_client as fc
from stocknet.envs.bc_env import BCEnv
from stocknet.nets.dense import ConvDense16, SimpleDense
from stocknet.train.rltrainer import RlTrainer

dtype = torch.float32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Lerning with device:", device)

model_name = "rl/bc_5min/macd/ConvDense16_mono_v1"
max_step = 1000
data_client = fc.CSVClient(file="data_source/bitcoin_5_2017T0710-2021T103022.csv", frame=fc.Frame.MIN5, idc_processes=[fc.utils.MACDpreProcess()])
env = BCEnv(data_client, columns=[], max_step=max_step, observationDays=3, useBudgetColumns=True)
processes = [fc.utils.DiffPreProcess(), fc.utils.MinMaxPreProcess(scale=(-1, 1))]
env.register_preprocesses(processes)

obs = env.reset()
inputDim, size = obs.shape
# model = SimpleDense(30,size, inputDim, 3, removeHistoryData=False, lr=True) #modelの宣言
model = ConvDense16(size)  # .to(device=device)
criterion = nn.MSELoss()  # 評価関数の宣言
batch_size = 32

optimizer = torch.optim.Adam(model.parameters(), eps=1e-5)
gamma = 0.9
explorer = pfrl.explorers.ConstantEpsilonGreedy(epsilon=0.2, random_action_func=env.action_space.sample)
replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=batch_size)
phi = lambda x: x.astype(numpy.float32, copy=False)
gpu = -1
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
    gpu=gpu,
)

trainer = RlTrainer()
trainer.add_end_time(0, 5)
trainer.training_loop(env, agent, model_name, 10000, max_step_len=max_step, render=False)
