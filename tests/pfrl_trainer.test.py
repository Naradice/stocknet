import numpy
import pfrl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pfrl import experiments, utils  # NOQA:E402
from pfrl.agents import a3c
from torch.optim import SGD

import stocknet.envs.utils.preprocess as process
from stocknet.envs.bc_env import BC5Env
from stocknet.envs.market_clients.csv.client import CSVClient
from stocknet.nets.dense import ConvDense16, SimpleDense
from stocknet.train.rltrainer import RlTrainer

dtype = torch.float32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Lerning with device:", device)

processes = 4
seed = 1017
process_seeds = numpy.arange(processes) + seed * processes


def make_env(process_idx, test):
    # Use different random seeds for train and test envs
    process_seed = process_seeds[process_idx]
    env_seed = 2**31 - 1 - process_seed if test else process_seed
    data_client = CSVClient("data_source/bitcoin_5_2017T0710-2021T103022.csv")
    env = BC5Env(data_client, columns=[], max_step=max_step, observationDays=3, useBudgetColumns=True, use_diff=True)
    env.seed = int(env_seed)
    env.add_indicater(process.MACDpreProcess())
    processes = [process.DiffPreProcess(), process.MinMaxPreProcess(scale=(-1, 1))]
    env.register_preprocesses(processes)
    return env


model_name = "rl/bc_5min/macd/ConvDense16_mono_v1"
max_step = 1000

env = make_env(0, False)
obs = env.reset()
inputDim, size = obs.shape
n_actions = env.action_space.n


# model = SimpleDense(30,size, inputDim, 3, removeHistoryData=False, lr=True)
model = ConvDense16(size)  # .to(device=device)
criterion = nn.MSELoss()
batch_size = 32

optimizer = torch.optim.Adam(model.parameters(), eps=1e-5)
gamma = 0.9
explorer = pfrl.explorers.ConstantEpsilonGreedy(epsilon=0.01, random_action_func=env.action_space.sample)
replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=batch_size)
phi = lambda x: x.astype(numpy.float32, copy=False)
gpu = 0
agent = a3c.A3C(
    model,
    optimizer,
    t_max=5,
    gamma=0.99,
    beta=1e-2,
    phi=phi,
    max_grad_norm=40.0,
)
trainer = RlTrainer()
experiments.train_agent_async(
    agent=agent,
    outdir="models",
    processes=5,
    make_env=make_env,
    steps=1000,
    eval_n_episodes=None,
    eval_n_steps=1000,
    save_best_so_far_agent=True,
)
# trainer.train_agent_async('/',5,make_env, agent=agent)
