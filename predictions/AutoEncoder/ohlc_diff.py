import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

import stocknet.envs.datasets.bc as bc
from stocknet.nets.ae import AELinearModel
import finance_client.finance_client as fc
import stocknet.trainer as trainer
dtype = torch.float32
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.set_default_dtype(dtype)
torch.manual_seed(1017)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("device:", device)

data_client = fc.CSVClient(file='data_source/bitcoin_5_2017T0710-2021T103022.csv')
#ds = bc.Dataset(data_client=data_client, observationDays=1, isTraining=True)
ds = bc.ShiftDataset(data_client=data_client, observationDays=1, floor=1,isTraining=True)
ds.register_preprocess(fc.utils.DiffPreProcess())
ds.register_preprocess(fc.utils.MinMaxPreProcess(scale=(-1,1)))
ds.run_preprocess()

batch_size=32
train_dl = DataLoader(ds, batch_size = batch_size, drop_last = True, shuffle=False, pin_memory=True)

i,o = ds[0]
input_size = i.shape[0]

model_name = 'bc_5min_ohlc_shift-1_v1'
model = AELinearModel(input_size, device=device)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-6)
#optimizer = optim.SGD(model.parameters(), lr=1e-6)
loss_fn = nn.MSELoss()

tr = trainer.Trainer()
trainer.save_model_architecture(model, i, batch_size,model_name )
tr.training_loop(model,model_name, 100, optimizer, loss_fn, train_dl, device=device)

