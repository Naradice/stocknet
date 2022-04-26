import torch
from torch.utils.data import DataLoader

import stocknet.envs.datasets.bc as bc
from stocknet.nets.ae import AELinearModel
from stocknet.envs.market_clients.csv.client import CSVClient
import stocknet.envs.utils.preprocess as process
import stocknet.trainer as trainer
import stocknet.validater as val
dtype = torch.float32
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.set_default_dtype(dtype)
torch.manual_seed(1017)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("device:", device)

data_client = CSVClient('data_source/bitcoin_5_2017T0710-2021T103022.csv')
ds = bc.Dataset(data_client=data_client, observationDays=1, isTraining=False)
ds.register_preprocess(process.DiffPreProcess())
ds.register_preprocess(process.MinMaxPreProcess(scale=(-1,1)))
ds.run_preprocess()

batch_size=32
val_dl = DataLoader(ds, batch_size = batch_size, drop_last = True, shuffle=False, pin_memory=True)

i,o = ds[0]
input_size = i.shape[0]

model_name = 'bc_5min_ohlc_AE-5-12_v1'
model = AELinearModel(input_size,hidden_layer_num=5,middle_layer_size=12, device=device)
model = model.to(device)
loss_fn = torch.nn.MSELoss()


trainer.load_model(model, model_name)
val.validate(model, val_dl, device, loss_fn)
