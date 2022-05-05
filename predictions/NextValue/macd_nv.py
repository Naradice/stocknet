import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

import stocknet.envs.datasets.bc as bc
from stocknet.nets.lstm import Predictor
from stocknet.envs.market_clients.csv.client import CSVClient
import stocknet.envs.utils.preprocess as process
import stocknet.trainer as trainer
dtype = torch.float32
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.set_default_dtype(dtype)
torch.manual_seed(1017)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

def training_auto_encoder(data_client, batch_size, observationDays, processes,epoc_num=-1, hidden_layer_num = 5, middle_layer_size = 48, version=1):
    frame = str(data_client.frame)
    kinds = str(data_client.kinds)
    macd_ps = process.MACDpreProcess()
    ds = bc.ShiftDataset(data_client=data_client, observationDays=observationDays, isTraining=True,floor=1)
    ds.add_indicater(macd_ps)
    ds.columns = macd_ps.columns
    ds.register_preprocesses(processes)
    ds.run_preprocess()
    
    
    batch_size=batch_size
    train_dl = DataLoader(ds, batch_size = batch_size, drop_last = True, shuffle=False, pin_memory=True)
    ds_val = bc.ShiftDataset(data_client=data_client, observationDays=observationDays, isTraining=False, floor=2)
    ds_val.add_indicater(macd_ps)
    ds_val.columns = macd_ps.columns
    ds_val.register_preprocesses(processes)
    ds_val.run_preprocess()
    val_loader = DataLoader(ds_val, batch_size = batch_size, drop_last = True, shuffle=False, pin_memory=True)
    i,o = ds[0]
    input_size = i.shape[1]
    print("input:", i.shape, "output:", o.shape)

    #model_name = 'bc_5min_ohlc_AE_v2'
    model_name = f'{kinds}_{frame}min/macd/LSTM16-{str(hidden_layer_num)}-{str(middle_layer_size).zfill(2)}_v{str(version)}'
    model = Predictor(input_size,hiddenDim=16,outputDim=o.shape[0],device=device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    #optimizer = optim.SGD(model.parameters(), lr=1e-6)
    loss_fn = nn.MSELoss()

    tr = trainer.Trainer(model_name, loss_fn, train_dl, val_loader, device)
    trainer.save_model_architecture(model, i, batch_size,model_name )
    tr.training_loop(model,optimizer, epoc_num)

if __name__ == "__main__":
    data_client = CSVClient('data_source/bitcoin_5_2017T0710-2021T103022.csv')
    ##hyper parameters##
    observationDays = 1
    processes = [process.DiffPreProcess(), process.MinMaxPreProcess(scale=(-1,1))]
    batch_size = 32
    hidden_layer_size = 5
    middle_layer_size = 96
    ####################
    #epoc_num = 50
    
    training_auto_encoder(data_client, batch_size, observationDays, processes, hidden_layer_num=hidden_layer_size, middle_layer_size=middle_layer_size)