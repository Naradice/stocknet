import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

import stocknet.envs.datasets.bc as bc
from stocknet.nets.lstm import Predictor
from stocknet.envs.market_clients.csv.client import CSVClient
import stocknet.envs.utils.idcprocess as indicater
import stocknet.envs.utils.preprocess as process
import stocknet.trainer as trainer
dtype = torch.float32
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.set_default_dtype(dtype)
torch.manual_seed(1017)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

def training_auto_encoder(data_client, batch_size, observationDays, processes,epoc_num=-1, hidden_layer_num = 5, middle_layer_size = 48, version=1):
    is_multi_output = False
    frame = str(data_client.frame)
    kinds = str(data_client.kinds)
    ema_ps_1 = indicater.EMApreProcess(key='e12', window=12)
    #ema_ps_2 = indicater.EMApreProcess(key='e26', window=12, is_output=is_multi_output)
    ids = [ema_ps_1]#, ema_ps_2]
    shift = 3
    ds = bc.ShiftDataset(data_client=data_client, observationDays=observationDays,out_ohlc__columns=[], isTraining=True, floor=shift)
    ds.add_indicaters(ids)
    ds.register_preprocesses(processes)
    ds.run_preprocess()
    
    batch_size=batch_size
    train_dl = DataLoader(ds, batch_size = batch_size, drop_last = True, shuffle=False, pin_memory=True)
    ds_val = bc.ShiftDataset(data_client=data_client, observationDays=observationDays, isTraining=False, out_ohlc__columns=[],floor=shift)
    ds_val.add_indicaters(ids)
    ds_val.register_preprocesses(processes)
    ds_val.run_preprocess()
    val_loader = DataLoader(ds_val, batch_size = batch_size, drop_last = True, shuffle=False, pin_memory=True)
    i,o = ds[0]
    input_size = i.shape[1]
    print("input:", i.shape, "output:", o.shape)

    #model_name = 'bc_5min_ohlc_AE_v2'
    if len(ids) > 1:
        if is_multi_output:
            model_name = f'{kinds}_{frame}min/{len(ids)}emas/shift{shift}_{int(observationDays*24)}h_LSTM8-{str(hidden_layer_num)}-{str(middle_layer_size).zfill(2)}_v{str(version)}'
        else:
            model_name = f'{kinds}_{frame}min/{len(ids)}emas-ema/shift{shift}_{int(observationDays*24)}h_LSTM8-{str(hidden_layer_num)}-{str(middle_layer_size).zfill(2)}_v{str(version)}'
    else:
        model_name = f'{kinds}_{frame}min/ema/shift{shift}_{int(observationDays*24)}h_LSTM8-{str(hidden_layer_num)}-{str(middle_layer_size).zfill(2)}_v{str(version)}'
    model = Predictor(input_size,hiddenDim=8,outputDim=o.shape[0],device=device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    #optimizer = optim.SGD(model.parameters(), lr=1e-6)
    loss_fn = nn.MSELoss()

    tr = trainer.Trainer(model_name, loss_fn, train_dl, val_loader, device)
    tr.save_architecture(model, i, batch_size)
    tr.save_params(ids, processes)    
    tr.training_loop(model,optimizer, epoc_num)
    tr.validate(model, val_loader)

if __name__ == "__main__":
    data_client = CSVClient('data_source/bitcoin_5_2017T0710-2021T103022.csv')
    ##hyper parameters##
    observationDays = 1
    #processes = [process.DiffPreProcess(), process.MinMaxPreProcess(scale=(-1,1))]
    processes = [process.MinMaxPreProcess(scale=(-1,1))]
    batch_size = 32
    hidden_layer_size = 5
    middle_layer_size = 96
    ####################
    #epoc_num = 50
    version = 2
    
    training_auto_encoder(data_client, batch_size, observationDays, processes, hidden_layer_num=hidden_layer_size, middle_layer_size=middle_layer_size, version=version)