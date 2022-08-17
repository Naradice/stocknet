import os, sys
finance_client_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../finance_client'))
sys.path.append(finance_client_module_path)
stocknet_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(stocknet_module_path)


import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

import stocknet.envs.datasets.finance as bc
from stocknet.nets.lstm import Predictor
import finance_client.utils.idcprocess as indicater
import stocknet.trainer as trainer
dtype = torch.float32
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.set_default_dtype(dtype)
torch.manual_seed(1017)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

def training_lstm_model(data_client, batch_size, observationLength, in_column, out_column, epoc_num=-1, hidden_layer_num = 5, middle_layer_size = 48, version=1):
    frame = data_client.frame
    kinds = str(data_client.kinds)
    batch_size=batch_size
    
    ds = bc.Dataset(data_client=data_client, observationLength=observationLength, in_columns=in_column, out_columns=out_column)
    train_dl = DataLoader(ds, batch_size=batch_size, drop_last = True, shuffle=False, pin_memory=True)
    
    ds_val = bc.Dataset(data_client=data_client, observationLength=observationLength, in_columns=in_column, out_columns=out_column, isTraining=False)
    val_loader = DataLoader(ds_val, batch_size = batch_size, drop_last = True, shuffle=False, pin_memory=True)
    
    i,o = ds[0]
    print("input:", i.shape, "output:", o.shape)
    input_size = i.shape[0]

    model_name = f'{kinds}_{str(frame)}min/next_trend/{int(observationLength/(60/frame))}h_LSTM8-{str(hidden_layer_num)}-{str(middle_layer_size).zfill(2)}_v{str(version)}'
    model = Predictor(input_size,hiddenDim=8,outputDim=o.shape[0],device=device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    #optimizer = optim.SGD(model.parameters(), lr=1e-6)
    loss_fn = nn.MSELoss()

    tr = trainer.Trainer(model_name, loss_fn, train_dl, val_loader, device)
    tr.save_architecture(model, i, batch_size)
    tr.save_client(data_client)
    tr.training_loop(model,optimizer, epoc_num)
    tr.validate(model, val_loader)

def training_lstm_with_shift(data_client, batch_size, observationLength, in_column, out_column, shift=1, epoc_num=-1, hidden_layer_num = 5, middle_layer_size = 48, version=1):
    is_multi_output = False
    frame = data_client.frame
    kinds = str(data_client.kinds)
    
    ds = bc.ShiftDataset(data_client=data_client, observationLength=observationLength, in_columns=in_column, out_columns=out_column, isTraining=True, floor=shift)
    train_dl = DataLoader(ds, batch_size = batch_size, drop_last = True, shuffle=False, pin_memory=True)
    ds_val = bc.ShiftDataset(data_client=data_client, observationLength=observationLength, isTraining=False, in_columns=in_column, out_columns=out_column, floor=shift)
    val_loader = DataLoader(ds_val, batch_size = batch_size, drop_last = True, shuffle=False, pin_memory=True)
    i,o = ds[0]
    input_size = i.shape[1]
    print("input:", i.shape, "output:", o.shape)

    #model_name = 'bc_5min_ohlc_AE_v2'
    model_name = f'{kinds}_{str(frame)}min/emas/shift{shift}_{int(observationLength / (60/frame))}h_LSTM8-{str(hidden_layer_num)}-{str(middle_layer_size).zfill(2)}_v{str(version)}'
    model = Predictor(input_size,hiddenDim=8,outputDim=o.shape[0],device=device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    #optimizer = optim.SGD(model.parameters(), lr=1e-6)
    loss_fn = nn.MSELoss()

    tr = trainer.Trainer(model_name, loss_fn, train_dl, val_loader, device)
    tr.save_architecture(model, i, batch_size)
    tr.save_client(data_client)
    tr.training_loop(model,optimizer, epoc_num)
    tr.validate(model, val_loader)