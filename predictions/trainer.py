import os, sys
finance_client_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../finance_client'))
sys.path.append(finance_client_module_path)
stocknet_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(stocknet_module_path)


import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

import stocknet.datasets as ds
import finance_client.utils.idcprocess as indicater
import stocknet.trainer as trainer
from stocknet.nets.lstm import LSTM
dtype = torch.float32

#torch.set_default_tensor_type('torch.cuda.FloatTensor')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

def __train_common(data_client, model, sample_iunput, batch_size, model_name, train_dl, val_dl, epoc_num=-1, loss_fn = nn.MSELoss(), optimizer = None):
    model = model.to(device)
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
    tr = trainer.Trainer(model_name, loss_fn, train_dl, val_dl, device)
    tr.save_architecture(model, sample_iunput, batch_size)
    tr.save_client(data_client)
    tr.training_loop(model,optimizer, epoc_num)
    tr.validate(model, val_dl)


def training_lstm_model(data_client, batch_size, observationLength, in_column, out_column, epoc_num=-1, hidden_layer_num=5, middle_layer_size=48, optimizer=None, loss_fn=nn.MSELoss(), version=1, model_name=None):
    frame = data_client.frame
    kinds = str(data_client.kinds)
    
    ds = ds.Dataset(data_client=data_client, observationLength=observationLength, in_columns=in_column, out_columns=out_column)
    train_dl = DataLoader(ds, batch_size=batch_size, drop_last = True, shuffle=False, pin_memory=True)
    
    ds_val = ds.Dataset(data_client=data_client, observationLength=observationLength, in_columns=in_column, out_columns=out_column, isTraining=False)
    val_loader = DataLoader(ds_val, batch_size = batch_size, drop_last = True, shuffle=False, pin_memory=True)
    
    i,o = ds[0]
    print("input:", i.shape, "output:", o.shape)
    input_size = i.shape[0]

    if model_name is None:
        model_name = f'{kinds}_{str(frame)}min/{int(observationLength/(60/frame))}h_LSTM{str(hidden_layer_num)}-{str(middle_layer_size).zfill(2)}_v{str(version)}'
    else:
        model_name = f'{kinds}_{str(frame)}min/{model_name}'
    model = LSTM(input_size, hiddenDim=hidden_layer_num, outputDim=o.shape[0], device=device)
    __train_common(data_client=data_client, model=model, model_name=model_name,
                   train_dl=train_dl, val_dl=val_loader, epoc_num=epoc_num,
                   optimizer=optimizer, loss_fn=loss_fn)
    
def training_lstm_with_shift(data_client, batch_size, observationLength, in_column, out_column, shift=1, epoc_num=-1, hidden_layer_num = 5, optimizer=None, loss_fn=nn.MSELoss(), version=1, model_name=None):
    frame = data_client.frame
    kinds = str(data_client.kinds)
    
    ds = ds.ShiftDataset(data_client=data_client, observationLength=observationLength, in_columns=in_column, out_columns=out_column, isTraining=True, floor=shift)
    train_dl = DataLoader(ds, batch_size = batch_size, drop_last = True, shuffle=False, pin_memory=True)
    ds_val = ds.ShiftDataset(data_client=data_client, observationLength=observationLength, isTraining=False, in_columns=in_column, out_columns=out_column, floor=shift)
    val_loader = DataLoader(ds_val, batch_size = batch_size, drop_last = True, shuffle=False, pin_memory=True)
    i,o = ds[0]
    input_size = i.shape[1]
    print("input:", i.shape, "output:", o.shape)

    #model_name = 'bc_5min_ohlc_AE_v2'
    if model_name is None:
        model_name = f'{kinds}_{str(frame)}min/shift{shift}_{int(observationLength / (60/frame))}h_LSTM_{str(hidden_layer_num)}_v{str(version)}'
    else:
        model_name = f'{kinds}_{str(frame)}min/{model_name}'
    model = LSTM(input_size,hiddenDim=hidden_layer_num,outputDim=o.shape[0],device=device)
    __train_common(data_client=data_client, model=model, model_name=model_name,
                   train_dl=train_dl, val_dl=val_loader, epoc_num=epoc_num,
                   optimizer=optimizer, loss_fn=loss_fn)
    
def training_auto_encoder(data_client, batch_size, observationDays, processes, columns, epoc_num=-1, hidden_layer_num = 5, middle_layer_size = 48, version=1):
    frame = str(data_client.frame)
    kinds = str(data_client.kinds)
    ds = ds.Dataset(data_client=data_client, observationDays=observationDays, isTraining=True)
    ds.columns = columns
    ds.register_preprocesses(processes)
    ds.run_preprocess()
    
    
    batch_size=batch_size
    train_dl = DataLoader(ds, batch_size = batch_size, drop_last = True, shuffle=False, pin_memory=True)
    ds_val = ds.Dataset(data_client=data_client, observationDays=observationDays, isTraining=False)
    ds_val.columns = columns
    ds_val.register_preprocesses(processes)
    ds_val.run_preprocess()
    val_loader = DataLoader(ds_val, batch_size = batch_size, drop_last = True, shuffle=False, pin_memory=True)
    i,o = ds[0]
    input_size = i.shape[0]

    #model_name = 'bc_5min_ohlc_AE_v2'
    model_name = f'{kinds}_{frame}min/ema/AE-{str(hidden_layer_num)}-{str(middle_layer_size).zfill(2)}_v{str(version)}'
    model = AELinearModel(input_size,hidden_layer_num=hidden_layer_num,middle_layer_size=middle_layer_size, device=device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    #optimizer = optim.SGD(model.parameters(), lr=1e-6)
    loss_fn = nn.MSELoss()

    tr = trainer.Trainer(model_name, loss_fn, train_dl, val_loader, device)
    trainer.save_model_architecture(model, i, batch_size,model_name )
    tr.training_loop(model,optimizer, epoc_num)
