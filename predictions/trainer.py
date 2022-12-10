import os, sys
finance_client_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../finance_client'))
sys.path.append(finance_client_module_path)
stocknet_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(stocknet_module_path)

import stocknet.datasets as ds
import finance_client.utils.idcprocess as indicater
import stocknet.trainer as trainer
from stocknet.nets.lstm import LSTM
from stocknet.nets.ae import AELinearModel

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

#torch.set_default_tensor_type('torch.cuda.FloatTensor')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

def __train_common(data_client, model, sample_input, batch_size, model_name, train_dl, val_dl, epoc_num=-1, loss_fn = nn.MSELoss(), optimizer = None):
    model = model.to(device)
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
    else:
        optimizer = optimizer(model.parameters(), lr=1e-5)
    tr = trainer.Trainer(model_name, loss_fn, train_dl, val_dl, device)
    tr.save_architecture(model, sample_input, batch_size)
    tr.save_client(data_client)
    tr.training_loop(model, optimizer, epoc_num)
    tr.validate(model, val_dl)

def __create_dataloaders(data_client, dataset, batch_size, observationLength, in_column, out_column, idc_processes=[], pre_processes=[]):
    if data_client is None and dataset is None:
        raise Exception("You need to provide either data_client or dataset")
    
    if dataset is not None:
        #if dataset is provided, use it even if data_client is also provided.
        data_client = dataset.data_client
        frame = data_client.frame
        kinds = str(data_client.kinds)
        val_ds_class = ds.available_dataset[dataset.key]
        ds_val = val_ds_class(*dataset.args, isTraining=False)
    elif data_client is not None:
        frame = data_client.frame
        kinds = str(data_client.kinds)
        
        dataset = ds.Dataset(data_client=data_client, idc_processes=idc_processes, pre_processes=pre_processes, observationLength=observationLength, in_columns=in_column, out_columns=out_column, merge_columns=False)
        ds_val = ds.Dataset(data_client=data_client, idc_processes=idc_processes, pre_processes=pre_processes, observationLength=observationLength, in_columns=in_column, out_columns=out_column, merge_columns=False, isTraining=False)
    input, output = dataset[0]
    train_dl = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=False, pin_memory=True)
    val_dl = DataLoader(ds_val, batch_size=batch_size, drop_last=True, shuffle=False, pin_memory=True)    
    return data_client, train_dl, val_dl, frame, kinds, input, output

def training_lstm_model(data_client=None, dataset=None, idc_processes=[], pre_processes=[], batch_size=32, observationLength=100, in_column=["open"], out_column=["close"], epoc_num=-1, hidden_layer_num=5, optimizer=None, loss_fn=nn.MSELoss(), version=1, model_name=None, activation_for_output=torch.tanh):

    data_client, train_dl, val_dl, frame, kinds, i, o = __create_dataloaders(data_client=data_client, dataset=dataset, idc_processes=idc_processes, pre_processes=pre_processes, 
                                                                             batch_size=batch_size, observationLength=observationLength, in_column=in_column, out_column=out_column)
    print("input:", i.shape, "output:", o.shape)
    # need to modify the dim when shape is unexpected
    input_size = i.shape[1]

    if model_name is None:
        model_name = f'{kinds}_{str(frame)}min/{int(observationLength/(60/frame))}h_LSTM{str(hidden_layer_num)}_v{str(version)}'
    else:
        model_name = f'{kinds}_{str(frame)}min/{model_name}'
    model = LSTM(input_size, hiddenDim=hidden_layer_num, outputDim=o.shape[0], device=device, activation_for_output=activation_for_output)
    __train_common(data_client=data_client, model=model, model_name=model_name,
                    sample_input=i, batch_size=batch_size,
                   train_dl=train_dl, val_dl=val_dl, epoc_num=epoc_num,
                   optimizer=optimizer, loss_fn=loss_fn)
    
def training_lenear_auto_encoder(data_client=None, dataset=None, batch_size=32, observationLength=100, in_column=["open"], out_column=None,
                                 epoc_num=-1, hidden_layer_num = 5, middle_layer_size = 48, version=1,
                                 optimizer=None, loss_fn=nn.MSELoss(), model_name=None):
    data_client, train_dl, val_dl, frame, kinds, input, output = __create_dataloaders(data_client=data_client, dataset=dataset, batch_size=batch_size, 
                                                          observationLength=observationLength, in_column=in_column, out_column=out_column)
    input_size = input.shape[0]
    if model_name is None:
        model_name = f'{kinds}_{frame}min/macd/AE-{str(hidden_layer_num)}-{str(middle_layer_size).zfill(2)}_v{str(version)}'
    else:
        model_name = f'{kinds}_{str(frame)}min/{model_name}'
    
    model = AELinearModel(input_size,hidden_layer_num=hidden_layer_num, middle_layer_size=middle_layer_size, device=device)
    model = model.to(device)
    
def training_lstm_auto_encoder(data_client=None, dataset=None, batch_size=32, observationLength=100, in_column=["open"], out_column=None,
                                 epoc_num=-1, hidden_layer_num = 5, middle_layer_size = 48, version=1,
                                 optimizer=None, loss_fn=nn.MSELoss(), model_name=None):
    data_client, train_dl, val_dl, frame, kinds, input, output = __create_dataloaders(data_client=data_client, dataset=dataset, batch_size=batch_size, 
                                                          observationLength=observationLength, in_column=in_column, out_column=out_column)

    
    input_size = input.shape[0]
    if model_name is None:
        model_name = f'{kinds}_{frame}min/LSTMAE-{str(hidden_layer_num)}_v{str(version)}'
    else:
        model_name = f'{kinds}_{str(frame)}min/{model_name}'
    
    model = LSTM(input_size,hidden_layer_num=hidden_layer_num, middle_layer_size=middle_layer_size, device=device)
    model = model.to(device)

    __train_common(data_client=data_client, model=model, model_name=model_name,
                    sample_input=input, batch_size=batch_size,
                   train_dl=train_dl, val_dl=val_dl, epoc_num=epoc_num,
                   optimizer=optimizer, loss_fn=loss_fn)
