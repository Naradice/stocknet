import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

import sys, os

finance_client_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../finance_client'))
sys.path.append(finance_client_module_path)

import finance_client as fc
import finance_client.utils.idcprocess as indicater

import stocknet.datasets as ds
from stocknet.nets.ae import AELinearModel
import finance_client.finance_client as fc
import stocknet.trainer as trainer
dtype = torch.float32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("device:", device)

# if __name__ == "__main__":
#     ema_ps = fc.utils.EMApreProcess(window = 12)
#     data_client = fc.CSVClient(file='data_source/bitcoin_5_2017T0710-2021T103022.csv', idc_processes=[ema_ps])
#     ##hyper parameters##
#     observationDays = 1
#     processes = [fc.utils.DiffPreProcess(), fc.utils.MinMaxPreProcess(scale=(-1,1))]
#     batch_size = 32
#     hidden_layer_size = 5
#     middle_layer_size = 96
#     ####################
#     #epoc_num = 50
#     training_auto_encoder(data_client, batch_size, observationDays, processes, hidden_layer_num=hidden_layer_size, middle_layer_size=middle_layer_size)
    
def train_auto_encoder_macd(file, client_columns, client_date_column, batch_size, observationDays, columns, epoc_num=-1, hidden_layer_num=5, middle_layer_size=48, version=1):
    macd_ps = fc.utils.MACDpreProcess()
    processes = [fc.utils.MinMaxPreProcess(scale=(-1,1))]
    data_client = fc.CSVClient(file=file, columns=client_columns, date_column=client_date_column, idc_processes=[macd_ps], post_process=processes)
    columns = []
    for key, value in macd_ps.columns:
        columns.append(value)
        
    ##hyper parameters##
    observationDays = observationDays
    batch_size = batch_size
    ####################
    #epoc_num = 50
    frame = str(data_client.frame)
    kinds = str(data_client.kinds)
    
    batch_size=batch_size
    dataset = ds.Dataset(data_client=data_client, observationDays=observationDays, in_columns=columns, out_columns=columns, isTraining=True)    
    train_dl = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=False, pin_memory=True)
    ds_val = ds.Dataset(data_client=data_client, observationDays=observationDays, in_columns=columns, out_columns=columns, isTraining=False)
    val_loader = DataLoader(ds_val, batch_size=batch_size, drop_last=True, shuffle=False, pin_memory=True)
    i,o = dataset[0]
    input_size = i.shape[0]

    model_name = f'{kinds}_{frame}min/macd/AE-{str(hidden_layer_num)}-{str(middle_layer_size).zfill(2)}_v{str(version)}'
    model = AELinearModel(input_size,hidden_layer_num=hidden_layer_num, middle_layer_size=middle_layer_size, device=device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    #optimizer = optim.SGD(model.parameters(), lr=1e-6)
    loss_fn = nn.MSELoss()

    tr = trainer.Trainer(model_name, loss_fn, train_dl, val_loader, device)
    trainer.save_model_architecture(model, i, batch_size,model_name )
    tr.training_loop(model,optimizer, epoc_num)

    
def train_diff():
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
    
def training_auto_encoder(data_client, batch_size, observationDays, processes,epoc_num=-1, hidden_layer_num = 5, middle_layer_size = 48, version=1):
    frame = str(data_client.frame)
    kinds = str(data_client.kinds)
    ds = bc.Dataset(data_client=data_client, observationDays=observationDays, isTraining=True)
    ds.register_preprocesses(processes)
    ds.run_preprocess()
    
    batch_size=batch_size
    train_dl = DataLoader(ds, batch_size = batch_size, drop_last = True, shuffle=False, pin_memory=True)
    ds_val = bc.Dataset(data_client=data_client, observationDays=observationDays, isTraining=False)
    ds_val.register_preprocesses(processes)
    ds_val.run_preprocess()
    val_loader = DataLoader(ds_val, batch_size = batch_size, drop_last = True, shuffle=False, pin_memory=True)
    i,o = ds[0]
    input_size = i.shape[0]


    #model_name = 'bc_5min_ohlc_AE_v2'
    model_name = f'{kinds}_{frame}min/ohlc/AE-{str(hidden_layer_num)}-{str(middle_layer_size).zfill(2)}_v{str(version)}'
    model = AELinearModel(input_size,hidden_layer_num=hidden_layer_num,middle_layer_size=middle_layer_size, device=device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    #optimizer = optim.SGD(model.parameters(), lr=1e-6)
    loss_fn = nn.MSELoss()

    tr = trainer.Trainer(model_name, loss_fn, train_dl, val_loader, device)
    trainer.save_model_architecture(model, i, batch_size,model_name )
    tr.training_loop(model,optimizer, epoc_num)

if __name__ == "__main__":
    data_client = fc.CSVClient(file='data_source/bitcoin_5_2017T0710-2021T103022.csv')
    ##hyper parameters##
    observationDays = 1
    processes = [fc.utils.DiffPreProcess(), fc.utils.MinMaxPreProcess(scale=(-1,1))]
    batch_size = 32
    hidden_layer_size = 5
    middle_layer_size = 96
    ####################
    #epoc_num = 50
    
    training_auto_encoder(data_client, batch_size, observationDays, processes, hidden_layer_num=hidden_layer_size, middle_layer_size=middle_layer_size)
    
    
file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../finance_client/finance_client/data_source/mt5/OANDA-Japan MT5 Live/mt5_USDJPY_d1.csv'))
