import unittest

from matplotlib.pyplot import sca
import stocknet.envs.datasets.bc as bc
from stocknet.envs.market_clients.csv.client import CSVClient
import stocknet.envs.datasets.preprocess as process
import datetime
import torch

import torch.optim as optim
import torch.nn as nn

class TestBCDataset(unittest.TestCase):
    
    data_client = CSVClient('data_source/bitcoin_5_2017T0710-2021T103022.csv')

    def test_get_obs(self):
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        ds = bc.Dataset(self.data_client,  observationDays=1, isTraining=True)
        ds.add_indicater(process.MACDpreProcess())
        ds.register_preprocess(process.MinMaxPreProcess(scale=(0,1)))
        ds.register_preprocess(process.DiffPreProcess())
        ds.run_preprocess()
        column_num = len(ds.columns)
        i, o = ds[0:10]
        dataLength = 12*24*column_num
        expected_shape = (10, dataLength)
        self.assertEqual(expected_shape, i.shape)
        self.assertEqual(expected_shape, o.shape)
        self.assertEqual(i[0][0], o[0][0])
        self.assertTrue(i.min() >= -1, f"{i.min()} is invalid")
        self.assertTrue(i.max() <= 1, f"{i.max()} is invalid")
        print(i.min())
    
    def test_get_diff(self):
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        ds = bc.ShiftDataset(self.data_client, observationDays=1, floor=1, isTraining=True)
        ds.add_indicater(process.MACDpreProcess())
        ds.register_preprocess(process.MinMaxPreProcess(scale=(0,1)))
        ds.register_preprocess(process.DiffPreProcess())
        ds.run_preprocess()
        i, o = ds[0:10]
        column_num = len(ds.columns)
        dataLength = 12*24*column_num
        expected_shape = (10, dataLength)
        self.assertEqual(expected_shape, i.shape)
        self.assertEqual(expected_shape, o.shape)
        self.assertEqual(i[0][1], o[0][0])
        self.assertTrue(i.min() >= -1, f"{i.min()} is invalid")
        self.assertTrue(i.max() <= 1, f"{i.max()} is invalid")
        
    def test_dl(self):
        from torch.utils.data import DataLoader
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        ds = bc.Dataset(self.data_client, observationDays=1, isTraining=True)
        ds.register_preprocess(process.DiffPreProcess())
        ds.register_preprocess(process.MinMaxPreProcess(scale=(-1,1)))
        ds.run_preprocess()
        batch_size=32
        train_dl = DataLoader(ds, batch_size = batch_size,  drop_last = True, shuffle=False)
        count = 1
        start_time = datetime.datetime.now()
        consumed_total_time = datetime.timedelta(0)
        loss_fn = nn.MSELoss()
        loss_train_mse  = 0.0
        i, o = ds[0:batch_size]
        for inputValues, ansValue in train_dl:
            end_time = datetime.datetime.now()
            diff = end_time - start_time
            if count == 1 or count % 100 == 0:
                print(f"{count} times finished. meaning consumed time is ", consumed_total_time/count)
                print("may end on", consumed_total_time/count * (len(train_dl) - count))
                print(loss_train_mse/count)
            consumed_total_time += diff
            start_time = end_time
            count+=1
            loss = loss_fn(inputValues, i)
            loss_train_mse += loss
        print("finished. meaning consumed time is ", consumed_total_time/count)
        
if __name__ == '__main__':
    unittest.main()