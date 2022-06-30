import unittest, os, json, sys, datetime
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(module_path)

finance_client_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../finance_client'))
sys.path.append(finance_client_module_path)

import unittest
from finance_client.csv.client import CSVClient
from stocknet.envs.bc_env import BCEnv
import time
import pandas as pd

file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data_source/bitcoin_5_2017T0710-2021T103022.csv'))

class TestRenderClient(unittest.TestCase):
    
    data_client = CSVClient(file=file_path)
    env = BCEnv(data_client, max_step = 1000)
        
    def test_step(self):
        self.env.reset()
        self.env.step(0)
        self.env.step(1)
        for i in range(100):
            self.env.step(0)
            self.env.render()
        self.env.step(2)
    
    def test_get_prices(self):
        ask = self.env.ask
        bid = self.env.bid
        
        self.assertGreater(ask, bid)
        
    def test_check_env_params(self):
        rewards = self.env.reward_range
        self.assertGreater(rewards[1], rewards[0])
        
    def test_reset(self):
        index = self.env.get_data_index()
        obs = self.env.reset()
        index_after_reset = self.env.get_data_index()
        self.assertNotEqual(index, index_after_reset)
        
    

        
if __name__ == '__main__':
    unittest.main()