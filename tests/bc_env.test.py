import unittest
from market_clients.csv.client import CSVClient
from bc_env import BC5Env
import time
import pandas as pd

class TestRenderClient(unittest.TestCase):
    
    data_client = CSVClient('data_source/bitcoin_5_2017T0710-2021T103022.csv')
    env = BC5Env(data_client)
        
    def test_step(self):
        self.env.reset()
        self.env.step(0)
        self.env.step(1)
        for i in range(100):
            self.env.step(0)
            self.env.render()
        self.env.step(2)
        
    """
    def test_update_observation(self):
        tick, done = self.data_client.get_next_tick()
        preState = self.env.dataSet.iloc[-1]
        self.env.__update_observation(tick)
        postState = self.env.dataSet.iloc[-1]
        self.assertEqual(preState.Close, self.env.dataSet.iloc[-2].Close)
        self.assertNotEqual(preState.Close, postState.Close)
    """
        
if __name__ == '__main__':
    unittest.main()