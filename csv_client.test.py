import unittest
from stocknet.envs.market_clients.csv.client import CSVClient
from stocknet.envs.market_clients.frames import Frame

class TestCSVClient(unittest.TestCase):

    client = CSVClient(file='data_source/bitcoin_5_2017T0710-2021T103022.csv')
    
    def test_get_rates(self):
        length = 10
        rates = self.client.get_rates(length, frame=Frame.MIN5)
        self.assertEqual(len(rates.Close), length)

    def test_get_next_tick(self):
        print(self.client.get_next_tick())
        print(self.client.get_next_tick())
    
    def test_get_current_ask(self):
        print(self.client.get_current_ask())

    def test_get_current_bid(self):
        print(self.client.get_current_bid())
        
    def test_get_30min_rates(self):
        length = 10
        rates = self.client.get_rates(length, frame=Frame.MIN30)
        self.assertEqual(len(rates.Close), length)
    
if __name__ == '__main__':
    unittest.main()