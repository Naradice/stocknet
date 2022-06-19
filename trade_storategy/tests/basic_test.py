import unittest, os, json, sys, datetime
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(module_path)
import trade_storategy as ts
from finance_client.client_base import Client
from finance_client.csv.client import CSVClient
from finance_client.utils.idcprocess import *

class Test(unittest.TestCase):
    
    def test_property(self):
        ts.LongTrend != ts.ShortTrend
        self.assertEqual(ts.LongTrend, ts.LongTrend)
    
    def test_storategy_base(self):
        client = Client()
        ts.Storategy(client, 1)
        
    def test_parallel_timer(self):
        client = Client()
        storategy1 = ts.Storategy(client, 1)
        storategy2 = ts.Storategy(client, 2)
        storategy3 = ts.Storategy(client, 3)
        sts = [storategy1, storategy2, storategy3]
        manager = ts.ParallelStorategyManager(sts, minutes=5)
        manager.start_storategies()
    
    def test_MACDCross_Registration(self):
        client = Client()
        st1 = ts.storategies.MACDCross(client, 2)
        option = {
            "column": "Close",
            "short_window": 9,
            "long_window": 18,
            "signal_window":9
        }
        macd = MACDpreProcess(key="test_macd", option=option)
        st2 = ts.storategies.MACDCross(client, 3)
        sts = [st1, st2]
        manager = ts.ParallelStorategyManager(sts, minutes=5)
    
if __name__ == '__main__':
    unittest.main()