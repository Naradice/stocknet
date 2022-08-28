import unittest, os, json, sys, datetime
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(module_path)

from stocknet.envs.render.graph import Rendere
import time
import pandas as pd

class TestRenderClient(unittest.TestCase):
    
    def test_add_subplot(self):
        print("test")
        
    def test_plot(self):
        r = Rendere()
        r.plot()
        
    def test_subplot(self):
        r = Rendere()
        r.add_subplot()
        r.plot()
        time.sleep(5)
        
    def test_subplots(self):
        r = Rendere()
        for i in range(9):
            r.add_subplot()
        r.plot()
        time.sleep(5)
        
    def test_plot_ohlc(self):
        r = Rendere()
        for i in range(5):
            r.add_subplot()
        data = {
            'Open': [10000,10000,10000,10000],
            'High': [11000,12000,13000,14000],
            'Low': [7000, 7000, 7000,7000],
            'Close': [9000, 9000, 9000, 9000]
        }
        df = pd.DataFrame(data)
        r.register_ohlc(df,4)
        r.plot()
        time.sleep(5)
        
if __name__ == '__main__':
    unittest.main()