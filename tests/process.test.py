import unittest

from torch import rand
import stocknet.envs.utils.preprocess as ps
import numpy
import pandas as pd

class TestProcess(unittest.TestCase):
    
    def test_diff(self):
        process = ps.DiffPreProcess()
        ds = pd.DataFrame({'input': [10, 20, 1], 'expect': [numpy.NaN, 10, -19]})
        diff_dict = process.run(ds)
        for index in range(1,len(ds)):
            self.assertEqual(diff_dict['input'].iloc[index], ds['expect'].iloc[index])
            
        new_data = pd.Series({'input': 10, 'expect': 9})
        standalized_new_data = process.update(new_data)
        standalized_ds = process.concat(ds, standalized_new_data)
        self.assertEqual(len(standalized_ds), 4)
        self.assertEqual(standalized_ds['input'].iloc[3], new_data['expect'])
    
    def test_macd(self):
        ##input
        ds = pd.DataFrame({'close':[120.000 + i*0.1 for i in range(30)]})
        ##ans
        short_ema = ds['close'].ewm(span=12, adjust=False).mean()
        long_ema = ds['close'].ewm(span=26, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.rolling(9).mean()
        
        process = ps.MACDpreProcess(option={'column':'close'})
        macd_dict = process.run(ds)
        self.assertEqual(macd_dict['ShortEMA'].iloc[-1], short_ema.iloc[-1])
        self.assertEqual(macd_dict['LongEMA'].iloc[-1], long_ema.iloc[-1])
        self.assertEqual(macd_dict['MACD'].iloc[-1], macd.iloc[-1])
        self.assertEqual(macd_dict['Signal'].iloc[-1], signal.iloc[-1])
    
    def test_macd_update(self):
        ### prerequisites
        ds = pd.DataFrame({'close':[120.000 + i*0.1 for i in range(30)]})
        process = ps.MACDpreProcess(option={'column':'close'})
        macd_dict = process.run(ds)
        
        ###input
        new_value = 120.000 + 30*0.1
        new_data = pd.Series({'close': new_value})
        
        ###output
        new_indicater_series = process.update(new_data)
        test_data = process.concat(ds, new_indicater_series)
        
        ###expect
        ex_data_org = process.concat(ds, new_data)
        another_ps = ps.MACDpreProcess(option={'column':'close'})
        macd_dict = another_ps.run(ex_data_org)
        
        self.assertEqual(macd_dict['ShortEMA'].iloc[-1], test_data['ShortEMA'].iloc[-1])
        self.assertEqual(macd_dict['LongEMA'].iloc[-1], test_data['LongEMA'].iloc[-1])
        self.assertEqual(macd_dict['MACD'].iloc[-1], test_data['MACD'].iloc[-1])
        self.assertEqual(macd_dict['Signal'].iloc[-1], test_data['Signal'].iloc[-1])
    
    def test_min_max(self):
        import random
        open = [random.random()*123 for index in range(100)]
        close = [o_value + random.random() -0.5 for o_value in open]
        ds = pd.DataFrame({'close':close, 'open': open} )
        mm = ps.MinMaxPreProcess(scale=(-1,1))
        
        result = mm.run(ds)
        self.assertTrue(len(result['close']) == 100)
        self.assertTrue(result['close'].min() >= -1)
        self.assertTrue(result['close'].max() <= 1)
        self.assertTrue(result['open'].min() >= -1)
        self.assertTrue(result['open'].max() <= 1)
        
        new_open_value = ds['open'].max() + random.random() + 0.1
        new_close_value =  new_open_value + random.random() - 0.5
        
        new_data = pd.Series({'close':new_close_value, 'open': new_open_value} )
        new_data_standalized = mm.update(new_data)
        new_ds = mm.concat(pd.DataFrame(result), new_data_standalized)
        self.assertTrue(new_data_standalized['open'] == 1)
        self.assertTrue(len(new_ds['close']) == 101)
        self.assertTrue(new_ds['close'].min() >= -1)
        self.assertTrue(new_ds['close'].max() <= 1)
        self.assertTrue(new_ds['open'].min() >= -1)
        self.assertTrue(new_ds['open'].max() <= 1)
    
if __name__ == '__main__':
    unittest.main()