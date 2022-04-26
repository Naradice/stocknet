import unittest
import stocknet.envs.utils.indicaters as indicaters
import numpy
import pandas as pd

class TestIndicaterUtils(unittest.TestCase):

    #def test_ema_error(self):
    #    indicaters.EMA([],12)

    #caliculate expected EMA  
    def __init__(self, methodName: str = ...) -> None:
        self.window = 4
        self.input = [1,2,3,4,5,6,7,8,9,10,11,12,13,20]
        self.out_ex = [self.input[0]]
        alpha = 2/(1 + self.window)
        
        for i in range(1, len(self.input)):
            self.out_ex.append(self.out_ex[i -1] * (1 - alpha) + self.input[i]*alpha)

        super().__init__(methodName)
    
    def test_ema(self):
        out = indicaters.EMA(self.input, self.window)
        self.assertListEqual(self.out_ex, out)
        
    def test_ema_series(self):
        input = pd.Series(self.input)
        out = indicaters.EMA(input, self.window)
        success = False
        self.assertEqual(len(out), len(self.out_ex))
        for i in range(0, len(self.out_ex)):
            if out[i] == self.out_ex[i]:
                success = True
            else:
                print(f'Failed on {i}: {out[i]} isnt {self.out_ex[i]}')
                success = False
        self.assertTrue(success)
        
    def test_sma(self):
        window = 4
        sample_array = [1,2,3,4,5,6,7,8,9,10,11,12,13,20]
        result_array = indicaters.SMA (sample_array,window)
        self.assertEqual(len(sample_array), len(result_array))
        self.assertEqual(result_array[-1], (11+12+13+20)/window)
    
    def test_bolinger(self):
        #print(indicaters.bolinger_from_ohlc([1,2,3,4,5,6,7,8,9,10,11,12,13,20],window=5))
        pass
        
    def test_revert_ema(self):
        ema = indicaters.EMA(self.input, self.window)
        suc, r_ema = indicaters.revert_EMA(ema, self.window)
        self.assertTrue(suc)
        for i in range(0, len(self.input)):
            self.assertAlmostEqual(self.input[i], r_ema[i])
            
    def test_revert_ema_series(self):
        input = pd.Series(self.input)
        ema = indicaters.EMA(input, self.window)
        suc, r_ema = indicaters.revert_EMA(ema, self.window)
        self.assertTrue(suc)
        for i in range(0, len(self.input)):
            self.assertAlmostEqual(self.input[i], r_ema[i])

if __name__ == '__main__':
    unittest.main()