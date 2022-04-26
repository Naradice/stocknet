import unittest

from matplotlib.pyplot import sca
import stocknet.envs.utils.standalization as std
import numpy
import pandas as pd

class TestStandalizationUtils(unittest.TestCase):

    def __init__(self, methodName: str = ...) -> None:
        self.window = 4
        self.input = [1,2,3,4,5,6,7,8,9,10,20]
        self.scale = (-1, 1)
        super().__init__(methodName)
            
    def test_minimax_array(self):
        input = self.input
        mm_inputs, _max, _min = std.mini_max_from_array(input, self.scale)
        self.assertEqual(self.scale[0], mm_inputs[0])
        self.assertEqual(self.scale[1], mm_inputs[-1])
        
    def test_minimax_vealue(self):
        input = self.input
        _min = min(input)
        _max = max(input)
        
        scaled = std.mini_max(input[0], _min, _max, self.scale)
        self.assertEqual(scaled, self.scale[0])
        scaled = std.mini_max(input[-1], _min, _max)
        self.assertEqual(scaled, self.scale[-1])
        
    
    def test_revert_minimax(self):
        input = self.input
        _min = min(input)
        _max = max(input)
        
        scaled = std.mini_max(input[0], _min, _max, self.scale)
        row = std.revert_mini_max(scaled, _min, _max, self.scale)
        self.assertEqual(row, input[0])
        
        scaled = std.mini_max(input[-3], _min, _max, self.scale)
        row = std.revert_mini_max(scaled, _min, _max, self.scale)
        self.assertEqual(row, input[-3])
        
    def test_mini_max_series(self):
        input = pd.Series(self.input)
        mm_inputs, _max, _min = std.mini_max_from_series(input, self.scale)
        self.assertEqual(self.scale[0], mm_inputs.iloc[0])
        self.assertEqual(self.scale[1], mm_inputs.iloc[-1])
    
    def test_revert_mini_max_series(self):
        input = pd.Series(self.input)
        mm_inputs, _max, _min = std.mini_max_from_series(input, self.scale)
        rows = std.revert_mini_max_from_series(mm_inputs, _min, _max, self.scale)
        self.assertEqual(rows.iloc[0], input.iloc[0])
        self.assertEqual(rows.iloc[-3], input.iloc[-3])
        
        
if __name__ == '__main__':
    unittest.main()