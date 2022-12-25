import random
import numpy
import torch
from stocknet.datasets.finance import Dataset

class RewardDataset(Dataset):
    
    key = "reward_ohlc"
    
    def __init__(self, data_client, observationDays=1, in_column = ["Open", "High", "Low", "Close"], out_column = "Close" , seed = None, isTraining=True):
        """ * Not yet implemented.
            Output how much you can obtain a profit

        Args:
            data_client (_type_): _description_
            observationDays (int, optional): _description_. Defaults to 1.
            in_column (list, optional): _description_. Defaults to ["Open", "High", "Low", "Close"].
            column (str, optional): _description_. Defaults to "Close".
            seed (_type_, optional): _description_. Defaults to None.
            isTraining (bool, optional): _description_. Defaults to True.
        """
        super().__init__(data_client, observationDays, out_columns=[], seed=seed, isTraining=isTraining)
        self.args = (data_client, observationDays, in_column, out_column, seed)
        self.column = out_column
    
    def init_indicies(self):
        length = len(self.data)
        if self.isTraining:
            self.fromIndex = self.observationLength
            self.toIndex = int(length*0.7)
        else:
            self.fromIndex = int(length*0.7)+1
            self.toIndex = length
        
        ##select random indices.
        k=length - self.observationLength*2 -1
        self.indices = random.choices(range(self.fromIndex, self.toIndex), k=k)
        
    def inputFunc(self, ndx):
        return self.getInputs(ndx, self.in_columns)
    
    def getInputs(self, ndx, columns):
        inputs = []
        if type(ndx) == int:
            indicies = slice(ndx, ndx+1)
            for index in self.indices[indicies]:
                temp = (self.data[columns].iloc[index-self.observationLength:index].values.tolist())
                inputs.append(temp)
            return inputs[0]
        elif type(ndx) == slice:
            indicies = ndx
            for index in self.indices[indicies]:
                temp = (self.data[columns].iloc[index-self.observationLength:index].values.tolist())
                inputs.append(temp)
            return inputs
    
    def caliculate_reward(self, index):
        window = 10
        values = self.data[self.column].iloc[index-window: index + 100]
        ewa = values.ewm(span=window, adjust=True).mean()
        sign = numpy.sign(ewa.diff())
        count = 0
        direction = sign.iloc[1]##change 1
        cp = [numpy.NaN]
        trend_count = 3
        for index in range(1, len(sign)):
            s = sign.iloc[index]
            if s != direction:
                count += 1
                if count == trend_count:
                    direction = s
                    count = 0
                    print(f"direction changed on {index}")
                    cp.append(numpy.NaN)
                    cp[-trend_count] = ewa.iloc[index - trend_count]
                else:
                    cp.append(numpy.NaN)
            else:
                count = 0
                cp.append(numpy.NaN)
        
    
    def getReward(self, ndx):
        """
        each index have 
        [
            reward if you bought coin
            reward if you sell coin
        ]
        """
        inputs = []
        if type(ndx) == int:
            indicies = slice(ndx, ndx+1)
            for index in self.indices[indicies]:
                temp = self.data[self.out_columns].iloc[index-1]
                inputs.append(temp)
            return inputs[0]
        elif type(ndx) == slice:
            indicies = ndx
            for index in self.indices[indicies]:
                temp = self.data[self.out_columns].iloc[index-1]
                inputs.append(temp)
            return inputs
     
    def outputFunc(self, ndx):
        return self.getReward(ndx)
    
    def __getitem__(self, ndx):
        return super().__getitem__(ndx)