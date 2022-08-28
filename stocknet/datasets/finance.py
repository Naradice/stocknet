import random
import numpy
import torch

class Dataset():
    """
    common dataset
    """
    
    key = "common"

    def __init__(self, data_client, observationLength:int, in_columns=["Open", "High", "Low", "Close"] ,out_columns=["Open", "High", "Low", "Close"], merge_columns = False, seed = None, isTraining = True):
        self.in_columns = list(in_columns)
        self.out_columns = list(out_columns)
        if merge_columns:
            self.create_column_func = self.__create_column_with_merge
        else:
            self.create_column_func = self.__create_column
        
        self.seed(seed)
        
        self.observationLength = observationLength
        self.isTraining = isTraining
 
        self.data = data_client.get_rate_with_indicaters()
        self.dataLength = observationLength
        self.isTraining = isTraining
        self.init_indicies()
        
    def init_indicies(self):
        length = len(self.data)
        if length < self.observationLength:
            raise Exception(f"date length {length} is less than observationLength {self.observationLength}")
        
        if self.isTraining:
            self.fromIndex = self.dataLength
            self.toIndex = int(length*0.7)
        else:
            self.fromIndex = int(length*0.7)+1
            self.toIndex = length
        
        training_data_length = self.__len__()
        ## length to select random indices.
        k = training_data_length
        ## if allow duplication
        #self.indices = random.choices(range(self.fromIndex, self.toIndex), k=k)
        ## if disallow duplication
        self.indices = random.sample(range(self.fromIndex, self.toIndex), k=k)
        
    ## overwrite
    def outputFunc(self, ndx):
        '''
        ndx: slice type or int
        return ans value array from actual tick data. data format (rate, diff etc) is depends on data initialization. default is diff
        '''
        return self.getInputs(ndx, self.out_columns)
    
    def inputFunc(self, ndx):
        return self.getInputs(ndx, self.in_columns)
    
    def __create_column_with_merge(self, columns, slice):
        temp = numpy.array([])
        for column in columns:
            temp = numpy.append(temp, self.data[column][slice].values.tolist())
        return temp

    def __create_column(self, columns, slice):
        return self.data[columns][slice].values.tolist()
    
    ## common functions
    def getInputs(self, batch_size: slice, columns:list):
        inputs = []
        if type(batch_size) == int:
            batch_indicies = slice(batch_size, batch_size+1)
            out_indicies = 0
        elif type(batch_size) == slice:
            batch_indicies = batch_size
            out_indicies = slice(0,None)
        for index in self.indices[batch_indicies]:
            temp = self.create_column_func(columns, slice(index-self.dataLength, index))
            inputs.append(temp)
        return inputs[out_indicies]
    
    def __len__(self):
        return self.toIndex - self.fromIndex
    
    def getActialIndex(self,ndx):
        inputs = []
        if type(ndx) == slice:
            for index in self.indices[ndx]:
                inputs.append(index)
        else:
            inputs = self.indices[ndx]

        return inputs
    
    def __getitem__(self, ndx):
        inputs = numpy.array(self.inputFunc(ndx), dtype=numpy.dtype('float32'))
        outputs = numpy.array(self.outputFunc(ndx), dtype=numpy.dtype('float32'))
        return torch.tensor(inputs), torch.tensor(outputs)
           
    def seed(self, seed=None):
        '''
        '''
        if seed == None:
            seed = 1192
        else:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        random.seed(seed)
        numpy.random.seed(seed)
        self.seed_value = seed
        
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)
        
    def get_params(self):
        params = {"observationLength": self.observationLength, "in_columns": self.in_columns, "out_columns": self.out_columns, "seed":self.seed_value, "isTraining": self.isTraining }
        additional = self.get_ds_params()
        params.update(additional)
        return params
        
    def render(self, mode='human', close=False):
        '''
        '''
        pass