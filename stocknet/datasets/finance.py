import random
import numpy
import torch
import finance_client as fc

class Dataset():
    """
    common dataset
    """
    
    key = "common"

    def __init__(self, data_client: fc.CSVClient, observationLength:int, idc_processes=[], pre_processes=[], in_columns=["Open", "High", "Low", "Close"] ,out_columns=["Open", "High", "Low", "Close"], merge_columns = False, seed = None, isTraining = True):
        self.in_columns = list(in_columns)
        self.out_columns = list(out_columns)
        if merge_columns:
            self.__get_data_func = self.__create_merge
        else:
            self.__get_data_func = self.__create_wo_merge
        
        self.seed(seed)
        
        self.observationLength = observationLength
        self.isTraining = isTraining
        self.data_client = data_client
        self.isTraining = isTraining
        self.idc_processes = idc_processes
        self.pre_processes = pre_processes
        self.args = (data_client, observationLength, in_columns ,out_columns, merge_columns, seed)
        self.init_indicies()
        
    def init_indicies(self):
        length = len(self.data_client)
        if length < self.observationLength:
            raise Exception(f"date length {length} is less than observationLength {self.observationLength}")
        
        if self.isTraining:
            #TODO: need to caliculate required length
            self.fromIndex = self.observationLength + 100
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
    def outputFunc(self, batch_size):
        '''
        ndx: slice type or int
        return ans value array from actual tick data. data format (rate, diff etc) is depends on data initialization. default is diff
        '''
        if type(batch_size) == int:
            batch_indicies = slice(batch_size, batch_size+1)
        elif type(batch_size) == slice:
            batch_indicies = batch_size
        
        chunk_data = []
        for index in self.indices[batch_indicies]:
            data = self.data_client.get_train_data(index, 1, self.out_columns, self.data_client.symbols, self.idc_processes, self.pre_processes)
            chunk_data.append(data.values.tolist())
        return self.__create_merge(batch_indicies, 1, self.out_columns, self.data_client.symbols)        
    
    def inputFunc(self, ndx):
        return self.getInputs(ndx, self.in_columns, self.observationLength)
    
    def __create_merge(self, ndx, length, columns, symbols):
        chunk_data = []
        for index in self.indices[ndx]:
            data = self.data_client.get_train_data(index, length, columns, symbols, self.idc_processes, self.pre_processes)
            chunk_data.append(data.values.tolist())
        return torch.tensor(chunk_data).reshape((len(chunk_data) * len(symbols)* len(columns) * length))

    def __create_wo_merge(self, ndx, length, columns, symbols):
        chunk_data = []
        for index in self.indices[ndx]:
            data = self.data_client.get_train_data(index, length, columns, symbols, self.idc_processes, self.pre_processes)
            chunk_data.append(data.values.tolist())
        return torch.tensor(chunk_data).reshape((len(chunk_data)*len(symbols)*len(columns), length))
    
    ## common functions
    def getInputs(self, batch_size: slice, columns:list, length:int):
        if type(batch_size) == int:
            batch_indicies = slice(batch_size, batch_size+1)
        elif type(batch_size) == slice:
            batch_indicies = batch_size
        
        return self.__get_data_func(batch_indicies, length, columns, self.data_client.symbols)
    
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
        return self.inputFunc(ndx), self.outputFunc(ndx)
           
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
        
    def render(self, mode='human', close=False):
        '''
        '''
        pass