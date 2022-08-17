import torch
import random
import numpy

class DatasetBase():
    """
    common dataset
    """
    
    key = "base"

    def __init__(self, observationLength:int, in_columns:list ,out_columns:list, seed = None, isTraining = True):
        self.seed(seed)
        self.columns = list(in_columns)
        self.out_columns = list(out_columns)
        
        self.observationLength = observationLength
        self.isTraining = isTraining
        
    def __getitem__(self, ndx):
        inputs = torch.tensor(self.inputFunc(ndx, self.columns))
        outputs = torch.tensor(self.outputFunc(ndx, self.out_columns))
        return inputs, outputs
    
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
        params = {"observationLength": self.observationLength, "in_columns": self.columns, "out_columns": self.out_columns, "seed":self.seed_value, "isTraining": self.isTraining }
        additional = self.get_ds_params()
        params.update(additional)
        return params

    """
    overwrite below methods
    """
    
    def get_ds_params(self):
        return {}
    
    def outputFunc(self, ndx, column):
        return []
    
    def inputFunc(self, ndx, column):
        return []
        
    def __len__(self):
        return 0
        
    def render(self, mode='human', close=False):
        '''
        '''
        pass