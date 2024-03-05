import random

import numpy
import torch

from stocknet.datasets.finance import ClientDataset


class LossDataset:
    key = "loss"

    def __init__(self, dataset: ClientDataset, model, loss_fn, seed=None, isTraining=True):
        self.dataset = dataset
        self.isTraining = isTraining
        self.model = model
        self.loss_fn = loss_fn
        self.args = (dataset, model, loss_fn, seed)

    def init_indicies(self):
        length = len(self.dataset)
        if self.isTraining:
            self.fromIndex = self.observationLength
            self.toIndex = int(length * 0.7)
        else:
            self.fromIndex = int(length * 0.7) + 1
            self.toIndex = length

        training_data_length = self.__len__()
        # length to select random indices.
        k = training_data_length
        # if allow duplication
        # self.indices = random.choices(range(self.fromIndex, self.toIndex), k=k)
        # if disallow duplication
        self.indices = random.sample(range(self.fromIndex, self.toIndex), k=k)

    def outputFunc(self, batch_size):
        pass

    def __len__(self):
        return self.toIndex - self.fromIndex

    def getActialIndex(self, ndx):
        inputs = []
        if type(ndx) == slice:
            for index in self.indices[ndx]:
                inputs.append(index)
        else:
            inputs = self.indices[ndx]

        return inputs

    def __getitem__(self, ndx):
        d_in, d_out = self.dataset[ndx]
        m_out = self.model(d_in)
        out = numpy.array([])
        for index in range(0, len(d_out)):
            out.append(self.loss_fn(d_out[index], m_out[index]))
        return d_out, torch.tensor(out)

    def seed(self, seed=None):
        """ """
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

    def render(self, mode="human", close=False):
        """ """
        pass
