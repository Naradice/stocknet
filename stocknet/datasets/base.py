# from multiprocessing import Pool
import random
from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import (k_fold_sampling, random_sampling, random_sampling_row,
                    read_csv)


class Dataset(Dataset):
    version = 10

    def __init__(
        self,
        source,
        columns: list,
        observation_length: int = 60,
        device=None,
        processes=None,
        prediction_length=10,
        seed=1017,
        is_training=True,
        randomize=True,
        index_sampler=None,
        split_ratio=0.8,
        indices=None,
        dtype=torch.float,
        batch_first=False,
        output_mask=True,
        volume_limit_ratio=None,
        **kwargs,
    ):
        self.seed(seed)
        self.mm_params = {}
        if volume_limit_ratio is None:
            volume_limit_ratio = 1.0
        self.volume_limit_ratio = volume_limit_ratio
        self.device = device
        self.dtype = dtype
        self.batch_first = batch_first
        min_length = [1]
        if isinstance(source, (pd.DataFrame, pd.Series)):
            data = source
            self.file_path = None
        elif isinstance(source, str):
            data = pd.read_csv(source, parse_dates=True, index_col=0)
            self.file_path = source
        elif isinstance(source, dict):
            data = read_csv(**source)
            self.file_path = source["file_path"]
        else:
            raise TypeError(f"{type(source)} is not supported as source")
        if processes is not None:
            if isinstance(processes, dict):
                from finance_client.fprocess import fprocess

                processes = fprocess.load_preprocess(processes)

            for process in processes:
                data = process(data)
                min_length.append(process.get_minimum_required_length())
        self.processes = processes

        self._min_index = max(min_length) - 1
        if self._min_index < 0:
            self._min_index = 0
        if index_sampler is None:
            self.index_sampler = random_sampling
        elif type(index_sampler) is str and "k" in index_sampler:
            self.index_sampler = k_fold_sampling
        else:
            self.index_sampler = index_sampler

        self.output_mask = output_mask
        self.observation_length = observation_length
        self.is_training = is_training
        self.columns = columns
        self._data = data[columns]
        self._prediction_length = prediction_length
        if indices is None:
            self._init_indicies(data.index, randomize, split_ratio=split_ratio)
        else:
            self._init_indicies_row(indices, randomize, split_ratio=split_ratio)

    def _apply_volume_limit(self, indices):
        limited_length = int(len(indices) * self.volume_limit_ratio)
        return indices[:limited_length]

    def update_volume_limit(self, volume_limit_ratio=None):
        if volume_limit_ratio is not None and volume_limit_ratio <= 1.0:
            self.volume_limit_ratio = volume_limit_ratio
        if self.is_training:
            self._indices = self._apply_volume_limit(self.train_indices)
        else:
            self._indices = self._apply_volume_limit(self.eval_indices)

    def _init_indicies(self, index, randomize=False, split_ratio=0.8):
        length = len(index) - self.observation_length - self._prediction_length
        if length <= 0:
            raise Exception(f"date length {length} is less than observation_length {self.observation_length}")

        self.train_indices, self.eval_indices = self.index_sampler(
            index, self._min_index, randomize, split_ratio, self.observation_length, self._prediction_length
        )
        self.update_volume_limit()

    def _init_indicies_row(self, index, randomize=False, split_ratio=0.8):
        length = len(index) - self.observation_length - self._prediction_length
        if length <= 0:
            raise Exception(f"date length {length} is less than observation_length {self.observation_length}")

        self.train_indices, self.eval_indices = random_sampling_row(
            index, self._min_index, randomize, split_ratio, self.observation_length, self._prediction_length
        )

        if self.is_training:
            self._indices = self.train_indices
        else:
            self._indices = self.eval_indices

    def get_params(self):
        from finance_client import fprocess

        if self.processes is None:
            process_params = None
        else:
            process_params = fprocess.preprocess_to_params(self.processes)
        params = {
            "source": self.file_path,
            "columns": self.columns,
            "observation_length": self.observation_length,
            "prediction_length": self._prediction_length,
            "processes": process_params,
            "seed": self.seed_value,
            "batch_first": self.batch_first,
            "output_mask": self.output_mask,
        }
        return params

    def output_indices(self, index):
        return slice(index + self.observation_length, index + self.observation_length + self._prediction_length)

    def _output_func(self, batch_size):
        if type(batch_size) == int:
            index = self._indices[batch_size]
            ndx = self.output_indices(index)
            ans = self._data.iloc[ndx].values.tolist()
            ans = torch.tensor(ans, device=self.device, dtype=self.dtype)
            return ans
        elif type(batch_size) == slice:
            batch_indices = batch_size
            chunk_data = []
            for index in self._indices[batch_indices]:
                ndx = self.output_indices(index)
                chunk_data.append(self._data.iloc[ndx].values.tolist())

            ans = torch.tensor(chunk_data, device=self.device, dtype=self.dtype)
            if self.batch_first:
                return ans
            else:
                return ans.transpose(0, 1)

    def input_indices(self, index):
        return slice(index, index + self.observation_length)

    def _input_func(self, batch_size):
        if type(batch_size) == int:
            index = self._indices[batch_size]
            ndx = self.input_indices(index)
            src = self._data[ndx].values.tolist()
            src = torch.tensor(src, device=self.device, dtype=self.dtype)
            return src
        elif type(batch_size) == slice:
            batch_indices = batch_size
            chunk_src = []
            for index in self._indices[batch_indices]:
                ndx = self.input_indices(index)
                chunk_src.append(self._data.iloc[ndx].values.tolist())

            src = torch.tensor(chunk_src, device=self.device, dtype=self.dtype)
            if self.batch_first:
                return src
            else:
                return src.transpose(0, 1)

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, ndx):
        src = self._input_func(ndx)
        tgt = self._output_func(ndx)
        if self.output_mask:
            mask_tgt = torch.nn.Transformer.generate_square_subsequent_mask(self._prediction_length).to(device=self.device)
            return src, tgt, mask_tgt
        else:
            return src, tgt

    def seed(self, seed=None):
        """ """
        if seed is None:
            seed = 1017
        else:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        self.seed_value = seed

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def eval(self):
        indices = self._apply_volume_limit(self.eval_indices)
        self._indices = random.sample(indices, k=len(indices))
        self.is_training = False

    def train(self):
        indices = self._apply_volume_limit(self.train_indices)
        self._indices = random.sample(indices, k=len(indices))
        self.is_training = True

    def get_index_range(self):
        return min(self._indices), max(self._indices)

    def get_date_range(self):
        min_index, max_index = self.get_index_range()
        return self._data.index[min_index], self._data.index[max_index]

    def get_actual_index(self, ndx):
        inputs = []
        if type(ndx) == slice:
            inputs = self._indices[ndx]
        elif isinstance(ndx, Iterable):
            for index in ndx:
                inputs.append(self._indices[index])
        else:
            return self._indices[ndx]

        return inputs

    def get_row_data(self, ndx):
        inputs = []
        if type(ndx) == slice:
            for index in self._indices[ndx]:
                df = self._data[index : index + self.observation_length]
                inputs.append(df)
        else:
            index = ndx
            inputs = df = self._data[index : index + self.observation_length]
        return inputs


class TimeDataset(Dataset):
    def __init__(
        self,
        df,
        columns: list,
        processes,
        time_column="index",
        observation_length: int = 60,
        device="cuda",
        prediction_length=10,
        seed=1017,
        is_training=True,
        randomize=True,
        index_sampler=None,
        indices=None,
        dtype=torch.float,
        batch_first=False,
        split_ratio=0.8,
        volume_limit_ratio=None,
        **kwargs,
    ):
        """return time data in addition to the columns data
        ((observation_length, CHUNK_SIZE, NUM_FEATURES), ((prediction_length, CHUNK_SIZE, 1)) as (feature_data, time_data) for source and target
        Args:
            df (pd.DataFrame): _description_
            columns (list): target columns like ["open", "high", "low", "close", "volume"]
            processes (list): list of process to add indicater and/or run standalization
            time_column (str, optional): specify column name or index. Defaults to "index"
            observation_length (int, optional): specify observation_length for source data. Defaults to 60.
            device (str, optional): Defaults to "cuda".
            prediction_length (int, optional): specify prediction_length for target data. Defaults to 10.
            seed (int, optional): specify random seed. Defaults to 1017.
            is_training (bool, optional): specify training mode or not. Defaults to True.
            randomize (bool, optional): specify randomize the index or not. Defaults to True.
            split_ratio (float, optional): specify ratio to split training data and validation data
            volume_limit_ratio (float, optional): to investigate how data volume affect to model_performance
        """

        self.time_column = time_column
        self._feature_columns = columns.copy()
        entire_columns = [*columns, time_column]

        super().__init__(
            df,
            entire_columns,
            observation_length,
            device,
            processes,
            prediction_length,
            seed,
            is_training,
            randomize,
            index_sampler,
            indices=indices,
            split_ratio=split_ratio,
            dtype=dtype,
            batch_first=batch_first,
            **kwargs,
        )

    def _output_func(self, batch_size):
        batch_indices = batch_size
        chunk_data = []
        time_chunk_data = []
        for index in self._indices[batch_indices]:
            ndx = self.output_indices(index)
            chunk_data.append(self._data[self._feature_columns].iloc[ndx].values.tolist())
            time_chunk_data.append(self._data[self.time_column].iloc[ndx].values.tolist())

        ans = torch.tensor(chunk_data, device=self.device, dtype=self.dtype)
        time = torch.tensor(time_chunk_data, device=self.device, dtype=torch.int)
        if self.batch_first:
            return (ans, time)
        else:
            return (ans.transpose(0, 1), time.transpose(0, 1))

    def _input_func(self, batch_size):
        batch_indices = batch_size
        chunk_src = []
        time_chunk_data = []
        for index in self._indices[batch_indices]:
            ndx = self.input_indices(index)
            chunk_src.append(self._data[self._feature_columns].iloc[ndx].values.tolist())
            time_chunk_data.append(self._data[self.time_column].iloc[ndx].values.tolist())

        src = torch.tensor(chunk_src, device=self.device, dtype=self.dtype)
        time = torch.tensor(time_chunk_data, device=self.device, dtype=torch.int)

        if self.batch_first:
            return (src, time)
        else:
            return (src.transpose(0, 1), time.transpose(0, 1))

    def __getitem__(self, ndx):
        src, src_time = self._input_func(ndx)
        tgt, tgt_time = self._output_func(ndx)
        if self.output_mask:
            mask_tgt = torch.nn.Transformer.generate_square_subsequent_mask(self._prediction_length).to(device=self.device)
            return src, tgt, src_time, tgt_time, mask_tgt
        else:
            return src, tgt, src_time, tgt_time

    def get_params(self):
        params = super().get_params()
        params.update({"columns": self._feature_columns, "time_column": self.time_column})
        return params
