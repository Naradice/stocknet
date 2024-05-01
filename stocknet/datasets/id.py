import math
import random

import numpy as np
import pandas as pd
import torch

from .utils import read_csv


class DiffIDDS:
    key = "seq2seq_did"

    def __init__(
        self,
        source,
        columns,
        observation_length=60,
        prediction_length=10,
        device="cuda",
        seed=1017,
        is_training=True,
        batch_first=True,
        clip_range=None,
        with_close_column: str = None,
        with_mean: int = None,
        output_mask: bool = True,
        **kwargs,
    ):
        self.seed(seed)
        self.columns = columns
        if isinstance(source, (pd.DataFrame)):
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
        self.ohlc_idf = self.__init_ohlc(data, columns, clip_range=clip_range, with_close=with_close_column, with_mean=with_mean)
        self.clip_range = clip_range
        self.with_close_column = with_close_column
        self.with_mean = with_mean

        self.output_mask = output_mask
        self.observation_length = observation_length
        self.device = device
        self.prediction_length = prediction_length
        self.is_training = is_training
        self.__init_indicies(self.ohlc_idf)
        self.batch_first = batch_first

    def get_params(self):
        params = {
            "source": self.file_path,
            "columns": self.columns,
            "observation_length": self.observation_length,
            "device": self.device,
            "prediction_length": self.prediction_length,
            "seed": self.seed_value,
            "batch_first": self.batch_first,
            "clip_range": self.clip_range,
            "with_close_column": self.with_close_column,
            "with_mean: int": self.with_mean,
            "utput_mask": self.output_mask,
        }
        return params

    def __init_indicies(self, data, split_ratio=0.8):
        length = len(data) - self.observation_length - self.prediction_length
        if length < 0:
            raise Exception(f"date length {length} is less than observation_length {self.observation_length}")

        to_index = int(length * split_ratio)
        from_index = 0
        train_indices = list(range(from_index, to_index))
        self.train_indices = random.sample(train_indices, k=to_index - from_index)

        from_index = int(length * split_ratio) + self.observation_length + self.prediction_length
        to_index = length
        eval_indices = list(range(from_index, to_index))
        self.eval_indices = random.sample(eval_indices, k=to_index - from_index)

        if self.is_training:
            self._indices = self.train_indices
        else:
            self._indices = self.eval_indices

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

    def revert_diff(self, prediction, ndx, last_values=None):
        pass

    def revert(self, diff):
        pass

    def __init_ohlc(self, df, ohlc_columns, decimal_digits=3, clip_range=None, with_close=False, with_mean=None):
        if with_mean is not None:
            df = df.rolling(window=with_mean).mean().dropna()
        if with_close is not None:
            close_column = [ohlc_columns[3]]
            ohlc_diff_df = df[ohlc_columns].iloc[1:] - df[close_column].iloc[:-1].values
        else:
            ohlc_diff_df = df[ohlc_columns].diff()
        ohlc_diff_df.dropna(inplace=True)
        if clip_range is not None:
            ohlc_diff_df = ohlc_diff_df.clip(lower=clip_range[0], upper=clip_range[1])
        min_value = ohlc_diff_df.min().min()
        min_value_abs = abs(min_value)

        lower_value = math.ceil(min_value_abs) * 10**decimal_digits
        upper_value = math.ceil(ohlc_diff_df.max().max()) * 10**decimal_digits
        id_df = ohlc_diff_df * 10**decimal_digits + lower_value
        self.ohlc_lower = lower_value
        id_df = id_df.astype("int64")
        vocab_size = lower_value + upper_value + 1
        if vocab_size % 2 == 0:
            self.vocab_size = vocab_size
        else:
            self.vocab_size = vocab_size + 1
        return id_df

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, ndx):
        ohlc_chunk_data = []

        for index in self._indices[ndx]:
            idx = slice(index, index + self.observation_length + self.prediction_length)
            ohlc_ids = self.ohlc_idf.iloc[idx].values.tolist()
            ohlc_chunk_data.append(ohlc_ids)

        ohlc_ids = torch.tensor(ohlc_chunk_data, device=self.device, dtype=torch.int64)
        if self.batch_first:
            src = ohlc_ids[:, : -self.prediction_length]
            tgt = ohlc_ids[:, -self.prediction_length - 1 :]
        else:
            ohlc_ids = ohlc_ids.transpose(0, 1)
            src = ohlc_ids[: -self.prediction_length]
            tgt = ohlc_ids[-self.prediction_length - 1 :]
        if self.output_mask:
            mask_tgt = torch.nn.Transformer.generate_square_subsequent_mask(self.prediction_length).to(device=self.device)
            return src, tgt, mask_tgt
        else:
            return src, tgt

    def seed(self, seed=None):
        """ """
        if seed is None:
            seed = 1192
        else:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        self.seed_value = seed

    def eval(self):
        self._indices = self.eval_indices
        self.is_training = False

    def train(self):
        self._indices = self.train_indices
        self.is_training = False
