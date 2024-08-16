import random

import numpy as np
import pandas as pd
import torch
from finance_client.fprocess import fprocess

from stocknet.datasets.base import Dataset


def k_means(src_df, label_num_k, initial_centers=None, max_iter=10000):
    np.random.seed(100)
    random.seed(100)

    count = 0

    labels = np.fromiter(random.choices(range(label_num_k), k=src_df.shape[0]), dtype=int)
    labels_prev = np.zeros(src_df.shape[0])
    if initial_centers is None:
        cluster_centers = np.eye(label_num_k, src_df.shape[1])
    else:
        initial_centers = np.array(initial_centers)
        if initial_centers.shape == (label_num_k, src_df.shape[1]):
            cluster_centers = initial_centers
        else:
            raise ValueError("invalid initial centeers")

    while not (labels == labels_prev).all():
        for i in range(label_num_k):
            clusters = src_df.iloc[labels == i]
            if len(clusters) > 0:
                cluster_centers[i, :] = clusters.mean(axis=0)
            else:
                cluster_centers[i, :] = np.ones(src_df.shape[1])
        dist = ((src_df.values[:, :, np.newaxis] - cluster_centers.T[np.newaxis, :, :]) ** 2).sum(axis=1)
        # dist = np.sqrt(dist)
        labels_prev = labels
        labels = dist.argmin(axis=1)
        count += 1
        if count > max_iter:
            break
    return labels, cluster_centers


# Freedman–Diaconis rule. Sometimes 0 count appeare due to outfliers.
def freedamn_diaconis_bins(data):
    q75, q25 = np.percentile(data, [75, 25])

    iqr = q75 - q25
    n = len(data)
    bin_width = 2.0 * iqr / (n ** (1 / 3))
    return bin_width


def prob_mass(data, bin_width=None):
    if bin_width is None:
        counts, bin_edges = np.histogram(data)
    else:
        try:
            bins = np.arange(min(data), max(data) + bin_width, bin_width)
            counts, bin_edges = np.histogram(data, bins=bins)
        except ValueError:
            counts, bin_edges = np.histogram(data)
    mass = counts / counts.sum()
    return mass, bin_edges


class ClusterDistDataset(Dataset):
    def __init__(
        self,
        src,
        columns: list,
        label_num_k: int = 30,
        freq=30,
        observation_length: int = 60,
        device="cuda",
        prediction_length=10,
        seed=1017,
        is_training=True,
        randomize=True,
        index_sampler=None,
        split_ratio=0.8,
        indices=None,
        batch_first=True,
        **kwargs
    ):
        df = src
        diff_p = fprocess.DiffPreProcess(columns=columns)
        src_df = df[columns].dropna()
        src_df = diff_p(src_df).dropna()
        processes = [fprocess.WeeklyIDProcess(freq=freq, time_column="index")]

        divisions = [i / (label_num_k - 1) for i in range(label_num_k)]
        ini_centers = [np.quantile(src_df, p, axis=0) for p in divisions]
        labels, centers = k_means(src_df, label_num_k=label_num_k, initial_centers=ini_centers)
        self.centers = centers
        dist = ((src_df.values[:, :, np.newaxis] - centers.T[np.newaxis, :, :]) ** 2).sum(axis=1)
        token_df = pd.DataFrame(dist, index=src_df.index)
        super().__init__(
            token_df,
            columns=token_df.columns,
            observation_length=observation_length,
            processes=processes,
            device=device,
            prediction_length=prediction_length,
            seed=seed,
            is_training=is_training,
            randomize=randomize,
            index_sampler=index_sampler,
            split_ratio=split_ratio,
            indices=indices,
            dtype=torch.float,
            batch_first=batch_first,
        )

    def to_labels(self, observations):
        if isinstance(observations, pd.DataFrame):
            observations = observations.values
        dist = ((observations[:, :, np.newaxis] - self.centers.T[np.newaxis, :, :]) ** 2).sum(axis=1)
        labels = dist.argmin(axis=1)
        return labels

    def output_indices(self, index):
        return slice(index + self.observation_length - 1, index + self.observation_length + self._prediction_length)

    def __getitem__(self, ndx):
        src, tgt, options = super().__getitem__(ndx)
        src = src.squeeze()
        tgt = tgt.squeeze()
        return src, tgt, options


class ClusterIDDataset(Dataset):
    def __init__(
        self,
        src,
        columns: list,
        label_num_k: int = 30,
        freq=30,
        observation_length: int = 60,
        device="cuda",
        prediction_length=10,
        seed=1017,
        is_training=True,
        randomize=True,
        index_sampler=None,
        split_ratio=0.8,
        indices=None,
        batch_first=True,
        **kwargs
    ):
        df = src
        diff_p = fprocess.DiffPreProcess(columns=columns)
        src_df = df[columns].dropna()
        src_df = diff_p(src_df).dropna()
        processes = [fprocess.WeeklyIDProcess(freq=freq, time_column="index")]
        # parameter for a trainer
        self.vocab_size = label_num_k

        divisions = [i / (label_num_k - 1) for i in range(label_num_k)]
        ini_centers = [np.quantile(src_df, p, axis=0) for p in divisions]
        labels, centers = k_means(src_df, label_num_k=label_num_k, initial_centers=ini_centers)
        self.centers = centers
        token_df = pd.DataFrame(labels, index=src_df.index, dtype=int)
        super().__init__(
            token_df,
            columns=token_df.columns,
            observation_length=observation_length,
            processes=processes,
            device=device,
            prediction_length=prediction_length,
            seed=seed,
            is_training=is_training,
            randomize=randomize,
            index_sampler=index_sampler,
            split_ratio=split_ratio,
            indices=indices,
            dtype=torch.float,
            batch_first=batch_first,
        )

    def output_indices(self, index):
        return slice(index + self.observation_length - 1, index + self.observation_length + self._prediction_length)

    def __getitem__(self, ndx):
        src, tgt, options = super().__getitem__(ndx)
        src = src.to(dtype=int)
        tgt = tgt.to(dtype=int)
        src = src.squeeze()
        tgt = tgt.squeeze()
        return src, tgt, options
