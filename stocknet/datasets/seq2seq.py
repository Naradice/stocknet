from .base import Dataset, TimeDataset


class FeatureDataset(Dataset):
    """Dataset to return (DataLength, BATCH_SIZE, FEATURE_SIZE), here FEATURE_SIZE means column size specified
    dataset[index: index+batch_size] returns (src, tgt)
    Note that using Dataloader is a bit slower than directly using in my env.
    """

    key = "seq2seq"

    def __init__(
        self,
        df,
        columns: list,
        observation_length: int = 60,
        device=None,
        processes=None,
        prediction_length=10,
        seed=1017,
        is_training=True,
        randomize=True,
        index_sampler=None,
        indices=None,
        batch_first=True,
        **kwargs
    ):
        super().__init__(
            df,
            columns,
            observation_length,
            device,
            processes,
            prediction_length,
            seed,
            is_training,
            randomize,
            index_sampler,
            indices=indices,
            batch_first=batch_first,
            **kwargs,
        )

    def output_indices(self, index):
        # output overrap with last input
        return slice(index + self.observation_length - 1, index + self.observation_length + self._prediction_length)


class TimeFeatureDataset(TimeDataset):
    """Dataset to return (DataLength, BATCH_SIZE, FEATURE_SIZE), here FEATURE_SIZE means column size specified
    dataset[index: index+batch_size] returns (src, tgt)
    Note that using Dataloader is a bit slower than directly using in my env.
    """

    key = "seq2seq_time"

    def __init__(
        self,
        df,
        columns: list,
        time_column: str,
        observation_length: int = 60,
        device=None,
        processes=None,
        prediction_length=10,
        seed=1017,
        is_training=True,
        randomize=True,
        indices=None,
        batch_first=True,
        **kwargs
    ):
        super().__init__(
            df,
            columns,
            time_column=time_column,
            processes=processes,
            observation_length=observation_length,
            device=device,
            prediction_length=prediction_length,
            seed=seed,
            is_training=is_training,
            randomize=randomize,
            index_sampler=None,
            indices=indices,
            batch_first=batch_first,
            **kwargs,
        )

        if batch_first:
            self.__fit_time_size = lambda tgt_time: tgt_time[:, :-1]
        else:
            self.__fit_time_size = lambda tgt_time: tgt_time[:-1]

    def _output_func(self, batch_size):
        tgt, tgt_time = super()._output_func(batch_size)
        if isinstance(batch_size, slice):
            tgt_time = self.__fit_time_size(tgt_time)
            return tgt, tgt_time
        else:
            return tgt, tgt_time

    def output_indices(self, index):
        return slice(index + self.observation_length - 1, index + self.observation_length + self._prediction_length)
