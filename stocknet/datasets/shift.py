import torch

from stocknet.datasets.finance import ClientDataset


class ShiftDataset(ClientDataset):
    key = "shit"

    def __init__(
        self,
        data_client,
        observationLength=1000,
        idc_processes=[],
        pre_processes=[],
        in_columns=["Open", "High", "Low", "Close"],
        out_columns=["Open", "High", "Low", "Close"],
        shift=1,
        merge_input_columns=False,
        seed=None,
        isTraining=True,
    ):
        self.shift = shift
        super().__init__(
            data_client,
            observationLength,
            idc_processes,
            pre_processes,
            in_columns=in_columns,
            out_columns=out_columns,
            merge_columns=merge_input_columns,
            seed=seed,
            isTraining=isTraining,
        )
        self.args = (data_client, observationLength, idc_processes, pre_processes, in_columns, out_columns, shift, merge_input_columns, seed)

    def getNextInputs(self, ndx):
        if type(ndx) == int:
            batch_indicies = slice(ndx, ndx + 1)
        elif type(ndx) == slice:
            batch_indicies = ndx
        chunk_data = []
        symbols = self.data_client.symbols
        for index in self.indices[batch_indicies]:
            data = self.data_client.get_train_data(index + self.shift, 1, self.out_columns, symbols, self.idc_processes, self.pre_processes)
            chunk_data.append(data.values.tolist())
        return torch.tensor(chunk_data).reshape((len(chunk_data) * len(symbols) * len(self.out_columns) * 1))

    def outputFunc(self, ndx):
        return self.getNextInputs(ndx)

    def __getitem__(self, ndx):
        return super().__getitem__(ndx)
