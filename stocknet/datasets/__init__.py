from stocknet.datasets.shift import ShiftDataset
from stocknet.datasets.finance import Dataset, MultiFrameDataset
from stocknet.datasets.highlow import HighLowDataset

available_dataset = {
    Dataset.key: Dataset,
    MultiFrameDataset.key: MultiFrameDataset,
    ShiftDataset.key: ShiftDataset,
    HighLowDataset.key: HighLowDataset
}

def dataset_to_params(ds):
    from finance_client.client_base import Client
    params = {}
    c_params = Client.client_to_params(ds.data_client)
    params["client"] = c_params
    params["args"] = ds.args
    params["kinds"] = ds.key
    return params

def load_dataset(params:dict):
    kinds = params["kinds"]
    if kinds != available_dataset:
        from finance_client.client_base import Client
        c_params = params["client"]
        c_kinds = c_params["kinds"]
        c_args = c_params["args"]
        data_client = Client.load_client(c_kinds, c_args)
        
        Dataset = available_dataset[kinds]
        args = params["args"]
        ds = Dataset(data_client, *args)
        return ds
    return None