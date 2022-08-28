from stocknet.datasets.shift import ShiftDataset
from stocknet.datasets.ohlc import OHLCDataset
from stocknet.datasets.finance import Dataset

available_dataset = {
    Dataset.key: Dataset,
    OHLCDataset.key: OHLCDataset,
    ShiftDataset.key: ShiftDataset,
}

def dataset_to_params(ds):
    from finance_client.finance_client.client_base import Client as m_client
    params = {}
    c_params = m_client.client_to_params(ds.data_client)
    params["client"] = c_params
    params["args"] = ds.args
    params["kinds"] = ds.key
    return params

def load_dataset(params:dict):
    kinds = params["kinds"]
    if kinds != available_dataset:
        from finance_client.finance_client.client_base import Client as m_client
        c_params = params["client"]
        c_kinds = c_params["kinds"]
        c_args = c_params["args"]
        data_client = m_client.load_client(c_kinds, c_args)
        
        Dataset = available_dataset[kinds]
        args = params["args"]
        ds = Dataset(data_client, *args)
        return ds
    return None