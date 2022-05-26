from matplotlib.style import available
from numpy import c_
from stocknet.envs.datasets.bc import Dataset, ShiftDataset
from stocknet.envs.datasets.fx import FXNextMEANDiffDataset, FXDataset

available_dataset = {
    Dataset.key : Dataset,
    ShiftDataset.key: ShiftDataset,
    FXDataset.key: FXDataset,
    FXNextMEANDiffDataset.key: FXNextMEANDiffDataset
}

def dataset_to_params(ds):
    import stocknet.envs.market_clients as m_client
    params = {}
    c_params = m_client.client_to_params(ds.data_client)
    params["client"] = c_params
    params["args"] = ds.args
    params["kinds"] = ds.key
    return params

def load_dataset(params:dict):
    kinds = params["kinds"]
    if kinds != available_dataset:
        import stocknet.envs.market_clients as m_client
        c_params = params["client"]
        c_kinds = c_params["kinds"]
        c_args = c_params["args"]
        data_client = m_client.load_client(c_kinds, c_args)
        
        Dataset = available_dataset[kinds]
        args = params["args"]
        ds = Dataset(data_client, *args)
        return ds
    return None