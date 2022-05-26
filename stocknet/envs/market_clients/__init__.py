from stocknet.envs.market_clients.csv.client import CSVClient, MultiFrameClient
import stocknet.envs.market_clients.coincheck
from stocknet.envs.market_clients.frames import Frame
from stocknet.envs.market_clients.market_client_base import MarketClientBase


available_clients = {
    CSVClient.kinds : CSVClient,
    MultiFrameClient.kinds: MultiFrameClient
    #BCCClient.kinds : BCCClient
}

def client_to_params(client):
    params = {}
    params["kinds"] = client.kinds
    params["args"] = client.args
    return params

def load_client(params:dict):
    kinds = params["kinds"]
    if kinds in available_clients:
        args = params["args"]
        Client = available_clients[kinds]
        data_client = Client(*args)
        return data_client
    return None
    