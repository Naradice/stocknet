from stocknet.nets.ae import AELinearModel
from stocknet.nets.dense import ConvDense16, SimpleDense
from stocknet.nets.lstm import LSTM

available_models = {
    AELinearModel.key: AELinearModel,
    ConvDense16.key: ConvDense16,
    SimpleDense.key: SimpleDense,
    LSTM.key: LSTM
}

def model_to_params(model):
    params = {}
    
    kinds = model.key    
    params["args"] = model.option
    params["kinds"] = kinds
    
    return params

def load_model(params:dict):
    kinds = params["kinds"]
    if kinds in available_models:
        Model = available_models[kinds]
        args = params["args"]
        model = Model(*args)
    return None