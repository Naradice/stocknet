from torch import nn

_params_key = "model_params_num"


def model_to_params(model, model_name=None):
    params = {}
    if hasattr(model, "key"):
        kinds = model.key
    else:
        kinds = type(model).__name__
    params["params"] = model.args.copy()
    params["key"] = kinds
    if model_name is not None and isinstance(model_name, str):
        params["model_name"] = model_name

    return params


def __get_tf_params_count(model):
    params_num = 0
    tf_params = 0

    for name, p in model.named_parameters():
        if p.requires_grad:
            params_num += p.numel()
            if "transformer" in name:
                tf_params += p.numel()

    return {_params_key: params_num, "tf_parms_num": tf_params}


def get_params_count(model):
    if "transformer" in type(model).__name__.lower():
        return __get_tf_params_count(model)
    else:
        params_num = 0

    for name, p in model.named_parameters():
        if p.requires_grad:
            params_num += p.numel()
    return {_params_key: params_num}
