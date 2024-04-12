from torch import nn


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


def get_params_count(model, save_each=True):
    total_num = 0
    params_num = {}

    for name, p in model.named_parameters():
        if p.requires_grad:
            count = p.numel()
            total_num += count
            if save_each:
                params_num[name] = count

    params_num["total_num"] = total_num
    return params_num
