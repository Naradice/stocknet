import json
from inspect import _empty, getmembers, isclass, signature

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


def get_class_args(instance):
    class_params = signature(type(instance)).parameters
    param_keys = list(class_params.keys())

    params = {}
    for key in param_keys:
        try:
            value = getattr(instance, key)
        except AttributeError:
            continue

        try:
            json.dumps(value)
        except (TypeError, OverflowError):
            continue
        try:
            default_value = class_params[key].default
            if default_value == _empty:
                params[key] = value
            else:
                if default_value == value:
                    continue
                else:
                    params[key] = value
        except Exception:
            params[key] = value
    return params


def load_nn_module(key: str, params: dict):
    _key = key.lower()
    for name, nn_class in getmembers(nn, isclass):
        if name.lower() == _key:
            return nn_class(**params)
