from inspect import _empty, signature


def __get_class_args(instance):
    class_params = signature(type(instance)).parameters
    param_keys = list(class_params.keys())

    params = {}
    for key in param_keys:
        try:
            value = getattr(instance, key)
        except AttributeError:
            continue
        try:
            default_value = class_params[key].default
            if isinstance(default_value, _empty):
                params[key] = value
            else:
                if default_value == value:
                    continue
                else:
                    params[key] = value
        except Exception:
            params[key] = value
    return params


def optimizer_to_params(optimizer):
    optim_key = type(optimizer).__name__
    if hasattr(optimizer, "defaults"):
        params = optimizer.defaults

        class_params = signature(type(optimizer)).parameters
        for key, value in params.copy().items():
            if key in class_params:
                default_value = class_params[key].default
                if default_value is _empty:
                    continue
                if default_value == value:
                    params.pop(key)
    else:
        params = __get_class_args(optimizer)
    params["key"] = optim_key
    return params


def criterion_to_params(criterion):
    key = type(criterion).__name__
    params = __get_class_args(criterion)
    params["key"] = key
    return params


def scheduler_to_params(scheduler):
    class_params = signature(type(scheduler)).parameters
    param_keys = list(class_params.keys())
    key = type(scheduler).__name__
    params = {"key": key}
    for key in param_keys:
        if key != "optimizer":
            try:
                value = getattr(scheduler, key)
            except AttributeError:
                continue
            try:
                default_value = class_params[key].default
                if isinstance(default_value, _empty):
                    params[key] = value
                else:
                    if default_value == value:
                        continue
                    else:
                        params[key] = value
            except Exception:
                params[key] = value
    return params


def tainer_options_to_params(optimizer, criterion, scheduler=None, **kwargs):
    params = {}
    params["optimizer"] = optimizer_to_params(optimizer)
    params["loss"] = criterion_to_params(criterion)
    if scheduler is not None:
        params["scheduler"] = scheduler_to_params(scheduler)
    params.update(kwargs)
    return params
