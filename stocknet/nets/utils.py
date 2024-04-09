def model_to_params(model, model_name=None):
    params = {}
    if hasattr(model, "key"):
        kinds = model.key
    else:
        kinds = type(model).__name__
    params["args"] = model.args.copy()
    params["key"] = kinds
    if model_name is not None and isinstance(model_name, str):
        params["model_name"] = model_name

    return params
