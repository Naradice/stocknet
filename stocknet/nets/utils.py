def model_to_params(model):
    params = {}
    if hasattr(model, "key"):
        kinds = model.key
    else:
        kinds = type(model).__name__
    params["args"] = model.args.copy()
    params["key"] = kinds

    return params
