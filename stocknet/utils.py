import inspect
import os


def replace_params_vars(params: dict, dataset):
    if dataset is None:
        return params
    new_params = {}
    for key, value_key in params.items():
        if isinstance(value_key, dict):
            replace_params_vars(value_key, dataset)
        elif isinstance(value_key, str):
            if value_key.startswith("$"):
                variable = value_key[1:]
                vars = variable.split(".")
                # currently dataset is supported only
                if len(vars) == 2:
                    if vars[0] == "dataset":
                        value = getattr(dataset, vars[1])
                        new_params[key] = value
                    else:
                        raise ValueError(f"unkown variable is specified in model params: {key}:{value_key}")
                else:
                    raise ValueError(f"unkown variable is specified in model params: {key}:{value_key}")
    params.update(new_params)


def get_caller_directory(depth=2):
    """get parent file directory

    Args:
        depth (int, optional): 0 is this module. 1 is module of caller of this function. Defaults to 2.

    Returns:
        str: file path of trace depth
    """
    stack = inspect.stack()
    caller_frame = stack[depth]
    caller_file_path = caller_frame.filename
    return os.path.abspath(caller_file_path)
