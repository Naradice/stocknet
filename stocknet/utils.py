import importlib
import inspect
import os
import sys


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


def load_a_custom_module(module_key: str, custom_key: str, module_path: str):
    if "." in module_key:
        custom_folder = os.path.join(module_path, "custom")
        dirs = module_key.split(".")
        # need to check submodules
        module_name = dirs[-2]
        attr_name = dirs[-1]
        sys.path.append(custom_folder)
        module = importlib.import_module(f"{custom_key}.{module_name}")
        attr = getattr(module, attr_name)
        return attr
    return None


def generate_args(params: dict):
    params = params.copy()
    if "args" in params:
        ds_args = params.pop("args")
    seq_params = {}
    for param_key, values in params.items():
        if isinstance(values, (list, tuple)):
            seq_params[param_key] = values
        else:
            ds_args[param_key] = values
    if len(seq_params) > 0:
        seq_args_set = []
        while len(seq_params) > 0:
            item = seq_params.popitem()
            if len(seq_args_set) == 0:
                seq_key, values = item
                for value in values:
                    seq_args_set.append({seq_key: value})
            else:
                new_seq_args_set = []
                while len(seq_args_set) > 0:
                    seq_args = seq_args_set.pop()
                    seq_key, values = item
                    for value in values:
                        new_seq_args = seq_args.copy()
                        new_seq_args.update({seq_key: value})
                        new_seq_args_set.append(new_seq_args)
                seq_args_set = new_seq_args_set
    for seq_args in seq_args_set:
        args = ds_args.copy()
        args.update(seq_args)
        yield args
