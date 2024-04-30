import glob
import json
import os
from inspect import getmembers, isclass
from typing import Sequence

from .. import nets


def load_a_model(params: dict, key: str = None, device=None):
    if key is None:
        kinds = params["key"].lower()
    else:
        kinds = key.lower()

    for name, model_class in getmembers(nets, isclass):
        if name.lower() == kinds:
            if "args" in params:
                args = params["args"]
            elif "params" in params:
                args = params["params"]
            else:
                args = params
            args["device"] = device
            if hasattr(model_class, "load"):
                # if args include another nn.Module, create it in load function.
                model = model_class.load(**args)
            else:
                model = model_class(*args)
            return model
        else:
            continue
    return None


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


def load_models(model_configs: dict, dataset=None, device=None):
    model_key = model_configs["key"]
    if "model_name" in model_configs:
        model_name = model_configs["model_name"]
    else:
        model_name = model_key
    model_version = None
    model_version_number = None
    if "increment_version" in model_configs:
        model_version_number = int(model_configs["increment_version"])

    # config is used when params are defined in another file(s)
    if "configs" in model_configs:
        config_files_path = model_configs["configs"]
        if isinstance(config_files_path, str):
            formatted_path = os.path.abspath(os.path.join(os.getcwd(), config_files_path))
            config_files = glob.glob(formatted_path)
        elif isinstance(config_files_path, Sequence):
            config_files = config_files_path
        else:
            raise TypeError(f"unsupported configs provided: {config_files_path}. Suppose file_path or its list.")
        for config_file in config_files:
            if config_file.endswith(".json"):
                with open(config_file, "r") as fp:
                    params = json.load(fp)
            elif config_file.endswith("yaml") or config_file.endswith("yml"):
                import yaml

                with open(config_file, "r") as fp:
                    params = yaml.safe_load(fp)
            else:
                print(f"unsupported file type for model: {config_file}")
                yield None, model_name, model_version
            replace_params_vars(params, dataset)
            model = load_a_model(params, model_key, device=device)

            if "model_version" in params:
                model_version = params["model_version"]
            else:
                if model_version_number is not None:
                    model_version = model_version_number
                    model_version_number += 1
            yield model, model_name, model_version
    # params is used when model args are defined in the same file
    if "params" in model_configs:
        model_params = model_configs["params"]

        if isinstance(model_params, dict):
            replace_params_vars(model_params, dataset)
            model = load_a_model(model_params, model_key, device=device)
            if "model_version" in model_params:
                model_version = model_params["model_version"]
            elif model_version_number is not None:
                model_version = model_version_number
                model_version_number += 1

            yield model, model_name, model_version
        elif isinstance(model_params, Sequence):
            for params in model_params:
                model = load_a_model(params, model_key, device=device)
                if "model_version" in params:
                    model_version = params["model_version"]
                elif model_version_number is not None:
                    model_version = model_version_number
                    model_version_number += 1
                yield model, model_name, model_version
        else:
            raise TypeError(f"unsupported params provided: {model_params}. Suppose dict or its list.")
    return None, model_name, model_version
