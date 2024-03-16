import glob
import json
from inspect import getmembers, isclass
from typing import Sequence

from . import __init__ as nets


def model_to_params(model):
    params = {}

    kinds = model.key
    params["args"] = model.option
    params["key"] = kinds

    return params


def load_a_model(params: dict, key: str = None):
    if key is None:
        kinds = params["key"].lower()
    else:
        kinds = key.lower()

    for name, model_class in getmembers(nets, isclass):
        if name.lower() == kinds:
            if "args" in params:
                args = params["args"]
            else:
                args = params
            if hasattr(model_class, "load"):
                # if args include another nn.Module, create it in load function.
                model = model_class.load(**args)
            else:
                model = model_class(*args)
            return model
        else:
            continue
    return None


def load_models(model_configs: dict):
    model_key = model_configs["key"]
    if "model_name" in model_configs:
        model_name = model_configs["model_name"]
    else:
        model_name = model_key
    model_version = None

    if "configs" in model_configs:
        config_files_path = model_configs["configs"]
        if isinstance(config_files_path, str):
            config_files = glob.glob(config_files_path)
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
            model = load_a_model(params, model_key)

            if "model_version" in params:
                model_version = params["model_version"]
            yield model, model_name, model_version
    if "params" in model_configs:
        model_params = model_configs["configs"]
        if isinstance(model_params, dict):
            model = load_a_model(model_params, model_key)
            yield model, model_name
        elif isinstance(model_params, Sequence):
            for params in model_params:
                model = load_a_model(model_params, model_key)
                yield model, model_name, model_version
        else:
            raise TypeError(f"unsupported params provided: {model_params}. Suppose dict or its list.")
    return None, model_name, model_version