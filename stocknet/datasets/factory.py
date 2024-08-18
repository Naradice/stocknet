from inspect import getmembers, isclass

import pandas as pd

from ..utils import load_a_custom_module
from . import utils
from .base import Dataset, TimeDataset
from .generator import (AgentSimulationTrainDataGenerator,
                        AgentSimulationWeeklyDataGenerator)
from .id import DiffIDDS
from .seq2seq import FeatureDataset, TimeFeatureDataset

seq2seq_pandas_dataset = [FeatureDataset, TimeFeatureDataset, DiffIDDS]
simulation_dataset = [AgentSimulationTrainDataGenerator, AgentSimulationWeeklyDataGenerator]
basic_dataset = [Dataset, TimeDataset]
DEFAULT_BATCH_SIZE = 64


def _list_file_info(source_info: dict):
    if isinstance(source_info, dict):
        files_info = [source_info]
    elif isinstance(source_info, list):
        files_info = source_info
    elif isinstance(source_info, str):
        files_info = [source_info]
    else:
        raise TypeError(f"invalid type specified as source: {type(source_info)}")

    return files_info


def _list_seqential_params(params: dict, key: str, alt: str = None, default_value=None, exp_type=None):
    if key in params:
        seq_info = params[key]
    elif alt is not None and alt in params:
        seq_info = params[alt]
    else:
        seq_info = None

    if isinstance(default_value, (list, tuple)):
        sequential = default_value
    else:
        sequential = [default_value]

    if seq_info is not None:
        if isinstance(seq_info, (int, float)):
            sequential = [seq_info]
        elif isinstance(seq_info, (list, tuple)):
            sequential = seq_info
        if exp_type is not None:
            exp_type = [exp_type(value) for value in sequential]
    return sequential


def _handle_files_info(files_info: list, params: dict, default_batch_sizes: list, device):
    for file_info in files_info:
        if "args" in params:
            args = params["args"].copy()
        else:
            args = params.copy()
        if "device" not in args:
            args["device"] = device

        if "version_suffix" in file_info:
            version_suffix = file_info["version_suffix"]
        else:
            version_suffix = None

        if "processes" in file_info:
            preprocesses_params_list = file_info["processes"]
        elif "processes" in args:
            preprocesses_params_list = args["processes"]
        else:
            preprocesses_params_list = None

        if "columns" in file_info:
            columns = file_info["columns"]
        elif "columns" in args:
            columns = args["columns"]
        elif "features" in args:
            columns = args.pop("features")
        else:
            columns = None

        if preprocesses_params_list is not None and len(preprocesses_params_list) > 0:
            pre_processes = utils.load_fprocesses(preprocesses_params_list, columns)
            args["processes"] = pre_processes
        if "columns" not in args and columns is not None:
            args["columns"] = columns

        batch_sizes = _list_seqential_params(params, "batch_size", default_value=default_batch_sizes, exp_type=int)

        volume_scale_set = []
        is_scaling = False
        if "scale_combinations" in file_info:
            is_scaling = True
            for scale_params in file_info["scale_combinations"]:
                volume_rate = float(scale_params["volume_rate"])
                if isinstance(scale_params["batch_size"], (list, set)):
                    batch_size = scale_params["batch_size"]
                else:
                    batch_size = [int(scale_params["batch_size"])]
                volume_scale_set.append((volume_rate, batch_size))
        else:
            if "scale_combinations" in args:
                is_scaling = True
                for scale_params in file_info["scale_combinations"]:
                    volume_rate = float(scale_params["volume_rate"])
                    if isinstance(scale_params["batch_size"], (list, set)):
                        batch_size = scale_params["batch_size"]
                    else:
                        batch_size = [int(scale_params["batch_size"])]
                    volume_scale_set.append((volume_rate, batch_size))
            else:
                is_scaling = False
                volume_scale_set = [(1.0, batch_sizes)]
        yield file_info, args, version_suffix, batch_sizes, is_scaling, volume_scale_set


def _handle_file_dict(file_dict):
    if isinstance(file_dict, (pd.DataFrame)):
        data = file_dict
    elif isinstance(file_dict, str):
        data = pd.read_csv(file_dict, parse_dates=True, index_col=0)
    elif isinstance(file_dict, dict):
        data = utils.read_csv(**file_dict)
    else:
        raise TypeError(f"{type(file_dict)} is not supported as source")
    return data


def _generate_suffix(observation=None, prediction=None, version_suffix=None, scale_id=None):
    suffix_strings = []
    if observation is not None and observation != "":
        suffix_strings.append(str(observation))
    if prediction is not None and prediction != "":
        suffix_strings.append(str(prediction))
    if version_suffix is not None:
        if type(version_suffix) is str and version_suffix.endswith("_"):
            version_suffix = version_suffix[:-1]
        suffix_strings.append(version_suffix)
    if scale_id is not None:
        if type(scale_id) is str and scale_id.startswith("_"):
            scale_id = scale_id[1:]
        suffix_strings.append(scale_id)
    suffix = "_".join(suffix_strings)
    return suffix


def load_finance_datasets(params: dict, device=None):
    from .finance import ClientDataset, FrameConvertDataset
    from .highlow import HighLowDataset
    from .shift import ShiftDataset

    finance_dataset = [ClientDataset, FrameConvertDataset, ShiftDataset, HighLowDataset]

    kinds = params["key"]
    Dataset = None
    batch_sizes = None
    version_suffix = None

    for ds_class in finance_dataset:
        if kinds == ds_class.key:
            Dataset = ds_class
            break

    if Dataset is not None:
        from finance_client import load_client

        c_params = params["client"]
        data_client = load_client(c_params)

        Dataset = finance_dataset[kinds]
        args = params["args"]
        ds = Dataset(data_client, device=device, **args)
        return ds, batch_sizes, version_suffix
    return None, None, None


def load_datasets(params: dict, device=None):
    params = params.copy()
    kinds = params["key"]
    Dataset = None
    batch_sizes = None
    version_suffix = None

    for ds_class in seq2seq_pandas_dataset:
        if kinds == ds_class.key:
            Dataset = ds_class
            break

    if Dataset is not None:
        source_info = params["source"]
        files_info = _list_file_info(source_info=source_info)
        observation_lengths = _list_seqential_params(params, "observation", "observation_length")
        prediction_lengths = _list_seqential_params(params, "prediction", "prediction_length")
        default_batch_sizes = _list_seqential_params(params, "batch_size", default_value=DEFAULT_BATCH_SIZE, exp_type=int)

        for options in _handle_files_info(files_info, params, default_batch_sizes, device):
            file_info, args, version_suffix, batch_sizes, is_scaling, volume_scale_set = options
            for obs_length in observation_lengths:
                for pre_length in prediction_lengths:
                    if obs_length is not None:
                        args["observation_length"] = obs_length
                    if pre_length is not None:
                        args["prediction_length"] = pre_length
                    ds = Dataset(file_info, **args)
                    for volume_rate, batch_sizes in volume_scale_set:
                        if is_scaling:
                            scale_id = f"_{volume_rate}"
                            ds.update_volume_limit(volume_rate)
                        else:
                            scale_id = ""
                        yield ds, batch_sizes, _generate_suffix(
                            observation=obs_length, prediction=pre_length, version_suffix=version_suffix, scale_id=scale_id
                        )
    return None, None, None


def load_simlation_datasets(params: dict, device=None):
    params = params.copy()
    kinds = params["key"]
    Dataset = None
    batch_sizes = None

    for ds_class in seq2seq_pandas_dataset:
        if kinds == ds_class.key:
            Dataset = ds_class
            break
    if Dataset is not None:
        pre_processes = utils.load_fproceses(params["processes"])
        data_generator = Dataset(processes=pre_processes, device=device, **params)
        return data_generator, batch_sizes
    return None, None, None


def load_custom_dataset(key: str, params: dict, base_path: str, device=None):
    ds_class = load_a_custom_module(key, "dataset", base_path)
    if ds_class is not None:
        default_batch_sizes = _list_seqential_params(params, "batch_size", default_value=DEFAULT_BATCH_SIZE, exp_type=int)
        ds_args = {"device": device}
        params = params.copy()
        if "args" in params:
            ds_args = params.pop("args")
        seq_params = {}
        if "source" in params:
            source_info = params.pop("source")
            files_info = _list_file_info(source_info=source_info)
            options = _handle_files_info(files_info, params, default_batch_sizes, device)
        else:
            options = [None]

        for param_key, values in params.items():
            if key == values:
                continue
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
        else:
            seq_args_set = [{}]

        for option in options:
            for seq_args in seq_args_set:
                args = ds_args.copy()
                args.update(seq_args)
                if option is not None:
                    file_info, file_args, version_suffix, batch_sizes, is_scaling, volume_scale_set = option
                    data = _handle_file_dict(file_info)

                    # handle multiple observation parameters
                    observation_key_candidates = ["observation", "observation_length"]
                    obs_key = None
                    for obs_key_candidate in observation_key_candidates:
                        if obs_key_candidate in file_args:
                            observations = file_args.pop(obs_key_candidate)
                            if type(observations) != list:
                                observations = [observations]
                            obs_key = obs_key_candidate
                            break
                    if obs_key is None:
                        observations = [None]
                        obs_key = ""

                    # handle multiple prediction parameters
                    prediction_key_candidates = ["prediction", "prediction_length"]
                    pre_key = None
                    for pre_key_candidate in prediction_key_candidates:
                        if pre_key_candidate in file_args:
                            predictions = file_args.pop(pre_key_candidate)
                            if type(predictions) != list:
                                predictions = [predictions]
                            pre_key = pre_key_candidate
                            break
                    if pre_key is None:
                        # dummy param
                        predictions = [None]
                        pre_key = ""

                    args.update(file_args)
                    for observation in observations:
                        for prediction in predictions:
                            args[obs_key] = observation
                            args[pre_key] = prediction
                            for volume_rate, batch_sizes in volume_scale_set:
                                if is_scaling:
                                    scale_id = f"_{volume_rate}"
                                    if hasattr(ds_class, "update_volume_limit"):
                                        ds = ds_class(data, **args)
                                        ds.update_volume_limit(volume_rate)
                                    else:
                                        length = len(data)
                                        temp_data = data.iloc[: int(length * volume_rate)].copy()
                                        ds = ds_class(temp_data, **args)
                                else:
                                    ds = ds_class(data, **args)
                                    scale_id = ""
                                yield ds, batch_sizes, _generate_suffix(
                                    observation=observation, prediction=prediction, version_suffix=version_suffix, scale_id=scale_id
                                )
                else:
                    ds = ds_class(**args)
                    yield ds, default_batch_sizes, f"{observation}_{prediction}"
    else:
        return None, None, None


def load_a_dataset(key: str, params: dict, device=None):
    from .. import datasets

    kinds = key.lower()
    for name, model_class in getmembers(datasets, isclass):
        if name.lower() == kinds:
            pass


def load(params: dict, device=None, base_path=None):
    dataset_key = params["key"]
    if dataset_key.startswith("seq2seq"):
        if "sim" in dataset_key:
            return load_simlation_datasets(params, device=device)
        else:
            return load_datasets(params, device=device)
    elif "fc" in dataset_key:
        return load_finance_datasets(params, device=device)
    else:
        return load_custom_dataset(dataset_key, params, base_path, device=device)
