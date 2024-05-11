from typing import Sequence

from . import utils
from .base import Dataset, TimeDataset
from .generator import AgentSimulationTrainDataGenerator, AgentSimulationWeeklyDataGenerator
from .id import DiffIDDS
from .seq2seq import FeatureDataset, TimeFeatureDataset

seq2seq_pandas_dataset = [FeatureDataset, TimeFeatureDataset, DiffIDDS]
simulation_dataset = [AgentSimulationTrainDataGenerator, AgentSimulationWeeklyDataGenerator]
basic_dataset = [Dataset, TimeDataset]


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


def load_seq2seq_datasets(params: dict, device=None):
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
        souce_info = params["source"]
        if isinstance(souce_info, dict):
            files_info = [souce_info]
        elif isinstance(souce_info, list):
            files_info = souce_info
        elif isinstance(souce_info, str):
            files_info = [souce_info]
        else:
            raise TypeError(f"invalid type specified as source: {type(souce_info)}")

        if "observation" in params:
            observation_info = params["observation"]
        elif "observation_length" in params:
            observation_info = params["observation_length"]
        else:
            raise ValueError("observation is required in dataset params")
        if isinstance(observation_info, (int, float)):
            observation_lengths = [observation_info]
        elif "length" in observation_info:
            observation_length_info = observation_info["length"]
            if isinstance(observation_length_info, (int, float)):
                observation_lengths = [int(observation_length_info)]
            elif isinstance(observation_length_info, list):
                observation_lengths = observation_length_info
            else:
                raise TypeError(f"invalid type specified as source: {type(observation_length_info)}")
        else:
            raise ValueError("length definition required in observation")

        if "prediction" in params:
            prediction_info = params["prediction"]
        elif "prediction_length" in params:
            prediction_info = params["prediction_length"]
        else:
            raise ValueError("prediction is required in dataset params")
        if isinstance(prediction_info, (int, float)):
            prediction_lengths = [prediction_info]
        elif "length" in prediction_info:
            prediction_length_info = prediction_info["length"]
            if isinstance(prediction_length_info, (int, float)):
                prediction_lengths = [int(prediction_length_info)]
            elif isinstance(prediction_length_info, list):
                prediction_lengths = prediction_length_info
            else:
                raise TypeError(f"invalid type specified as source: {type(prediction_length_info)}")
        else:
            raise ValueError("length definition required in prediction")

        if "batch_size" in params:
            default_batch_sizes = params["batch_size"]
            if isinstance(default_batch_sizes, (int, float)):
                default_batch_sizes = [int(default_batch_sizes)]
            elif not isinstance(default_batch_sizes, Sequence):
                default_batch_sizes = batch_sizes
        else:
            default_batch_sizes = [64]

        for file_info in files_info:
            args = params["args"].copy()

            if "version_suffix" in file_info:
                version_suffix = file_info["version_suffix"]

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

            if "batch_size" in file_info:
                specific_batch_sizes = file_info["batch_size"]
                if isinstance(specific_batch_sizes, (int, float)):
                    batch_sizes = [int(specific_batch_sizes)]
                if isinstance(specific_batch_sizes, Sequence):
                    batch_sizes = specific_batch_sizes
                else:
                    raise TypeError(f"{type(specific_batch_sizes)} is not supported as batch_size")
            else:
                batch_sizes = default_batch_sizes

            if preprocesses_params_list is not None and len(preprocesses_params_list) > 0:
                pre_processes = utils.load_fprocesses(preprocesses_params_list, columns)
                args["processes"] = pre_processes

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

            args["columns"] = columns
            args["device"] = device
            for obs_length in observation_lengths:
                for pre_length in prediction_lengths:
                    args["observation_length"] = obs_length
                    args["prediction_length"] = pre_length
                    ds = Dataset(file_info, **args)
                    for volume_rate, batch_sizes in volume_scale_set:
                        if is_scaling:
                            scale_id = f"_{volume_rate}"
                            ds.update_volume_limit(volume_rate)
                        else:
                            scale_id = ""
                        if version_suffix is None:
                            yield ds, batch_sizes, f"{obs_length}_{pre_length}{scale_id}"
                        else:
                            yield ds, batch_sizes, f"{obs_length}_{pre_length}_{version_suffix}{scale_id}"
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
