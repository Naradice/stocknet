from typing import Sequence

from . import utils
from .generator import AgentSimulationTrainDataGenerator, AgentSimulationWeeklyDataGenerator
from .seq2seq import FeatureDataset, TimeFeatureDataset

seq2seq_pandas_dataset = [FeatureDataset, TimeFeatureDataset]
simulation_dataset = [AgentSimulationTrainDataGenerator, AgentSimulationWeeklyDataGenerator]


def load_finance_dataset(params: dict, device=None):
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


def load_seq2seq_dataset(params: dict, device=None):
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
        else:
            raise TypeError(f"invalid type specified as source: {type(souce_info)}")

        observation_info = params["observation"]
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

        prediction_info = params["prediction"]
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
            df = utils.read_csv(**file_info)
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
                columns = df.columns.copy()

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
                pre_processes = []
                for preprocesses_params in preprocesses_params_list:
                    process = utils.load_fprocesses(preprocesses_params, columns)
                    if process is None:
                        raise ValueError(f"can't load {preprocesses_params} with {columns}")
                    if isinstance(process, Sequence):
                        pre_processes.extend(process)
                    else:
                        pre_processes.append(process)
                args["processes"] = pre_processes

            volume_scale_set = []
            is_scaling = False
            if "scale_combinations" in file_info:
                is_scaling = True
                for scale_params in file_info["scale_combinations"]:
                    volume_rate = float(scale_params["volume_rate"])
                    batch_size = int(scale_params["batch_size"])
                    volume_scale_set.append((volume_rate, [batch_size]))
            else:
                if "scale_combinations" in args:
                    is_scaling = True
                    for scale_params in file_info["scale_combinations"]:
                        volume_rate = float(scale_params["volume_rate"])
                        batch_size = int(scale_params["batch_size"])
                        volume_scale_set.append((volume_rate, [batch_size]))
                else:
                    is_scaling = False
                    volume_scale_set = [(1.0, batch_sizes)]

            args["columns"] = columns
            args["device"] = device
            for obs_length in observation_lengths:
                for pre_length in prediction_lengths:
                    for volume_rate, batch_sizes in volume_scale_set:
                        args["observation_length"] = obs_length
                        args["prediction_length"] = pre_length
                        data_volume = int(len(df) * volume_rate)
                        data = df.iloc[:data_volume].copy()
                        ds = Dataset(data, **args)
                        if is_scaling:
                            scale_id = f"_{volume_rate}"
                        else:
                            scale_id = ""
                        if version_suffix is None:
                            yield ds, batch_sizes, f"{obs_length}_{pre_length}{scale_id}"
                        else:
                            yield ds, batch_sizes, f"{obs_length}_{pre_length}_{version_suffix}{scale_id}"
    return None, None, None


def load_simlation_dataset(params: dict, device=None):
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
