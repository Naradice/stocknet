import random

import numpy as np
import pandas as pd
from finance_client.fprocess import fprocess


def random_sampling(index, min_index, randomize, split_ratio, observation_length, prediction_length, params=None):
    length = len(index) - observation_length - prediction_length
    to_index = int(length * split_ratio)
    from_index = min_index
    train_indices = list(range(from_index, to_index))
    if randomize:
        train_indices = random.sample(train_indices, k=to_index - from_index)
    else:
        train_indices = train_indices

    # Note: If unique value exits in validation data only, validation loss would be grater than expected
    from_index = int(length * split_ratio) + observation_length + prediction_length
    to_index = length
    eval_indices = list(range(from_index, to_index))
    if randomize:
        eval_indices = random.sample(eval_indices, k=to_index - from_index)
    else:
        eval_indices = eval_indices
    return train_indices, eval_indices


def random_sampling_row(index, min_index, randomize, split_ratio, observation_length, prediction_length, params=None):
    length = len(index) - observation_length - prediction_length
    to_index = int(length * split_ratio)
    train_indices = index[:to_index]
    if randomize:
        train_indices = random.sample(train_indices, k=to_index)

    from_index = int(length * split_ratio) + observation_length + prediction_length

    eval_indices = index[from_index:]
    if randomize:
        eval_indices = random.sample(eval_indices, k=len(eval_indices))
    return train_indices, eval_indices


def k_fold_sampling(index, min_index, randomize, split_ratio, observation_length, prediction_length, params: dict = None):
    n = len(index)
    if params is None or "k" not in params:
        k = 100
    else:
        k = int(params["k"])

    if randomize:
        train_fold_index = random.sample(range(k), int(k * split_ratio))
    else:
        train_fold_index = list(range(int(k * split_ratio)))

    # create fold index
    split_idx = np.linspace(min_index, n, k + 1, dtype=int)

    train_idx = []
    val_idx = []
    for i in range(k):
        if i in train_fold_index:
            train_idx.extend(list(range(split_idx[i], split_idx[i + 1] - prediction_length - observation_length)))
        else:
            val_idx.extend(list(range(split_idx[i], split_idx[i + 1] - prediction_length - observation_length)))
    return train_idx, val_idx


def dataset_to_params(ds):
    params = {}
    args = ds.get_params()
    source = args.pop("source")
    params["source"] = source
    if "device" in args:
        device = str(args["device"])
        args["device"] = device
    params["args"] = args
    if hasattr(ds, "key"):
        params["key"] = ds.key
    else:
        ds_class_name = type(ds).__name__
        params["key"] = ds_class_name
    return params


def read_csv(file_path, index_col=0, **kwargs):
    return pd.read_csv(file_path, parse_dates=True, index_col=index_col)


def load_fprocesses(params: dict, columns: list):
    if isinstance(params, dict):
        return fprocess.load_preprocess(params)
    elif isinstance(params, list):
        processes = []
        for param in params:
            processes.extend(load_fprocesses(param, columns))
        return processes
    elif isinstance(params, str):
        if columns is not None and len(columns) > 0:
            if "columns" in params:
                params = params.copy()
                params.pop("columns")
            process = fprocess.load_default_preprocess(params, columns)
            if process is None:
                return []
            else:
                return [process]
        else:
            raise ValueError("valid column should be provided to load default proceess")
    else:
        raise TypeError(f"unsupported type {type(params)} provided to load fprocess")


def revert(dataset, values, ndx, is_tgt=False, columns=None):
    r_data = values
    indices = dataset.get_actual_index(ndx)
    if is_tgt:
        tgt_indices = []
        for __index in indices:
            ndx = dataset.output_indices(__index)
            tgt_indices.append(ndx.start)
        indices = tgt_indices
    # print(f"start revert procress for {[__process.kinds for __process in dataset.processes]}")
    for p_index in range(len(dataset.processes)):
        r_index = len(dataset.processes) - 1 - p_index
        process = dataset.processes[r_index]
        if hasattr(process, "revert_params"):
            # print(f"currently: {r_data[0, 0]}")
            params = process.revert_params
            if len(params) == 1:
                r_data = process.revert(r_data)
            else:
                params = {}
                if process.kinds == fprocess.MinMaxPreProcess.kinds:
                    r_data = process.revert(r_data, columns=columns)
                elif process.kinds == fprocess.SimpleColumnDiffPreProcess.kinds:
                    close_column = process.base_column
                    if p_index > 0:
                        processes = dataset.processes[:p_index]
                        required_length = [1]
                        base_processes = []
                        for base_process in processes:
                            if close_column in base_process.columns:
                                base_processes.append(base_process)
                                required_length.append(base_process.get_minimum_required_length())
                        if len(base_processes) > 0:
                            raise Exception("Not implemented yet")
                    base_indices = [index - 1 for index in indices]
                    base_values = dataset.org_data[close_column].iloc[base_indices]
                    r_data = process.revert(r_data, base_value=base_values)
                elif process.kinds == fprocess.DiffPreProcess.kinds:
                    if columns is None:
                        target_columns = process.columns
                    else:
                        target_columns = columns
                    if r_index > 0:
                        processes = dataset.processes[:r_index]
                        required_length = [process.get_minimum_required_length()]
                        base_processes = []
                        for base_process in processes:
                            if len(set(target_columns) & set(base_process.columns)) > 0:
                                base_processes.append(base_process)
                                required_length.append(base_process.get_minimum_required_length())
                        if len(base_processes) > 0:
                            required_length = max(required_length)
                            batch_base_indices = [index - required_length for index in indices]
                            batch_base_values = pd.DataFrame()
                            # print(f"  apply {[__process.kinds for __process in base_processes]} to revert diff")
                            for index in batch_base_indices:
                                target_data = dataset.org_data[target_columns].iloc[index : index + required_length]
                                for base_process in base_processes:
                                    target_data = base_process(target_data)
                                batch_base_values = pd.concat([batch_base_values, target_data.iloc[-1:]], axis=0)
                            batch_base_values = batch_base_values.values.reshape(1, *batch_base_values.shape)
                        else:
                            base_indices = [index - 1 for index in indices]
                            batch_base_values = dataset.org_data[target_columns].iloc[base_indices]
                    else:
                        base_indices = [index - 1 for index in indices]
                        batch_base_values = dataset.org_data[target_columns].iloc[base_indices].values
                    r_data = process.revert(r_data, base_values=batch_base_values)
                else:
                    raise Exception(f"Not implemented: {process.kinds}")
    return r_data
