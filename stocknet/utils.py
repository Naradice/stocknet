import datetime
import json
import os

import torch
from fprocess import idcprocess as ips
from fprocess import preprocess as pps


def get_validate_filename(model_name, extension="png"):
    dir_name, version = __remove_version_str(model_name)
    check_directory(dir_name)
    file_name = f"models/{dir_name}/validation_result_{version}.{extension}"
    return file_name


def save_result_as_txt(model_name: str, result: str, use_date_to_filename=True):
    dir_name, version = __remove_version_str(model_name)
    check_directory(dir_name)
    if use_date_to_filename:
        now = datetime.datetime.now()
        current_datetime_txt = str(now.date()) + "T" + str(now.hour)  # YYYY-MM-DDThh
        txt_file_name = f"models/{dir_name}/result_{version}_{current_datetime_txt}.txt"
        mode = "w"
    else:
        txt_file_name = f"models/{dir_name}/result_{version}.txt"
        mode = "a+"
    with open(txt_file_name, mode, encoding="utf-8") as f:
        if use_date_to_filename is False:
            # move read cursor to top, then read to know if there are content already
            f.seek(0)
            data = f.read(100)
            if len(data) > 0:
                f.write("\n")
        f.write(result)


def save_model(model, model_name):
    dir_name, version = __remove_version_str(model_name)
    check_directory(dir_name)
    torch.save(model.state_dict(), f"models/{dir_name}/model_{version}.torch")


def save_tarining_params(params: dict, model_name):
    check_directory(model_name)
    params_str = json.dumps(params)
    with open(f"models/{model_name}.param", "w", encoding="utf-8") as f:
        f.write(params_str)


def __remove_version_str(name: str):
    contents = name.split("_")
    removed = "_".join(contents[:-1])
    version = contents[-1]
    return removed, version


def check_directory(model_name: str) -> None:
    if "/" in model_name:
        names = model_name.split("/")
        path_ = os.path.join("models", *names)
        if os.path.exists(path_) is False:
            os.makedirs(path_)
    elif os.path.exists("models") is False:
        os.makedirs("models")


def save_client_params(model_name, client):
    from finance_client import client_to_params

    dir_name, version = __remove_version_str(model_name)
    check_directory(dir_name)
    txt_file_name = f"models/{dir_name}/client_{version}.json"
    client_params = client_to_params(client)
    with open(txt_file_name, "w", encoding="utf-8") as f:
        json.dump(client_params, f)


def save_params(model_name, params: dict):
    dir_name, version = __remove_version_str(model_name)
    check_directory(dir_name)
    txt_file_name = f"models/{dir_name}/params_{version}.json"
    with open(txt_file_name, "w", encoding="utf-8") as f:
        json.dump(params, f)


def processes_to_params(indicaters: list, preprocesses: list):
    ind_params = ips.to_param_dict(indicaters)
    pre_params = pps.to_params_dict(preprocesses)
    params = {"indicaters": ind_params, "preprocesses": pre_params}
    return params


def save_processes(model_name, indicaters: list, preprocesses: list):
    params = processes_to_params(indicaters, preprocesses)
    save_params(model_name, params)


def load_params(model_name):
    dir_name, version = __remove_version_str(model_name)
    check_directory(dir_name)
    txt_file_name = f"models/{dir_name}/params_{version}.json"
    with open(txt_file_name, "r", encoding="utf-8") as f:
        params = json.load(f)
    return params


def dataset_to_params(model_name, ds):
    from finance_client import client_to_params

    client_params = client_to_params(ds.data_client)
    return client_params


def load_model(model, model_name):
    dir_name, version = __remove_version_str(model_name)
    model_path = f"models/{dir_name}/model_{version}.torch"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        print("model name doesn't exist. new model will be created.")
