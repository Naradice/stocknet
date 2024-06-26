import importlib
import os
import sys
from glob import glob
from inspect import getmembers, isclass

from torch import nn, optim
from torch.optim import lr_scheduler

from .. import nets
from . import sltrainer

models_4_seq2seq = [nets.Seq2SeqTransformer.__name__]
models_4_seq2seq_sim = []


def __load_a_module(module_key: str, custom_key: str, module_path: str):
    if "." in module_key:
        dirs = module_key.split(".")
        # need to check submodules
        module_name = dirs[-2]
        attr_name = dirs[-1]
        sys.path.append(module_path)
        module = importlib.import_module(f"{custom_key}.{module_name}")
        attr = getattr(module, attr_name)
        return attr
    return None


def load_custom_trainer(trainer_config: dict, base_path: str):
    custom_folder = os.path.join(base_path, "custom")
    if "train_key" in trainer_config:
        train_key = trainer_config["train_key"]
        trainer_func = __load_a_module(train_key, "trainer", custom_folder)
    else:
        trainer_func = None
    if "eval_key" in trainer_config:
        eval_key = trainer_config["eval_key"]
        eval_func = __load_a_module(eval_key, "trainer", custom_folder)
    else:
        eval_func = None
    return trainer_func, eval_func


def load_trainers(model_key: str, configs: dict, base_path: str):
    # initialize options
    options = {}
    if "batch_first" in configs:
        batch_first = bool(configs["batch_first"])
        options["batch_first"] = batch_first

    # load trainer
    if "trainer" in configs:
        trainer_config = configs["trainer"]
        trainer_func, eval_func = load_custom_trainer(trainer_config, base_path)
        return trainer_func, eval_func, options
    elif model_key in models_4_seq2seq:
        return sltrainer.seq2seq_train, sltrainer.seq2seq_eval, options
    elif model_key in models_4_seq2seq_sim:
        # return sltrainer.seq2seq_train, sltrainer.seq2seq_eval, options
        return None, None, options
    else:
        # dummy
        trainer = sltrainer.Trainer()
        return trainer, None, {}


def load_trainer_options(model, params, base_path):
    if "optimizer" in params:
        optim_params = params["optimizer"].copy()
        if "key" in optim_params:
            optim_key = optim_params.pop("key")
            optimizer = load_an_optimizer(optimizer_key=optim_key, model=model, params=optim_params)
        else:
            optimizer = None
    else:
        optimizer = None

    if "loss" in params:
        loss_params = params["loss"].copy()
        if "key" in loss_params:
            loss_key = loss_params.pop("key")
            criterion = load_a_criterion(loss_key, loss_params)
            if criterion is None:
                criterion = load_a_custom_criterion(loss_key, loss_params, base_path)
        else:
            criterion = None
    else:
        criterion = None

    if "scheduler" in params and optimizer is not None:
        schl_params = params["scheduler"].copy()
        if "key" in schl_params:
            schl_key = schl_params.pop("key")
            scheduler = load_a_scheduler(scheduler_key=schl_key, optimizer=optimizer, params=schl_params)
        else:
            scheduler = None
    else:
        scheduler = None

    return optimizer, criterion, scheduler


def load_an_optimizer(optimizer_key: str, model, params: dict):
    key = optimizer_key.lower()
    for name, opt_class in getmembers(optim, isclass):
        if name.lower() == key:
            return opt_class(model.parameters(), **params)
    return None


def load_a_scheduler(scheduler_key: str, optimizer, params: dict):
    key = scheduler_key.lower()
    for name, shl_class in getmembers(lr_scheduler, isclass):
        if name.lower() == key:
            params["last_epoch"] = -1
            # The verbose parameter is deprecated
            if "verbose" in params:
                params.pop("verbose")
            return shl_class(optimizer, **params)
    return None


def load_a_criterion(criterion_key: str, params: dict):
    key = criterion_key.lower()
    for name, nn_class in getmembers(nn, isclass):
        if name.endswith("Loss"):
            if name.lower() == key:
                return nn_class(**params)


def load_a_custom_criterion(criterion_key: str, params: dict, base_dir: str):
    """load a custom criterion which user defined

    Args:
        criterion_key (str): expected format is {directory}.{class_name}
        params (dict): params to initialize a criterion
        base_dir (str): parent folder path. cutom loss function must exist in {base_dir}/loss/{your_class}.py
    """

    if len(criterion_key) > 0:
        custom_folder = os.path.join(base_dir, "custom")
        if "." in criterion_key:
            custom_loss = __load_a_module(criterion_key, "loss", custom_folder)
            return custom_loss(**params)
        else:
            # since folder name was not specified, search class name from all files.
            pass
    else:
        return None
