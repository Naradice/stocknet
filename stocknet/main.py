import inspect
import math
import os
import time
from typing import Sequence

import torch

from . import logger
from .datasets import factory as ds_factory
from .datasets import utils as ds_utils
from .nets import factory as mdl_factory
from .nets import utils as mdl_utils
from .trainer import factory as tr_factory
from .trainer import utils as tr_utils


def train_from_config(training_config_file: str):
    if os.path.exists(training_config_file):
        training_config_file_path = training_config_file
    else:
        stacks = inspect.stack()
        parent_path = stacks[1].filename
        parent_dir_name = os.path.dirname(parent_path)
        training_config_file_path = os.path.abspath(os.path.join(parent_dir_name, training_config_file))
    if training_config_file_path.endswith(".yaml") or training_config_file_path.endswith(".yml"):
        import yaml

        with open(training_config_file_path, "r") as fp:
            config = yaml.safe_load(fp)
    elif training_config_file_path.endswith(".json"):
        import json

        with open(training_config_file_path, "r") as fp:
            config = json.load(fp)
    else:
        raise ValueError("unsupported extension")

    dataset_config = config["dataset"]
    model_config = config["model"]
    train_config = config["trainer"]
    logger_config = config["log"]
    log_path = logger_config["path"]
    global_model_version = 0
    storage_handler = None

    if "epoch" in train_config:
        epoch = train_config["epoch"]
    else:
        epoch = 100
    if "patience" in train_config:
        patience = train_config["patience"]
    else:
        patience = 2
    if "device" in train_config:
        device = train_config["device"]
        device = torch.device(device)
    else:
        device = None

    # create dataset
    dataset_key = dataset_config["key"]
    if "seq2seq" in dataset_key:
        if "sim" in dataset_key:
            datasets = ds_factory.load_simlation_datasets(dataset_config, device=device)
        else:
            datasets = ds_factory.load_seq2seq_datasets(dataset_config, device=device)
    elif "fc" in dataset_key:
        datasets = ds_factory.load_finance_datasets(dataset_config, device=device)
    else:
        raise ValueError("invalid dataset key")

    for dataset, batch_sizes_4_ds, version_suffix in datasets:
        if dataset is None:
            continue
        print("new dataset loaded")
        model_key = model_config["key"]
        models = mdl_factory.load_models(model_config.copy(), device=device)
        for model, model_name, model_version in models:
            if model is None:
                continue
            if model_version is None:
                global_model_version += 1
                model_version = global_model_version
            if version_suffix is None:
                model_version_str = f"v{model_version}"
            else:
                model_version_str = f"{version_suffix}_v{model_version}"

            print(f"new model created: {model_name}_{model_version_str}")
            training_logger = logger.TrainingLogger(model_name, model_version_str, log_path, storage_handler)
            optimizer, criterion, scheduler = tr_factory.load_trainer_options(model=model, params=train_config.copy())
            if optimizer is None:
                print("optimizer not found.")
                continue
            if criterion is None:
                print("loss function not found.")

            if batch_sizes_4_ds is None or len(batch_sizes_4_ds) == 0:
                if "batch_size" in train_config:
                    common_batch_size = train_config["batch_size"]
                    if isinstance(common_batch_size, (int, float)):
                        batch_sizes = [int(common_batch_size)]
                    elif isinstance(common_batch_size, Sequence):
                        batch_sizes = common_batch_size
                    else:
                        raise TypeError(f"{type(common_batch_size)} is not supported as batch_size")
                else:
                    batch_sizes = [64]
            else:
                batch_sizes = batch_sizes_4_ds

            trainer_func, eval_func, train_options = tr_factory.load_trainers(model_key, train_config)
            succ, model, optimizer, scheduler, best_loss = logger.load_model_checkpoint(
                model, model_name, model_version_str, optimizer, scheduler, log_path, eval_func is None, storage_handler
            )
            save_params(epoch, model, dataset, patience, optimizer, criterion, scheduler, batch_sizes, training_logger)
            for batch_size in batch_sizes:
                print(f"start training model with batch_size: {batch_size}")
                epoch_trainer(
                    epoch=epoch,
                    model=model,
                    dataset=dataset,
                    patience=patience,
                    train_method=trainer_func,
                    eval_method=eval_func,
                    batch_size=batch_size,
                    train_options=train_options,
                    optimizer=optimizer,
                    criterion=criterion,
                    scheduler=scheduler,
                    logger=training_logger,
                )


def save_params(epoch, model, dataset, patience, optimizer, criterion, scheduler, batch_size, logger):
    training_params = {}

    ds_params = ds_utils.dataset_to_params(dataset)
    training_params["dataset"] = ds_params

    model_params = mdl_utils.model_to_params(model, logger.model_name)
    training_params["model"] = model_params
    model_info = mdl_utils.get_params_count(model)
    training_params["model_info"] = model_info
    dataset.train()
    training_params["data_volume"] = len(dataset)

    train_option_params = tr_utils.tainer_options_to_params(optimizer, criterion, scheduler, epoch=epoch, patience=patience, batch_size=batch_size)
    training_params["trainer"] = train_option_params
    training_params["log"] = {"path": logger.base_path}
    logger.save_params(training_params)


def epoch_trainer(
    epoch, model, dataset, patience, train_method, eval_method, batch_size, train_options, optimizer, criterion, scheduler, logger, **kwargs
):
    if "best_train_loss" in kwargs:
        train_to_best = bool(kwargs["best_train_loss"])
    else:
        if eval_method is None:
            train_to_best = True
        else:
            train_to_best = False
    best_train_loss, best_valid_loss = logger.get_min_losses()
    model_name = logger.model_name
    model_version = logger.version

    best_model = model
    best_train_model = model
    counter = 0

    for loop in range(1, epoch + 1):
        start_time = time.time()

        loss_train = train_method(model=model, ds=dataset, optimizer=optimizer, criterion=criterion, batch_size=batch_size, **train_options)

        elapsed_time = time.time() - start_time

        if eval_method is not None:
            loss_valid = eval_method(model=model, ds=dataset, criterion=criterion, batch_size=batch_size, **train_options)
        else:
            loss_valid = 0.0

        elapsed_mins = math.floor(elapsed_time / 60)
        logger.add_training_log(loss_train, loss_valid, elapsed_time)
        log = "[{}/{}] train loss: {:.10f}, valid loss: {:.10f}  [{}{:.0f}s] count: {}, {}".format(
            loop,
            epoch,
            loss_train,
            loss_valid,
            str(int(elapsed_mins)) + "m" if elapsed_mins > 0 else "",
            elapsed_time % 60,
            counter,
            "**" if best_valid_loss > loss_valid else "",
        )
        print(log)

        if best_train_loss > loss_train:
            best_train_loss = loss_train
            best_train_model = model
            logger.save_checkpoint(best_train_model, optimizer, scheduler, f"{model_name}_train", model_version, best_train_loss)
            if train_to_best is True:
                counter = 0
        else:
            if train_to_best is True:
                counter += 1
                scheduler.step()
        if best_valid_loss > loss_valid:
            best_valid_loss = loss_valid
            best_model = model
            if train_to_best is False:
                counter = 0
            logger.save_checkpoint(best_model, optimizer, scheduler, model_name, model_version, best_valid_loss)
        else:
            if train_to_best is False:
                counter += 1
                scheduler.step()

        if counter > patience:
            break

    logger.save_logs()


if __name__ == "__main__":
    pass
