import inspect
import math
import os
import time
from typing import Sequence

from . import logger
from .datasets import factory as ds_factory
from .nets import factory as mdl_factory
from .trainer import factory as tr_factory


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
    else:
        device = None

    # create dataset
    dataset_key = dataset_config["key"]
    if "seq2seq" in dataset_key:
        if "sim" in dataset_key:
            datasets = ds_factory.load_simlation_dataset(dataset_config, device=device)
        else:
            datasets = ds_factory.load_seq2seq_dataset(dataset_config, device=device)
    elif "fc" in dataset_key:
        datasets = ds_factory.load_finance_dataset(dataset_config, device=device)
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

            print(f"new model created: {model_name}")
            training_logger = logger.TrainingLogger(model_name, model_version_str, log_path, storage_handler)
            opt_config = train_config["optimizer"].copy()
            opt_key = opt_config.pop("key")
            optimizer = tr_factory.load_an_optimizer(opt_key, model, opt_config)
            if optimizer is None:
                print(f"optimizer {opt_key} not found.")
                continue
            if "scheduler" in train_config:
                schl_config = train_config["scheduler"].copy()
                schl_key = schl_config.pop("key")
                scheduler = tr_factory.load_a_scheduler(schl_key, optimizer, schl_config)
                if scheduler is None:
                    print(f"scheduler {schl_key} not found.")
            else:
                scheduler = None

            criterion_config = train_config["loss"].copy()
            criterion_key = criterion_config.pop("key")
            criterion = tr_factory.load_a_criterion(criterion_key, criterion_config)
            if criterion is None:
                print(f"criterion {criterion_key} not found.")

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
