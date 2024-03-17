from inspect import getmembers, isclass
from typing import Sequence

from torch import nn, optim
from torch.optim import lr_scheduler

from .. import nets
from . import sltrainer

models_4_seq2seq = [nets.Seq2SeqTransformer.__name__]
models_4_seq2seq_sim = []


def load_trainers(model_key: str, configs: dict):
    if model_key in models_4_seq2seq:
        options = {}
        if "batch_first" in configs:
            batch_first = bool(configs["batch_first"])
            options["batch_first"] = batch_first
        return sltrainer.seq2seq_train, sltrainer.seq2seq_eval, options.copy()
    elif model_key in models_4_seq2seq_sim:
        # return sltrainer.seq2seq_train, sltrainer.seq2seq_eval, options.copy()
        return None, None, options.copy()
    else:
        # dummy
        trainer = sltrainer.Trainer()
        return trainer, None, {}


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
            return shl_class(optimizer, **params)
    return None


def load_a_criterion(criterion_key: str, params: dict):
    key = criterion_key.lower()
    for name, nn_class in getmembers(nn, isclass):
        if name.endswith("Loss"):
            if name.lower() == key:
                return nn_class(**params)
