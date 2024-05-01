import numpy as np
import torch
from tqdm import tqdm


def __seq2seq_main(index, model, ds, criterion, batch_size, split_tgt_func):
    temp = ds[index : index + batch_size]
    src, tgt = temp[:2]
    option_args = temp[2:]

    input_tgt, output_tgt = split_tgt_func(tgt)
    logits_c = model(src, input_tgt, *option_args)
    loss = criterion(logits_c, output_tgt)
    return loss


def seq2seq_eval(model, ds, criterion, batch_size, batch_first=True):
    model.eval()
    ds.eval()
    losses = []

    if batch_first is True:
        split_tgt = lambda tensor: (tensor[:, :-1], tensor[:, 1:])
    else:
        split_tgt = lambda tensor: (tensor[:-1], tensor[1:])
    end_index = len(ds)
    for index in tqdm(range(0, end_index - batch_size, batch_size)):
        loss = __seq2seq_main(index, model, ds, criterion, batch_size, split_tgt)
        losses.append(loss.item())

    return np.mean(losses)


def seq2seq_train(model, ds, optimizer, criterion, batch_size, batch_first=True):
    model.train()
    ds.train()
    losses = []

    if batch_first is True:
        split_tgt = lambda tensor: (tensor[:, :-1], tensor[:, 1:])
    else:
        split_tgt = lambda tensor: (tensor[:-1], tensor[1:])
    end_index = len(ds)
    for index in tqdm(range(0, end_index - batch_size, batch_size)):
        optimizer.zero_grad()
        loss = __seq2seq_main(index, model, ds, criterion, batch_size, split_tgt)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return np.mean(losses)
