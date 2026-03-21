import numpy as np
from tqdm import tqdm


def meanvar_transformer_train(model, ds, optimizer, criterion, batch_size, batch_first=True, **kwargs):
    model.train()
    ds.train()
    losses = []
    split_tgt = (lambda t: (t[:, :-1], t[:, 1:])) if batch_first else (lambda t: (t[:-1], t[1:]))
    end_index = len(ds)
    for index in tqdm(range(0, end_index - batch_size, batch_size)):
        temp = ds[index : index + batch_size]
        src, tgt = temp[:2]
        option_args = temp[2:]
        input_tgt, output_tgt = split_tgt(tgt)
        mean, log_var = model(src, input_tgt, *option_args)
        loss = criterion(mean, output_tgt, log_var.exp())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)


def meanvar_transformer_eval(model, ds, criterion, batch_size, batch_first=True, **kwargs):
    model.eval()
    ds.eval()
    losses = []
    split_tgt = (lambda t: (t[:, :-1], t[:, 1:])) if batch_first else (lambda t: (t[:-1], t[1:]))
    end_index = len(ds)
    for index in tqdm(range(0, end_index - batch_size, batch_size)):
        temp = ds[index : index + batch_size]
        src, tgt = temp[:2]
        option_args = temp[2:]
        input_tgt, output_tgt = split_tgt(tgt)
        mean, log_var = model(src, input_tgt, *option_args)
        loss = criterion(mean, output_tgt, log_var.exp())
        losses.append(loss.item())
    return np.mean(losses)
