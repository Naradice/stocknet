import numpy as np
from tqdm import tqdm


def meanvar_lstm_train(model, ds, optimizer, criterion, batch_size, **kwargs):
    model.train()
    ds.train()
    losses = []
    end_index = len(ds)
    for index in tqdm(range(0, end_index - batch_size, batch_size)):
        src, tgt = ds[index : index + batch_size][:2]
        mean, log_var = model(src)
        target = tgt[:, -1, :]
        loss = criterion(mean, target, log_var.exp())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)


def meanvar_lstm_eval(model, ds, criterion, batch_size, **kwargs):
    model.eval()
    ds.eval()
    losses = []
    end_index = len(ds)
    for index in tqdm(range(0, end_index - batch_size, batch_size)):
        src, tgt = ds[index : index + batch_size][:2]
        mean, log_var = model(src)
        target = tgt[:, -1, :]
        loss = criterion(mean, target, log_var.exp())
        losses.append(loss.item())
    return np.mean(losses)
