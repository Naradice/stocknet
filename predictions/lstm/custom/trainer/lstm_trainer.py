import numpy as np
from tqdm import tqdm


def lstm_train(model, ds, optimizer, criterion, batch_size, **kwargs):
    model.train()
    ds.train()
    losses = []
    end_index = len(ds)
    for index in tqdm(range(0, end_index - batch_size, batch_size)):
        src, tgt = ds[index : index + batch_size][:2]
        pred = model(src)
        target = tgt[:, -1, :]
        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)


def lstm_eval(model, ds, criterion, batch_size, **kwargs):
    model.eval()
    ds.eval()
    losses = []
    end_index = len(ds)
    for index in tqdm(range(0, end_index - batch_size, batch_size)):
        src, tgt = ds[index : index + batch_size][:2]
        pred = model(src)
        target = tgt[:, -1, :]
        loss = criterion(pred, target)
        losses.append(loss.item())
    return np.mean(losses)
