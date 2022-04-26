import numpy as np
import matplotlib.pyplot as plt
import torch

def validate(model, val_loader, device, loss_fn,index=0):
    mean_loss = 0
    out_ = np.array([])
    ans_ = np.array([])

    with torch.no_grad():
        correct = 0
        count = 0
        for values, ans in val_loader:
            outputs = model(values).to(device)
            ans = ans.to(device)
            loss = loss_fn(outputs, ans)
            mean_loss += loss.item()
            #output: [batchDim, outputDim]
            out_ = np.append(out_, outputs.to('cpu').detach().numpy().copy())
            ans_ = np.append(ans_, ans.to('cpu').detach().numpy().copy())
            count += 1
    print('--------------------------------------------------')
    print(f'man loss: {mean_loss/count}')
    print(f'mean dif ({index}): {(out_ - ans_).mean()}, var: {(out_ - ans_).var()}')
    print('--------------------------------------------------')
    return ans_, out_


def rl_validate():
    pass