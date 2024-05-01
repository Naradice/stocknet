import torch
from torch import nn


class NumericalLoss(nn.Module):
    def __init__(self, label_size, device=None, **kwargs):
        super(NumericalLoss, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.indices = torch.arange(1, label_size + 1, device=device)

    def forward(self, pred, target_labels):
        logits = self.softmax(pred)
        e = torch.sum(torch.mul(logits, self.indices), dim=-1)
        e_diff = torch.abs(torch.sub(e, target_labels + 1))
        std_diff = torch.div(e_diff, target_labels + 1)
        return torch.mean(std_diff)
