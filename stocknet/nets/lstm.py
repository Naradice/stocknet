import torch.nn as nn
from torch import tanh


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, activation_for_output=tanh, batch_first=True, **kwargs):
        self.args = {"input_dim": input_dim, "hidden_dim": hidden_dim, "output_dim": output_dim}
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=batch_first, device=device)
        self.output_layer = nn.Linear(hidden_dim, output_dim, device=device)
        self.activation_for_output = activation_for_output
        self.device = device

    def forward(self, inputs, hidden0=None):
        output, (hidden, cell) = self.rnn(inputs.to(self.device), hidden0)
        l_inpuuts = output[:, -1, :]
        output = self.output_layer(l_inpuuts.to(self.device))
        output = self.activation_for_output(output)

        return output
