import torch.nn as nn
from torch import tanh

class LSTM(nn.Module):
    
    key = "lstm"
    
    def __init__(self, inputDim, hiddenDim, outputDim, device, activation_for_output = tanh):
        self.args = (inputDim, hiddenDim, outputDim, device)
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(input_size = inputDim,
                            hidden_size = hiddenDim,
                            batch_first = True,
                            device=device)
        self.output_layer = nn.Linear(hiddenDim, outputDim, device=device)
        self.activation_for_output = activation_for_output
        self.device = device
    
    def forward(self, inputs, hidden0=None):
        #batch_size, seq_len = inputs.shape[0], inputs.shape[1]
        #print(f"batch_size {batch_size}", f"seq_len: {seq_len}")
        output, (hidden, cell) = self.rnn(inputs.to(self.device), hidden0)
        # output = self.output_layer(output)
        l_inpuuts = output[:, -1, :]
        output = self.output_layer(l_inpuuts.to(self.device))
        output = self.activation_for_output(output)

        return output