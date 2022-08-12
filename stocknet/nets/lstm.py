import torch
import torch.nn as nn
from torch.optim import SGD

##Added from FX NEXT MACD
class Predictor(nn.Module):
    
    key = "lstm"
    
    def __init__(self, inputDim, hiddenDim, outputDim, device):
        self.args = (inputDim, hiddenDim, outputDim, device)
        super(Predictor, self).__init__()

        self.rnn = nn.LSTM(input_size = inputDim,
                            hidden_size = hiddenDim,
                            batch_first = True)
        self.rnn.to(device)
        self.output_layer = nn.Linear(hiddenDim, outputDim)
        self.output_layer.to(device)
    
    def forward(self, inputs, hidden0=None):
        batch_size, seq_len = inputs.shape[0], inputs.shape[1]
        #print(f"batch_size {batch_size}", f"seq_len: {seq_len}")
        output, (hidden, cell) = self.rnn(inputs, hidden0) #LSTM層
        #print(output.shape)
        #output = self.output_layer(output) #全結合層
        output = self.output_layer(output[:, -1, :]) #全結合層

        return output