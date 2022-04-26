import torch
import torch.nn as nn
from torch.optim import SGD

dtype = torch.float32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

##Added from FX NEXT MACD
class Predictor(nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim):
        super(Predictor, self).__init__()

        self.rnn = nn.LSTM(input_size = inputDim,
                            hidden_size = hiddenDim,
                            batch_first = True)
        self.rnn.to(device)
        self.output_layer = nn.Linear(hiddenDim, outputDim)
        self.output_layer.to(device)
    
    def forward(self, inputs, hidden0=None):
        batch_size, seq_len = inputs.shape[0], inputs.shape[1]
        output, (hidden, cell) = self.rnn(inputs, hidden0) #LSTM層
        output = self.output_layer(output[:, -1, :]) #全結合層

        return output