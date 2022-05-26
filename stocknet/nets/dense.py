
import torch
import torch.nn as nn
import numpy as np
import pfrl

class SimpleDense(nn.Module):
    
    key = "dense"
    
    def __init__(self, layer_num, size, inputDim, n_actions, removeHistoryData = True, lr=True):
        super().__init__()
        self.args = (layer_num, size, inputDim, n_actions, removeHistoryData, lr)
        self.__lr = lr
        self.size = size
        self.rhd = removeHistoryData
        self.ActionHistoryDim = 2
        if removeHistoryData:
            input_dims = inputDim - self.ActionHistoryDim
        else:
            input_dims = inputDim
        self.layerDips = size * input_dims
        self.layers = nn.ModuleList()
        for i in range(0, layer_num):
            layer = nn.Linear(self.layerDips, self.layerDips)
            self.layers.append(layer)
        
        out_in_dims = self.layerDips
        if self.rhd:
            out_in_dims += self.ActionHistoryDim
        self.output_layer = nn.Linear( out_in_dims , n_actions)

    def forward(self, inputs):
        batch_size, feature_len,seq_len = inputs.shape[0], inputs.shape[1],inputs.shape[2]
        if self.rhd:
            feature_len = feature_len - self.ActionHistoryDim
            last_actions = inputs[:, -self.ActionHistoryDim:,-1] # [1, ActionHistoryDim] (ex.torch.Size([1, 3]))
        out = inputs[:, :feature_len, :]
        layerDips = feature_len * seq_len
        #assert layerDips == self.layerDips
        out = out.view(-1, layerDips)
        for layer in self.layers:
            out = torch.tanh(layer(out))
        if self.rhd:
            out = torch.cat((out, last_actions), dim=1)
        #out = torch.tanh()
        out = self.output_layer(out)
        if self.__lr:
            return pfrl.action_value.DiscreteActionValue(out)
        else:
            return out
        
class ConvDense16(nn.Module):
    
    key = "conv16"
    
    def __init__(self, size, channel=5, out_size=3,lr=True):
        super().__init__()
        self.args = (size, channel, out_size, lr)
        self.__lr = lr
        self.size = size
        dtype = torch.float32
        self.preprocess = nn.Sequential(
            nn.Conv1d(channel, 64, kernel_size=3, stride=1, padding=1, dtype=dtype),
            nn.Tanh(),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, dtype=dtype),
            nn.MaxPool1d(3, stride=2, padding=1),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, dtype=dtype),
            nn.Tanh(),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1, dtype=dtype),
            nn.Tanh(),
            nn.MaxPool1d(3, stride=2, padding=1),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1, dtype=dtype),
            nn.Tanh(),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1, dtype=dtype),
            nn.Tanh(),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1, dtype=dtype),
            nn.Tanh(),
            nn.MaxPool1d(3, stride=2, padding=1),
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1, dtype=dtype),
            nn.Tanh(),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1, dtype=dtype),
            nn.Tanh(),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1, dtype=dtype),
            nn.Tanh(),
            nn.MaxPool1d(3, stride=2, padding=1),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1, dtype=dtype),
            nn.Tanh(),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1, dtype=dtype),
            nn.Tanh(),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1, dtype=dtype),
            nn.Tanh()
        )
        #self.avgpool = nn.AdaptiveAvgPool1d(int(size/(2**5)))
        self.avgpool = nn.AdaptiveAvgPool1d(int(size/(2)))
        
        self.classifier = nn.Sequential(
            #nn.Linear(int(size/(2**5)) * 512, int(size/(2**8)) * 512),
            nn.Linear(int(size/(2)) * 512, int(size) * 512),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            #nn.Linear(int(size/(2**8)) * 512, 512),
            nn.Linear(int(size) * 512, 512),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(512, out_size)# 3 is action space
        )

    def forward(self, inputs):
        out = self.preprocess(inputs)
        out = self.avgpool(out)
        out = self.classifier(out.view(-1, int(self.size/(2)) * 512))
        if self.__lr:
            return pfrl.action_value.DiscreteActionValue(out)
        else:
            return out
        
