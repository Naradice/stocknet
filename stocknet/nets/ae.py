import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_activation_func(func_type: str):
    if func_type == "Tanh()":
        return nn.Tanh()
    else:
        print("please create...")


class AELinearModel(nn.Module):
    key = "ae"

    def default_setup(self, input_size):
        i_size = input_size
        digits = 0
        while i_size > 9:
            i_size = i_size / 10
            digits += 1
        i_size = input_size
        o_size = round(input_size / (2 + digits - 1))
        self.middle_out_index = digits
        for d in range(1, digits + 1):
            layer = nn.Linear(i_size, o_size).to(self.device)
            i_size = o_size
            o_size = round(i_size / (2 + digits - d - 1))
            self.layers.append(layer)
        temp_layers = self.layers
        for index in range(len(temp_layers) - 1, -1, -1):
            i_size = temp_layers[index].out_features
            o_size = temp_layers[index].in_features
            layer = nn.Linear(i_size, o_size).to(self.device)
            self.layers.append(layer)
        self.layers.to(self.device)

    def create_layers(self, input_size, layer_num, middle_size):
        if input_size < middle_size:
            raise Exception("middle layer size is lager than input size")
        if layer_num % 2 == 0:
            half_hidden_layer_size = layer_num / 2
            self.middle_out_index = half_hidden_layer_size
            even = True
        else:
            half_hidden_layer_size = math.ceil(layer_num / 2)
            self.middle_out_index = half_hidden_layer_size
            even = False

        step_reduce_size = (input_size - middle_size) / half_hidden_layer_size
        self.middle_out_index

        half_layer_sizes = [input_size - math.floor(i * step_reduce_size) for i in range(0, half_hidden_layer_size)]
        layer_sizes = half_layer_sizes.copy()
        layer_sizes.append(middle_size)
        if even:
            layer_sizes.append(middle_size)
        for value in reversed(half_layer_sizes):
            layer_sizes.append(value)

        layer_input_size = input_size
        for layer_out_size in layer_sizes[1:]:
            layer = nn.Linear(layer_input_size, layer_out_size).to(self.device)
            layer_input_size = layer_out_size
            self.layers.append(layer)

    def __init__(self, input_size, hidden_layer_num=-1, middle_layer_size=-1, activation_func=nn.Tanh(), out_middle_layer=False, device=None):
        super().__init__()
        if type(activation_func) == str:
            activation_func = initialize_activation_func(activation_func)
        self.args = (input_size, hidden_layer_num, middle_layer_size, str(activation_func), out_middle_layer, device)
        if device == None:
            self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.activation = activation_func.to(self.device)
        self.layers = nn.ModuleList().to(self.device)
        self.out_middle_layer = out_middle_layer
        if hidden_layer_num == -1 and middle_layer_size == -1:
            self.default_setup(input_size=input_size)
        else:
            if hidden_layer_num > 1:
                if middle_layer_size == -1:
                    # create layers with hidden_layer_num and default middle_layer_size
                    raise Exception("Need implementation.")
                elif middle_layer_size > 1:
                    # create layers with hidden_layer_num and middle_layer_size
                    self.create_layers(input_size=input_size, layer_num=hidden_layer_num, middle_size=middle_layer_size)
                else:
                    raise Exception("middle_layer_size should be grater than 2.")
            elif hidden_layer_num == -1:
                if middle_layer_size > 1:
                    # create layers with default hidden_layer_num and middle_layer_size
                    raise Exception("Need implementation.")
                else:
                    raise Exception("middle_layer_size should be greater than 2.")
            else:
                raise Exception("hidden_layer_num should be grater than 2.")

    def forward(self, x):
        out = x.to(self.device)
        index = 1
        for layer in self.layers:
            layer = layer.to(self.device)
            out = self.activation(layer(out))
            if index == self.middle_out_index:
                self.middle_layer_output = out
            index += 1
        if self.out_middle_layer == True:
            return out, self.out_middle_layer
        else:
            return out
