from torch import nn


class Perceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, device=None, **kwargs):
        super(Perceptron, self).__init__()

        self.args = {"input_dim": input_dim, "hidden_dim": hidden_dim, "output_dim": output_dim, "num_layers": num_layers}

        if num_layers > 1:
            layers = []
            layers.append(nn.Linear(input_dim, hidden_dim, device=device))
            layers.append(nn.ReLU())

            for i in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim, device=device))
                layers.append(nn.ReLU())

            layers.append(nn.Linear(hidden_dim, output_dim, device=device))
            self.layers = nn.Sequential(*layers)
        else:
            self.layers = nn.Linear(input_dim, output_dim, device=device)

    def forward(self, x):
        out = self.layers(x)
        return out
