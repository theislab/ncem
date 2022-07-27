"""
MLPModel module
Vanilla module for consecutive dense layers.
"""
import torch.nn as nn


class MLPModel(nn.Module):

    def __init__(self, in_channels=34, hidden_dims=None, out_channels=34, dp_rate=0.1):
        """
        Inputs:
            in_channels - Dimension of input features
            out_channels - Dimension of the output features.
            hidden_dims - List of hidden dimensions
            dp_rate - Dropout rate to apply throughout the network
        """
        super().__init__()
        layers = []

        prev_dim = in_channels
        if hidden_dims is not None:
            for dim in hidden_dims:
                layers += [
                    nn.Linear(prev_dim, dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dp_rate)
                ]
                prev_dim = dim

        layers += [nn.Linear(prev_dim, out_channels), nn.ReLU(inplace=True)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x, *args, **kwargs):
        """
        Inputs:
            x - Input features per node
        """
        return self.layers(x)
