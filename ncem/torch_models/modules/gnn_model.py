"""
GNN module
contains consecutive layers
"""
import torch.nn as nn
import torch_geometric.nn as geom_nn


class GNNModel(nn.Module):
    def __init__(self, in_channels=11, hidden_dims=None, out_channels=34, **kwargs):
        """
        Inputs:
            in_channels - Dimension of input features
            out_channels - Dimension of the output features.
            hidden_dims - List of hidden dimensions
        """
        super().__init__()

        layers = []
        prev_dim = in_channels
        if hidden_dims is not None:
            for dim in hidden_dims:
                layers += [
                    geom_nn.GCNConv(
                        in_channels=prev_dim,
                        out_channels=dim),
                    nn.ReLU(inplace=True),
                    nn.Softmax(dim=1)
                ]
                prev_dim = dim

        layers += [
            geom_nn.GCNConv(in_channels=prev_dim,
                            out_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Softmax(dim=1)
        ]

        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        for lyr in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(lyr, geom_nn.MessagePassing):
                x = lyr(x, edge_index)
            else:
                x = lyr(x)
        return x
