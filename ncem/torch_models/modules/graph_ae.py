"""
Graph VAE module
g(A,X) from the paper
"""
import torch
from torch_geometric.nn import SAGEConv


# taken from https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial6/Tutorial6.ipynb
class GraphAE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim):
        super(GraphAE, self).__init__()
        self.conv1 = SAGEConv(in_channels, latent_dim * 2, root_weight=False, normalize=True)
        self.conv2 = SAGEConv(latent_dim * 2, latent_dim, root_weight=False, normalize=True)


        self.conv3 = SAGEConv(latent_dim, latent_dim * 2, root_weight=False, normalize=True)
        self.conv4 = SAGEConv(latent_dim * 2, out_channels, root_weight=False, normalize=False)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.conv4(x, edge_index)
        x = self.softmax(x)
        return x
