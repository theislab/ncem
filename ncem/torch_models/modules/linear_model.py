"""
Linear baseline model.
"""
import torch.nn as nn
import torch
import torch_geometric.nn as geom_nn
from torch_geometric.utils import to_dense_adj

#TODO: generalize in_channels, out_channels to work for any dataset 

class LinearNonspatial(nn.Module):

    def __init__(self, in_channels=8, out_channels=36):
        """
        Inputs:
            in_channels - Dimension of input features
            out_channels - Dimension of the output features.
        """
        super().__init__()

        self.linear=nn.Linear(in_channels, out_channels)


    def forward(self, x):
        """
        Inputs:
            x - Input features per node
        """

        return self.linear(x)

class LinearSpatial(nn.Module):

    def __init__(self, in_channels=8, out_channels=36):
        """
        Inputs:
            in_channels - Dimension of input features
            out_channels - Dimension of the output features.
        """
        super().__init__()
        
        
        #self.linear=nn.Linear(in_channels, out_channels)

        self.linear=geom_nn.GCNConv(in_channels, out_channels)


    def forward(self, x, edge_index):
        """
        Inputs:
            x - Input features per node
            edge_index - input edge indices. Shape 2 x 2*(No. edges)
        """

        #TODO: Spatial model as done in paper. Outer product implies in_channels dependent on N 
        #adj_matrix=to_dense_adj(edge_index) #NxN x: NxL
        #Xs=adj_matrix*x
        #mask=adj_matrix*x>0
        #Xs=Xs[mask] #NxL
        #Xts=torch.outer(x,Xs) #NxL^2 or NxN ?
        #Xd=torch.cat(x,Xts) #Nx(L+L^2)?
        #out=self.linear(Xd)

        return self.linear(x,edge_index)