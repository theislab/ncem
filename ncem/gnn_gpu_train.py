from gnn_gpu import NodeLevelGNN, train_node_classifier, print_results


import scanpy as sc
import squidpy as sq
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch_geometric.data import Data, LightningNodeData
import torch_sparse
import torch_geometric.nn as geom_nn
from torch_geometric.loader import NeighborLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch.optim as optim
import os

def get_data():
    #Load dataset 
    adata = sq.datasets.imc()

    #Get adjacency matrix
    sq.gr.spatial_neighbors(adata, coord_type="generic")
    A=adata.obsp['spatial_connectivities']

    #Get features of nodes 
    X=adata.obs
    X=pd.get_dummies(X)
    X=X.to_numpy()
    X=torch.FloatTensor(X)

    #Get labels of nodes
    Y=adata.X
    Y=torch.FloatTensor(Y)

    Acoo = A.tocoo()
    A_sparse = torch.sparse.FloatTensor(torch.LongTensor([Acoo.row.tolist(), Acoo.col.tolist()]),
                                torch.FloatTensor(Acoo.data)).coalesce()

    data = Data(x=X,edge_index=A_sparse.indices(), y=Y, train_mask=torch.arange(A.shape[0]), val_mask=torch.arange(A.shape[0]))

    return data

def main():
    cwd=os.getcwd()
    data=get_data()
    model, result = train_node_classifier(cur_dir=cwd, model_name="nonlinear NCEM",data=data, c_hidden=34,num_layers=2,dp_rate=0.1)

    return print(result)

if __name__ == "__main__":
    main()