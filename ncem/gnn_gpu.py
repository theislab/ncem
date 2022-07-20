
import scanpy as sc
import squidpy as sq
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch_geometric.data import Dataset, Data, HeteroData, LightningNodeData
import torch_sparse
import torch_geometric.nn as geom_nn
from torch_geometric.loader import NeighborLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch.optim as optim
import os

class GNNModel(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, num_layers=2, layer_name="GCN", dp_rate=0.1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        gnn_layer = geom_nn.GCNConv

        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers-1):
            layers += [
                gnn_layer(in_channels=in_channels,
                          out_channels=out_channels,
                          **kwargs),
                nn.ReLU(inplace=True),
                nn.Softmax()
            ]
            in_channels = c_hidden
        layers += [gnn_layer(in_channels=in_channels,
                             out_channels=c_out,
                             **kwargs)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        for l in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index)
            else:
                x = l(x)
        return x

class MLPModel(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, num_layers=2, dp_rate=0.1):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of hidden layers
            dp_rate - Dropout rate to apply throughout the network
        """
        super().__init__()
        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers-1):
            layers += [
                nn.Linear(in_channels, out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = c_hidden
        layers += [nn.Linear(in_channels, c_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x, *args, **kwargs):
        """
        Inputs:
            x - Input features per node
        """
        return self.layers(x)

class NodeLevelGNN(pl.LightningModule):

    def __init__(self, model_name, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()

        if model_name=="nonlinear NCEM":
            encoder= GNNModel(**model_kwargs)
            decoder= MLPModel(**model_kwargs)

        self.encoder = encoder
        self.decoder = decoder
        self.loss_module = nn.GaussianNLLLoss()

    def forward(self, data, mode="train"):
        x, edge_index = data.x, data.edge_index
        x = self.encoder(x, edge_index)
        x = self.decoder(x)

        ''' # Only calculate the loss on the nodes corresponding to the mask
        if mode == "train":
            mask = data.train_mask
        elif mode == "val":
            mask = data.val_mask
        elif mode == "test":
            mask = data.test_mask
        else:
            assert False, f"Unknown forward mode: {mode}"'''


        var=torch.ones(x.shape[0], x.shape[1])
        loss = self.loss_module(x, data.y, var)
        return loss

    def configure_optimizers(self):
        # We use SGD here, but Adam works as well
        optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch, mode="train")
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss= self.forward(batch, mode="val")
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self.forward(batch, mode="test")
        self.log('test_loss', loss)

def train_node_classifier(cur_dir, model_name, data, **model_kwargs):
    pl.seed_everything(42)
    node_data_loader = LightningNodeData(data, data.train_mask, loader='neighbor',
                                   num_neighbors=[30] * 2, batch_size=128,
                                   num_workers=8)
    strategy = pl.plugins.DDPSpawnPlugin(find_unused_parameters=False)
    '''NeighborLoader(
    data,
    # Sample 30 neighbors for each node for 2 iterations
    num_neighbors=[30] * 2,
    # Use a batch size of 128 for sampling training nodes
    batch_size=128)'''

    CHECKPOINT_PATH = cur_dir + "checkpoints"

    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(strategy=strategy, default_root_dir=root_dir, callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss")],
                         max_epochs=200,
                         progress_bar_refresh_rate=0) # 0 because epoch size is 1


    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"{model_name}.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = NodeLevelGNN.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything()
        model = NodeLevelGNN(model_name=model_name, c_in=data.x.shape[1], c_out=data.y.shape[1], **model_kwargs)
        trainer.fit(model, node_data_loader)
        model = NodeLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on the test set
    test_result = trainer.test(model, node_data_loader, verbose=False)
    batch = next(iter(node_data_loader))
    batch = batch.to(model.device)
    train_loss = model.forward(batch, mode="train")
    val_loss = model.forward(batch, mode="val")
    result = {"train": train_loss,
              "val": val_loss,
              "test": test_result['test_loss']}
    return model, result

def print_results(result_dict):
    if "train" in result_dict:
        print(f"Train loss: {result_dict['train']}")
    if "val" in result_dict:
        print(f"Val loss:   {result_dict['val']}")
    print(f"Test loss:  {result_dict['test']}")




