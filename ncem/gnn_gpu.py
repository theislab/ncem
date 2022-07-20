
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
from sklearn.metrics import r2_score

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
                    nn.Softmax()
                ]
                prev_dim = dim

        layers += [geom_nn.GCNConv(in_channels=prev_dim,
                                   out_channels=out_channels)]

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


class NonLinearNCEM(pl.LightningModule):
    def __init__(self, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters(model_kwargs)

        self.encoder = GNNModel(
            in_channels=self.hparams.in_channels,
            hidden_dims=self.hparams.encoder_hidden_dims,
            out_channels=self.hparams.latent_dim,
        )

        self.decoder = MLPModel(
            in_channels=self.hparams.latent_dim,
            out_channels=self.hparams.out_channels * 2,  # one for means one for vars
            hidden_dims=self.hparams.decoder_hidden_dims,
        )

        self.loss_module = nn.GaussianNLLLoss(eps=1e-5)

        def init_weights(m):
            if isinstance(m, geom_nn.GCNConv): 
                # TODO: how to init weights of GNN's?
                pass
                # torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                # m.bias.data.fill_(0.01)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                m.bias.data.fill_(0.01)

        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("NonLinearNCEM")
        parser.add_argument("--lr", type=float, default=0.1, help="the initial learning rate")
        parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
        parser.add_argument("--weight-decay", type=float, default=2e-3, help="the weight decay")
        parser.add_argument("--latent_dim", type=int, default=30, help="the weight decay")
        parser.add_argument("--decoder_hidden_dims", action='append',
                            default=None, help="Decoder Hidden Layer Dim")  # TODO Test
        parser.add_argument("--encoder_hidden_dims", action='append',
                            default=None, help="Encoder Hidden Layer Dim")  # TODO Test
        parser.add_argument("--dp_rate", type=float, default=0.1, help="Dropout Rate")
        return parent_parser

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.encoder(x, edge_index)
        x = self.decoder(x)
        latent_dim = x.shape[-1]
        assert latent_dim == self.hparams.out_channels * 2
        mu, sigma = x[:, :latent_dim // 2], x[:, latent_dim // 2:]
        sigma = sigma + 1 # for numeric stability, TODO: is this OK?
        return mu, sigma

    def configure_optimizers(self):
        # We use SGD here, but Adam works as well
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.hparams["lr"],
            momentum=self.hparams["momentum"],
            weight_decay=self.hparams["weight_decay"],
        )
        return optimizer

    def training_step(self, batch, _):
        mu, sigma = self.forward(batch)
        loss = self.loss_module(mu, batch.y, sigma)
        self.log('train_loss', loss, batch_size=batch.batch_size)
        return loss

    def validation_step(self, batch, _):
        mu, sigma = self.forward(batch)
        val_loss = self.loss_module(mu, batch.y, sigma)
        val_r2_score = r2_score(batch.y, mu)
        self.log('val_r2_score', val_r2_score, batch_size=batch.batch_size, prog_bar=True)
        self.log('val_loss', val_loss, batch_size=batch.batch_size, prog_bar=True)

    def test_step(self, batch, _):
        mu, sigma = self.forward(batch)
        self.log('test_loss', loss, batch_size=batch.batch_size)
