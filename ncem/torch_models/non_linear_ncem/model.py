"""
NonLinearNCEM module
"""
import torch
import torch.nn as nn
import torch_geometric.nn as geom_nn
import pytorch_lightning as pl
import torch.optim as optim
from sklearn.metrics import r2_score
from torch_models.modules.gnn_model import GNNModel
from torch_models.modules.mlp_model import MLPModel
from ncem.utils.init_weights import init_weights


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

        self.decoder_sigma = MLPModel(
            in_channels=self.hparams.latent_dim,
            out_channels=self.hparams.out_channels,
            hidden_dims=self.hparams.decoder_hidden_dims,
        )
        self.decoder_mu = MLPModel(
            in_channels=self.hparams.latent_dim,
            out_channels=self.hparams.out_channels,
            hidden_dims=self.hparams.decoder_hidden_dims,
        )

        self.loss_module = nn.GaussianNLLLoss(eps=1e-5)

        self.encoder.apply(init_weights)
        self.decoder_mu.apply(init_weights)
        self.decoder_sigma.apply(init_weights)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("NonLinearNCEM")
        parser.add_argument("--lr", type=float, default=0.1, help="the initial learning rate")
        parser.add_argument("--weight_decay", type=float, default=2e-3, help="the weight decay")
        parser.add_argument("--latent_dim", type=int, default=30, help="the weight decay")
        parser.add_argument("--decoder_hidden_dims", nargs='+', type=int,
                            default=None, help="Decoder Hidden Layer Dim")  #
        parser.add_argument("--encoder_hidden_dims", nargs='+', type=int,
                            default=None, help="Encoder Hidden Layer Dim")  #
        parser.add_argument("--dp_rate", type=float, default=0.1, help="Dropout Rate")
        return parent_parser

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.encoder(x, edge_index)
        sigma = self.decoder_sigma(x)
        mu = self.decoder_mu(x)
        return mu, sigma

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams["lr"],
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
        val_r2_score = r2_score(batch.y.cpu(), mu.cpu())
        self.log('val_r2_score', val_r2_score, batch_size=batch.batch_size, prog_bar=True)
        self.log('val_loss', val_loss, batch_size=batch.batch_size, prog_bar=True)

    def test_step(self, batch, _):
        mu, sigma = self.forward(batch)
        test_loss = self.loss_module(mu, batch.y, sigma)
        test_r2_score = r2_score(batch.y.cpu(), mu.cpu())
        self.log('test_r2_score', test_r2_score, batch_size=batch.batch_size, prog_bar=True)
        self.log('test_loss', test_loss, batch_size=batch.batch_size, prog_bar=True)
