"""
Graph VAE module
g(A,X) from the paper
"""
import torch
from ncem.utils.init_weights import init_weights
import pytorch_lightning as pl
from torch_geometric.nn import VGAE, GCNConv


# taken from https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial6/Tutorial6.ipynb
class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        # cached only for transductive learning
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=False)
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=False)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class GraphVAE(pl.LightningModule):
    def __init__(self, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters(model_kwargs)
        self.model = VGAE(VariationalGCNEncoder(self.hparams['num_features'], self.hparams['latent_dim']))

        self.model.apply(init_weights)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GraphVAE")
        parser.add_argument("--lr", type=float, default=0.1, help="the initial learning rate")
        parser.add_argument("--weight_decay", type=float, default=1e-4, help="the weight decay")
        parser.add_argument("--latent_dim", type=int, default=30, help="Latent dim")
        return parent_parser

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.model(x, edge_index)
        return x

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(),
                                  lr=self.hparams["lr"],
                                  weight_decay=self.hparams["weight_decay"])
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', factor=0.2, patience=20, min_lr=5e-5)
        optim = {"optimizer": optim, "lr_scheduler": sch, "monitor": "train_loss"}
        return optim

    def training_step(self, batch, _):
        loss = torch.zeros(1, device=batch[0].x.device, requires_grad=True)
        for data in batch:
            z = self.model.encode(data.x, data.edge_index)
            recon_loss = self.model.recon_loss(z, data.edge_index)
            kl_loss = self.model.kl_loss()
            train_loss = recon_loss + (1.0 / data.num_nodes) * kl_loss

            loss = loss + train_loss

            self.log('train_recon_loss', recon_loss, batch_size=1)
            self.log('train_kl_loss', kl_loss, batch_size=1)
            self.log('train_loss', train_loss, batch_size=1)
        return loss

    def validation_step(self, batch, _):
        for data in batch:
            z = self.model.encode(data.x, data.edge_index)
            recon_loss = self.model.recon_loss(z, data.edge_index)
            kl_loss = self.model.kl_loss()
            loss = recon_loss + (1.0 / data.num_nodes) * kl_loss
            self.log('val_recon_loss', recon_loss, batch_size=1)
            self.log('val_kl_loss', kl_loss, batch_size=1)
            self.log('val_loss', loss, batch_size=1)

    def test_step(self, batch, _):
        for data in batch:
            z = self.model.encode(data.x, data.edge_index)
            recon_loss = self.model.recon_loss(z, data.edge_index)
            kl_loss = self.model.kl_loss()
            loss = recon_loss + (1.0 / data.num_nodes) * kl_loss

            self.log('train_recon_loss', recon_loss, batch_size=1)
            self.log('train_kl_loss', kl_loss, batch_size=1)
            self.log('train_loss', loss, batch_size=1)
