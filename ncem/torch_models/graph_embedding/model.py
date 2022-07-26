"""
Graph VAE module
g(A,X) from the paper
"""
import torch
from ncem.utils.init_weights import init_weights
from ncem.torch_models.modules.graph_ae import GraphAE
import pytorch_lightning as pl
from torch_geometric.data import Batch


class GraphEmbedding(pl.LightningModule):
    def __init__(self, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters(model_kwargs)

        self.model = GraphAE(self.hparams['num_features'], self.hparams['num_features'], self.hparams['latent_dim'])

        self.loss_fn = torch.nn.MSELoss()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GraphEmbedding")
        parser.add_argument("--lr", type=float, default=0.1, help="the initial learning rate")
        parser.add_argument("--weight_decay", type=float, default=1e-4, help="the weight decay")
        parser.add_argument("--latent_dim", type=int, default=30, help="Latent dim")
        return parent_parser

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.model(x, edge_index)
        return x

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(),
                                 lr=self.hparams["lr"],
                                 weight_decay=self.hparams["weight_decay"])
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', factor=0.2, patience=20, min_lr=5e-5)
        optim = {"optimizer": optim, "lr_scheduler": sch, "monitor": "train_loss"}
        return optim

    def general_step(self, data_list, batch_idx, mode):
        batch_size = len(data_list)
        batch = Batch.from_data_list(data_list)

        x = self.forward(batch)
        recon_loss = self.loss_fn(x, batch.x)/batch_size
        return recon_loss, batch_size

    def training_step(self, data_list, batch_idx):
        loss, batch_size = self.general_step(data_list, batch_idx, "train")
        self.log('train_loss', loss, batch_size=batch_size)
        return loss

    def validation_step(self, data_list, batch_idx):
        loss, batch_size = self.general_step(data_list, batch_idx, "val")
        self.log('val_loss', loss, batch_size=batch_size, prog_bar=True)

    def test_step(self, data_list, batch_idx):
        loss, batch_size = self.general_step(data_list, batch_idx, "test")
        self.log('test_loss', loss, batch_size=batch_size, prog_bar=True)
