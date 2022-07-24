
import scanpy as sc
from sklearn.utils import shuffle
import squidpy as sq
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from dataset import HartmannWrapper
from torch_geometric.data import Data, LightningNodeData
from torch_geometric.transforms import RandomLinkSplit

import torch_sparse
import torch_geometric.nn as geom_nn
from torch.utils.data import DataLoader
from torch_geometric.loader import NeighborLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch.optim as optim
import argparse
import os

from gnn_gpu import NonLinearNCEM


def parse_args():
    """
    Program arguments
    """
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument("--data_path", default="./data/", help="data path")
    parser.add_argument("--dataset", type=str, default="hartmann", help="dataset to load", choices=["hartmann"])
    parser.add_argument("--init_model", default=None, help="initial model to load")
    parser.add_argument("--batch_size", type=int, default=128, help="train batch size")
    parser.add_argument("--num_workers", type=int, default=12, help="the num of training process")
    parser = NonLinearNCEM.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = argparse.Namespace(**group_dict)
    return args, arg_groups


def main():
    args, arg_groups = parse_args()

    # Checkpoint settings
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(os.path.dirname(__file__), 'checkpoints'),
        save_top_k=1,
        verbose=False,
        monitor="val_loss",
        mode="min",
    )
    # TODO: improve checkpointing stuff.
    # Create a PyTorch Lightning trainer with the generation callback

    # Choosing the dataset
    if args.dataset == "hartmann":
        dataset = HartmannWrapper(args.data_path)
        data = dataset[0]
        # TODO: For now use the first node of the dataset

        # TODO: Test step
        data.train_mask = torch.arange(data.num_nodes)
        transform = RandomLinkSplit(is_undirected=True, add_negative_train_samples=False)
        train_data, val_data, test_data = transform(data)

        n_input_features, n_output_features = train_data.num_features, train_data.y.shape[1]

    else:
        raise NotImplementedError()

    trainer: pl.Trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])

    train_dataloader = NeighborLoader(
        data=train_data,
        # input_nodes=train_data,
        num_neighbors=[120] * 2,
        batch_size=args.batch_size,
        num_workers=12,
        shuffle=True,
    )
    val_dataloader = NeighborLoader(
        data=val_data,
        # input_nodes=val_data.label_mask,
        num_neighbors=[120] * 2,
        batch_size=args.batch_size,
        num_workers=12,
        shuffle=False
    )

    model = NonLinearNCEM(in_channels=n_input_features, out_channels=n_output_features,
                          **vars(arg_groups["NonLinearNCEM"]))
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    main()
