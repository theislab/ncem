
import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch_data.datasets.hartmann import Hartmann
from torch_models.non_linear_ncem import NonLinearNCEM
from torch_geometric import loader
from torch_geometric.data import Dataset, Data
from torch.utils.data import DataLoader
import numpy as np


def parse_args():
    """
    Program arguments
    """
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument("--data_path", default="../example_data/hartmann", help="data path")
    parser.add_argument("--dataset", type=str, default="hartmann", help="dataset to load", choices=["hartmann"])
    parser.add_argument("--init_model", default=None, help="initial model to load")
    parser.add_argument("--batch_size", type=int, default=1, help="train batch size")
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
    # TODO: improve check-pointing stuff.
    # Create a PyTorch Lightning trainer with the generation callback

    # Choosing the dataset
    if args.dataset == "hartmann":

        # A transform function just for this use case
        def transform_hartmann_ncem(data: Data):
            if data.transform_done:
                return data
            return Data(
                x=data.cell_type,
                y=data.gene_expression,
                edge_index=data.edge_index,
                transform_done=True,
                num_nodes=data.cell_type.shape[0],
            )

        dataset = Hartmann(args.data_path, transform=transform_hartmann_ncem)
        # TODO: Do test
        # TODO: Distribute better
        idxs = np.arange(len(dataset))
        np.random.shuffle(idxs)
        split = int(0.8 * len(dataset))
        train_dataset, val_dataset = dataset[:split], dataset[split:]

        train_dataloader = loader.DataListLoader(
            train_dataset,
            num_workers=args.num_workers,
            shuffle=True,
            batch_size=args.batch_size
        )
        val_dataloader = loader.DataListLoader(
            val_dataset,
            num_workers=args.num_workers,
            batch_size=args.batch_size
        )

        n_input_features, n_output_features = train_dataset[0].num_features, train_dataset[0].y.shape[1]

    else:
        raise NotImplementedError()

    trainer: pl.Trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])

    model = NonLinearNCEM(
        in_channels=n_input_features,
        out_channels=n_output_features,
        **vars(arg_groups["NonLinearNCEM"]))

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    main()
