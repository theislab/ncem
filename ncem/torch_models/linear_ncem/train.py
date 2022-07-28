import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from model import LinearNCEM
from torch_geometric import loader
from torch_geometric.data import Data
import numpy as np
from ncem.utils.argparser import parse_args
from ncem.torch_data.datasets.hartmann import Hartmann


def main():
    args, arg_groups = parse_args(LinearNCEM)

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
                sf=data.scale_factor,
                edge_index=data.edge_index,
                transform_done=True,
                num_nodes=data.cell_type.shape[0],
            )

        dataset = Hartmann(args.data_path, transform=transform_hartmann_ncem)
        # TODO: Do test
        # TODO: Distribute better (add 2nd option: eg train on 10 nodes per graph for each batch)
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

    model = LinearNCEM(
        in_channels=n_input_features,
        out_channels=n_output_features,
        **vars(arg_groups["LinearNCEM"]))
    
    trainer: pl.Trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    main()
