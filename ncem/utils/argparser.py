
import os
import argparse
import pytorch_lightning as pl


def parse_args(model_cls):
    """
    Program arguments
    """
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument("--data_path", default=os.path.join("..", "example_data", "hartmann"), help="data path")
    parser.add_argument("--dataset", type=str, default="hartmann", help="dataset to load", choices=["hartmann"])
    parser.add_argument("--init_model", default=None, help="initial model to load")
    parser.add_argument("--batch_size", type=int, default=1, help="train batch size")
    parser.add_argument("--num_workers", type=int, default=12, help="the num of training process")
    parser = model_cls.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = argparse.Namespace(**group_dict)
    return args, arg_groups
