import os.path as osp
import pathlib
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import RandomLinkSplit

from ncem.interpretation.interpreter import InterpreterInteraction


class Hartmann(Dataset):
    """ Wrapper class of mibitof to pytorch_geometric Dataset class.

    Args:
        root (str): Data path
    """

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.img_count = 58
        self.root = pathlib.Path(root)
        super().__init__(root, transform, pre_transform, pre_filter)

    # raw file name
    @property
    def raw_file_names(self):
        return [self.root / "scMEP_MIBI_singlecell" / "scMEP_MIBI_singlecell.csv"]

    @property
    def processed_file_names(self):
        return [f"data_{idx}.pt" for idx in range(self.img_count)]

    def process(self):
        # Read from already implemented DataLoader to load to np/cpu
        ip = InterpreterInteraction()

        # Read data from `raw_path`.
        ip.get_data(
            data_origin="hartmann",
            data_path=self.root,
            radius=35,
            node_label_space_id="type",
            node_feature_space_id="standard",
        )

        for idx, k in enumerate(ip.a.keys()):

            a, h_0, h_1, domains, node_covar, sf = (
                ip.a[k],
                ip.h_0[k],
                ip.h_1[k],
                ip.domains[k],
                ip.node_covar[k],
                ip.size_factors[k],
            )

            # we are not going to use it
            del node_covar

            # Edge list (2,n_edges)
            row, col = a.nonzero()
            edge_index = torch.LongTensor([row.tolist(), col.tolist()])
            # X_c from the paper (n_domains=58)
            # one hot vector
            g = torch.zeros(ip.n_domains)
            g[domains] = 1
            # Note: Not used since dataloader encodes the batch id

            # X_l from paper (n_node, n_celltypes=8) n_celltypes: count of distinct cell-type labels
            h_0 = torch.from_numpy(h_0).to(torch.float32)

            # size factor
            sf = torch.from_numpy(sf).to(torch.float32)

            # Y from paper (n_node, n_genes)
            h_1 = torch.from_numpy(h_1).to(torch.float32)

            #choose n=10 random nodes to evaluate per image 
            #mask=torch.zeros(h_0.shape[0])
            #mask[:10]=torch.ones(10)
            #idx = torch.randperm(mask.shape[0])
            #mask=mask[idx]


            # x for pygeometric convention is node to features
            # x = torch.hstack((h_0, h_1)).to(torch.float32)

            # creating data object
            data = Data(
                edge_index=edge_index,
                scale_factor=sf,
                cell_type=h_0,
                gene_expression=h_1,
                domain=g,
                transform_done=False,
                num_nodes=h_1.shape[0]
            )
            # saving it as file
            torch.save(data, osp.join(self.processed_dir, f"data_{idx}.pt"))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f"data_{idx}.pt"))
        if self.transform is not None:
            data = self.transform(data)
        return data
