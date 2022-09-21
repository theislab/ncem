from ncem.tools.fit.constants import UNS_KEY_NCEM


def read_uns(adata, k):
    if UNS_KEY_NCEM not in adata.uns.keys():
        raise ValueError("could not read {k} from .uns, it seems that NCEM was not run on this adata instance")
    if k not in adata.uns[UNS_KEY_NCEM].keys():
        raise ValueError("could not find {k} in .uns[{UNS_KEY_NCEM}]")
    return adata.uns[UNS_KEY_NCEM][k]


def write_uns(adata, k, v):
    if UNS_KEY_NCEM not in adata.uns.keys():
        adata.uns[UNS_KEY_NCEM] = {}
    adata.uns[UNS_KEY_NCEM][k] = v
