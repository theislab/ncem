import anndata
import numpy as np
import pandas as pd


def vardecomp(
    adata: anndata.AnnData,
    key_type: str,
    library_key: Optional[str] = None,
):

    # replace by get obs_df function in scanpy
    df = pd.DataFrame(adata.X, columns=adata.var_names)
    df[key_type] = pd.Series(list(adata.obs[key_type]), dtype="category")
    if library_key:
        df[library_key] = pd.Series(list(adata.obs[library_key]), dtype="category")
        images = np.unique(df[library_key])


"""

variance_decomposition = []
with tqdm(total=len(images)) as pbar:
    for img in images:
        mean_img_genes = np.mean(df[df["image_col"] == img], axis=0)
        mean_img_global = np.mean(mean_img_genes)

        intra_ct_var = []
        inter_ct_var = []
        gene_var = []
        for ct in np.unique(df["cluster_col_preprocessed"]):
            img_celltype = np.array(df[(df["image_col"] == img) & (df["cluster_col_preprocessed"] == ct)])[
                :, :-2
            ]
            if img_celltype.shape[0] == 0:
                continue
            mean_image_celltype = np.mean(img_celltype, axis=0)

            for i in range(img_celltype.shape[0]):
                intra_ct_var.append((img_celltype[i, :] - mean_image_celltype) ** 2)
                inter_ct_var.append((mean_image_celltype - mean_img_genes) ** 2)
                gene_var.append((mean_img_genes - mean_img_global) ** 2)

        intra_ct_var = np.sum(intra_ct_var)
        inter_ct_var = np.sum(inter_ct_var)
        gene_var = np.sum(gene_var)
        variance_decomposition.append(np.array([img, intra_ct_var, inter_ct_var, gene_var]))
        pbar.update(1)
df = (
    pd.DataFrame(
        variance_decomposition, columns=["image_col", "intra_celltype_var", "inter_celltype_var", "gene_var"]
    )
    .astype(
        {
            "image_col": str,
            "intra_celltype_var": "float32",
            "inter_celltype_var": "float32",
            "gene_var": "float32",
        }
    )
    .set_index("image_col")
)

df["total"] = df.intra_celltype_var + df.inter_celltype_var + df.gene_var
df["intra cell type variance"] = df.intra_celltype_var / df.total
df["inter cell type variance"] = df.inter_celltype_var / df.total
df["gene variance"] = df.gene_var / df.total
"""
