#%% 
import os 
import h5py
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
#%%
data_path = "/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Xenium_Prime_Human_Lung_Cancer_FFPE_outs"

file_path = os.path.join(data_path, "cell_feature_matrix.h5")

# Open the file
with h5py.File(file_path, "r") as h5_file:
    # Recursively print all groups and datasets in the file
    def print_structure(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name} (Shape: {obj.shape}, Dtype: {obj.dtype})")
    
    
    h5_file.visititems(print_structure)


#%% Load data
use_rank_encoding = True
filter_hvg = False    # filter highly variable genes

adata = sc.read_10x_h5(file_path)
adata.var_names_make_unique()

# Annotate mitochondrial genes
adata.var['mt'] = adata.var_names.str.startswith('MT-')  # Use 'mt-' for mouse data
# Calculate QC metrics
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)

# Basic preprocessing
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# Normalize the data
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Optional: Identify highly variable genes
if filter_hvg:
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable].copy()

# Scale the data

if use_rank_encoding:
    # If your data is in a sparse matrix format
    # Convert SparseCSRView to a concrete sparse CSR matrix
    if isinstance(adata.X, sp.spmatrix):
        adata.X = adata.X.copy()
        print("Converted SparseCSRView to a concrete sparse CSR matrix.")

    # Convert the sparse matrix to a dense array
    adata.X = adata.X.toarray()
    print("Converted sparse matrix to a dense NumPy array.")

    print(f"Type of adata.X: {type(adata.X)}")
    print(f"Shape of adata.X: {adata.X.shape}")
    print(f"Dtype of adata.X: {adata.X.dtype}")

    # Convert adata.X to a DataFrame for easier manipulation
    df_expr = pd.DataFrame(
        adata.X,
        index=adata.obs_names,
        columns=adata.var_names
    )

    # Compute ranks across genes for each cell
    df_ranks = df_expr.rank(axis=1, method='average')

    # Replace the expression matrix with ranked values
    adata.X = df_ranks.values
else:
    sc.pp.scale(adata, max_value=10)

#%% View distributions
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts'], jitter=0.4)
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')

#%% Dim reduction and visualizing
sc.tl.pca(adata, svd_solver='arpack')

sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata)
sc.pl.umap(adata, color=['leiden'])

#%%
# Specify the filename and path where you want to save the file
if filter_hvg:
    output_file = os.path.join(data_path, 'processed_xenium_data.h5ad') 
else: 
    output_file = os.path.join(data_path, 'processed_xenium_data-all_genes.h5ad') 


# Save the AnnData object
adata.write(output_file)
#%%

import h5py

file_path = "/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Visium_HD_Human_Lung_Cancer_post_Xenium_v1_Experiment1/Visium_HD_Human_Lung_Cancer_post_Xenium_v1_Experiment1_feature_slice.h5"

with h5py.File(file_path, "r") as f:
    print(list(f.keys()))  # Lists the top-level groups in the file

with h5py.File(file_path, "r") as f:
    for group in f.keys():
        print(f"Group: {group}")
        print(list(f[group].keys()))  # List datasets within the group
#%%
import scanpy as sc
data_path = "/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Visium_HD_Human_Lung_Cancer_post_Xenium_v1_Experiment1"

adata = sc.read_10x_h5(f"{data_path}/Visium_HD_Human_Lung_Cancer_post_Xenium_v1_Experiment1_feature_slice.h5", genome="features")
print(adata)
