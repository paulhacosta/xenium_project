#%%
import os 
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np 
#%%
visium_hd_folder = "Visium_HD_Human_Lung_Cancer_post_Xenium_Prime_5k_Experiment2"
data_path = f"/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics"
project_path = f"{data_path}/{visium_hd_folder}"
path002 = f"{project_path}/binned_outputs/square_002um"


# %%
adata = sc.read_h5ad(os.path.join(path002, "2um_lung.h5ad"))
bdata = sc.read_h5ad(os.path.join(path002, "cells_matched_spatial_refined_v2.h5ad"))

fdata = sc.read_h5ad(os.path.join(path002, "preprocessed", "002um", "filtered_feature_bc_matrix_preprocessed.h5ad"))

#%%
def visualize_bin2cell_data(bdata, show=False):
    """
    Visualize QC metrics for Bin2Cell data.
    """
    # QC Metric Distributions
    sc.pl.violin(
        bdata,
        ["n_genes_by_counts", "bin_count", "pct_counts_mt"],
        jitter=0.4,
        multi_panel=True,
        save="_bin2cell_qc_distributions.png"
    )
    if show:
        plt.show()

    # Scatter plot: Total counts vs Percentage mitochondrial counts
    plt.figure(figsize=(8, 6))
    plt.scatter(bdata.obs['total_counts'], bdata.obs['pct_counts_mt'], alpha=0.5)
    plt.axhline(20, c="r", linestyle="--", label="Mitochondrial % Threshold")
    plt.xlabel("Total Counts")
    plt.ylabel("Percentage Mitochondrial Counts")
    plt.title("Total Counts vs. Mitochondrial Counts")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Spatial distribution of pseudo-cells
    plt.figure(figsize=(8, 8))
    plt.scatter(
        bdata.obsm['spatial'][:, 0],
        bdata.obsm['spatial'][:, 1],
        c=bdata.obs['bin_count'],
        cmap='viridis',
        s=1,
        alpha=0.7
    )
    plt.xlabel("X Coordinate (Full Resolution)")
    plt.ylabel("Y Coordinate (Full Resolution)")
    plt.title("Spatial Distribution of Pseudo-Cells (Bin Counts)")
    plt.colorbar(label="Bin Count")
    plt.gca().invert_yaxis()  # Match image orientation
    plt.grid(False)
    plt.show()


# %% Preprocessing for B2C data 

# Extract spatial coordinates
spatial_coords = bdata.obsm['spatial']

# Add as new columns in obs
bdata.obs['pxl_row_in_fullres'] = spatial_coords[:, 1]  # Y-coordinate
bdata.obs['pxl_col_in_fullres'] = spatial_coords[:, 0]  # X-coordinate

# Identify mitochondrial genes (if using standard names)
bdata.var['mt'] = bdata.var_names.str.startswith('MT-')

# Calculate QC metrics
sc.pp.calculate_qc_metrics(bdata, qc_vars=['mt'], inplace=True)
bdata.obs["n_counts"] = bdata.obs["total_counts"]
visualize_bin2cell_data(bdata, show=True)

# QC metric quantiles
q10_counts = bdata.obs['bin_count'].quantile(0.10)
q10_genes = bdata.obs['n_genes_by_counts'].quantile(0.10)
q99_genes = bdata.obs['n_genes_by_counts'].quantile(0.99)
q95_mt = bdata.obs['pct_counts_mt'].quantile(0.95)

# Ensure reasonable thresholds
min_total_counts = max(int(q10_counts), 1)
min_genes = max(int(q10_genes), 1)
max_genes = int(q99_genes)
max_mt = min(20, int(np.ceil(q95_mt)))
min_bin_counts = 4

qc_params = {
    "min_total_counts": min_total_counts,
    "min_genes": min_genes,
    "max_genes": max_genes,
    "max_mt": max_mt,
    "min_bin_counts": min_bin_counts
    }

bdata = bdata[bdata.obs['total_counts'] >= qc_params['min_total_counts'], :]
bdata = bdata[(bdata.obs['n_genes_by_counts'] >= qc_params['min_genes']) & 
                (bdata.obs['n_genes_by_counts'] <= qc_params['max_genes']), :]
bdata = bdata[bdata.obs['pct_counts_mt'] <= qc_params['max_mt'], :]

# Apply min_bin_counts filter
bdata = bdata[bdata.obs['bin_count'] >= qc_params['min_bin_counts'], :]

# Normalize and log-transform
sc.pp.normalize_total(bdata, target_sum=1e4)
sc.pp.log1p(bdata)
bdata.var.rename(columns={"gene_ids": "ensembl_id"}, inplace=True)

# Convert numeric obs columns to float
for col in bdata.obs.select_dtypes(include=["int", "float"]).columns:
    bdata.obs[col] = bdata.obs[col].astype(np.float32)

# Convert numeric var columns to float
for col in bdata.var.select_dtypes(include=["int", "float", "bool"]).columns:
    bdata.var[col] = bdata.var[col].astype(np.float32)

# Also ensure the main matrix is float
bdata.X = bdata.X.astype(np.float32)



# bdata.write(os.path.join(path002, "preprocessed", "bin2cell", "preprocessed_bin2cell_matched.h5ad"))


# %%
