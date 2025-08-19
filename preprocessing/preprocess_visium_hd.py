#%%
import scanpy as sc
import yaml
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.signal import find_peaks
import argparse

#%%


def load_visium_hd_data(bin_path):
    """
    Load bin data 
    """
    
    filtered_matrix_path = os.path.join(bin_path, 'filtered_feature_bc_matrix.h5')
    # Load data similar to standard Visium
    adata = sc.read_10x_h5(filtered_matrix_path)
    adata.var_names_make_unique()

    # Identify mitochondrial genes
    adata.var['mt'] = adata.var_names.str.startswith('MT-')

    # Load spatial coords if provided (Visium HD may use different spot layouts)
    # Assuming tissue_positions_list or equivalent file:
    tissue_positions_path = os.path.join(bin_path, 'spatial', 'tissue_positions.parquet')
    spatial_coords = pd.read_parquet(tissue_positions_path)
    spatial_coords.set_index("barcode", inplace=True)

    adata.obs = adata.obs.join(spatial_coords, how='left')
    adata.obs['in_tissue'] = adata.obs['in_tissue'].astype(str)
    adata = adata[adata.obs['in_tissue'] == '1', :]

    return adata

def load_bin2cell_hd_data(bin_path):
    adata_path = os.path.join(bin_path, "cells_matched_spatial_refined_v2.h5ad")
    adata = sc.read_h5ad(adata_path)
    return adata

def suggest_qc_params_for_bin_size(adata, bin_size):
    """
    Suggest QC parameters based on bin size.
    These heuristics are chosen as starting points and may be refined:
    - Smaller bin sizes have fewer transcripts, so lower min_total_counts.
    - Use quantiles to adapt upper thresholds dynamically.
    """

    # Calculate QC metrics
    if bin_size == "b2c":
        sc.pp.calculate_qc_metrics(adata, inplace=True)

    else:
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)

   # Compute quantiles
    q10_counts = adata.obs['total_counts'].quantile(0.10)
    q10_genes = adata.obs['n_genes_by_counts'].quantile(0.10)
    q99_genes = adata.obs['n_genes_by_counts'].quantile(0.99)
    if bin_size != "b2c":
        q95_mt = adata.obs['pct_counts_mt'].quantile(0.95)

    # Ensure thresholds are at least reasonable (no negative or zero if data extremely sparse)
    min_total_counts = max(int(q10_counts), 1) 
    min_genes = max(int(q10_genes), 1)
    max_genes = int(q99_genes)

    if bin_size != "b2c":
        max_mt = int(np.ceil(q95_mt))
        # Check mitochondrial fraction cutoff
        # If most spots have very low MT%, consider lowering cutoff. If very high, set a reasonable max.
        if max_mt > 20:
            max_mt = 20
    else: 
        max_mt = 0
    

    # Return suggested parameters
    return {
        "min_total_counts": min_total_counts,
        "min_genes": min_genes,
        "max_genes": int(max_genes),
        "max_mt": int(max_mt)
    }

def visualize_qc_metrics_hd(project_path, bin_size, show=False):
    assert bin_size in ["002", "008", "016", "b2c"], "bin_size does not matcha available bin sizes"

    if bin_size == "b2c":
        bin_path =  os.path.join(project_path, "binned_outputs", f"square_002um")
        adata = load_bin2cell_hd_data(bin_path)
        sc.pp.calculate_qc_metrics(adata, inplace=True) # no longer contins 'mt'
    else:
        bin_path =  os.path.join(project_path, "binned_outputs", f"square_{bin_size}um")
        adata = load_visium_hd_data(bin_path)
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)


    results_dir = os.path.join(bin_path, "preprocessed")
    os.makedirs(results_dir, exist_ok=True)
    sc.settings.figdir = results_dir
    
    if bin_size == "b2c":
        qc_list = ["n_genes_by_counts", "total_counts"]
    else:
        qc_list = ["n_genes_by_counts", "total_counts", "pct_counts_mt"]
    
    
    # Plot QC distributions
    sc.pl.violin(
        adata, 
        qc_list,
        jitter=0.4,
        multi_panel=True,
        save="_hd_distributions.png"
    )
    if show:
        plt.show()

    # Suggest parameters dynamically
    qc_params = suggest_qc_params_for_bin_size(adata, bin_size)

    # Print suggested parameters
    print("Suggested QC Parameters:")
    for k,v in qc_params.items():
        print(f"{k}: {v}")

    print("\nNumber of spots in tissue:", adata.obs.shape[0])
    print("Number of genes:", adata.var.shape[0])

    # Total counts
    print("Total counts range:", adata.obs["total_counts"].min(), "-", adata.obs["total_counts"].max())
    print("Mean total counts:", adata.obs["total_counts"].mean())
    print("Median total counts:", adata.obs["total_counts"].median())

    # Number of genes by counts
    print("Number of genes by counts range:", adata.obs["n_genes_by_counts"].min(), "-", adata.obs["n_genes_by_counts"].max())
    print("Mean number of genes by counts:", adata.obs["n_genes_by_counts"].mean())
    print("Median number of genes by counts:", adata.obs["n_genes_by_counts"].median())

    if bin_size != "b2c":
        # Percentage of mitochondrial counts
        print("Percentage of mitochondrial counts range:", adata.obs["pct_counts_mt"].min(), "-", adata.obs["pct_counts_mt"].max())
        print("Mean percentage of mitochondrial counts:", adata.obs["pct_counts_mt"].mean())
        print("Median percentage of mitochondrial counts:", adata.obs["pct_counts_mt"].median())

        # Scatter plot: Total counts vs. Percentage of mitochondrial counts
        max_mt = qc_params["max_mt"]
        plt.figure(figsize=(8, 6))
        plt.scatter(adata.obs['total_counts'], adata.obs['pct_counts_mt'], alpha=0.5)
        plt.axhline(max_mt,c="r")
        plt.xlabel('Total Counts')
        plt.ylabel('Percentage of Mitochondrial Counts')
        plt.title('Total Counts vs. Percentage of Mitochondrial Counts')
        plt.grid(True)
        plt.show()

    # Scatter plot: Total counts vs. Number of genes by counts
    max_genes = adata.obs['n_genes_by_counts'].quantile(0.95)
    plt.figure(figsize=(8, 6))
    plt.scatter(adata.obs['total_counts'], adata.obs['n_genes_by_counts'], alpha=0.5)
    plt.axhline(max_genes,c="r")
    plt.xlabel('Total Counts')
    plt.ylabel('Number of Genes by Counts')
    plt.title('Total Counts vs. Number of Genes by Counts')
    plt.grid(True)
    plt.show()

    return qc_params

def apply_preprocessing_hd(project_path, bin_size, qc_params, save_data = True):
    assert bin_size in ["002", "008", "016", "b2c"], "bin_size does not matcha available bin sizes"
   
    if bin_size == "b2c":
        bin_path =  os.path.join(project_path, "binned_outputs", f"square_002um")
        adata = load_bin2cell_hd_data(bin_path)
        sc.pp.calculate_qc_metrics(adata, inplace=True) # no longer contins 'mt'
        results_dir = os.path.join(bin_path, "preprocessed", "002um")

    else:
        bin_path =  os.path.join(project_path, "binned_outputs", f"square_{bin_size}um")
        adata = load_visium_hd_data(bin_path)
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
        results_dir = os.path.join(bin_path, "preprocessed", f"{bin_size}um")


    os.makedirs(results_dir, exist_ok=True)
    adata.obs["n_counts"] = adata.obs["total_counts"]

    min_genes = qc_params['min_genes']
    max_genes =  qc_params['max_genes']
    if bin_size != "b2c":
        max_mt =  qc_params['max_mt']
    threshold_lower_counts =  qc_params['min_total_counts']

    adata = adata[adata.obs['total_counts'] >= threshold_lower_counts, :]
    adata = adata[(adata.obs['n_genes_by_counts'] >= min_genes) & 
                  (adata.obs['n_genes_by_counts'] <= max_genes), :]
    if bin_size != "b2c":
        adata = adata[adata.obs['pct_counts_mt'] <= max_mt, :]
    adata.var.rename(columns={"gene_ids": "ensembl_id"}, inplace=True)
    
    # Normalize and log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if bin_size != "b2c":
        preprocessed_data_dir = os.path.join(results_dir, "filtered_feature_bc_matrix_preprocessed.h5ad")
    else:
         preprocessed_data_dir = os.path.join(results_dir, "filtered_b2c_preprocessed.h5ad")
    if save_data == True:
        adata.write(preprocessed_data_dir) 
    return adata
#%% Run for standard bin sizes

# bin_size = "002"  # 002 008 or 016
all_bins = ["002", "008", "016"]

for bin_size in all_bins:
    visium_hd_folder = "Visium_HD_Human_Lung_Cancer_post_Xenium_Prime_5k_Experiment2"
    data_path = f"/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics"
    project_path = f"{data_path}/{visium_hd_folder}"

    qc_params = visualize_qc_metrics_hd(project_path, bin_size, show=True)

    apply_preprocessing_hd(project_path, bin_size, qc_params)

#%% Run for bin2cell 
bin_size="b2c"
visium_hd_folder = "Visium_HD_Human_Lung_Cancer_post_Xenium_Prime_5k_Experiment2"
data_path = f"/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics"
project_path = f"{data_path}/{visium_hd_folder}"

qc_params = visualize_qc_metrics_hd(project_path, bin_size, show=True)
qc_params["max_genes"] = 5000
adata = apply_preprocessing_hd(project_path, bin_size, qc_params, save_data=True)
# %%
