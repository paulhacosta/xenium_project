#%%
import os 
import h5py
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import yaml
import seaborn as sns
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt

#%% Visisum

hd = True
bin_size = "008"

#%%
if hd:
    data_path = "/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Visium_HD_Human_Lung_Cancer_post_Xenium_Prime_5k_Experiment2"
    # Path to your feature slice file
    feature_file = f"{data_path}/binned_outputs/square_{bin_size}um/filtered_feature_bc_matrix.h5"
    spatial_file = f"{data_path}/binned_outputs/square_{bin_size}um/spatial/tissue_positions.parquet"
    # Load data into AnnData format
    spatial_data = pd.read_parquet(spatial_file)
else:
    feature_file = "/rsrch9/home/plm/idso_fa1_pathology/TIER1/public_data/10xpublic_st/lung/Human Lung Cancer (FFPE)_lusc/output/CytAssist_FFPE_Human_Lung_Squamous_Cell_Carcinoma_filtered_feature_bc_matrix.h5"
    spatial_file = "/rsrch9/home/plm/idso_fa1_pathology/TIER1/public_data/10xpublic_st/lung/Human Lung Cancer (FFPE)_lusc/output/spatial/tissue_positions.csv"
    spatial_data = pd.read_csv(spatial_file)

adata = sc.read_10x_h5(feature_file)

#%% Visualize QC 
adata.var_names_make_unique()

# Identify mitochondrial genes
adata.var['mt'] = adata.var_names.str.startswith('MT-')

# Add spatial coordinates
spatial_data.set_index('barcode', inplace=True)
adata.obs = adata.obs.join(spatial_data, how='left')

# Add spatial coordinates as additional features
adata.obs["x_centroid"] = adata.obs["pxl_col_in_fullres"]
adata.obs["y_centroid"] = adata.obs["pxl_row_in_fullres"] 
adata.obs["x_scaled"] = adata.obs["x_centroid"] / adata.obs["x_centroid"].max()
adata.obs["y_scaled"] = adata.obs["y_centroid"] / adata.obs["y_centroid"].max()


# Convert 'in_tissue' to string if necessary
adata.obs['in_tissue'] = adata.obs['in_tissue'].astype(str)
# Initial filtering: spots over tissue
adata = adata[adata.obs['in_tissue'] == '1', :]

# Calculate QC metrics
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
sc.pl.violin(
adata,
["n_genes_by_counts", "total_counts", "pct_counts_mt"],
jitter=0.4,
multi_panel=True,
)

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

# Percentage of mitochondrial counts
print("Percentage of mitochondrial counts range:", adata.obs["pct_counts_mt"].min(), "-", adata.obs["pct_counts_mt"].max())
print("Mean percentage of mitochondrial counts:", adata.obs["pct_counts_mt"].mean())
print("Median percentage of mitochondrial counts:", adata.obs["pct_counts_mt"].median())

# Scatter plot: Total counts vs. Percentage of mitochondrial counts
max_mt = np.ceil(adata.obs['pct_counts_mt'].quantile(0.95))
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


# Automatic threshold detection for total counts
hist, bin_edges = np.histogram(adata.obs['total_counts'], bins=500)
peaks, _ = find_peaks(hist, distance=20)
if len(peaks) > 1:
    threshold_lower_counts = bin_edges[peaks[1]]
else:
    print("Not enought peaks, using manual threshold")

# Visualization of histogram with peaks
plt.figure(figsize=(6, 4))
plt.plot(bin_edges[:-1], hist)
plt.plot(bin_edges[peaks], hist[peaks], "x")
plt.axvline(threshold_lower_counts, c="r")
plt.title('Histogram of Total Counts with Peaks')
plt.xlabel('Total Counts')
plt.ylabel('Frequency')
plt.show()

print(f"Suggested total counts threshold: {int(threshold_lower_counts)}")
print(f"Suggested max genes: {int(max_genes)}")
print(f"Suggested max mt pcnt: {int(max_mt)}")
if max_mt > 20:
    print("Suggested max mt pcnt is too high. Changing to default max: 20")
    max_mt = 20




#%% Apply Preprocessing
print("Before filtering, adata shape is:", adata.shape)
# Calculate QC metrics
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)

# Filtering based on thresholds
min_genes = 200

adata = adata[adata.obs['total_counts'] >= threshold_lower_counts, :]
adata = adata[(adata.obs['n_genes_by_counts'] >= min_genes) & 
                (adata.obs['n_genes_by_counts'] <= max_genes), :]
adata = adata[adata.obs['pct_counts_mt'] <= max_mt, :]

print("After filtering, adata shape is:", adata.shape)
    
# Normalize and log-transform the data
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# # Optional: Highly variable genes
# sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=5000)
# adata = adata[:, adata.var['highly_variable']].copy()

if not hd:
    preprocessed_data_dir = "/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/TEST"
    preprocessed_file = f"{data_path.split('/')[-3]}_preprocessed.h5ad".replace(" ", "_")
    if not os.path.exists(preprocessed_data_dir):
        os.mkdir(preprocessed_data_dir)

else:
    preprocessed_data_dir = data_path
    preprocessed_file = f"preprcessed_visium_HD_{bin_size}um.h5ad"


# Rename 'gene_ids' to 'ensembl_id' in adata.var
adata.var.rename(columns={"gene_ids": "ensembl_id"}, inplace=True)
adata.obs["n_counts"] = adata.obs["total_counts"]

# Save the preprocessed data
adata.write(os.path.join(preprocessed_data_dir, preprocessed_file))


#%%