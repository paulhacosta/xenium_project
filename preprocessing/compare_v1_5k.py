#%%
import scanpy as sc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import issparse
#%%
preprocessed = False

#%
#%%
# 1. Read/Load AnnData Objects

if preprocessed: 
        
    adata_v1 = sc.read_h5ad("/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Xenium_V1_Human_Lung_Cancer_FFPE_outs/processed_xenium_data.h5ad")
    adata_5k = sc.read_h5ad("/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Xenium_Prime_Human_Lung_Cancer_FFPE_outs/preprocessed/pretrained/processed_xenium_data.h5ad")
    
else:
    adata_v1 = sc.read_10x_h5("/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Xenium_V1_Human_Lung_Cancer_FFPE_outs/cell_feature_matrix.h5")
    adata_5k = sc.read_10x_h5("/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Xenium_Prime_Human_Lung_Cancer_FFPE_outs/cell_feature_matrix.h5")
    print(adata_v1.var_names[adata_v1.var_names.str.startswith("MT-")])
    print(adata_5k.var_names[adata_5k.var_names.str.startswith("MT-")])

    sc.pp.calculate_qc_metrics(adata_v1,percent_top = [50, 100, 200],  inplace=True)
    sc.pp.calculate_qc_metrics(adata_5k, inplace=True)


# 2. Identify Overlapping Genes
common_genes = adata_v1.var_names.intersection(adata_5k.var_names)
print(f"Number of overlapping genes: {len(common_genes)}")

# Sort or leave as is. Sorting can help with consistent ordering:
common_genes = sorted(common_genes)

# 3. Subset Each AnnData Object to Overlapping Genes
adata_v1_common = adata_v1[:, common_genes].copy()
adata_5k_common = adata_5k[:, common_genes].copy()

print("adata_v1_common shape:", adata_v1_common.shape)
print("adata_5k_common shape:", adata_5k_common.shape)

# 4. Function to Get Total Counts for Each Gene
#    (Summing across all cells)
def get_total_counts(adata):
    """Return a pd.Series of total counts per gene, summing across all cells."""
    X = adata.X
    if issparse(X):
        total_counts = np.array(X.sum(axis=0)).ravel()  # shape = (n_genes,)
    else:
        total_counts = X.sum(axis=0)  # if it's a dense matrix
    return pd.Series(total_counts, index=adata.var_names, name="total_counts")

# 5. Calculate Total Counts for V1 & 5k Overlapping Genes
v1_counts = get_total_counts(adata_v1_common)
x5k_counts = get_total_counts(adata_5k_common)

# Merge into one DataFrame
counts_df = pd.concat([v1_counts, x5k_counts], axis=1)
counts_df.columns = ["V1_total_counts", "5K_total_counts"]

# Calculate difference and ratio columns
counts_df["difference"] = counts_df["5K_total_counts"] - counts_df["V1_total_counts"]
counts_df["ratio"] = (
    counts_df["5K_total_counts"] / (counts_df["V1_total_counts"] + 1e-9)
)

# Preview the counts DataFrame
print("\nMerged gene counts comparison:\n", counts_df.head())

# 6. Plot V1 vs. 5k Counts on Log-Log Scale
plt.figure(figsize=(6, 6))
# Adding +1 to avoid log(0)
plt.scatter(
    counts_df["V1_total_counts"] + 1,
    counts_df["5K_total_counts"] + 1,
    alpha=0.7,
    edgecolor="none"
)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("V1 Total Gene Counts (log scale)")
plt.ylabel("5k Total Gene Counts (log scale)")
plt.title("Comparison of Overlapping Gene Counts: V1 vs. 5k")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.show()

# 7. Further Analyses
# For instance, sorting by difference to see the genes with the biggest gap:
sorted_by_diff = counts_df.sort_values("difference", ascending=False)
print("\nTop 10 genes with largest difference (5k minus V1):")
print(sorted_by_diff.head(10))

# Or sorting by ratio:
sorted_by_ratio = counts_df.sort_values("ratio", ascending=False)
print("\nTop 10 genes with largest ratio (5k / V1):")
print(sorted_by_ratio.head(10))

# %%
# Identify genes present in V1 but missing in 5K
missing_genes_v1 = adata_v1.var_names.difference(adata_5k.var_names)

# Print the number of missing genes
print(f"Number of genes present in V1 but missing in 5K: {len(missing_genes_v1)}")

# Create a subset of adata_v1 containing only the missing genes
if len(missing_genes_v1) > 0:
    adata_v1_missing = adata_v1[:, missing_genes_v1].copy()
    
    # Calculate total counts for these missing genes in V1
    missing_v1_counts = get_total_counts(adata_v1_missing)

    # Convert to DataFrame
    missing_genes_df = pd.DataFrame(missing_v1_counts, columns=["V1_total_counts"])

    # Drop genes with zero expression
    missing_genes_df = missing_genes_df[missing_genes_df["V1_total_counts"] > 0]

    # Sort by expression level in V1 (descending)
    missing_genes_df = missing_genes_df.sort_values("V1_total_counts", ascending=False)

    # Print final number of nonzero genes
    print(f"Number of missing genes with nonzero expression in V1: {len(missing_genes_df)}")

    # Display the top 10 missing genes
    print("\nTop 10 genes present in V1 but missing in 5K (sorted by expression level in V1):")
    print(missing_genes_df.head(10))
else:
    print("No genes are uniquely present in V1.")

# %%

# %%
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import issparse

# %%
preprocessed = False

# %%
# 1. Read/Load AnnData Objects
if preprocessed: 
    adata_v1 = sc.read_h5ad("/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Xenium_V1_Human_Lung_Cancer_FFPE_outs/processed_xenium_data.h5ad")
    adata_5k = sc.read_h5ad("/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Xenium_Prime_Human_Lung_Cancer_FFPE_outs/preprocessed/pretrained/processed_xenium_data.h5ad")
else:
    adata_v1 = sc.read_10x_h5("/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Xenium_V1_Human_Lung_Cancer_FFPE_outs/cell_feature_matrix.h5")
    adata_5k = sc.read_10x_h5("/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Xenium_Prime_Human_Lung_Cancer_FFPE_outs/cell_feature_matrix.h5")
    print(adata_v1.var_names[adata_v1.var_names.str.startswith("MT-")])
    print(adata_5k.var_names[adata_5k.var_names.str.startswith("MT-")])

    sc.pp.calculate_qc_metrics(adata_v1, percent_top=[50, 100, 200], inplace=True)
    sc.pp.calculate_qc_metrics(adata_5k, inplace=True)

# %%
# Summary statistics function
def summarize_adata(adata, dataset_name):
    """Computes and prints summary statistics for an AnnData object."""
    num_spots = adata.shape[0]
    num_genes = adata.shape[1]
    
    # Total counts per spot
    total_counts_per_spot = adata.X.sum(axis=1).A1 if issparse(adata.X) else adata.X.sum(axis=1)
    total_counts_min, total_counts_max = total_counts_per_spot.min(), total_counts_per_spot.max()
    total_counts_mean = total_counts_per_spot.mean()
    total_counts_median = np.median(total_counts_per_spot)

    # Genes detected per spot
    detected_genes_per_spot = (adata.X > 0).sum(axis=1).A1 if issparse(adata.X) else (adata.X > 0).sum(axis=1)
    genes_min, genes_max = detected_genes_per_spot.min(), detected_genes_per_spot.max()
    genes_mean = detected_genes_per_spot.mean()
    genes_median = np.median(detected_genes_per_spot)

    print(f"\nSummary statistics for {dataset_name}:")
    print(f"Number of spots in tissue: {num_spots}")
    print(f"Number of genes: {num_genes}")
    print(f"Total counts range: {total_counts_min} - {total_counts_max}")
    print(f"Mean total counts: {total_counts_mean:.0f}")
    print(f"Median total counts: {total_counts_median:.0f}")
    print(f"Number of genes by counts range: {genes_min} - {genes_max}")
    print(f"Mean number of genes by counts: {genes_mean:.0f}")
    print(f"Median number of genes by counts: {genes_median:.0f}")

# Run the summary statistics for both datasets
summarize_adata(adata_v1, "Xenium V1")
summarize_adata(adata_5k, "Xenium 5K")

# %%
# 2. Identify Overlapping Genes
common_genes = adata_v1.var_names.intersection(adata_5k.var_names)
print(f"\nNumber of overlapping genes: {len(common_genes)}")

# Sort or leave as is. Sorting can help with consistent ordering:
common_genes = sorted(common_genes)

# 3. Subset Each AnnData Object to Overlapping Genes
adata_v1_common = adata_v1[:, common_genes].copy()
adata_5k_common = adata_5k[:, common_genes].copy()

print("\nadata_v1_common shape:", adata_v1_common.shape)
print("adata_5k_common shape:", adata_5k_common.shape)

# 4. Function to Get Total Counts for Each Gene
def get_total_counts(adata):
    """Return a pd.Series of total counts per gene, summing across all cells."""
    X = adata.X
    if issparse(X):
        total_counts = np.array(X.sum(axis=0)).ravel()  # shape = (n_genes,)
    else:
        total_counts = X.sum(axis=0)  # if it's a dense matrix
    return pd.Series(total_counts, index=adata.var_names, name="total_counts")

# 5. Calculate Total Counts for V1 & 5k Overlapping Genes
v1_counts = get_total_counts(adata_v1_common)
x5k_counts = get_total_counts(adata_5k_common)

# Merge into one DataFrame
counts_df = pd.concat([v1_counts, x5k_counts], axis=1)
counts_df.columns = ["V1_total_counts", "5K_total_counts"]

# Calculate difference and ratio columns
counts_df["difference"] = counts_df["5K_total_counts"] - counts_df["V1_total_counts"]
counts_df["ratio"] = counts_df["5K_total_counts"] / (counts_df["V1_total_counts"] + 1e-9)

# Preview the counts DataFrame
print("\nMerged gene counts comparison:\n", counts_df.head())

# 6. Plot V1 vs. 5k Counts on Log-Log Scale
plt.figure(figsize=(6, 6))
plt.scatter(
    counts_df["V1_total_counts"] + 1,
    counts_df["5K_total_counts"] + 1,
    alpha=0.7,
    edgecolor="none"
)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("V1 Total Gene Counts (log scale)")
plt.ylabel("5K Total Gene Counts (log scale)")
plt.title("Comparison of Overlapping Gene Counts: V1 vs. 5K")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.show()

# %%
# Identify genes present in V1 but missing in 5K
missing_genes_v1 = adata_v1.var_names.difference(adata_5k.var_names)
print(f"\nNumber of genes present in V1 but missing in 5K: {len(missing_genes_v1)}")

if len(missing_genes_v1) > 0:
    adata_v1_missing = adata_v1[:, missing_genes_v1].copy()
    missing_v1_counts = get_total_counts(adata_v1_missing)
    missing_genes_df = pd.DataFrame(missing_v1_counts, columns=["V1_total_counts"])
    missing_genes_df = missing_genes_df[missing_genes_df["V1_total_counts"] > 0]
    missing_genes_df = missing_genes_df.sort_values("V1_total_counts", ascending=False)
    print(f"Number of missing genes with nonzero expression in V1: {len(missing_genes_df)}")
    print("\nTop 10 genes present in V1 but missing in 5K:")
    print(missing_genes_df.head(10))
else:
    print("No genes are uniquely present in V1.")

# %%

