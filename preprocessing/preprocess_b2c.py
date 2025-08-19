#%%
import scanpy as sc
import scipy.sparse as sp
import numpy as np

#%%
# ------------------------------------------------------------------------
# 1. Load or have your AnnData object
# ------------------------------------------------------------------------
adata = sc.read_h5ad("/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Visium_HD_Human_Lung_Cancer_post_Xenium_Prime_5k_Experiment2/binned_outputs/square_002um/b2c_cellvit_refined_fullres_v2.h5ad")

# ------------------------------------------------------------------------
# 2. Label mitochondrial genes and calculate QC metrics
# ------------------------------------------------------------------------
# For human data, mitochondrial genes usually start with "MT-"
# Adjust as necessary for other species (e.g. "mt-" for mouse).
adata.var["mt"] = adata.var_names.str.startswith("MT-")

sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

# ------------------------------------------------------------------------
# 3. Print basic stats as requested
# ------------------------------------------------------------------------
print("\nNumber of spots (cells) in tissue:", adata.obs.shape[0])
print("Number of genes:", adata.var.shape[0])

# Total counts
print("Total counts range:", adata.obs["total_counts"].min(), "-", adata.obs["total_counts"].max())
print("Mean total counts:", adata.obs["total_counts"].mean())
print("Median total counts:", adata.obs["total_counts"].median())

# Number of genes by counts
print("Number of genes by counts range:", adata.obs["n_genes_by_counts"].min(), 
      "-", adata.obs["n_genes_by_counts"].max())
print("Mean number of genes by counts:", adata.obs["n_genes_by_counts"].mean())
print("Median number of genes by counts:", adata.obs["n_genes_by_counts"].median())

# Percentage of mitochondrial counts
print("Percentage of mitochondrial counts range:", adata.obs["pct_counts_mt"].min(), 
      "-", adata.obs["pct_counts_mt"].max())
print("Mean percentage of mitochondrial counts:", adata.obs["pct_counts_mt"].mean())
print("Median percentage of mitochondrial counts:", adata.obs["pct_counts_mt"].median())

# ------------------------------------------------------------------------
# 4. (Optional) Filter out low-quality cells and genes
# ------------------------------------------------------------------------
# Remove cells with too few genes
sc.pp.filter_cells(adata, min_genes=10)

# (Optional) Remove cells with too many genes (possible doublets)
# sc.pp.filter_cells(adata, max_genes=6000)

# Filter out genes expressed in too few cells
sc.pp.filter_genes(adata, min_cells=3)

# Remove cells with too high mitochondrial fraction
# Here we filter out cells where >5% of reads come from mt-genes
adata = adata[adata.obs["pct_counts_mt"] < 10].copy()

# ------------------------------------------------------------------------
# 5. (Optional) Store raw counts for scGPT or other raw-based methods
# ------------------------------------------------------------------------
# If scGPT specifically needs raw counts, you can store them before normalization:
# adata.layers["raw_counts"] = adata.X.copy()

# ------------------------------------------------------------------------
# 6. Normalize, log-transform, etc. (typical for standard scRNA-seq analysis)
# ------------------------------------------------------------------------
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# ------------------------------------------------------------------------
# 7. Ensure adata.X is not a view; convert to real CSR matrix if needed
# ------------------------------------------------------------------------
adata = adata.copy()  # break potential "view" references
# if not isinstance(adata.X, sp.csr_matrix):
#     adata.X = sp.csr_matrix(adata.X)

adata.write_h5ad("/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Visium_HD_Human_Lung_Cancer_post_Xenium_Prime_5k_Experiment2/binned_outputs/square_002um/b2c_cellvit_preprocessed_refined_fullres_v2.h5ad")



#%%
