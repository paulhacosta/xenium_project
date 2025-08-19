#%%
import os 
import h5py
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
#%%
task = "pretrained" # pretrained or fine_tune
ground_truth = "refined" # aitil, refined, cellvit
#%%
cancer = "lung"
xenium_folder_dict = {"lung": "Xenium_Prime_Human_Lung_Cancer_FFPE_outs",
                      "breast":"Xenium_Prime_Breast_Cancer_FFPE_outs",
                      "lymph_node": "Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_outs",
                      "prostate": "Xenium_Prime_Human_Prostate_FFPE_outs",
                      "skin": "Xenium_Prime_Human_Skin_FFPE_outs",
                      "ovarian": "Xenium_Prime_Ovarian_Cancer_FFPE_outs",
                      "cervical": "Xenium_Prime_Cervical_Cancer_FFPE_outs"
                      "lung_v1": "Xenium_V1_Human_Lung_Cancer_FFPE_outs"
                      }

xenium_folder = xenium_folder_dict[cancer]

data_path = f"/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/{xenium_folder}"

#%% # Load cell metadata

if task == "pretrained":
    cells_df = pd.read_csv(os.path.join( data_path, "cells.csv.gz"))  # or pd.read_parquet("path_to/cells.parquet")
    # spatial_data = cells_df[["cell_id", "x_centroid", "y_centroid", "transcript_counts", "total_counts", "cell_area"]]
elif task == "fine_tune":
    if ground_truth=="refined":
         csv_file = os.path.join( data_path, "cells_matched_spatial_refined_v2.csv")
    elif ground_truth=="aitil" :
        csv_file = os.path.join( data_path, "cells_matched_spatial.csv")
    elif ground_truth=="cellvit":
        csv_file = os.path.join( data_path, "cells_matched_spatial_cellvit.csv")
    elif ground_truth=="segmask":
        csv_file = os.path.join( data_path, "cells_matched_spatial_segmask.csv")
cells_df =  pd.read_csv(csv_file)
#%%
adata = sc.read_10x_h5(os.path.join(data_path, "cell_feature_matrix.h5"))

#%%
# Check alignment between cell IDs
print("First few cell IDs in AnnData:", adata.obs_names[:5])
print("First few cell IDs in cells.csv.gz:", cells_df["cell_id"].head())

# Check how many match
matching_ids = adata.obs_names.isin(cells_df["cell_id"]).sum()
print(f"Number of matching cell IDs: {matching_ids}/{len(adata.obs_names)}")

#%%

# Merge spatial metadata into adata.obs
adata.obs = adata.obs.merge(
    cells_df.set_index("cell_id"),  # Set cell_id as index in the metadata
    left_index=True,  # Use cell barcodes from adata
    right_index=True,
    how="left"
)

# Check merged metadata
# print(adata.obs.head())

# Add a quality control metric for mitochondrial content
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

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



#%%

plt.scatter(adata.obs["x_centroid"], adata.obs["y_centroid"], 
            c=adata.obs["total_counts"], s=5, cmap="viridis", alpha=0.7)
plt.colorbar(label="Total Transcript Counts")
plt.xlabel("X Centroid")
plt.ylabel("Y Centroid")
plt.title("Spatial Distribution of Cells with Transcript Counts")
plt.show()

#%% Simple preprocessing
# Filter cells with too few or too many genes
sc.pp.filter_cells(adata, min_genes=200)  # Keep cells with at least 200 genes
sc.pp.filter_genes(adata, min_cells=3)   # Keep genes expressed in at least 3 cells

# Filter cells with high mitochondrial percentages or very high total counts
adata = adata[adata.obs["pct_counts_mt"] < 10]  # Remove cells with >10% MT content

# Filter cells without reliable cell type (ground truth for fine tuning only)
if task == "fine_tune":
    adata = adata[adata.obs["qc_match"] == 1]  # Remove cells that were not matched to AITIL based on distance
    adata = adata[adata.obs["qc_exclusive"] == 1]  # Remove cells that were not matched to AITIL based on unique pairs



#%% Simple Normalization:
# Normalize and log-transform
sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize to 10,000 counts per cell
sc.pp.log1p(adata)  # Log-transform for smoother distributions

#%% Adding spatial coordinates as features 
# Add spatial coordinates as additional features
adata.obs["x_scaled"] = adata.obs["x_centroid"] / adata.obs["x_centroid"].max()
adata.obs["y_scaled"] = adata.obs["y_centroid"] / adata.obs["y_centroid"].max()

#%%
# Retain only necessary columns in adata.obs
# adata.obs = adata.obs[["x_centroid", "y_centroid", "total_counts", "x_scaled", "y_scaled"]]

# Retain only the 'gene_ids' column in adata.var
adata.var = adata.var[["gene_ids"]]

# Verify the updated AnnData structure
print(adata)
print("obs columns:", adata.obs.columns)
print("var columns:", adata.var.columns)
 
#%%
# Rename 'gene_ids' to 'ensembl_id' in adata.var
adata.var.rename(columns={"gene_ids": "ensembl_id"}, inplace=True)
adata.obs["n_counts"] = adata.obs["total_counts"]

# Save the processed AnnData object to an h5ad file

if task == "pretrained":
    adata.write_h5ad(os.path.join(data_path, "processed_xenium_data.h5ad"))
else: 
    # Define the test size and random state for reproducibility
    random_state = 42

    # Ensure 'class' column exists and is non-empty
    if 'class' not in adata.obs.columns or adata.obs['class'].isnull().all():
        raise ValueError("The 'class' column is missing or empty in adata.obs.")

    # Exclude any class that has less than 100 cells
    # adata = adata[adata.obs["class"].map(adata.obs["class"].value_counts()) > 100]
    adata = adata[adata.obs["class"].astype(str).map(adata.obs["class"].value_counts()) > 100]

    # Add a 'split' column and initialize with None
    adata.obs['split'] = None

    # First split: Train (70%) and Temp (30% for test+eval)
    train_data, temp_data = train_test_split(
        adata.obs,
        test_size=0.3,
        stratify=adata.obs['class'],
        random_state=random_state
    )

    # Assign 'train' to the split column for the training set
    adata.obs.loc[train_data.index, 'split'] = 'train'

    # Second split: Split Temp into Test (15%) and Eval (15%)
    test_data, eval_data = train_test_split(
        temp_data,
        test_size=0.5,  # Split Temp equally
        stratify=temp_data['class'],
        random_state=random_state
    )

    # Assign 'test' and 'eval' to the split column
    adata.obs.loc[test_data.index, 'split'] = 'test'
    adata.obs.loc[eval_data.index, 'split'] = 'eval'

    # Verify split sizes
    print("Train size:", (adata.obs['split'] == 'train').sum())
    print("Test size:", (adata.obs['split'] == 'test').sum())
    print("Eval size:", (adata.obs['split'] == 'eval').sum())

    # Verify the stratification
    print("Train class distribution:\n", adata.obs.loc[train_data.index, 'class'].value_counts())
    print("Test class distribution:\n", adata.obs.loc[test_data.index, 'class'].value_counts())
    print("Eval class distribution:\n", adata.obs.loc[eval_data.index, 'class'].value_counts())

    if ground_truth == "aitil":
        adata.write_h5ad(os.path.join(data_path, "preprocessed", task, "processed_xenium_data_fine_tune.h5ad"))
    else:
        os.makedirs(os.path.join(data_path, "preprocessed", task+f"_{ground_truth}"), exist_ok=True)
        adata.write_h5ad(os.path.join(data_path, "preprocessed", task+f"_{ground_truth}", f"processed_xenium_data_fine_tune_{ground_truth}_v2.h5ad"))

#%%
# from datasets import load_from_disk

# # Load the dataset
# tokenized_dataset = load_from_disk(f"{data_path}/tokenize_output/processed_xenium_fine_tune.dataset")

# # Inspect the dataset
# print(tokenized_dataset)
# print(tokenized_dataset.column_names)  # Columns available in the dataset
# print(tokenized_dataset[0])  # First row to check structure

# # Filter for the train split
# train_dataset = tokenized_dataset.filter(lambda example: example["split"] == "train")

# # Access the "input_ids" column for the train split
# train_input_ids = train_dataset["input_ids"]

# # Verify
# print("Number of training examples:", len(train_input_ids))
# %%
