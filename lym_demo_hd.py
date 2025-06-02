#%%
import os 
import scanpy as sc
import pandas as pd 
import numpy as np
import cv2 
import tifffile as tf 
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns 
import scipy.sparse as sparse
#%%

r9 = "/rsrch9/home/plm/idso_fa1_pathology"
aitil_root = "/rsrch9/home/plm/idso_fa1_pathology/TIER2/paul-xenium/aitil_outputs"
file_name = "Xenium_Prime_Human_Lung_Cancer_FFPE_he_image.ome.tif"

adata_path = f"{r9}/TIER1/paul-xenium/public_data/10x_genomics/Visium_HD_Human_Lung_Cancer_post_Xenium_Prime_5k_Experiment2/binned_outputs/square_002um/preprocessed/bin2cell/to_tokenize/corrected_cells_matched_preprocessed_refined_v2.h5ad"
segmask_label_dir = f"{r9}/TIER2/paul-xenium/mit-b3-finetuned-TCGAbcssWsss10xLuadMacroMuscle-40x896-20x512-10x256re/mask_cws512/{file_name}"

he_image_path = f"/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Xenium_Prime_Human_Lung_Cancer_FFPE_outs/{file_name}"

# Load AnnData
adata = sc.read_h5ad(adata_path)

param = pickle.load(open(f"{aitil_root}/1_cws_tiling/{file_name}/param.p", "rb"))


# %% Set Rescale factor
rescale_factor = param["rescale"]


# %%
he_image = tf.imread(he_image_path)  # imF image
target_shape = he_image.shape

wsi_seg_targets = np.zeros(
    (int(round(target_shape[0] / rescale_factor)), int(round(target_shape[1] / rescale_factor))),
    dtype=np.uint8
)
    # Load cell position data
cellpos_file = f"{aitil_root}/4_cell_class/CellPos/{file_name}_cellPos.csv"
cellPos_df = pd.read_csv(cellpos_file)

# Create a DataFrame of unique tile coordinates
tile_coords_df = cellPos_df[['tile_file', 'tile_x', 'tile_y']].drop_duplicates()

# Initialize global color-to-class mapping
global_color_to_class = {}

# Process each tile
for _, row in tqdm(tile_coords_df.iterrows(), total=len(tile_coords_df)):
    tile_name = row.tile_file
    label_image_file = os.path.join(segmask_label_dir, tile_name.replace("csv", "png"))
    
    image_bgr = cv2.imread(label_image_file, cv2.IMREAD_UNCHANGED)
    label_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Extract unique colors for this tile
    unique_colors = np.unique(label_image.reshape(-1, 3), axis=0)

    # Map each unique color to a class label (assign new labels as needed)
    for color in unique_colors:
        color_tuple = tuple(color)
        if color_tuple not in global_color_to_class:
            global_color_to_class[color_tuple] = len(global_color_to_class) + 1

    # Create a class map for this tile
    class_map_tile = np.zeros(label_image.shape[:2], dtype=np.uint8)
    for color, class_label in global_color_to_class.items():
        mask = np.all(label_image == color, axis=-1)
        class_map_tile[mask] = class_label

    # Assign class_map_tile into the corresponding region of `wsi_seg_targets`
    tile_x = int(row.tile_x)
    tile_y = int(row.tile_y)
    wsi_seg_targets[tile_y:tile_y + class_map_tile.shape[0], tile_x:tile_x + class_map_tile.shape[1]] = class_map_tile

final_wsi_seg_targets = cv2.resize(wsi_seg_targets, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
final_wsi_seg_targets[final_wsi_seg_targets != 0] -= 1

class_to_color = {
        0: 'background',  # Black
        1: 'alveoli',  # Dark Green
        2: 'bronchi_epithelial',  # Cyan
        3: 'macrophage_area',  # Purple
        4: 'stroma',  # Yellow
        5: 'microvessel',  # Dark Blue
        6: 'tumor',  # Maroon
        7: 'inflammatory',  # Red
        8: 'necrosis',  # Pink
        9: 'adipose',  # Olive
        10: 'muscle'  # Navy Blue
    }


# Create an RGB image with the same shape as final_wsi_seg_targets
segmentation_vis = np.zeros((*final_wsi_seg_targets.shape, 3), dtype=np.uint8)

# Assign colors to specific classes
segmentation_vis[final_wsi_seg_targets == 4] = [128, 0, 0]    # Maroon (Stroma)
segmentation_vis[final_wsi_seg_targets == 9] = [255, 255, 0]  # Yellow (Adipose)

# Plot the segmentation map
plt.figure(figsize=(20, 20))
plt.imshow(segmentation_vis)
plt.axis("off")
plt.show()

# %%
# Extract Lymphocytes from Visium HD
lymphocytes = adata[adata.obs['class'] == 'l'].copy()
lymphocyte_coords = lymphocytes.obsm['spatial']

# Convert spatial coordinates to image space
lymphocyte_x = np.round(lymphocyte_coords[:, 0] / rescale_factor).astype(int)
lymphocyte_y = np.round(lymphocyte_coords[:, 1] / rescale_factor).astype(int)

# Ensure coordinates are within bounds
h, w = final_wsi_seg_targets.shape
valid_indices = (lymphocyte_x >= 0) & (lymphocyte_x < w) & (lymphocyte_y >= 0) & (lymphocyte_y < h)
lymphocyte_x, lymphocyte_y = lymphocyte_x[valid_indices], lymphocyte_y[valid_indices]

#  Assign Lymphocytes to Tumor (IT) or Stroma
lymphocyte_regions = final_wsi_seg_targets[lymphocyte_y, lymphocyte_x]
lymphocyte_tissue_labels = np.where(lymphocyte_regions == 4, "Intratumoral", 
                                    np.where(lymphocyte_regions == 9, "Stromal", "Other"))

lymphocytes.obs['Tissue_Region'] = lymphocyte_tissue_labels

#  Extract Gene Expression (CD8A, CD8B2, FOXP3)
def extract_gene_expression(adata, lymphocytes, gene):
    matched_genes = [g for g in adata.var.index if g.upper() == gene.upper()]
    if matched_genes:
        expr = lymphocytes[:, matched_genes[0]].X
        return expr.toarray().flatten() if sparse.issparse(expr) else expr.flatten()
    else:
        print(f"Warning: {gene} not found in dataset. Returning empty array.")
        return np.zeros(len(lymphocytes), dtype=float)

cd8a_expr = extract_gene_expression(adata, lymphocytes, "CD8A")
cd8b_expr = extract_gene_expression(adata, lymphocytes, "CD8B2")  # CD8B is stored as CD8B2
foxp3_expr = extract_gene_expression(adata, lymphocytes, "FOXP3")

#  Classify Lymphocytes as CD8+ or FOXP3+
cd8_positive = (cd8a_expr > 0) | (cd8b_expr > 0)
foxp3_positive = foxp3_expr > 0

# lymphocytes.obs['Subtype'] = np.where(cd8_positive, "CD8+", 
#                                      np.where(foxp3_positive, "FOXP3+", "Other"))

lymphocytes.obs["Subtype"] = np.where(cd8_positive & foxp3_positive, "CD8+FOXP3+",
                                      np.where(cd8_positive, "CD8+",
                                      np.where(foxp3_positive, "FOXP3+", "Other")))

#  Compare CD8+ vs FOXP3+ in IT vs Stroma
lymphocytes_filtered = lymphocytes[lymphocytes.obs['Tissue_Region'].isin(["Intratumoral", "Stromal"])]
lymphocytes_filtered = lymphocytes_filtered[lymphocytes_filtered.obs['Subtype'].isin(["CD8+", "FOXP3+", "CD8+FOXP3+"])]

comparison_df = pd.DataFrame({
    "Tissue_Region": lymphocytes_filtered.obs['Tissue_Region'],
    "Subtype": lymphocytes_filtered.obs['Subtype']
})

# Compute proportions
proportions = comparison_df.groupby(["Tissue_Region", "Subtype"]).size().unstack().fillna(0)
proportions = proportions.div(proportions.sum(axis=1), axis=0)  # Normalize to proportions


# Plot side-by-side bar plot
proportions.plot(kind='bar', stacked=False, figsize=(8, 6), colormap="coolwarm", width=0.7)

plt.title("Proportion of CD8+ and FOXP3+ Lymphocytes in IT vs Stroma")
plt.ylabel("Proportion")
plt.xlabel("Tissue Region")
plt.xticks(rotation=0)
plt.legend(title="Lymphocyte Subtype")
plt.show()

# %%

# Ensure only Tumor (IT) and Stroma lymphocytes are included
lymphocytes_filtered = lymphocytes[lymphocytes.obs['Tissue_Region'].isin(["Intratumoral", "Stromal"])].copy()

# Perform differential expression analysis using Scanpy
sc.tl.rank_genes_groups(lymphocytes_filtered, groupby='Tissue_Region', method='wilcoxon')

# Convert Scanpy DEGs result into a Pandas DataFrame
deg_results = sc.get.rank_genes_groups_df(lymphocytes_filtered, group="Intratumoral")  # Compare IT vs Stroma

# Sort by p-value (smallest first)
deg_results = deg_results.sort_values(by="pvals")

# Display top 20 DEGs
print("\nTop 20 Differentially Expressed Genes (IT vs. Stroma):")
print(deg_results.head(20))

# Save results to CSV for further analysis

# --- Visualization: Volcano Plot ---
plt.figure(figsize=(8,6))
plt.scatter(deg_results['logfoldchanges'], -np.log10(deg_results['pvals']), alpha=0.7, edgecolors="k")
plt.axhline(-np.log10(0.05), linestyle='dashed', color='red', label="p = 0.05 cutoff")
plt.xlabel("Log Fold Change (logFC)")
plt.ylabel("-log10(p-value)")
plt.title("Differentially Expressed Genes (IT vs. Stroma)")
plt.legend()
plt.show()


# %%
