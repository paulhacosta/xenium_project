#%% 
import os 
import pandas as pd 
from scipy.io import loadmat
from PIL import Image
import numpy as np
from glob import glob
import pickle 
import json 
import tifffile as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2 
from scipy.spatial import KDTree
from matplotlib.colors import ListedColormap, BoundaryNorm
import geojson
import scanpy as sc

#%% 
save_new = True 
bin_size = "b2c"  #  008 or b2c
ground_truth = "refined"  # "aitil", "refined", "cellvit", or "segmask"

## Define paths
aitil_root = "/rsrch5/home/plm/phacosta/aitil_t6/output_pa2"
file_name = "Xenium_Prime_Human_Lung_Cancer_FFPE_he_image_coregistered_pyramid.ome.tif"
aitil_out_folders = ["1_cws_tiling", "2_tissue_seg", "3_cell_seg", "4_5_stitch_output", "4_cell_class", "step5_spatial"]
aitil_path_dict = {f: f"{aitil_root}/{f}" for f in aitil_out_folders}
param = pickle.load(open(f"{aitil_root}/1_cws_tiling/{file_name}/param.p", "rb"))
r9 = "/rsrch9/home/plm/idso_fa1_pathology"
segmask_label_dir = f"{r9}/TIER2/paul-xenium/mit-b3-finetuned-TCGAbcssWsss10xLuadMacroMuscle-40x896-20x512-10x256re/mask_cws512/Xenium_Prime_Human_Lung_Cancer_FFPE_he_image.ome.tif/"
refined_label_file = f"{r9}/TIER2/paul-xenium/aitil_outputs/4_cell_class_segformerMacro/CellPos/Xenium_Prime_Human_Lung_Cancer_FFPE_he_image.ome.csv"
cellvit_labels_file = "/rsrch5/home/plm/phacosta/CellViT/example/output/preprocessing/coregistered/Xenium_Prime_Human_Lung_Cancer_FFPE_he_image_coregistered_pyramid.ome/cell_detection/cell_detection.geojson"


# %% Set Rescale factor
rescale_factor = param["rescale"]

# %%
experiment = "post5k"
visium_folder_dict = {"post5k": "Visium_HD_Human_Lung_Cancer_post_Xenium_Prime_5k_Experiment2",
                      "postV1": "Visium_HD_Human_Lung_Cancer_post_Xenium_v1_Experiment1",
                      "control1": "Visium_HD_Human_Lung_Cancer_HD_Only_Experiment1",
                      "control2": "Visium_HD_Human_Lung_Cancer_HD_Only_Experiment2",
                      }

visium_folder = visium_folder_dict[experiment]

data_path = f"{r9}/TIER1/paul-xenium/public_data/10x_genomics/{visium_folder}/binned_outputs/square_002um/"

# Load cell data 
if bin_size == "b2c":
    # data_path = f"{r9}/TIER1/paul-xenium/public_data/10x_genomics/{visium_folder}/binned_outputs/square_002um/preprocessed/bin2cell"
    data_path = f"{r9}/TIER1/paul-xenium/public_data/10x_genomics/{visium_folder}/binned_outputs/square_002um/"

    # Use Bin2Cell Aggregation
    # cells_df = sc.read_h5ad(os.path.join(data_path, "preprocessed_bin2cell.h5ad"))  
    cells_df = sc.read_h5ad(os.path.join(data_path, "b2c_cellvit_fullres.h5ad"))  
    cell_coords = np.array(cells_df.obsm["spatial"])  # Shape: (N, 2)
    cells_df.obs["x_centroid"] = cell_coords[:,0]
    cells_df.obs["y_centroid"] = cell_coords[:,1]
else:
    print("Not yet implemented")

# %%
# For aitil, refined, and cellvit only 
micron_threshold = 10 
mpp = 0.2738
pixel_threshold = micron_threshold / mpp

if ground_truth=="refined":
 
    refined_df = pd.read_csv(refined_label_file)

    # Extract coordinates from both dataframes
    # cell_coords = cells_df[["x_centroid", "y_centroid"]].to_numpy()  # From spatial_data
    nuclei_coords = refined_df[["x_fullres", "y_fullres"]].to_numpy()    # From aitil_df

    # Build KDTree for nuclei
    nuclei_tree = KDTree(nuclei_coords)

    # Query all nearest neighbors (k=2 to allow fallback matching)
    distances, indices = nuclei_tree.query(cell_coords, k=2)
    print("Matching Indices Prior to Exlusivity:", indices.shape[0])

    # Array to track exclusivity (1 if matched, 0 if dropped)
    exclusivity_qc = np.ones(len(cell_coords), dtype=int)

    # Dictionary to track which nucleus is assigned and to whom
    assigned_nuclei = {}

    # Final match arrays
    final_indices = np.full(len(cell_coords), -1, dtype=int)  # Default to unmatched
    final_distances = np.full(len(cell_coords), np.inf)       # Default to no distance

    # First pass: Assign closest matches
    for cell_idx, (nucleus_idx, distance) in enumerate(zip(indices[:, 0], distances[:, 0])):
        if nucleus_idx not in assigned_nuclei:
            # If nucleus is unassigned, assign it
            assigned_nuclei[nucleus_idx] = cell_idx
            final_indices[cell_idx] = nucleus_idx
            final_distances[cell_idx] = distance
        else:
            # If nucleus is already assigned, check distance
            prev_cell_idx = assigned_nuclei[nucleus_idx]
            prev_distance = final_distances[prev_cell_idx]
            if distance < prev_distance:
                # Reassign nucleus to closer cell
                assigned_nuclei[nucleus_idx] = cell_idx
                final_indices[cell_idx] = nucleus_idx
                final_distances[cell_idx] = distance
                exclusivity_qc[prev_cell_idx] = 0  # Mark previous cell as unmatched
                final_indices[prev_cell_idx] = -1
                final_distances[prev_cell_idx] = np.inf
            else:
                # Current cell is farther; mark as unmatched
                exclusivity_qc[cell_idx] = 0

    # Second pass: Handle unmatched cells
    for cell_idx in range(len(cell_coords)):
        if final_indices[cell_idx] == -1:  # Unmatched cell
            # Assign second-best match (indices[:, 1])
            second_nucleus_idx = indices[cell_idx, 1]
            second_distance = distances[cell_idx, 1]

            # Ensure second-best match isn't already taken
            if second_nucleus_idx not in assigned_nuclei:
                final_indices[cell_idx] = second_nucleus_idx
                final_distances[cell_idx] = second_distance
                assigned_nuclei[second_nucleus_idx] = cell_idx
                exclusivity_qc[cell_idx] = 1

    print("Indices after exclusivity:", np.sum(final_indices!=-1))

    # Update cells_df with match information
    if bin_size == "b2c":
        cells_df.obs["class"] = np.where(
            final_indices != -1, refined_df.iloc[final_indices]["class2"].values, np.nan)
        cells_df.obs["distance"] = np.where(
            final_indices != -1, final_distances, np.inf)
        cells_df.obs["x_refined"] = np.where(
            final_indices != -1, refined_df.iloc[final_indices]["x_fullres"].values, np.nan)
        cells_df.obs["y_refined"] = np.where(
            final_indices != -1, refined_df.iloc[final_indices]["y_fullres"].values, np.nan)

        # Add QC and threshold information
        cells_df.obs["pixel_threshold"] = pixel_threshold
        cells_df.obs["micron_threshold"] = micron_threshold
        cells_df.obs["qc_match"] = (cells_df.obs['distance'] < pixel_threshold).astype(int)  # Binary flag for exclusivity match success
        cells_df.obs["qc_exclusive"] = exclusivity_qc  # New column for exclusivity QC


# # Save the matched results
if save_new:
    if bin_size == "b2c":
        # Ensure all columns in cells_df.obs are string-compatible
        for col in cells_df.obs.columns:
            if isinstance(cells_df.obs[col].dtype, pd.CategoricalDtype):  # Check for CategoricalDtype
                cells_df.obs[col] = cells_df.obs[col].astype(str)  # Convert Categorical to string
            elif not np.issubdtype(cells_df.obs[col].dtype, np.object_):  # Check for other non-string types
                cells_df.obs[col] = cells_df.obs[col].astype(str)  # Convert other non-string types to string

        # Save the AnnData object
        output_path = os.path.join(data_path, f"b2c_cellvit_{ground_truth}_fullres_v2.h5ad")
        cells_df.write(output_path)
        print(f"Updated AnnData saved to {output_path}")



        # cells_df.to_csv(f"{data_path}/cells_matched_preprocessed_{ground_truth}_v2.csv", index=False)


# %% Testing Geojson files match 

# Initialize a list to store GeoJSON features
geojson_features = []


# Define colors
pred_color = [255, 0, 0]  # Red for CellViT/Refined centroids
line_color = [0, 0, 0]       # Black for lines connecting matched centroids

xenium_color = [0, 0, 255]   # Blue for Xenium centroids


# Create GeoJSON features for CellViT centroids
for coord in nuclei_coords:
    feature = geojson.Feature(
        geometry=geojson.Point(coord.tolist()),
        properties={
            "type": ground_truth,
            "color": pred_color
        }
    )
    geojson_features.append(feature)

# Create GeoJSON features for Xenium cell coordinates
for coord in cell_coords:
    feature = geojson.Feature(
        geometry=geojson.Point(coord.tolist()),
        properties={
            "type": "xenium",
            "color": xenium_color
        }
    )
    geojson_features.append(feature)

# Add a column to cells_df to store the matched CellViT index
cells_df.obs[f"{ground_truth}_index"] = final_indices  # Indices from KDTree query

# Create GeoJSON features for lines connecting matched nuclei with QC match
for i, row in cells_df.obs.iterrows():
    if int(row['qc_match']) == 1 and int(row["qc_exclusive"]) == 1:  # Only for QC-matched pairs
        nuc_coord = nuclei_coords[row[f'{ground_truth}_index']].tolist()  # Matched CellViT centroid
        visium_coord = [int(float(row["x_centroid"])), int(float(row["y_centroid"]))]  # Xenium centroid
        
        # Create a LineString feature connecting the two centroids
        line_feature = geojson.Feature(
            geometry=geojson.LineString([nuc_coord, visium_coord]),
            properties={
                "type": "MatchLine",
                "color": line_color
            }
        )
        geojson_features.append(line_feature)

# Combine all features into a GeoJSON FeatureCollection
combined_geojson = geojson.FeatureCollection(geojson_features)

# Save to a GeoJSON file
output_path = f"{data_path}/combined_coordinates_{ground_truth}_v2.geojson"
with open(output_path, "w") as f:
    geojson.dump(combined_geojson, f)

print(f"Combined GeoJSON with lines saved to {output_path}")

# %% Save Refined Geojson dectection and classes

# Define color mapping for each class
class_color_mapping = {
    'o': [255, 255, 255],  # White
    't': [255, 0, 0],      # Red
    'f': [0, 0, 255],      # Blue
    'l': [0, 255, 0]       # Green
}

# Initialize a list to store GeoJSON features
geojson_features = []

# Iterate through refined_df to create GeoJSON features with colors
for _, row in refined_df.iterrows():
    cell_class = row['class2']
    feature = geojson.Feature(
        geometry=geojson.Point((row['x_fullres'], row['y_fullres'])),
        properties={
            "class": cell_class,
            "color": class_color_mapping[cell_class]
        }
    )
    geojson_features.append(feature)

# Combine all features into a GeoJSON FeatureCollection
geojson_data = geojson.FeatureCollection(geojson_features)

# Save the GeoJSON file
output_geojson_path = f"{data_path}/refined_cells_colored.geojson"
with open(output_geojson_path, "w") as f:
    geojson.dump(geojson_data, f)

print(f"Colored GeoJSON file saved to {output_geojson_path}")


#%% # %% Save Refined classes, Xenium Detection Geojson 

# Define color mapping for each class
class_color_mapping = {
    'o': [255, 255, 255],  # White
    't': [255, 0, 0],      # Red
    'f': [0, 0, 255],      # Blue
    'l': [0, 255, 0],       # Green
    'nan': [0,0,0]   # black for unpaired detections
}

# Initialize a list to store GeoJSON features
geojson_features = []

# Iterate through refined_df to create GeoJSON features with colors
for _, row in cells_df.obs.iterrows():
    cell_class = str(row['class'])
    if cell_class != "nan":
        feature = geojson.Feature(
            geometry=geojson.Point((int(float(row['x_centroid'])), int(float(row['y_centroid'])))),
            properties={
                "class": cell_class,
                "color": class_color_mapping[cell_class]
            }
        )
        geojson_features.append(feature)

# Combine all features into a GeoJSON FeatureCollection
geojson_data = geojson.FeatureCollection(geojson_features)

# Save the GeoJSON file
output_geojson_path = f"{data_path}/refined_cells_xenium_detect_v2.geojson"
with open(output_geojson_path, "w") as f:
    geojson.dump(geojson_data, f)

print(f"Colored GeoJSON file saved to {output_geojson_path}")

# %%  Save 