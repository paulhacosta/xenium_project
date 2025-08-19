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
#%% 
save_new = False 
ground_truth = "refined"  # "aitil", "refined", "cellvit", or "segmask"
r9 = "/rsrch9/home/plm/idso_fa1_pathology"


# %%
cancer = "lung"
xenium_folder_dict = {"lung": "Xenium_Prime_Human_Lung_Cancer_FFPE_outs",
                      "breast":"Xenium_Prime_Breast_Cancer_FFPE_outs",
                      "lymph_node": "Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_outs",
                      "prostate": "Xenium_Prime_Human_Prostate_FFPE_outs",
                      "skin": "Xenium_Prime_Human_Skin_FFPE_outs",
                      "ovarian": "Xenium_Prime_Ovarian_Cancer_FFPE_outs",
                      "cervical": "Xenium_Prime_Cervical_Cancer_FFPE_outs"
                      }

xenium_folder = xenium_folder_dict[cancer]

prefix = xenium_folder.rsplit("_",1)[0]
data_path = f"{r9}/TIER1/paul-xenium/public_data/10x_genomics/{xenium_folder}"

file_name = f"{prefix}_he_image_registered.ome.tif"

param = pickle.load(open(f"{r9}/TIER2/paul-xenium/aistil_outputs/1_cws_tiling/{file_name}/param.p", "rb"))
refined_label_file = f"{r9}/TIER2/paul-xenium/aistil_outputs/4_cell_class_segformer/CellPos/{file_name.replace('tif','csv')}"

# Load the JSON file
with open(f"{data_path}/experiment.xenium", "r") as f:
    experiment_metadata = json.load(f) # get experiment metadata
pixel_size = experiment_metadata["pixel_size"] # μm per pixel

# Load cell data 
cells_df = pd.read_parquet(os.path.join( data_path, "cells.parquet"))  

# Coordinates are saved in microns
cells_df = cells_df.rename(columns={"x_centroid":"x_centroid_um", "y_centroid":"y_centroid_um"})

# Convert to pixel coordinates
cells_df["x_centroid"] = cells_df["x_centroid_um"] / pixel_size
cells_df["y_centroid"] = cells_df["y_centroid_um"] / pixel_size


# %% Set Rescale factor
rescale_factor = param["rescale"]
# %%

# For aitil, refined, and cellvit only 
micron_threshold = 10 
pixel_threshold = micron_threshold / pixel_size
 
refined_df = pd.read_csv(refined_label_file)

# Extract coordinates from both dataframes
cell_coords = cells_df[["x_centroid", "y_centroid"]].to_numpy()  # From spatial_data
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
cells_df["class"] = np.where(
    final_indices != -1, refined_df.iloc[final_indices]["class2"].values, None)

cells_df["distance"] = np.where(
    final_indices != -1, final_distances, np.inf)
cells_df["x_refined"] = np.where(
    final_indices != -1, refined_df.iloc[final_indices]["x_fullres"].values, None)

cells_df["y_refined"] = np.where(
    final_indices != -1, refined_df.iloc[final_indices]["y_fullres"].values, None)

# Add QC and threshold information
cells_df["pixel_threshold"] = pixel_threshold
cells_df["micron_threshold"] = micron_threshold
cells_df["qc_match"] = (cells_df['distance'] < pixel_threshold).astype(int)  # Binary flag for exclusivity match success
cells_df["qc_exclusive"] = exclusivity_qc  # New column for exclusivity QC


# # Save the matched results
if save_new:

    cells_df.to_csv(f"{data_path}/cells_matched_spatial_{ground_truth}_v2.csv", index=False)
    print("File saved to:", f"{data_path}/cells_matched_spatial_{ground_truth}_v2.csv")

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
cells_df[f"{ground_truth}_index"] = final_indices  # Indices from KDTree query

# Create GeoJSON features for lines connecting matched nuclei with QC match
for i, row in cells_df.iterrows():
    if row['qc_match'] == 1 and row["qc_exclusive"] == 1:  # Only for QC-matched pairs
        nuc_coord = nuclei_coords[row[f'{ground_truth}_index']].tolist()  # Matched CellViT centroid
        xenium_coord = [row["x_centroid"], row["y_centroid"]]  # Xenium centroid
        
        # Create a LineString feature connecting the two centroids
        line_feature = geojson.Feature(
            geometry=geojson.LineString([nuc_coord, xenium_coord]),
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
# class_color_mapping = {
#     'o': [255, 255, 255],  # White
#     't': [255, 0, 0],      # Red
#     'f': [0, 0, 255],      # Blue
#     'l': [0, 255, 0]       # Green
# }
class_color_mapping = {
    'o': [255, 255, 255],  
    't': [0, 255, 0],      
    'f': [255, 255, 0],      
    'l': [0, 0, 255]       
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
output_geojson_path = f"{data_path}/refined_cells_aistil_detect.geojson"
with open(output_geojson_path, "w") as f:
    geojson.dump(geojson_data, f)

print(f"Colored GeoJSON file saved to {output_geojson_path}")


#%% # %% Save Refined classes, Xenium Detection Geojson 

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
for _, row in cells_df.iterrows():
    if row['qc_match'] == 1 and row["qc_exclusive"] == 1:  # Only for QC-matched pairs
        cell_class = row['class']
        feature = geojson.Feature(
            geometry=geojson.Point((row['x_centroid'], row['y_centroid'])),
            properties={
                "class": cell_class,
                "color": class_color_mapping[cell_class]
            }
        )
        geojson_features.append(feature)

# Combine all features into a GeoJSON FeatureCollection
geojson_data = geojson.FeatureCollection(geojson_features)

# Save the GeoJSON file
output_geojson_path = f"{data_path}/refined_cells_xenium_detect.geojson"
with open(output_geojson_path, "w") as f:
    geojson.dump(geojson_data, f)

print(f"Colored GeoJSON file saved to {output_geojson_path}")

# %%  Save 