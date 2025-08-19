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
aitil_root = "/rsrch5/home/plm/phacosta/aitil_t6/output_pa2"
file_name = "Xenium_Prime_Human_Lung_Cancer_FFPE_he_image_coregistered_pyramid.ome.tif"
aitil_out_folders = ["1_cws_tiling", "2_tissue_seg", "3_cell_seg", "4_5_stitch_output", "4_cell_class", "step5_spatial"]
aitil_path_dict = {f: f"{aitil_root}/{f}" for f in aitil_out_folders}
param = pickle.load(open(f"{aitil_root}/1_cws_tiling/{file_name}/param.p", "rb"))
r9 = "/rsrch9/home/plm/idso_fa1_pathology"
segmask_label_dir = f"{r9}/TIER2/paul-xenium/mit-b3-finetuned-TCGAbcssWsss10xLuadMacroMuscle-40x896-20x512-10x256re/mask_cws512/Xenium_Prime_Human_Lung_Cancer_FFPE_he_image_coregistered_pyramid.ome.tif/"
refined_label_file = f"{r9}/TIER2/paul-xenium/aitil_outputs/4_cell_class_segformerMacro/CellPos/Xenium_Prime_Human_Lung_Cancer_FFPE_he_image_coregistered_pyramid.ome.csv"
cellvit_labels_file = "/rsrch5/home/plm/phacosta/CellViT/example/output/preprocessing/coregistered/Xenium_Prime_Human_Lung_Cancer_FFPE_he_image_coregistered_pyramid.ome/cell_detection/cell_detection.geojson"


# %% Set Rescale factor
rescale_factor = param["rescale"]

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

data_path = f"{r9}/TIER1/paul-xenium/public_data/10x_genomics/{xenium_folder}"

# Load the JSON file
with open(f"{data_path}/experiment.xenium", "r") as f:
    experiment_metadata = json.load(f) # get experiment metadata
pixel_size = experiment_metadata["pixel_size"] # μm per pixel

# Load cell data 
cells_df = pd.read_parquet(os.path.join( data_path, "cells.parquet"))  
# spatial_data = cells_df[["cell_id", "x_centroid", "y_centroid", "transcript_counts", "total_counts", "cell_area"]]

# Coordinates are saved in microns
cells_df = cells_df.rename(columns={"x_centroid":"x_centroid_um", "y_centroid":"y_centroid_um"})

# Convert to pixel coordinates
cells_df["x_centroid"] = cells_df["x_centroid_um"] / pixel_size
cells_df["y_centroid"] = cells_df["y_centroid_um"] / pixel_size

# %%

# For aitil, refined, and cellvit only 
micron_threshold = 10 
pixel_threshold = micron_threshold / pixel_size

if ground_truth=="aitil":
    aitil_df = pd.read_csv(os.path.join(aitil_path_dict["4_cell_class"], "CellPos", f"{file_name}_cellPos.csv"))

    aitil_df[['x', 'y']] = aitil_df[['x', 'y']].apply(lambda x: (x * 16) * rescale_factor)
    aitil_df.columns = ["class", "x_centroid", "y_centroid", "class2", "source", "x_tile", "y_tile"]

    # Extract coordinates from both dataframes
    cell_coords = cells_df[["x_centroid", "y_centroid"]].to_numpy()  # From spatial_data
    nuclei_coords = aitil_df[["x_centroid", "y_centroid"]].to_numpy()    # From aitil_df

    # Build KDTree for aitil_df (nuclei) coordinates
    nuclei_tree = KDTree(nuclei_coords)

    # Query nearest neighbors for each cell in spatial_data
    distances, indices = nuclei_tree.query(cell_coords)

    # Add results to spatial_data
    cells_df["class"] = aitil_df.iloc[indices]["class"].values  # Class of matched nucleus
    cells_df["distance"] = distances                            # Distance to matched nucleus
    cells_df["x_aitil"] = aitil_df.iloc[indices]["x_centroid"].values  # Matched x-coordinate
    cells_df["y_aitil"] = aitil_df.iloc[indices]["y_centroid"].values  # Matched y-coordinate

    # Add new columns to the DataFrame
    cells_df['pixel_threshold'] = pixel_threshold
    cells_df['micron_threshold'] = micron_threshold
    cells_df['qc_match'] = (cells_df['distance'] < pixel_threshold).astype(int)

elif ground_truth=="segmask":
    imf_image_path = f"{data_path}/morphology.ome.tif"
    imf_image = tf.imread(imf_image_path)  # imF image
    target_shape = np.array([imf_image.shape[1], imf_image.shape[2], 3])
    
    del imf_image # delete to free up memory

    # wsi_seg_targets = np.zeros((imf_image.shape[1],imf_image.shape[2],3), dtype=np.uint8)
    wsi_seg_targets = np.zeros((int(target_shape[0] / rescale_factor), int(target_shape[1] / rescale_factor)), dtype=np.uint8)

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

    cell_coords = np.round(cells_df[["x_centroid", "y_centroid"]].to_numpy()).astype(int) # From spatial_data
    cell_classes = final_wsi_seg_targets[cell_coords[:, 1], cell_coords[:, 0]]  # y comes first, then x

    class_to_temp_cat = class_to_color = {
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
    cells_df['class_original'] = cell_classes
    cells_df['class'] = cells_df['class_original'].map(class_to_temp_cat)


    cells_df['qc_match'] = (cells_df['class'] !=0 ).astype(int)

elif ground_truth == "cellvit":
    print("Using CellViT as ground truth")

    # Load the CellViT labels (GeoJSON)
    with open(cellvit_labels_file, "r") as f:
        cellvit_geojson = geojson.load(f)

    # Initialize lists to store centroids and class labels
    cellvit_centroids = []
    cellvit_classes = []

    # Loop through each annotation in the GeoJSON
    for annotation in cellvit_geojson:
        class_label = annotation["properties"]["classification"]["name"]  # Extract class name
        coordinates_list = annotation["geometry"]["coordinates"]  # List of centroid coordinates
        
        cellvit_centroids.extend(coordinates_list)  # Add all centroids
        cellvit_classes.extend([class_label] * len(coordinates_list))  # Repeat class label

    # Convert to NumPy arrays for KDTree usage
    nuclei_coords = np.array(cellvit_centroids)  # Shape: (N, 2)
    cellvit_classes = np.array(cellvit_classes)      # Shape: (N,)

    # Build KDTree for CellViT centroids
    cellvit_tree = KDTree(nuclei_coords)

    # Query nearest neighbors for each cell in Xenium data
    cell_coords = cells_df[["x_centroid", "y_centroid"]].to_numpy()
    distances, indices = cellvit_tree.query(cell_coords)

    # Add results to Xenium DataFrame
    cells_df["class"] = cellvit_classes[indices]
    cells_df["distance"] = distances
    cells_df["qc_match"] = (distances < pixel_threshold).astype(int)


elif ground_truth=="refined":
    refined_df = pd.read_csv(refined_label_file)

    # Extract coordinates from both dataframes
    cell_coords = cells_df[["x_centroid", "y_centroid"]].to_numpy()  # From spatial_data
    nuclei_coords = refined_df[["x_fullres", "y_fullres"]].to_numpy()    # From aitil_df

    # Build KDTree for aitil_df (nuclei) coordinates
    nuclei_tree = KDTree(nuclei_coords)

    # Query nearest neighbors for each cell in spatial_data
    distances, indices = nuclei_tree.query(cell_coords)

    # Add results to spatial_data
    cells_df["class"] = refined_df.iloc[indices]["class2"].values  # Class of matched nucleus
    cells_df["distance"] = distances                            # Distance to matched nucleus
    cells_df["x_refined"] = refined_df.iloc[indices]["x_fullres"].values  # Matched x-coordinate
    cells_df["y_refined"] = refined_df.iloc[indices]["y_fullres"].values  # Matched y-coordinate

    # Add new columns to the DataFrame
    cells_df['pixel_threshold'] = pixel_threshold
    cells_df['micron_threshold'] = micron_threshold
    cells_df['qc_match'] = (cells_df['distance'] < pixel_threshold).astype(int)


# # Save the matched results
if save_new:

    cells_df.to_csv(f"{data_path}/cells_matched_spatial_{ground_truth}_v2.csv", index=False)


# %% Testing visualization for segmask labels
# color_to_class_fixed = {c:global_color_to_class[c]-1 for c in global_color_to_class}
# colors = [np.array(rgb) / 255 for rgb, cls in sorted(color_to_class_fixed.items(), key=lambda x: x[1])]

# # Create a ListedColormap
# custom_cmap = ListedColormap(colors)

# # Define boundaries for each class (e.g., 0.5 to 1.5, 1.5 to 2.5, etc.)
# boundaries = list(range(len(colors) + 1))
# norm = BoundaryNorm(boundaries, custom_cmap.N)

# plt.figure(figsize=(10,10))
# plt.imshow(final_wsi_seg_targets,cmap=custom_cmap)
# plt.title("Converted Class Mask")

# plt.figure(figsize=(10, 7))  # Set the figure size
# plt.scatter(
#     cell_coords[:, 0],  # X-coordinates
#     cell_coords[:, 1],  # Y-coordinates
#     c=cell_classes,     # Colors based on class
#     cmap=custom_cmap,
#     s=3,                # Marker size
#     alpha=0.8           # Transparency for better visibility
# )

# # Invert the y-axis
# plt.ylim(0,37348)
# plt.xlim(0,54086)
# plt.gca().invert_yaxis()
# plt.title("Cell Classes")
# # Display the plot
# plt.show()

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
cells_df[f"{ground_truth}_index"] = indices  # Indices from KDTree query

# Create GeoJSON features for lines connecting matched nuclei with QC match
for i, row in cells_df.iterrows():
    if row['qc_match'] == 1:  # Only for QC-matched pairs
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
output_path = f"{data_path}/combined_coordinates_{ground_truth}.geojson"
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
    'l': [0, 255, 0]       # Green
}

# Initialize a list to store GeoJSON features
geojson_features = []

# Iterate through refined_df to create GeoJSON features with colors
for _, row in cells_df.iterrows():
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