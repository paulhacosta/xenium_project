#%%
#%% 
import os
import numpy as np
import pandas as pd
from glob import glob
from openslide import open_slide
from scipy.io import loadmat
from skimage.measure import label, regionprops
import pickle 
from tqdm import tqdm
from tqdm import tqdm
from skimage.transform import resize
import matplotlib.pyplot as plt
import pickle
import matplotlib.colors as mcolors
from skimage.measure import regionprops


#%% Load slide and tile coordinates
def stitch_mask(slide_path, aitil_path):
    file_name = os.path.basename(slide_path)
    wsi_file = slide_path
    slide = open_slide(wsi_file)

    mat_dir = f"{aitil_path}/3_cell_seg/mat/{file_name}"

    # Load cell position data
    cellpos_file = f"{aitil_path}/4_cell_class/CellPos/{file_name}_cellPos.csv"
    cellPos_df = pd.read_csv(cellpos_file)
    cellPos_df[['x_scaled', 'y_scaled']] = cellPos_df[['x', 'y']].apply(lambda col: col * 16)

    class_dict = {'o': 1, 'f': 2, 't': 3, 'l': 4}

    # Create a DataFrame of unique tile coordinates
    tile_coords_df = cellPos_df[['tile_file', 'tile_x', 'tile_y']].drop_duplicates()

    # Load rescale factor
    param_file = f"{aitil_path}/1_cws_tiling/{file_name}/param.p"
    with open(param_file, "rb") as f:
        params = pickle.load(f)
    rescale_factor = params['rescale']

    # Initialize the downsampled whole slide images
    wsi_ds = np.zeros((int(slide.dimensions[1] / rescale_factor), int(slide.dimensions[0] / rescale_factor)), dtype=np.uint32)
    wsi_ds_class = np.zeros_like(wsi_ds, dtype=np.uint8)  # For class values

    # Initialize label counter
    label_counter = 1  # Start labeling from 1

    # Process each tile
    for _, row in tqdm(tile_coords_df.iterrows(), total=tile_coords_df.shape[0], desc="Processing Tiles"):
        mat_file_path = os.path.join(mat_dir, row['tile_file'].replace(".csv", ".mat"))
        mat_data = loadmat(mat_file_path)
        cell_seg_mask = mat_data['mat'][0][0][1]

        cell_labels = label(cell_seg_mask)

        # Filter cell positions for this tile
        tile_classes = cellPos_df[cellPos_df['tile_file'] == row['tile_file']]

        # Assign class values based on cell positions
        cell_classes = np.zeros_like(cell_labels, dtype=np.uint8)
        for _, cell in tile_classes.iterrows():
            # Scale cell positions to match mask coordinates
            cell_x = int(cell['x_scaled'] - row['tile_x'])
            cell_y = int(cell['y_scaled'] - row['tile_y'])

            if 0 <= cell_x < cell_seg2_mask.shape[1] and 0 <= cell_y < cell_seg_mask.shape[0]:
                cell_label = cell_labels[cell_y, cell_x]
                if cell_label > 0:
                    cell_classes[cell_labels == cell_label] = class_dict.get(cell['class'], 0)

        # Adjust for global placement in the stitched image
        unique_tile_labels = np.where(cell_labels > 0, cell_labels + label_counter, 0)

        y_start, x_start = row['tile_y'], row['tile_x']
        y_end = y_start + cell_seg_mask.shape[0]
        x_end = x_start + cell_seg_mask.shape[1]
        
        # Handle edge case where tile exceeds bounds
        if y_end > wsi_ds.shape[0] or x_end > wsi_ds.shape[1]:
            y_end = min(y_end, wsi_ds.shape[0])
            x_end = min(x_end, wsi_ds.shape[1])
            cell_seg_mask = cell_seg_mask[: y_end - y_start, : x_end - x_start]
            unique_tile_labels = unique_tile_labels[: y_end - y_start, : x_end - x_start]
            cell_classes = cell_classes[: y_end - y_start, : x_end - x_start]
        
        # Place the tile into the downsampled whole slide
        wsi_ds[y_start:y_end, x_start:x_end] = unique_tile_labels
        wsi_ds_class[y_start:y_end, x_start:x_end] = cell_classes

        # Update label counter
        label_counter += cell_labels.max()

    # Upsample to original dimensions
    wsi_full_res = resize(
        wsi_ds, 
        (slide.dimensions[1], slide.dimensions[0]), 
        order=0, preserve_range=True, anti_aliasing=False
    ).astype(np.uint32)

    wsi_fullres_class = resize(
        wsi_ds_class, 
        (slide.dimensions[1], slide.dimensions[0]), 
        order=0, preserve_range=True, anti_aliasing=False
    ).astype(np.uint8)

    return wsi_full_res, wsi_fullres_class, slide


#%%
if __name__ == "__main__":
    file = "Xenium_Prime_Human_Lung_Cancer_FFPE_he_image_coregistered_pyramid.ome.tif"
    aitil_path = "/rsrch5/home/plm/phacosta/aitil_t6/output_pa2"
    slide_path = f"/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Xenium_Prime_Human_Lung_Cancer_FFPE_outs/{file}"

    # wsi_fullres_obj, wsi_fullres_class, slide = stitch_mask(slide_path, aitil_path)



#%%%


file_name = os.path.basename(slide_path)
wsi_file = slide_path
slide = open_slide(wsi_file)

mat_dir = f"{aitil_path}/3_cell_seg/mat/{file_name}"
csv_dir = f"{aitil_path}/4_cell_class/csv/{file_name}"

# Load cell position data
cellpos_file = f"{aitil_path}/4_cell_class/CellPos/{file_name}_cellPos.csv"
cellPos_df = pd.read_csv(cellpos_file)
cellPos_df[['x_scaled', 'y_scaled']] = cellPos_df[['x', 'y']].apply(lambda col: col * 16)

class_dict = {'o': 1, 'f': 2, 't': 3, 'l': 4}

# Create a DataFrame of unique tile coordinates
tile_coords_df = cellPos_df[['tile_file', 'tile_x', 'tile_y']].drop_duplicates()

# Load rescale factor
param_file = f"{aitil_path}/1_cws_tiling/{file_name}/param.p"
with open(param_file, "rb") as f:
    params = pickle.load(f)
rescale_factor = params['rescale']

# Initialize the downsampled whole slide images
wsi_ds = np.zeros((int(slide.dimensions[1] / rescale_factor), int(slide.dimensions[0] / rescale_factor)), dtype=np.uint32)
wsi_ds_class = np.zeros_like(wsi_ds, dtype=np.uint8)  # For class values

# Initialize label counter
label_counter = 1  # Start labeling from 1

# Process each tile
for _, row in tqdm(tile_coords_df.iterrows(), total=tile_coords_df.shape[0], desc="Processing Tiles"):
    mat_file_path = os.path.join(mat_dir, row['tile_file'].replace(".csv", ".mat"))
    csv_file_path = os.path.join(csv_dir, row['tile_file'])
    mat_data = loadmat(mat_file_path)
    csv_data = pd.read_csv(csv_file_path)

    cell_seg_mask = mat_data['mat'][0][0][1]
    
    # Label the cell segmentation mask
    cell_ids = label(cell_seg_mask)

    # Extract contours (regions) of the labeled cells
    regions = regionprops(cell_ids)

    # Initialize class mask for this tile
    cell_classes = np.zeros_like(cell_ids, dtype=np.uint8)

    # Iterate through each point in the CSV
    for _, point in csv_data.iterrows():
        x_point, y_point = int(point['V2']), int(point['V3'])  # Coordinates from the CSV
        cell_class = class_dict.get(point['V1'], 0)  # Class value from the CSV
        
        # Check if the point is inside any region
        for region in regions:
            if (y_point, x_point) in region.coords:
                # Assign the class to all pixels in this region
                cell_classes[cell_ids == region.label] = cell_class
                break

    # Map the tile-specific masks into the global masks
    y_start, x_start = row['tile_y'], row['tile_x']
    y_end = y_start + cell_seg_mask.shape[0]
    x_end = x_start + cell_seg_mask.shape[1]
    
    # Ensure the tile fits within the global mask
    if y_end > wsi_ds.shape[0] or x_end > wsi_ds.shape[1]:
        y_end = min(y_end, wsi_ds.shape[0])
        x_end = min(x_end, wsi_ds.shape[1])
        cell_ids = cell_ids[: y_end - y_start, : x_end - x_start]
        cell_classes = cell_classes[: y_end - y_start, : x_end - x_start]

    # Update the global masks
    wsi_ds[y_start:y_end, x_start:x_end] = np.where(cell_ids > 0, cell_ids + label_counter, 0)
    wsi_ds_class[y_start:y_end, x_start:x_end] = cell_classes

    # Update label counter
    label_counter += cell_ids.max()

# Upsample to original dimensions
wsi_fullres_obj = resize(
    wsi_ds, 
    (slide.dimensions[1], slide.dimensions[0]), 
    order=0, preserve_range=True, anti_aliasing=False
).astype(np.uint32)

wsi_fullres_class = resize(
    wsi_ds_class, 
    (slide.dimensions[1], slide.dimensions[0]), 
    order=0, preserve_range=True, anti_aliasing=False
).astype(np.uint8)


    
#%%

test_tile = "Da172.csv"

# Define the starting point for the 1000x1000 window
x_start, y_start = cellPos_df[cellPos_df.tile_file == test_tile][["tile_x", "tile_y"]].iloc[0].values
x_start, y_start = int( x_start*rescale_factor), int(y_start*rescale_factor)
rescaled_size = int(2000*rescale_factor)

# Extract the 1000x1000 region from the slide at the original resolution
slide_region = slide.read_region((x_start, y_start), 0, (rescaled_size, rescaled_size)).convert("RGB")

#% Extract the corresponding region from the stitched mask (already upscaled to match the slide dimensions)
mask_region = wsi_fullres_obj[y_start:y_start + rescaled_size, x_start:x_start + rescaled_size]

class_region = wsi_fullres_class[y_start:y_start + rescaled_size, x_start:x_start + rescaled_size]

# Plot the slide region and mask overlay for sanity check
plt.figure(figsize=(20, 10))

# Show the slide region
plt.subplot(1, 2, 1)
plt.imshow(slide_region)
plt.title("Slide Region")

# Show the corresponding mask region overlayed on the slide region
plt.subplot(1, 2, 2)
plt.imshow(slide_region)
plt.imshow(mask_region, alpha=0.8, cmap="viridis")  # Overlay mask with transparency
plt.title("Slide with Mask Overlay")

# Define a custom colormap with 5 colors from tab10
# cmap = mcolors.ListedColormap(plt.cm.tab10.colors[:np.unique(class_region).size])  # Use the first 5 colors
# norm = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, 5.5, 1), ncolors=5)  # Define boundaries for discrete values

# # Show the corresponding class region overlayed on the slide region
# plt.subplot(1, 3, 3)
# im = plt.imshow(slide_region)
# plt.imshow(class_region, alpha=0.5, cmap=cmap, norm=norm)  # Use custom colormap and normalization
# plt.title("Slide with Class Overlay")
# cbar = plt.colorbar( ticks=np.arange(0, 5))  # Add a colorbar with ticks for 0–4
# cbar.set_label("Class Labels")  # Label the colorbar

plt.tight_layout()
plt.show()


#%%
class_mat_dir = "/rsrch5/home/plm/phacosta/aitil_t6/output_pa2/4_cell_class/mat/Xenium_Prime_Human_Lung_Cancer_FFPE_he_image_coregistered_pyramid.ome.tif"
class_mats = glob(os.path.join(class_mat_dir, "*.mat"))

# %%
import scanpy as sc
data_file = "/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Xenium_Prime_Human_Lung_Cancer_FFPE_outs/preprocessed/fine_tune_refined/processed_xenium_data_fine_tune_refined.h5ad"
adata = sc.read_h5ad(data_file)

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Define custom colors for each class
class_colors = {'t': 'yellow', 'f': 'red', 'o': 'green', 'l': 'blue'}

# Map the class column to corresponding colors
colors = adata.obs['class'].map(class_colors)

# Create scatter plot
plt.figure(figsize=(11, 8))
plt.scatter(
    adata.obs['x_centroid'], 
    adata.obs['y_centroid'], 
    c=colors,              # Use the mapped colors
    s=5,                   # Size of points
    alpha=0.8              # Transparency
)

plt.ylim(0,37348)
plt.xlim(0,54086)
# Add legend with class labels and custom colors
for label, color in class_colors.items():
    plt.scatter([], [], c=color, label=label, s=50)  # Dummy points for legend
plt.legend(title='Class', loc='best')

# Set axis labels and title
plt.xlabel('x_centroid')
plt.ylabel('y_centroid')
plt.title('Scatter Plot of Cells Colored by Class')

plt.gca().invert_yaxis()  # Optional: Invert y-axis if necessary
plt.show()



# %%
