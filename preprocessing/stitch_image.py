#%% 
import os
import numpy as np
import pandas as pd
from glob import glob
from openslide import open_slide
from scipy.io import loadmat
from skimage.measure import label, regionprops
from utils import load_config, full_path    
import pickle 
from tqdm import tqdm
from tqdm import tqdm
from skimage.transform import resize
import matplotlib.pyplot as plt
import pickle

#%% Load slide and tile coordinates
def stitch_mask():
    config_file = "config.yaml"
    config = load_config(config_file)
    aitil_config = config["aitil"]
    project_paths = full_path(config)

    # Define paths to .mat files and corresponding tile images
    mat_dir = os.path.join(
        aitil_config["cell_seg"], "mat", aitil_config["image"]
    )
    mat_files = glob(os.path.join(mat_dir, "*.mat"))
    tile_dir = aitil_config["tiles"]
    tile_files = [
        f.replace(mat_dir, tile_dir).replace(".mat", ".jpg")
        for f in mat_files
    ]

    # Load scale factors
    wsi_file = project_paths['wsi_image']
    slide = open_slide(wsi_file)

    # Load cell position data
    cellpos_file = os.path.join(
        aitil_config["cell_class"], "CellPos",
        f"{aitil_config['image']}_cellPos.csv"
    )
    cellPos_df = pd.read_csv(cellpos_file)
    cellPos_df[['x_scaled', 'y_scaled']] = cellPos_df[['x', 'y']].apply(lambda col: col * 16)


    # Create a DataFrame of unique tile coordinates
    tile_coords_df = cellPos_df[['tile_file', 'tile_x', 'tile_y']].drop_duplicates()

    #% Load rescale factor
    param_file = aitil_config["params"]
    with open(param_file, "rb") as f:
        params = pickle.load(f)
    rescale_factor = params['rescale']

    #% Initialize the downsampled whole slide and label counter
    # Create an empty array for the downsampled whole-slide image
    wsi_ds = np.zeros((int(slide.dimensions[1] / rescale_factor), int(slide.dimensions[0] / rescale_factor)), dtype=np.uint32)

    # Initialize label counter
    label_counter = 1  # Start labeling from 1

    #% Process each tile
    for _, row in tqdm(tile_coords_df.iterrows(), total=tile_coords_df.shape[0], desc="Processing Tiles"):
        mat_file_path = os.path.join(mat_dir, row['tile_file'].replace(".csv", ".mat"))
        mat_data = loadmat(mat_file_path)
        cell_seg_mask = mat_data['mat'][0][0][1]

        cell_labels = label(cell_seg_mask)
        unique_tile_labels = np.where(cell_labels > 0, cell_labels + label_counter, 0)
        
        y_start, x_start = row['tile_y'], row['tile_x']
        y_end = y_start + cell_seg_mask.shape[0]
        x_end = x_start + cell_seg_mask.shape[1]
        
        # Handle edge case where tile exceeds bounds
        if y_end > wsi_ds.shape[0] or x_end > wsi_ds.shape[1]:
            # Debugging dimensions
            print(f"Tile {row['tile_file']} exceeds bounds. Clipping.")
            print(f"Mask: {unique_tile_labels.shape}, Crop: {wsi_ds[y_start:y_end, x_start:x_end].shape}")
            y_end = min(y_end, wsi_ds.shape[0])
            x_end = min(x_end, wsi_ds.shape[1])
            cell_seg_mask = cell_seg_mask[: y_end - y_start, : x_end - x_start]
            unique_tile_labels = unique_tile_labels[: y_end - y_start, : x_end - x_start]
        
        # Place the tile into the downsampled whole slide
        wsi_ds[y_start:y_end, x_start:x_end] = unique_tile_labels
        
        # Update label counter
        label_counter += cell_labels.max()


    #% Upsample to original dimensions if needed
    wsi_full_res = resize(wsi_ds, (slide.dimensions[1],slide.dimensions[0]), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint32)

    full_mask_file = os.path.join(aitil_config["full_mask"], aitil_config["image"].replace("tif", "pkl"))

    if not os.path.exists(os.path.dirname((full_mask_file))):
        os.mkdir(os.path.dirname(full_mask_file))

    pickle.dump(wsi_full_res, open(full_mask_file, "wb"))

if __name__ == "__main__":
    stitch_mask()
    
#%%

# Define the starting point for the 1000x1000 window
# x_start, y_start = 12866,30000  # Example coordinates, adjust as needed

# # Extract the 1000x1000 region from the slide at the original resolution
# slide_region = slide.read_region((x_start, y_start), 0, (1000, 1000)).convert("RGB")

# #% Extract the corresponding region from the stitched mask (already upscaled to match the slide dimensions)
# mask_region = wsi_full_res[y_start:y_start + 1000, x_start:x_start + 1000]

# # Plot the slide region and mask overlay for sanity check
# plt.figure(figsize=(10, 5))

# # Show the slide region
# plt.subplot(1, 2, 1)
# plt.imshow(slide_region)
# plt.title("Slide Region")

# # Show the corresponding mask region overlayed on the slide region
# plt.subplot(1, 2, 2)
# plt.imshow(slide_region)
# plt.imshow(mask_region, alpha=0.5, cmap="viridis")  # Overlay mask with transparency
# plt.title("Slide with Mask Overlay")

# plt.show()


#%%