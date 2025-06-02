#%% 
import os 
import openslide
import numpy as np 
from PIL import Image, ImageDraw, ImageOpss
import matplotlib.pyplot as plt
import re 

#%%

image_path = "/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Xenium_Prime_Human_Lung_Cancer_FFPE_outs/Xenium_Prime_Human_Lung_Cancer_FFPE_he_image_coregistered_pyramid.ome.tif"

slide = openslide.open_slide(image_path)

save_dir = "/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/patches/Xenium_Prime_Human_Lung_Cancer_FFPE_outs/"
os.makedirs(save_dir, exist_ok=True)

# %%

# Define parameters
patch_size = 256
num_rows = 10  # Number of patches down
num_columns = 5  # Number of patches across
start_x, start_y = 5122, 20734  # Starting point 

# Create a blank image to hold the stitched patches
stitched_image = Image.new("RGB", (num_columns * patch_size, num_rows * patch_size))
patches = []
# Create the 5x10 grid of patches and paste them into the stitched image
for col in range(num_columns):  # Loop over columns
    for row in range(num_rows):  # Loop over rows
        # Calculate the top-left corner of the current patch
        x = start_x + col * patch_size
        y = start_y + row * patch_size
        
        # Extract the patch (level 0 for full resolution)
        patch = slide.read_region((x, y), level=0, size=(patch_size, patch_size))
        
        # Convert to RGB (optional)
        patch = patch.convert("RGB")
        patches.append(patch)
        patch.save(f"{save_dir}/Yasin_Model/patch_{col}_{row}.png")

        # Paste the patch into the stitched image
        stitched_image.paste(patch, (col * patch_size, row * patch_size))


# Draw a grid on the stitched image
draw = ImageDraw.Draw(stitched_image)
for col in range(1, num_columns):  # Vertical grid lines
    draw.line([(col * patch_size, 0), (col * patch_size, num_rows * patch_size)], fill="red", width=2)

for row in range(1, num_rows):  # Horizontal grid lines
    draw.line([(0, row * patch_size), (num_columns * patch_size, row * patch_size)], fill="red", width=2)

# Save or display the stitched image
stitched_image.save(f"{save_dir}/patch_reference.png")
stitched_image.show()

# %% Stitch predictions into reference image 
predictions_dir = os.path.join(save_dir, "Yasin_Model", "predictions")

# Set paths
stitched_save_path = os.path.join(save_dir, "stitched_prediction_reference.png")

# Define parameters
patch_size = 256
num_rows = 10
num_columns = 5

# Create a blank image for stitching
stitched_image = Image.new("RGB", (num_columns * patch_size, num_rows * patch_size))

# Process and stitch patches
for col in range(num_columns):
    for row in range(num_rows):
        # Load the prediction image
        img_path = os.path.join(predictions_dir, f"prediction_patch_{col}_{row}.png")
        img = Image.open(img_path).convert("RGBA")  # Ensure alpha channel is included

        # Step 1: Crop to 2500x2500 (removing legend and extra space)
        img_cropped = img.crop((0, 0, 2500, 2500))

        # Step 2: Remove white space around the patch
        img_no_border = ImageOps.crop(img_cropped, border=0)  # Automatically trims white borders

        # Step 3: Resize to 256x256
        img_resized = img_no_border.resize((patch_size, patch_size), Image.ANTIALIAS)

        # Paste the resized patch into the stitched image
        stitched_image.paste(img_resized, (col * patch_size, row * patch_size))

# Save or display the stitched image
stitched_image.save(stitched_save_path)
stitched_image.show()



# %%
