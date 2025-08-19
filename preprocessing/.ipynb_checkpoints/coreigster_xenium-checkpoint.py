#%%
import os 
import pandas as pd 
import numpy as np 
import tifffile as tf
from skimage.transform import AffineTransform, warp

#%%

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

data_path = f"/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/{xenium_folder}"

#%% Parse Alignment file 
# Load the alignment file
alignment_path = f"{data_path}/Xenium_Prime_Human_Lung_Cancer_FFPE_he_imagealignment.csv"

alignment = pd.read_csv(alignment_path)

# Alignment matrix
alignment_matrix = np.array([
    [float(num) for num in alignment.columns.values],
    list(alignment.iloc[0].values),
    list(alignment.iloc[1].values),
])

#%% Load Images

# Load the H&E and imF images
he_image_path = f"{data_path}/Xenium_Prime_Human_Lung_Cancer_FFPE_he_image.ome.tif"
imf_image_path = f"{data_path}/morphology.ome.tif"

he_image = tf.imread(he_image_path)  # H&E image
imf_image = tf.imread(imf_image_path)  # imF image


#%% Apply transforms
from skimage.transform import resize

# Rescale H&E image to approximate spatial dimensions of the imF image
scaled_he_image = resize(
    he_image,
    (imf_image.shape[1], imf_image.shape[2]),  # Target dimensions: height, width
    anti_aliasing=True,
    preserve_range=True  # Preserve original intensity range
)

print("Scaled H&E image shape:", scaled_he_image.shape)

#%%
from skimage.transform import AffineTransform, warp

# Extract the transformation parameters from the matrix
alignment_transform = AffineTransform(matrix=alignment_matrix)

# Apply the affine transformation to the rescaled H&E image
aligned_he_image = warp(
    scaled_he_image,
    alignment_transform.inverse,
    output_shape=(imf_image.shape[1], imf_image.shape[2]),
    preserve_range=True
)

print("Aligned H&E image shape:", aligned_he_image.shape)

# %%
import matplotlib.pyplot as plt

# Select a fluorescence channel to overlay
imf_channel = imf_image[0]  # Use the first channel

plt.figure(figsize=(10, 10))
plt.imshow(imf_channel, cmap='gray', alpha=0.5)  # Fluorescence image in grayscale
plt.imshow(aligned_he_image[..., 0], cmap='Reds', alpha=0.5)  # Aligned H&E red channel
plt.title("Overlay of Aligned H&E and imF Image")
plt.axis("off")
plt.show()

# %%
