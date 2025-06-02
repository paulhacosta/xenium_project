#%%
import os 
import pandas as pd 
import numpy as np 
import tifffile as tf
from skimage.transform import AffineTransform, warp
from skimage.transform import rotate
from tqdm import tqdm
import matplotlib.pyplot as plt

#%% Define data paths

cancer = "lung" # Only implemented for lung so far
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

print("H&E image shape:", he_image.shape)
print("iMF image shape:", imf_image.shape)

#%%  Extract the transformation parameters from the matrix
alignment_transform = AffineTransform(matrix=alignment_matrix)
he_image_float = he_image/255

# Apply the affine transformation to the rescaled H&E image
registered_he_image = warp(
    he_image_float, 
    inverse_map=alignment_transform.inverse,  # apply the inverse of the given transform
    preserve_range=True,            # maintain original intensity range
    output_shape=(imf_image.shape[1], imf_image.shape[2],3),  # match shape of imf image
    mode='constant',                # how to handle boundaries
    cval=0                          # fill value outside boundaries
)

print("Aligned H&E image shape:", registered_he_image.shape)

output_path = f"{data_path}/Xenium_Prime_Human_Lung_Cancer_FFPE_he_image_coregistered.ome.tif"

#%% Extract original OME XML from original image
with tf.TiffFile(he_image_path) as original_tif:
    original_ome_xml = original_tif.ome_metadata
    
registered_he_image_uint8 = (registered_he_image * 255).astype(np.uint8)

tf.imwrite(
    output_path,
    registered_he_image_uint8,
    photometric='rgb',
    description=original_ome_xml, # reuse original OME-XML metadata
    metadata={'axes': 'YXC'},     # ensure axes are known
    ome=True
)