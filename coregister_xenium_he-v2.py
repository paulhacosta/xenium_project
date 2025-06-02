#!/usr/bin/env python
# coding: utf-8

import os 
import glob
import json
import pandas as pd
import tifffile as tf
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from skimage.transform import warp, AffineTransform
import cv2
import dask.array as da
from dask_image.ndinterp import affine_transform
import pyvips


# -------------------------------
# Helper function for image resizing (for pyramid levels)
# -------------------------------
def img_resize(img, scale_factor):
    width = int(np.floor(img.shape[1] * scale_factor))
    height = int(np.floor(img.shape[0] * scale_factor))
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

# -------------------------------
# Function to write OME-TIFF with pyramid levels.
# -------------------------------
def write_ome_tif(filename, image, channel_names, photometric_interp, metadata, subresolutions, tile_size=(1024, 1024), sub_res=True):
    fn = filename + ".ome.tif"
    with tf.TiffWriter(fn, bigtiff=True) as tif:
        # Use new pixel sizes from the metadata
        px_size_x = metadata['PhysicalSizeX']
        px_size_y = metadata['PhysicalSizeY']
        options = dict(
            photometric=photometric_interp,
            tile=tile_size,
            maxworkers=16,
            compression=None, # jpeg, jpeg2000 or None
            # compressionargs={'level': 85},
            resolutionunit='CENTIMETER',
        )
        
        print("Writing pyramid level 0")
        tif.write(
            image,
            subifds=subresolutions,
            resolution=(1e4 / px_size_x, 1e4 / px_size_y),
            metadata=metadata,
            **options
        )
        
        if sub_res:
            scale = 1
            for i in range(subresolutions):
                scale /= 2
                # Special handling for grayscale images (minisblack)
                if photometric_interp == 'minisblack':
                    if image.shape[0] < image.shape[-1]:
                        image = np.moveaxis(image, 0, -1)
                        image = img_resize(image, 0.5)
                        image = np.moveaxis(image, -1, 0)
                else:
                    image = img_resize(image, 0.5)

                print("Writing pyramid level {}".format(i + 1))
                tif.write(
                    image,
                    subfiletype=1,
                    resolution=(1e4 / scale / px_size_x, 1e4 / scale / px_size_y),
                    **options
                )


# Define cancer type and associated folder mapping
cancer = "breast"  # or "lung", "prostate", etc.

xenium_folder_dict = {
    "lung": "Xenium_Prime_Human_Lung_Cancer_FFPE_outs",
    "breast": "Xenium_Prime_Breast_Cancer_FFPE_outs",
    "lymph_node": "Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_outs",
    "prostate": "Xenium_Prime_Human_Prostate_FFPE_outs",
    "skin": "Xenium_Prime_Human_Skin_FFPE_outs",
    "ovarian": "Xenium_Prime_Ovarian_Cancer_FFPE_outs",
    "cervical": "Xenium_Prime_Cervical_Cancer_FFPE_outs"
}


    
xenium_folder = xenium_folder_dict[cancer]
data_path = f"/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/{xenium_folder}"
experiment_file = os.path.join(data_path, "experiment.xenium")

# Parse the alignment CSV file and create the affine transform
alignment_path = glob.glob(f"{data_path}/*he_imagealignment.csv")[0]
alignment = pd.read_csv(alignment_path)

# Build the alignment matrix:
alignment_matrix = np.array([
    [float(num) for num in alignment.columns.values],
    list(alignment.iloc[0].values),
    list(alignment.iloc[1].values),
])

print("Alignment Matrix:")
print(alignment_matrix)

# Create the affine transformation object
alignment_transform = AffineTransform(matrix=alignment_matrix)

# Load images
print("Loading Images")
he_image_path = os.path.join(data_path, xenium_folder.replace('outs', 'he_image.ome.tif'))
imf_image_path = os.path.join(data_path, "morphology.ome.tif")
with tf.TiffFile(imf_image_path) as tif:
    # Get the dimensions from the first page.
    page = tif.pages[0]
    imf_shape = page.shape
    print("imF image dimensions (shape):", imf_shape)

he_image = tf.imread(he_image_path)   # Original H&E image (OME-TIFF)

print("H&E image shape:", he_image.shape)
#

# Load the experiment JSON data and extract the new pixel size
with open(experiment_file, 'r') as f:
    experiment_data = json.load(f)
pixel_size = experiment_data.get("pixel_size")
print("New pixel size from experiment data:", pixel_size)

print(f"Using pyvips to perform affine transformation")


he_vips = pyvips.Image.new_from_file(he_image_path, access='sequential')

# Invert alignment matrix, which should be 3x3
# inv_matrix = np.linalg.inv(alignment_matrix) # Nevermind
inv_matrix = alignment_matrix   # Pyvips expects non-inverted matrix

# Suppose the inverted matrix is:
#  [ a  b  tx ]
#  [ d  e  ty ]
#  [ 0  0   1 ]
a, b, tx = inv_matrix[0, 0], inv_matrix[0, 1], inv_matrix[0, 2]
d, e, ty = inv_matrix[1, 0], inv_matrix[1, 1], inv_matrix[1, 2]

# Construct the 2x2 part as a single list
matrix_2x2 = [a, b, d, e]

# Create an interpolator (bilinear, nearest, bicubic, etc.)
interp = pyvips.Interpolate.new("bilinear")

# Run the transform in pyvips
# - Pass the 2x2 matrix as the first argument
# - Pass translation offsets as odx=... and ody=...
warped_vips = he_vips.affine(
    matrix_2x2,
    odx=tx,
    ody=ty,
    oarea=[0, 0, imf_shape[1], imf_shape[0]],  # [left, top, width, height]
    interpolate=interp
)

# Check the basic properties
width = warped_vips.width
height = warped_vips.height
bands = warped_vips.bands  # number of channels (3 for RGB, etc.)
pixel_format = warped_vips.format  # e.g. 'uchar' (8-bit), 'ushort', 'float', etc.

print("Warped image shape in pyvips:", (height, width, bands))
print("Pixel format in pyvips:", pixel_format)

# Convert pyvips image to a bytes buffer
raw_bytes = warped_vips.write_to_memory()

# Create a NumPy array from the buffer. 
# For an 8-bit (uchar) image, use np.uint8.
registered_he_image_uint8 = np.frombuffer(raw_bytes, dtype=np.uint8)

# Reshape to (height, width, bands)
registered_he_image_uint8 = registered_he_image_uint8.reshape((height, width, bands))

print("Registered image shape as NumPy array:", registered_he_image_uint8.shape)
print("Dtype:", registered_he_image_uint8.dtype)

# -------------------------------
# Prepare metadata for output file
# -------------------------------
# Extract tiling information and pyramid level count from original H&E image
with tf.TiffFile(he_image_path) as original_tif:
    tile_width = original_tif.pages[0].tilewidth
    tile_height = original_tif.pages[0].tilelength
    num_subifds = len(original_tif.pages[0].subifds)
    print("Tile width:", tile_width, "Tile height:", tile_height, "Pyramid levels:", num_subifds)
    original_ome_xml = original_tif.ome_metadata

    meta = tf.xml2dict(original_tif.ome_metadata)
    res = (meta['OME']['Image']['Pixels']['PhysicalSizeX'],meta['OME']['Image']['Pixels']['PhysicalSizeY'])
    unit = meta['OME']['Image']['Pixels']['PhysicalSizeXUnit']

    try:
        channel_names=[]
        for i, element in enumerate(meta['OME']['Image']['Pixels']['Channel']):
            channel_names.append(meta['OME']['Image']['Pixels']['Channel'][i]['Name'])
    except KeyError:
                channel_names=None

# Expand this dictionary with more metadata if needed.
metadata = {
    'axes': 'YXC',  # Y = height, X = width, C = channels
    'PhysicalSizeX': pixel_size,
    'PhysicalSizeXUnit': unit,
    'PhysicalSizeY': pixel_size,
    'PhysicalSizeYUnit': unit,
    'Channel': {'Name': channel_names}
}

# For H&E (color) images, use the 'rgb' photometric interpretation.
photometric_interp = 'rgb'

# Define the output filename (appending a suffix to the original filename)
output_filename = he_image_path.rsplit('.', 2)[0] + "_registered"

# -------------------------------
# Save the new registered OME‑TIFF with updated metadata
# -------------------------------
write_ome_tif(
    output_filename,
    registered_he_image_uint8,
    channel_names=None,
    photometric_interp=photometric_interp,
    metadata=metadata,
    subresolutions=num_subifds,
    tile_size=(tile_width, tile_height),
    sub_res=True
)

print("New registered OME‑TIFF saved as:", output_filename + ".ome.tif\n")




