#%%

import os 
import palom
import numpy as np
import pyvips
import tifffile as tf
import cv2
import openslide 
import geopandas as gpd

# ## KEY CHANGE ##: Add scikit-image for color deconvolution
from skimage.color import rgb2hed
from skimage import img_as_ubyte
import dask.array as da

# %% Define slide 
file_num = 0

root_dir = "/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/TMP-IL-Pilot/20250515__183240__CIMAC_Validation"
data_dict = {
        "output-XETG00522__0066398__Region_1__20250515__183305":"Xenium H&E Meso1-ICON2 TMA 5-21-2025_matching_orientation.ome.tif" ,
        "output-XETG00522__0066402__Region_1__20250515__183305":"Xenium H&E PCF TMA 5-28-2025_matching_orientation.ome.tif"
        }

xenium_folder = list(data_dict.keys())[file_num]
slide_name = data_dict[xenium_folder]
slide_file = os.path.join(root_dir, slide_name)
# morph_file = os.path.join(root_dir, xenium_folder, "morphology_focus", "morphology_focus_0000.ome.tif")

annot_path = "/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/TMP-IL-Pilot/20250515__183240__CIMAC_Validation/registration"

if "Meso" in os.path.basename(slide_name):
    morph_prefix = "meso1"
elif "PCF" in os.path.basename(slide_name):
    morph_prefix = "pcf"
    
slide_core_path = os.path.join(annot_path, "HnE", slide_name)
morph_core_path = os.path.join(annot_path, "morphology_focus", morph_prefix)

# morph_file = os.path.join(root_dir, xenium_folder, "morphology_focus", "morphology_focus_0000.ome.tif")
morph_annot = os.path.join(annot_path, "tma_annotations", f"{morph_prefix}_morphology_focus_0000_annot.geojson")
# load core polygons ---------------------------------------
gdf = gpd.read_file(morph_annot)
# gdf = gdf[gdf['isMissing'] == False].reset_index(drop=True)

#%%

def img_resize(img, scale_factor):
    width  = int(np.floor(img.shape[1] * scale_factor))
    height = int(np.floor(img.shape[0] * scale_factor))
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

def write_ome_tif(filename, image, compression,
                  photometric_interp, metadata, subresolutions,):
    
    with tf.TiffWriter(filename, bigtiff=True) as tif:
        px_size_x = metadata['PhysicalSizeX']
        px_size_y = metadata['PhysicalSizeY']

        if compression == "lzw":
            options = dict(
                photometric=photometric_interp,
                tile=(1024, 1024),
                maxworkers=4,
                compression='lzw',
                resolutionunit='CENTIMETER',
            )
        elif compression == "jpeg":
            options = dict(
                photometric=photometric_interp,
                tile=(1024, 1024),
                maxworkers=4,
                compression='jpeg',
                compressionargs={'level':85},
                resolutionunit='CENTIMETER',
            )

        print("Writing pyramid level 0")
        tif.write(
            image,
            subifds=subresolutions,
            resolution=(1e4 / px_size_x, 1e4 / px_size_y),
            metadata=metadata,
            **options,
        )

        scale = 1
        for i in range(subresolutions):
            scale *= 0.5
            # down‑sample by 2×
            if photometric_interp == 'minisblack':
                if image.shape[0] < image.shape[-1]:
                    image = np.moveaxis(image,0,-1)
                    image = img_resize(image,0.5)
                    image = np.moveaxis(image,-1,0)
            else:
                image = img_resize(image,0.5)

            print("Writing pyramid level {}".format(i+1))
            tif.write(
                image,
                subfiletype=1,
                resolution=(1e4 / scale / px_size_x, 1e4 / scale / px_size_y),
                **options
            )
            
    print("Saved:", filename)




# %%
target_core = None  # core_name or None


for idx, row in gdf.iterrows():
    core_name = row.get('name', f'core_{idx + 1}')
    if target_core:
        print("Targeting specific core:", target_core)
        if core_name != target_core:
            continue

    slide_core = os.path.join(slide_core_path, f"he_{core_name}.ome.tif")
    slide_HED_core = os.path.join(slide_core_path, f"he_{core_name}_HED.ome.tif")

    morph_core = os.path.join(morph_core_path, f"morphology_focus_{core_name}.ome.tif")
    # morph_core = morph_file
    # --- File Paths ---
    output_dir = os.path.join(annot_path, "palom_registration", slide_name.split(".")[0], "registered_cores").replace(" ", "_")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{core_name}_registered.ome.tif")

    if os.path.exists(output_path) and not target_core:
        print(f"{output_path} already exists. Skipping registration")
        continue
    # --- Load readers ---
    ref_reader = palom.reader.OmePyramidReader(morph_core)
    moving_reader = palom.reader.OmePyramidReader(slide_HED_core)

    # Workaround to prevent "file closed" issues
    _ = ref_reader.pyramid[0].blocks.ravel()[0].persist()
    _ = moving_reader.pyramid[0].blocks.ravel()[0].persist()

    # --- Level selection ---
    LEVEL = 0  # use level 1 for alignment
    THUMBNAIL_LEVEL = 3
    CHANNEL_REF = 0  # DAPI
    CHANNEL_MOVING = 0  # G channel often gives better contrast in H&E

    # --- Create aligner ---
    aligner = palom.align.Aligner(
        ref_img=ref_reader.read_level_channels(LEVEL, CHANNEL_REF),
        moving_img=moving_reader.read_level_channels(LEVEL, CHANNEL_MOVING),
        ref_thumbnail=ref_reader.read_level_channels(THUMBNAIL_LEVEL, CHANNEL_REF).compute(),
        moving_thumbnail=moving_reader.read_level_channels(THUMBNAIL_LEVEL, CHANNEL_MOVING).compute(),
        ref_thumbnail_down_factor=ref_reader.level_downsamples[THUMBNAIL_LEVEL] / ref_reader.level_downsamples[LEVEL],
        moving_thumbnail_down_factor=moving_reader.level_downsamples[THUMBNAIL_LEVEL] / moving_reader.level_downsamples[LEVEL],
    )

    # --- Affine + block-based alignment ---
    aligner.coarse_register_affine(n_keypoints=5000)
    aligner.compute_shifts()
    aligner.constrain_shifts()

    he_reader = palom.reader.OmePyramidReader(slide_core)

    # --- Apply transform to H&E only ---
    transformed_HE = palom.align.block_affine_transformed_moving_img(
        ref_img=ref_reader.read_level_channels(LEVEL, CHANNEL_REF),  # used only for shape/template
        moving_img=moving_reader.pyramid[LEVEL],  # full RGB channels from H&E
        mxs=aligner.block_affine_matrices_da
    )

    transformed_HE = palom.align.block_affine_transformed_moving_img(
        ref_img=ref_reader.read_level_channels(LEVEL, CHANNEL_REF),
        moving_img=he_reader.pyramid[LEVEL], # The image to be warped
        mxs=aligner.block_affine_matrices_da # The transformation matrices
    )

    # Compute and convert to channels-last (YXC)
    level0 = np.moveaxis(transformed_HE.compute(), 0, -1)  # (H, W, 3)

    # Get dimensions
    height, width = level0.shape[:2]
    pixel_size = ref_reader.pixel_size
    metadata = {'axes': 'YXC',
                'PhysicalSizeX': pixel_size,
                'PhysicalSizeY': pixel_size,
                'PhysicalSizeXUnit': 'µm',
                'PhysicalSizeYUnit': 'µm',
                'Channel': {'Name': None}}

    write_ome_tif(output_path, 
                level0, 
                compression="jpeg", 
                photometric_interp="rgb", 
                metadata=metadata, 
                subresolutions = 7)


# %%
