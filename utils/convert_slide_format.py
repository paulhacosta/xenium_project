#%% 
import os 
import openslide 
import numpy 
import pyvips 
import tifffile as tf
import numpy as np 

#%%
original_format = "svs"

target_format = "tiff"

slide_file = "/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-utuc/68253/{SLIDE ID}_UTUC_001_114750.svs"
# ['{SLIDE ID}_UTUC_001_114750.svs'
# '{SLIDE ID}_UTUC_002_114836.svs' 
# '{SLIDE ID}_UTUC_003_114916.svs'
# '{SLIDE ID}_UTUC_004_115000.svs']

output_path = slide_file.replace("svs", "tiff")

#%% Set parametsers
compression = "jpeg2000"
tile_size = 256


# %%

# Open slide 
slide = openslide.open_slide(slide_file)

if original_format == "svs":
    # Extract resolution levels
    num_levels = slide.level_count
    level_dims = [slide.level_dimensions[i] for i in range(num_levels)]
    level_downsamples = [slide.level_downsamples[i] for i in range(num_levels)]

    # Extract metadata (PhysicalSizeX and PhysicalSizeY)
    metadata = slide.properties
    mpp_x = float(metadata.get("openslide.mpp-x", 0.25))  # Default to 0.25 µm
    mpp_y = float(metadata.get("openslide.mpp-y", 0.25))

    try:
        magnification = metadata["openslide.objective-power"]
    except IndexError:
        magnification = str(round(10 / mpp_x))  # Example: 0.25μm -> 40X

# Define metadata dictionary
ome_metadata = {
    'axes': "YXC",
    "PhysicalSizeX": mpp_x,
    "PhysicalSizeY": mpp_y,
    "PhysicalSizeXUnit": "µm",
    "PhysicalSizeYUnit": "µm",
    "Magnification": magnification,
    "Levels": num_levels,
}

options = dict(
    photometric="rgb",  # SVS is usually RGB
    tile=tile_size,
    maxworkers=16,
    compression=compression,
    compressionargs={'level': 90} if compression == "jpeg2000" else {},
    resolutionunit="CENTIMETER",
)

# Open the OME-TIFF writer
with tf.TiffWriter(output_path, bigtiff=True) as tif:
    print(f"Writing Level 0 (Full Resolution): {level_dims[0]}")

    # Read and write Level 0 (full-resolution)
    full_res_img = slide.read_region((0, 0), 0, level_dims[0]).convert("RGB")
    full_res_np = np.array(full_res_img)

    tif.write(
            full_res_np,
            subifds=num_levels - 1,  # Define the number of pyramid levels
            resolution=(1e4 / mpp_x, 1e4 / mpp_y),
            metadata=ome_metadata,  # Storing metadata properly
            **options
        )

    # Write sub-resolution levels, matching SVS downsampling
    for i in range(1, num_levels):
        print(f"Writing Level {i}: {level_dims[i]} (Downsample: {level_downsamples[i]})")

        # Read lower resolution level
        low_res_img = slide.read_region((0, 0), i, level_dims[i]).convert("RGB")
        low_res_np = np.array(low_res_img)

        tif.write(
            low_res_np,
            subfiletype=1,  # Sub-resolution level
            resolution=(
                1e4 / (mpp_x * level_downsamples[i]),
                1e4 / (mpp_y * level_downsamples[i])
            ),
            **options
        )

print(f"OME-TIFF saved to {output_path}")
