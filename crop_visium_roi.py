#%%
#!/usr/bin/env python3
"""
Crop a Visium-HD H&E image to its on-tissue 6.5 mm × 6.5 mm ROI (plus margin)
and record the offsets needed to translate spot coordinates. Can be run on IDE or terminal

Example
-------
python crop_visium_hd_roi.py \
    --image   /path/to/full_slide.ome.tif \
    --adata   /path/to/filtered_feature_bc_matrix.h5 \
    --output  /path/to/output/sample_cropped_roi \
    --margin  500
            
Notes:

output: Will be used to name saved files:
    - .../sample_cropped_roi.ome.tiff
    - .../sample_cropped_roi.pkl
    - .../sample_cropped_roi.csv

margin: Amount of pixels to pad around Visium HD 6.5 area. 

"""
import os
import argparse
from pathlib import Path

import numpy as np
import tifffile as tf
import scanpy as sc
import pandas as pd
import pickle
import cv2
from types import SimpleNamespace


#%%
# -----------------------------------------------------------------------------#
# Helpers: write pyramid OME-TIFF
# -----------------------------------------------------------------------------

def _img_resize(img, scale=0.5):
    """Area-resize by a constant scale factor (RGB Y,X,C)."""
    h_next = max(1, int(np.floor(img.shape[0] * scale)))
    w_next = max(1, int(np.floor(img.shape[1] * scale)))
    return cv2.resize(img, (w_next, h_next), interpolation=cv2.INTER_AREA)


def write_ome_tif(
    filename, image, photometric_interp, metadata,
    subresolutions, tile_size=(1024, 1024)
):
    """
    Save `image` as multiresolution OME-TIFF, but stop adding levels once either
    dimension would drop below 1 pixel.
    """
    fn = f"{filename}.ome.tif"
    px_x = metadata["PhysicalSizeX"]
    px_y = metadata["PhysicalSizeY"]

    opts = dict(
        photometric=photometric_interp,
        tile=tile_size,
        maxworkers=16,
        compression=None,
        resolutionunit="CENTIMETER",
    )

    with tf.TiffWriter(fn, bigtiff=True) as tif:
        # level 0
        tif.write(
            image,
            subifds=subresolutions,
            resolution=(1e4 / px_x, 1e4 / px_y),
            metadata=metadata,
            **opts,
        )

        # pyramid
        lvl_img = image
        scale   = 1.0
        written = 0
        for _ in range(subresolutions):
            # stop if next half-size would be 0 in either dim
            if min(lvl_img.shape[0], lvl_img.shape[1]) // 2 == 0:
                break
            lvl_img = _img_resize(lvl_img, 0.5)
            scale  *= 2
            tif.write(
                lvl_img,
                subfiletype=1,
                resolution=(1e4 / (px_x * scale),
                            1e4 / (px_y * scale)),
                **opts,
            )
            written += 1

        # correct the subIFD count the main IFD advertises
        if written < subresolutions:
            tif.overwrite_description(0, {"SubIFDs": written})


# -----------------------------------------------------------------------------#
# Workflow helpers
# -----------------------------------------------------------------------------#
def load_spot_xy(h5_path):
    """Return full-resolution pixel coordinates of spots as int32 ndarray (N×2)."""
    adata = sc.read_10x_h5(h5_path)
    bin_path = os.path.dirname(h5_path)
    tissue_positions_path = os.path.join(bin_path, 'spatial', 'tissue_positions.parquet')
    spatial_coords = pd.read_parquet(tissue_positions_path)
    spatial_coords.set_index("barcode", inplace=True)

    adata.obs = adata.obs.join(spatial_coords, how='left')
    adata.obs['in_tissue'] = adata.obs['in_tissue'].astype(str)
    adata = adata[adata.obs['in_tissue'] == '1', :]

    # Visium-HD stores coordinates in these columns
    for col in ("pxl_col_in_fullres", "pxl_row_in_fullres"):
        if col not in adata.obs:
            raise RuntimeError(f"`{col}` not found in {h5_path}")
    xy = adata.obs[["pxl_col_in_fullres", "pxl_row_in_fullres"]].astype(int).values
    # Optionally keep only on-tissue bins (comment out if not present)
    if "is_tissue" in adata.obs:
        xy = xy[adata.obs["is_tissue"].values == 1]
    return xy, adata  # return adata to optionally save translated coords


def bounding_box(xy, margin, img_shape):
    """Tight bounding box around `xy` plus `margin` pixels, clipped to image."""
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)
    x0 = max(0, x_min - margin)
    y0 = max(0, y_min - margin)
    x1 = min(img_shape[1], x_max + margin)
    y1 = min(img_shape[0], y_max + margin)
    return int(x0), int(y0), int(x1), int(y1)


def crop_level0(path, bbox):
    """
    Read level-0 RGB data, coerce to Y,X,C, crop to bbox,
    and return:  crop, meta_dict, tile_w, tile_h, n_subifds
    """
    x0, y0, x1, y1 = bbox
    with tf.TiffFile(path) as tif:
        page0 = tif.pages[0]
        arr = page0.asarray()

        # ensure Y,X,C order
        if arr.ndim == 3 and arr.shape[0] in (3, 4):   # C,Y,X
            arr = np.moveaxis(arr, 0, -1)

        crop = arr[y0:y1, x0:x1, :]

        tile_w   = page0.tilewidth or crop.shape[1]     # fall-backs for untiled
        tile_h   = page0.tilelength or crop.shape[0]
        
        subifds  = page0.subifds or []
        n_subifd = len(subifds)

        # ------------------------------------------------------------------
        # build a **minimal** meta-dict that always has pixel sizes + units
        # ------------------------------------------------------------------
        if tif.is_ome:                                   # easy case
            meta_xml = tif.ome_metadata
            meta = tf.xml2dict(meta_xml)
            px = float(meta["OME"]["Image"]["Pixels"]["PhysicalSizeX"])
            unit = meta["OME"]["Image"]["Pixels"]["PhysicalSizeXUnit"]
        else:                                            # plain TIFF
            # use resolution tags if present, else default to 0.24 µm/px
            res_unit_tag = page0.tags.get("ResolutionUnit")
            res_x_tag    = page0.tags.get("XResolution")
            res_y_tag    = page0.tags.get("YResolution")

            if res_x_tag and res_y_tag and res_unit_tag:
                ppu_x = res_x_tag.value[0] / res_x_tag.value[1]   # pixels per unit
                ppu_y = res_y_tag.value[0] / res_y_tag.value[1]
                ppu   = (ppu_x + ppu_y) / 2                      # average
                if res_unit_tag.value == 2:      # inches
                    px = 25_400 / ppu            # µm per pixel
                    unit = "micron"
                elif res_unit_tag.value == 3:    # centimetres
                    px = 10_000 / ppu
                    unit = "micron"
                else:                            # unknown unit
                    px   = 0.25              # fallback
                    unit = "micron"
            else:
                px   = 0.25                  # fallback

                unit = "micron"
        meta_min = {
            "axes": "YXC",
            "PhysicalSizeX": px,
            "PhysicalSizeXUnit": unit,
            "PhysicalSizeY": px,
            "PhysicalSizeYUnit": unit,
        }

    return crop, meta_min, tile_w, tile_h, n_subifd


def save_offsets(path, x0, y0, pixel_size):
    """
    Store offsets so original coordinates can be translated to the cropped
    image.  Saves a .pkl instead of JSON.
    """
    # ensure we use a .pkl extension
    path = Path(path).with_suffix(".pkl")

    payload = {
        "offset_x_px": int(x0),
        "offset_y_px": int(y0),
        "scale_factor": 1.0,
        "pixel_size_microns": float(pixel_size),
        "note": (
            "new_x = original_x - offset_x_px ; "
            "new_y = original_y - offset_y_px"
        ),
    }
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)  

#%%
# -----------------------------------------------------------------------------#
# Main
# -----------------------------------------------------------------------------#
def main(args):
    print("Reading spot coordinates …")
    h5 = args.h5
    image = args.image
    output = args.output
    margin = args.margin
    save_csv= args.save_csv

    xy, adata = load_spot_xy(h5)

    print("Opening H&E image header …")
    with tf.TiffFile(image) as t:
        h, w = t.pages[0].shape[:-1]

    bbox = bounding_box(xy, margin, (h, w))
    x0, y0, x1, y1 = bbox
    print(f"Crop box (level 0): ({x0}, {y0}) – ({x1}, {y1})")

    print("Cropping image …")
    crop, meta, tile_w, tile_h, n_subifds = crop_level0(image, bbox)
    print("Cropped image shape:", crop.shape)

    print("Meta data:", meta)

    out_prefix = Path(output).expanduser()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    print("Writing cropped OME-TIFF …")

    # Compute a safe number of sub-IFDs for pyramid before calling the writer
    max_halvings = int(np.floor(np.log2(min(crop.shape[0], crop.shape[1]))))
    safe_subifds = min(n_subifds, max_halvings)

    write_ome_tif(
        out_prefix,
        crop,
        photometric_interp="rgb",
        metadata=meta,
        subresolutions=safe_subifds,
        tile_size=(tile_w, tile_h),
    )

    print("Saving offset metadata …")
    save_offsets(out_prefix, x0, y0, meta['PhysicalSizeX'])

    # (Optional) save translated coordinates for QC / plotting
    # as csv now but can rewrite as h5 usin scanpy
    if save_csv:
        adata.obs["x_crop"] = adata.obs["pxl_col_in_fullres"] - x0
        adata.obs["y_crop"] = adata.obs["pxl_row_in_fullres"] - y0
        adata.obs[["x_crop", "y_crop"]].to_csv(out_prefix.with_suffix(".csv"))
        print("Translated coordinates CSV saved.")

    print("Done!")

#%%
if __name__ == "__main__":
    # -----------------------------------------------------------------------------#
    # Run from Terminal
    # -----------------------------------------------------------------------------#
    # parser = argparse.ArgumentParser(
    #     description="Crop Visium-HD H&E slide to ROI and record offsets."
    # )
    # parser.add_argument("--image", required=True, help="Full-res H&E: TIFF or OME-TIFF")
    # parser.add_argument("--h5", required=True,
    #                     help="Visium-HD filtered_feature_bc_matrix.h5")
    # parser.add_argument("--output", required=True,
    #                     help="Output prefix (without extension)")
    # parser.add_argument("--margin", type=int, default=500,
    #                     help="Padding (pixels) around the ROI (default: 500)")
    # parser.add_argument("--save_csv", action="store_true",
    #                     help="Also write CSV of translated spot coords")
    # args = parser.parse_args()
    # main(args)


    # -----------------------------------------------------------------------------#
    # Run from an IDE
    # -----------------------------------------------------------------------------#
    if __name__ == "__main__":
        # ---- edit these four lines to define paths ----
        image = '/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-utuc/68253/240727_UTUC_002_114836.tiff'      # TIFF or OME-TIFF
        h5 = '/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-utuc/UTUC_B1_VisiumHD/spaceranger311/UTUC2025_240727-HD01/outs/binned_outputs/square_002um/filtered_feature_bc_matrix.h5'
        output = '/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-utuc/UTUC_B1_VisiumHD/spaceranger311/UTUC2025_240727-HD01/240727_UTUC_002_114750_cropped'      # no extension
        margin = 500   # pixel padding
        save_csv = False  # set False to skip CSV

        # build a lightweight “args” object to satisfy main()
        args = SimpleNamespace(
            image=image,
            h5=h5,
            output=output,
            margin=margin,
            save_csv=save_csv,
        )

        main(args)

# %%
