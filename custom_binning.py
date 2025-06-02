#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import anndata
import geopandas as gpd
import scanpy as sc

from tifffile import imread, imwrite
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from shapely.geometry import Polygon, Point
from scipy import sparse
from matplotlib.colors import ListedColormap

import geojson
import json
from shapely.geometry import shape
import geopandas as gpd

import scanpy as sc
import pandas as pd
import h5py
import openslide

import seaborn as sns
from shapely.geometry import Point, Polygon
from sklearn.decomposition import PCA
from scipy.sparse import issparse

#%%
# Filepaths
cellvit_file = "/rsrch5/home/plm/phacosta/CellViT/example/output/preprocessing/original/Xenium_Prime_Human_Lung_Cancer_FFPE_he_image.ome/cell_detection/cells.geojson"
feature_matrix_file = "/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Visium_HD_Human_Lung_Cancer_post_Xenium_Prime_5k_Experiment2/binned_outputs/square_002um/filtered_feature_bc_matrix.h5"
tissue_positions_file = "/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Visium_HD_Human_Lung_Cancer_post_Xenium_Prime_5k_Experiment2/binned_outputs/square_002um/spatial/tissue_positions.parquet"


#%%

def bins_to_pixels(n_bins=2, bin_size_um=2.0, mpp=0.2738):
    """
    Convert a desired number of bins to a pixel distance, 
    given each bin is bin_size_um µm wide and the image has mpp µm/pixel.
    """
    distance_um = n_bins * bin_size_um  # total microns
    distance_pixels = distance_um / mpp
    return distance_pixels

def expand_cell_polygons(nuclei_gdf, n_bins=2, bin_size_um=2.0, mpp=0.2738):
    """
    Expand each nucleus polygon by `n_bins` * `bin_size_um` microns, 
    accounting for mpp (µm/pixel).
    """
    distance_pixels = (n_bins * bin_size_um) / mpp
    # or use the helper function from above

    nuclei_gdf["expanded_geom"] = nuclei_gdf.geometry.buffer(distance_pixels)
    return nuclei_gdf

def expand_cell(
    adata_bins,
    bin_gdf,
    nucleus_gdf,
    n_bins=2,
    bin_size_um=2.0,
    mpp=0.2738,
    tie_break_method="pca",
    grouped_adata=None,
    n_pcs=10
):
    """
    Expand each nucleus polygon by (n_bins * bin_size_um) microns in pixel coords,
    then assign bins that fall inside the expanded region to that nucleus (cell).

    If a bin is claimed by multiple expanded polygons, we tie-break via:
      - "geometry": pick the polygon whose centroid is nearest
      - "pca": pick the cell with minimal distance in PCA embedding of expression

    Parameters
    ----------
    adata_bins : AnnData
        The *bin-level* AnnData. Rows=barcodes, columns=genes. 
        Must have .obs_names matching bin_gdf (or have bin_gdf["barcode"]).
    bin_gdf : GeoDataFrame
        Each row is a bin with .geometry=Point in pixel coords. 
        Typically includes a "barcode" column or else row index aligns with adata_bins.obs_names.
    nucleus_gdf : GeoDataFrame
        Each row is a nucleus polygon in pixel coords, with columns "id" and "geometry".
    n_bins : int, default=2
        Number of bin widths to expand by.
    bin_size_um : float, default=2.0
        Each bin is bin_size_um microns wide (e.g., 2µm).
    mpp : float, default=0.2738
        Microns per pixel. If your polygons/bins are in raw pixel coords, use this to convert.
    tie_break_method : str, default="pca"
        "geometry" -> choose cell with nearest centroid in pixels,
        "pca"      -> choose cell with nearest embedding in PCA space.
    grouped_adata : AnnData, optional
        If provided, we use it to get each cell’s expression (one row per 'id') 
        for the PCA tie-break. This presupposes you’ve *already* made a “grouped” 
        or “summed” AnnData with .obs["id"] for each cell.
        If None, we do a naive sum-of-bins approach inside this function to build cell vectors.
    n_pcs : int, default=10
        Number of principal components used if tie_break_method="pca".

    Returns
    -------
    pd.DataFrame with ["barcode", "cell_id"] for each bin, post-expansion.
    """

    # 1) Convert "n_bins" expansion to pixel distance
    expansion_distance_pixels = (n_bins * bin_size_um) / mpp

    # 2) Create expanded polygon for each nucleus
    nuc_gdf_expanded = nucleus_gdf.copy()
    nuc_gdf_expanded["expanded_geom"] = nuc_gdf_expanded.geometry.buffer(expansion_distance_pixels)

    # We'll build a GDF with geometry=expanded_geom for the spatial join
    expanded_nuc_gdf = nuc_gdf_expanded.drop(columns="geometry").rename(columns={"expanded_geom": "geometry"})

    # 3) Spatial join: which bins are in which expanded polygon
    bin_polygon_join = gpd.sjoin(bin_gdf, expanded_nuc_gdf, how="inner", predicate="within")
    # bin_polygon_join now has columns from bin_gdf + expanded_nuc_gdf,
    # including "id" (the cell ID) from expanded_nuc_gdf and "index_right" references.

    # 4) Identify bins that appear in multiple polygons
    duplicates = bin_polygon_join.groupby(bin_polygon_join.index).size()
    multi_bins = duplicates[duplicates > 1].index

    bin_to_cell = {}

    # a) Bins that appear in exactly 1 polygon => direct assignment
    single_bin_polygon_join = bin_polygon_join.loc[~bin_polygon_join.index.isin(multi_bins)]
    for bin_idx, row in single_bin_polygon_join.iterrows():
        bin_to_cell[bin_idx] = row["id"]

    # b) Bins in multiple polygons => tie-break
    multi_bin_polygon_join = bin_polygon_join.loc[bin_polygon_join.index.isin(multi_bins)]

    if tie_break_method == "geometry":
        # We'll define centroid for each expanded polygon
        expanded_nuc_gdf["centroid_x"] = expanded_nuc_gdf.geometry.centroid.x
        expanded_nuc_gdf["centroid_y"] = expanded_nuc_gdf.geometry.centroid.y

        # group by bin_idx (index=bin_idx), find nearest centroid
        for bin_idx, group_df in multi_bin_polygon_join.groupby(level=0):
            # bin coords
            bin_pt = bin_gdf.loc[bin_idx, "geometry"]
            bin_x, bin_y = bin_pt.x, bin_pt.y

            best_cell = None
            best_dist = np.inf
            for _, row in group_df.iterrows():
                cell_id = row["id"]
                # row["index_right"] points to the row in expanded_nuc_gdf
                centroid_x = expanded_nuc_gdf.loc[row["index_right"], "centroid_x"]
                centroid_y = expanded_nuc_gdf.loc[row["index_right"], "centroid_y"]
                dist = np.hypot(bin_x - centroid_x, bin_y - centroid_y)
                if dist < best_dist:
                    best_dist = dist
                    best_cell = cell_id

            bin_to_cell[bin_idx] = best_cell

    elif tie_break_method == "pca":
        # We do a PCA-based approach:
        # (1) Build PCA embedding for *all bins* in adata_bins.
        # (2) Build PCA embedding for each cell (from grouped_adata if given,
        #     or sum up bin expression ourselves).
        # (3) For each ambiguous bin, pick the cell with min distance in PCA space.

        # Step (1) Fit PCA on bin-level data
        #   We'll do a simple log1p and PCA. This can be memory heavy for large data.
        #   Alternatively, subset to only ambiguous bins + single bins that define each cell.
        #   But let's do full for clarity.

        X = adata_bins.X
        if issparse(X):
            X = X.toarray()
        X_log = np.log1p(X)
        pca_model = PCA(n_components=n_pcs)
        pca_scores_bins = pca_model.fit_transform(X_log)  # shape=(nBins, n_pcs)

        # We'll map bin_idx -> PCA embedding
        # If bin_gdf "barcode" column matches adata_bins.obs_names, we do:
        bin_idx_to_pca = {}
        # Build a quick lookup from barcode -> PCA row
        barcode2row = {bc: i for i, bc in enumerate(adata_bins.obs_names)}

        for b_idx in bin_gdf.index:
            if "barcode" in bin_gdf.columns:
                bcode = bin_gdf.loc[b_idx, "barcode"]
            else:
                # fallback if no "barcode" col
                bcode = adata_bins.obs_names[b_idx]
            # find row in adata_bins
            row_i = barcode2row[bcode]
            bin_idx_to_pca[b_idx] = pca_scores_bins[row_i, :]

        # Step (2) Build PCA embedding for each cell
        # If user gave us grouped_adata => each row is a cell, .obs["id"] is the ID
        # Then we just transform. Otherwise, sum bin expression ourselves.
        if grouped_adata is not None:
            # We'll do the same log1p and apply pca_model.transform
            X_cells = grouped_adata.X
            if issparse(X_cells):
                X_cells = X_cells.toarray()
            X_cells_log = np.log1p(X_cells)
            pca_scores_cells = pca_model.transform(X_cells_log)
            # map cell_id -> PCA vector
            cell_ids = grouped_adata.obs["id"].values
            cell_id_to_pca = {cid: pca_scores_cells[i, :] for i, cid in enumerate(cell_ids)}
        else:
            # Summation approach: for each cell, sum expression from bins that originally belonged to it
            # We'll do a naive approach: we assume "id" in adata_bins.obs is the original nucleus assignment
            # That might not exist yet. If it does, we can do something like:
            if "id" not in adata_bins.obs:
                raise ValueError("No grouped_adata given and adata_bins.obs['id'] missing. Can't sum bin expression.")
            # group by adata_bins.obs["id"], sum the expression
            # Then transform with the same PCA model
            bin_id_series = adata_bins.obs["id"]
            # sum approach
            cell_ids_uniq = bin_id_series.unique()
            cell_expr = []
            for cid in cell_ids_uniq:
                bin_indices = np.where(bin_id_series == cid)[0]
                # sum their expression in X_log space (or do sum in X space then log?)
                # We'll do sum in normal space, then log1p to be consistent. 
                # But that conflicts with the bin-level pca approach. 
                # We'll do a simpler approach: we sum X, do log1p, then transform.
                expr_sum = X[bin_indices, :].sum(axis=0)
                expr_sum_log = np.log1p(expr_sum)
                # transform
                expr_sum_pca = pca_model.transform(expr_sum_log.reshape(1, -1))[0]
                cell_expr.append((cid, expr_sum_pca))
            cell_id_to_pca = dict(cell_expr)

        # Step (3) For each ambiguous bin, pick cell with min distance in PCA space
        for bin_idx, group_df in multi_bin_polygon_join.groupby(level=0):
            bin_pca_vec = bin_idx_to_pca[bin_idx]
            best_cell, best_dist = None, float("inf")
            # group_df -> multiple rows, each w/ a different cell "id"
            for _, row in group_df.iterrows():
                cid = row["id"]
                if cid not in cell_id_to_pca:
                    # skip or handle error
                    continue
                cell_vec = cell_id_to_pca[cid]
                dist = np.linalg.norm(bin_pca_vec - cell_vec)
                if dist < best_dist:
                    best_dist = dist
                    best_cell = cid
            bin_to_cell[bin_idx] = best_cell

    else:
        # If user hasn't provided a known method, do nothing or error
        raise ValueError(f"Unknown tie_break_method: {tie_break_method}")

    # 5) Build final DataFrame: for each bin_idx => (barcode, cell_id)
    results = []
    for b_idx, c_id in bin_to_cell.items():
        if "barcode" in bin_gdf.columns:
            bcode = bin_gdf.loc[b_idx, "barcode"]
        else:
            bcode = adata_bins.obs_names[b_idx]
        results.append((bcode, c_id))

    bin_to_cell_df = pd.DataFrame(results, columns=["barcode", "cell_id"])
    return bin_to_cell_df


# %%
# -----------------------------------------------------------------------------
# 2) LOAD & PARSE THE GEOJSON (NUCLEI POLYGONS) - WITHOUT CLASSIFICATION
# -----------------------------------------------------------------------------
with open(cellvit_file, "r") as f:
    data = json.load(f)

# The GeoJSON usually has {"type": "FeatureCollection", "features": [...]}:
features = data["features"] if "features" in data else data

all_polygons = []
all_ids = []

cell_counter = 0  # We'll make a new "Cell_0", "Cell_1", etc.

for feat_idx, feat in enumerate(features):
    geom = shape(feat["geometry"])  # could be MultiPolygon or Polygon

    # If it's a single Polygon:
    if geom.geom_type == "Polygon":
        # Make a brand new ID for this sub-polygon (which is presumably one cell)
        cell_id = f"Cell_{cell_counter}"
        cell_counter += 1

        all_polygons.append(geom)
        all_ids.append(cell_id)

    # If it's a MultiPolygon, we iterate each polygon inside it
    elif geom.geom_type == "MultiPolygon":
        for subpoly in geom.geoms:
            cell_id = f"Cell_{cell_counter}"
            cell_counter += 1

            all_polygons.append(subpoly)
            all_ids.append(cell_id)

    else:
        # If there are other geometry types, skip or handle them
        pass

# Now each sub-polygon is treated as a separate "cell".
# Build a GeoDataFrame with your newly created IDs:
nuclei_gdf = gpd.GeoDataFrame(
    {"id": all_ids},
    geometry=all_polygons
).reset_index(drop=True)

print("Number of per-cell polygons in nuclei_gdf:", len(nuclei_gdf))
print(nuclei_gdf.head())

# -----------------------------------------------------------------------------
# 3) LOAD VISIUM HD DATA (EXPRESSION + SPATIAL POSITIONS)
# -----------------------------------------------------------------------------
# 3A) Load gene expression into AnnData
adata = sc.read_10x_h5(feature_matrix_file)
print("Original adata shape:", adata.shape)

# 3B) Load tissue_positions and merge into adata.obs
df_positions = pd.read_parquet(tissue_positions_file)
df_positions = df_positions.set_index("barcode")
df_positions["index"] = df_positions.index  # optional clarity

# Merge the coordinate info onto adata.obs
adata.obs = pd.merge(
    adata.obs,
    df_positions,
    left_index=True,
    right_index=True,
    how="left"
)

# 3C) Create a GeoDataFrame of barcodes (each a Point)
coords = [
    Point(xy) 
    for xy in zip(
        df_positions["pxl_col_in_fullres"],
        df_positions["pxl_row_in_fullres"]
    )
]
gdf_coordinates = gpd.GeoDataFrame(
    df_positions, geometry=coords
).reset_index(drop=True)

print("Number of barcode coordinates in gdf_coordinates:", len(gdf_coordinates))
print(gdf_coordinates.head())

# -----------------------------------------------------------------------------
# 4) SPATIAL JOIN: BARCODE -> NUCLEUS
# -----------------------------------------------------------------------------
# We'll use a left-join to match barcodes with the polygon they fall in
result_sjoin = gpd.sjoin(
    gdf_coordinates, 
    nuclei_gdf, 
    how="left", 
    predicate="within"
)
# 'index_right' tells which polygon row matched. If NaN => outside any polygon
result_sjoin["is_within_polygon"] = ~result_sjoin["index_right"].isna()

# Identify barcodes that land in multiple polygons
duplicate_barcodes = pd.unique(result_sjoin[result_sjoin.duplicated(subset=["index"])]["index"])

# Mark barcodes that are not in overlapping polygons
result_sjoin["is_not_in_an_polygon_overlap"] = ~result_sjoin["index"].isin(duplicate_barcodes)

# Keep only barcodes in exactly one polygon
barcodes_in_one_polygon = result_sjoin[
    result_sjoin["is_within_polygon"] & 
    result_sjoin["is_not_in_an_polygon_overlap"]
]

print("Barcodes uniquely mapped to one polygon:", len(barcodes_in_one_polygon))

# Filter adata to these unique barcodes
filt_mask = adata.obs_names.isin(barcodes_in_one_polygon["index"])
filtered_adata = adata[filt_mask,:].copy()
print("filtered_adata shape:", filtered_adata.shape)

# Merge polygon info into .obs
filtered_adata.obs = pd.merge(
    filtered_adata.obs,
    barcodes_in_one_polygon[["index", "geometry", "id", "is_within_polygon", "is_not_in_an_polygon_overlap"]],
    left_index=True,
    right_on="index",
    how="left"
).set_index("index")

# -----------------------------------------------------------------------------
# 5) SUM GENE COUNTS PER POLYGON (PER NUCLEUS)
# -----------------------------------------------------------------------------
# Group barcodes by polygon ID
grouped = filtered_adata.obs.groupby("id", observed=True)
counts = filtered_adata.X  # This is (barcodes x genes)

n_groups = grouped.ngroups
n_genes = counts.shape[1]

summed_counts = sparse.lil_matrix((n_groups, n_genes))
polygon_ids = []

row = 0
for nucleus_id, b_idx in grouped.indices.items():
    # Sum the gene counts across all barcodes in this nucleus
    summed_counts[row] = counts[b_idx].sum(axis=0)

    polygon_ids.append(nucleus_id)
    row += 1

summed_counts = summed_counts.tocsr()

# # Construct a new AnnData
grouped_adata = anndata.AnnData(
    X=summed_counts,
    obs=pd.DataFrame({"id": polygon_ids}, index=polygon_ids),
    var=filtered_adata.var.copy()
)

nuclei_gdf["x_centroid"] = nuclei_gdf.geometry.centroid.x
nuclei_gdf["y_centroid"] = nuclei_gdf.geometry.centroid.y

# For merging, ensure you only keep the unique columns
#   you need from nuclei_gdf – typically 'id', 'x_centroid', 'y_centroid'.
#   Then do a merge on 'id'.
nuclei_df_for_merge = nuclei_gdf[["id", "x_centroid", "y_centroid"]].drop_duplicates(subset=["id"])

grouped_adata.obs = (
    grouped_adata.obs
    .merge(
        nuclei_df_for_merge, 
        how="left", 
        on="id"
    )
    .set_index(grouped_adata.obs.index)
)


print("grouped_adata shape:", grouped_adata.shape)
print(grouped_adata.obs.head())
print("Done! Now you have per-polygon (nucleus) expression data.")


#%% 
# Testing functions 
# Suppose your original code produced:
#   adata_bins : an AnnData with each bin's expression
#   bin_gdf : a GeoDataFrame of bins in pixel coords
#   nucleus_gdf : each row is "id" + "geometry" for the nucleus

bin_to_cell_df = expand_cell(
    adata_bins=adata,
    bin_gdf=gdf_coordinates,
    nucleus_gdf=nuclei_gdf,
    n_bins=2,          # expand by 2 bins
    mpp=0.2738,        # image scale
    tie_break_method="pca",  # use expression-based tie break
    grouped_adata=grouped_adata,  # optional, for cell embeddings
    n_pcs=10
)

print(bin_to_cell_df.head())




#%%
##########################################

# All code below this point is for visualization purposes only

##########################################


# %%
slide_file = "/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Xenium_Prime_Human_Lung_Cancer_FFPE_outs/Xenium_Prime_Human_Lung_Cancer_FFPE_he_image.ome.tif"
slide = openslide.open_slide(slide_file)
start_x, start_y = 7696, 3484  # top left point 

width, height = 500, 500
end_x = start_x + width
end_y = start_y + height

region = slide.read_region(
    (start_x, start_y),
    level=0,
    size=(width, height)
).convert("RGB")

# ---------------------------------------
# 2) FILTER YOUR CELL POLYGONS TO THIS REGION
# ---------------------------------------
# Suppose you have 'nuclei_gdf' with .geometry as Polygons in slide coords
# e.g. nuclei_gdf = gpd.read_file("my_cells.geojson") or from your pipeline

bbox_polygon = Polygon([
    (start_x,        start_y),
    (start_x+width,  start_y),
    (start_x+width,  start_y+height),
    (start_x,        start_y+height)
])

roi_cells = nuclei_gdf[nuclei_gdf.geometry.intersects(bbox_polygon)].copy()

# Shift cell polygons so top-left corner is (0,0) in our local patch
roi_cells.geometry = roi_cells.geometry.translate(-start_x, -start_y)

# ---------------------------------------
# 3) FILTER 2µm BIN SPOTS TO CELLS
# ---------------------------------------
# a) Convert your adata.obs to a GeoDataFrame of bin points

bin_x = adata.obs["pxl_col_in_fullres"].values
bin_y = adata.obs["pxl_row_in_fullres"].values
barcodes = adata.obs_names

# First limit to bins in the bounding box
in_bbox = (
    (bin_x >= start_x) & (bin_x < end_x) &
    (bin_y >= start_y) & (bin_y < end_y)
)

bin_x_roi = bin_x[in_bbox]
bin_y_roi = bin_y[in_bbox]
barcodes_roi = barcodes[in_bbox]

bin_points_roi = gpd.GeoDataFrame(
    {"barcode": barcodes_roi},
    geometry=[
        Point(x, y) for x, y in zip(bin_x_roi, bin_y_roi)
    ],
    crs="EPSG:4326"  # or some dummy if you don't have a real CRS
)

# b) Spatial join: which bins fall inside (within) at least one cell polygon?
#    We can do that with 'predicate="within"'. 
#    We'll do an *inner* join, so only those bin points inside polygons remain.

# Because 'roi_cells' geometry is shifted, we need to either:
#   1) also shift bin_points, or
#   2) do the sjoin in the original coordinates (and shift after).
#
# Let's do the sjoin in the *original* coordinates (before shifting cells).
# So we use 'nuclei_gdf' BEFORE the translate, but still filtered by bounding box.

roi_cells_original = nuclei_gdf[nuclei_gdf.geometry.intersects(bbox_polygon)].copy()

# We'll do a spatial join: the bins that are "within" the polygons
# But first we need the bin points also in the *original* slide coords
bin_points_bbox = gpd.GeoDataFrame(
    {"barcode": barcodes_roi},
    geometry=[
        Point(x, y) for x, y in zip(bin_x_roi, bin_y_roi)
    ],
    crs="EPSG:4326"
)

# Perform the join
bins_in_cells = gpd.sjoin(
    bin_points_bbox,
    roi_cells_original,  # polygons in original coords
    how="inner",
    predicate="within"
)

# 'bins_in_cells' will contain only the bins that fall inside a cell polygon
# We now shift them so that (start_x, start_y) => (0,0)
bins_in_cells["x_shifted"] = bins_in_cells.geometry.x - start_x
bins_in_cells["y_shifted"] = bins_in_cells.geometry.y - start_y

# ---------------------------------------
# 4) PLOT THE SLIDE, CELLS, AND BINS
# ---------------------------------------
fig, ax = plt.subplots(figsize=(10, 10))
# The background image
ax.imshow(region, origin="upper")

# The shifted cell polygons (red outlines)
roi_cells.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=1)

# The bin spots that are inside the polygons (cyan dots)
ax.scatter(
    bins_in_cells["x_shifted"], 
    bins_in_cells["y_shifted"], 
    s=1, 
    c="cyan"
)

ax.set_axis_off()
plt.tight_layout()
plt.show()


# %%
bin_size = "8"
bdata_file  = f"/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Visium_HD_Human_Lung_Cancer_post_Xenium_Prime_5k_Experiment2/binned_outputs/square_00{bin_size}um/filtered_feature_bc_matrix.h5"
b_positions_file = f"/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Visium_HD_Human_Lung_Cancer_post_Xenium_Prime_5k_Experiment2/binned_outputs/square_00{bin_size}um/spatial/tissue_positions.parquet"

bdata = sc.read_10x_h5(bdata_file)
print("Original adata shape:", bdata.shape)

# 3B) Load tissue_positions and merge into adata.obs
bdf_positions = pd.read_parquet(b_positions_file)
bdf_positions = bdf_positions.set_index("barcode")
bdf_positions["index"] = bdf_positions.index  # optional clarity

# Merge the coordinate info onto adata.obs
bdata.obs = pd.merge(
    bdata.obs,
    bdf_positions,
    left_index=True,
    right_index=True,
    how="left"
)

# start_x, start_y = 8696, 3484  # top left point 
width, height = 500, 500       # desired region size in pixels
end_x = start_x + width
end_y = start_y + height


region = slide.read_region(
    (start_x, start_y),  # top-left corner in slide coords
    level=0,
    size=(width, height) # width, height in pixels
).convert("RGB")

# -------------------------------------------------------------------
# 2) PREPARE 2um BIN LOCATIONS (FROM YOUR ADATA)
# -------------------------------------------------------------------
# Suppose your full AnnData object is 'adata', with each spot's pixel coords
# in adata.obs["pxl_col_in_fullres"] and adata.obs["pxl_row_in_fullres"].
bin_x = bdata.obs["pxl_col_in_fullres"].values
bin_y = bdata.obs["pxl_row_in_fullres"].values

# Filter bins to the bounding box (5122..(5122+2000), 20734..(20734+2000))
in_roi = (
    (bin_x >= start_x) & (bin_x < end_x) &
    (bin_y >= start_y) & (bin_y < end_y)
)

bin_x_roi = bin_x[in_roi]
bin_y_roi = bin_y[in_roi]

# SHIFT these coordinates to the local 0..width, 0..height space 
# so they align with 'region' in a matplotlib image.
bin_x_shift = bin_x_roi - start_x
bin_y_shift = bin_y_roi - start_y

# -------------------------------------------------------------------
# 3) PREPARE CELL CENTROIDS (FROM NUCLEI POLYGONS)
# -------------------------------------------------------------------
# Suppose your cell polygons are in a GeoDataFrame 'nuclei_gdf' with .geometry
#  each polygon in the same (full-res) coordinate space.
# We find polygons intersecting the same bounding box:
bbox_polygon = Polygon([
    (start_x,         start_y),
    (start_x + width, start_y),
    (start_x + width, start_y + height),
    (start_x,         start_y + height)
])

cells_in_roi = nuclei_gdf[
    nuclei_gdf.geometry.intersects(bbox_polygon)
].copy()

# Compute each cell's centroid and shift to local coordinates
cells_in_roi["centroid_x"] = cells_in_roi.geometry.centroid.x - start_x
cells_in_roi["centroid_y"] = cells_in_roi.geometry.centroid.y - start_y

# -------------------------------------------------------------------
# 4) PLOT: 1x2 SUBPLOTS
# -------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Left Subplot: Slide patch + 2µm bin locations
ax1.imshow(region, origin="upper")
ax1.scatter(bin_x_shift, bin_y_shift, s=2, c="cyan")
ax1.set_title(f"{bin_size}µm Bins")
ax1.set_axis_off()

# Right Subplot: Same slide patch + cell centroids
ax2.imshow(region, origin="upper")
ax2.scatter(
    cells_in_roi["centroid_x"], 
    cells_in_roi["centroid_y"], 
    s=5, 
    c="red"
)
ax2.set_title("Cell Detections")
ax2.set_axis_off()

plt.tight_layout()
plt.show()

# %%
nuc_size_threshold = 1500
# 1a) Calculate each nucleus polygon's area
nuclei_gdf["area"] = nuclei_gdf.geometry.area

# 1b) (Optional) Visualize the distribution of nucleus area
#     Similar to the 10x code's "plot_nuclei_area" function:
def plot_nuclei_area(gdf, area_cut_off):
    fig, axs = plt.subplots(1, 2, figsize=(15, 4))
    # Left: entire distribution
    axs[0].hist(gdf['area'], bins=50, edgecolor='black')
    axs[0].set_title('Nuclei Area')

    # Right: restricted distribution
    axs[1].hist(gdf[gdf['area'] < area_cut_off]['area'], bins=50, edgecolor='black')
    axs[1].set_title('Nuclei Area < ' + str(area_cut_off))

    plt.tight_layout()
    plt.show()

# Example usage:
plot_nuclei_area(nuclei_gdf, area_cut_off=nuc_size_threshold)  # adjust cutoff as needed

# %%
umi_threshold = 1

# 2a) Calculate QC metrics (like total_counts, n_genes_by_counts, etc.)
#     This adds columns in grouped_adata.obs: "total_counts", "n_genes_by_counts", etc.
sc.pp.calculate_qc_metrics(grouped_adata, inplace=True)

# 2b) Visualize the distribution of total UMIs
def total_umi(adata_, cut_off):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Left: all data
    axs[0].boxplot(
        adata_.obs["total_counts"],
        vert=False, widths=0.7, patch_artist=True,
        boxprops=dict(facecolor='skyblue')
    )
    axs[0].set_title('Total Counts')

    # Right: only total_counts > cut_off
    axs[1].boxplot(
        adata_.obs["total_counts"][adata_.obs["total_counts"] > cut_off],
        vert=False, widths=0.7, patch_artist=True,
        boxprops=dict(facecolor='skyblue')
    )
    axs[1].set_title('Total Counts > ' + str(cut_off))

    # Remove y-axis ticks
    for ax in axs:
        ax.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.show()

# Example usage:
total_umi(grouped_adata, umi_threshold)  # choose a cutoff

# 3a) Create a mask for nucleus area
#     We'll keep only nucleus IDs (the same 'id' in grouped_adata.obs)
#     that appear in nuclei_gdf with area < size threshhold.
mask_area = grouped_adata.obs['id'].isin(
    nuclei_gdf[nuclei_gdf['area'] < nuc_size_threshold]["id"]
)

# 3b) Create a mask for total UMI > 100
mask_count = grouped_adata.obs["total_counts"] > umi_threshold

# 3c) Combine masks
filtered_adata = grouped_adata[mask_area & mask_count, :].copy()
print("filtered_adata shape:", filtered_adata.shape)

# 3d) Re-calculate QC metrics for the filtered data, if desired
sc.pp.calculate_qc_metrics(filtered_adata, inplace=True)

# %%
# # 4a) Normalize total counts to 1e4 (or another total) per nucleus
# sc.pp.normalize_total(filtered_adata, target_sum=1e4, inplace=True)

# # 4b) Log-transform the data
# sc.pp.log1p(filtered_adata)

# # 4c) Identify highly variable genes
# #     "Seurat" flavor picks top n genes by variance
# sc.pp.highly_variable_genes(filtered_adata, flavor="seurat", n_top_genes=2000)

# # 4d) PCA on the HVGs
# sc.pp.pca(filtered_adata, use_highly_variable=True)

# # 4e) Build a neighborhood graph
# sc.pp.neighbors(filtered_adata, n_pcs=20)

# # 4f) Leiden clustering
# #     Adjust resolution depending on how many clusters you want
# sc.tl.leiden(filtered_adata, resolution=0.35, key_added="clusters")


# %%
#  Normalize total counts to 1e4 (or another total) per nucleus
sc.pp.normalize_total(filtered_adata, target_sum=1e4, inplace=True)
# Log-transform the data
sc.pp.log1p(filtered_adata)

# 1a) PCA
sc.pp.pca(filtered_adata, n_comps=50)

# 1b) Build neighbors graph
sc.pp.neighbors(filtered_adata, n_neighbors=10, n_pcs=30)

# 1c) Leiden clustering (resolution can be tweaked)
sc.tl.leiden(filtered_adata, resolution=0.25, key_added="leiden")

# 1d) UMAP (optional, for 2D embedding)
sc.tl.umap(filtered_adata)

# 1e) Now you can do e.g. sc.pl.umap to see the embedding
sc.pl.umap(filtered_adata, color=["leiden"])

# 2a) Ensure you have "x_centroid" and "y_centroid" in filtered_adata.obs
#     If not, you can merge them from your nuclei_gdf or compute them:
#         nuclei_gdf["x_centroid"] = nuclei_gdf.geometry.centroid.x
#         nuclei_gdf["y_centroid"] = nuclei_gdf.geometry.centroid.y
#     Then filtered_adata.obs = filtered_adata.obs.merge(...)

# 2b) Create a color palette for each cluster
unique_clusters = filtered_adata.obs["leiden"].unique().tolist()
cluster_palette = sns.color_palette("bright", n_colors=len(unique_clusters))

cluster_colors = {
    str(cluster): cluster_palette[i]
    for i, cluster in enumerate(unique_clusters)
}
# Assign colors to each cell according to its cluster
filtered_adata.obs["cluster_color"] = filtered_adata.obs["leiden"].astype(str).map(cluster_colors)

# 2c) Plot the spatial distribution
plt.figure(figsize=(14, 10))

plt.scatter(
    filtered_adata.obs["x_centroid"],
    filtered_adata.obs["y_centroid"],
    c=filtered_adata.obs["cluster_color"].tolist(),
    alpha=0.6,
    s=5
)

plt.xlabel("X Centroid")
plt.ylabel("Y Centroid")
plt.title("Spatial Distribution of Clusters (Filtered)")

# Invert y-axis if your coordinate system has (0,0) top-left
plt.gca().invert_yaxis()

# Construct a legend for the clusters
legend_elements = [
    plt.Line2D(
        [0], [0],
        marker='o',
        color='w',
        markerfacecolor=color,
        markersize=8,
        label=f"Cluster {cluster}"
    )
    for cluster, color in cluster_colors.items()
]
plt.legend(
    handles=legend_elements,
    title="Cluster",
    bbox_to_anchor=(1.05, 1),
    loc='upper left'
)

plt.axis("off")
plt.show()

# %%
