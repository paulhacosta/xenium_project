#%%
import os
import json
import scanpy as sc
import pandas as pd
from collections import defaultdict
import random

def hex_to_rgb_int(color_hex):
    value = int(color_hex.lstrip('#'), 16)
    if value > 0x7FFFFFFF:
        value -= 0x100000000
    return value

def random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def create_geojson(adata, label_column, output_path, qc_column=None):
    features = []
    label_to_cells = defaultdict(list)
    label_to_color = {}

    # Apply QC filter if provided
    if qc_column is not None:
        adata = adata[adata.obs[qc_column] == 1]

    # Group cells by label
    for _, row in adata.obs.iterrows():
        label = row[label_column]
        x = row['x_centroid']
        y = row['y_centroid']
        if pd.notnull(label) and pd.notnull(x) and pd.notnull(y):
            label_to_cells[label].append([float(x), float(y)])

    # Assign a color to each label
    for label in label_to_cells:
        label_to_color[label] = hex_to_rgb_int(random_color())

    # Create a MultiPoint feature for each label
    for label, coords in label_to_cells.items():
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "MultiPoint",
                "coordinates": coords
            },
            "properties": {
                "classification": {
                    "name": label,
                    "colorRGB": label_to_color[label]
                }
            }
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)

    print(f"Saved grouped GeoJSON: {output_path}")


#%%
# Paths
root_dir = "/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Xenium_Prime_Human_Lung_Cancer_FFPE_outs/preprocessed/fine_tune_refined_v2"
celltypist_path = os.path.join(root_dir, "processed_xenium_data_fine_tune_refined_ImmuneHigh_v2.h5ad")
singleR_path = os.path.join(root_dir, "processed_xenium_data_fine_tune_refined_v2_annotated.h5ad")

# Read data
cdata = sc.read_h5ad(celltypist_path)
sdata = sc.read_h5ad(singleR_path)

#%%
# Output files
ct_geojson_path = os.path.join(root_dir, "celltypist_annotations.geojson")
sr_geojson_path = os.path.join(root_dir, "singleR_annotations.geojson")

# Create GeoJSONs
create_geojson(cdata, label_column="majority_voting", output_path=ct_geojson_path)
create_geojson(sdata, label_column="singleR_class", output_path=sr_geojson_path)

# %%
# Output files
ct_geojson_qc_path = os.path.join(root_dir, "celltypist_annotations_qc.geojson")
sr_geojson_qc_path = os.path.join(root_dir, "singleR_annotations_qc.geojson")

# Filtered by QC
create_geojson(cdata, label_column="majority_voting", output_path=ct_geojson_qc_path, qc_column="qc_celltypist")
create_geojson(sdata, label_column="singleR_class", output_path=sr_geojson_qc_path, qc_column="qc_singleR")


# %%
