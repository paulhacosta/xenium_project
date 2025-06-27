#%%
import matplotlib.pyplot as plt
import numpy as np
import openslide
import scanpy as sc 

import numpy as np
from PIL import Image
import cv2

import pandas as pd
import os
from matplotlib.lines import Line2D

#% 
save_dir = "/rsrch9/home/plm/idso_fa1_pathology/TIER2/paul-xenium/classification_results/public_data/figures"
#%% Random heatmaps

# Create random matrices
np.random.seed(42)
gene_embeddings = np.random.rand(5, 10)  # 10 cells x 10 features
morphology_embeddings = np.random.rand(5, 10)  # 10 cells x 10 features


# Plot gene embeddings heatmap (blue colormap)
plt.figure(figsize=(8, 4))
plt.imshow(gene_embeddings, cmap='Blues', aspect='auto')
plt.colorbar(ticks=[])
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()

# Plot morphology embeddings heatmap (pink colormap)
plt.figure(figsize=(8, 4))
plt.imshow(morphology_embeddings, cmap='PuRd', aspect='auto')
plt.colorbar(ticks=[])
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()

#%% Single line heatmap 


# Generate a random 1x12 array
data = np.random.rand(256, 1)

# Create the heatmap
fig, ax = plt.subplots(figsize=(1, 40))  # wider and short
cax = ax.imshow(data, cmap='PuRd', aspect='auto')

# Remove ticks
ax.set_xticks([])
ax.set_yticks([])

# Remove spines
for spine in ax.spines.values():
    spine.set_visible(False)

# Optional: Add colorbar if you want
# fig.colorbar(cax, orientation='horizontal')

plt.show()


# Generate a random 1x12 array
data = np.random.rand(12, 1)

# Create the heatmap
fig, ax = plt.subplots(figsize=(1, 6))  # wider and short
cax = ax.imshow(data, cmap='PuRd', aspect='auto')

# Remove ticks
ax.set_xticks([])
ax.set_yticks([])

# Remove spines
for spine in ax.spines.values():
    spine.set_visible(False)

# Optional: Add colorbar if you want
# fig.colorbar(cax, orientation='horizontal')

plt.show()


# %% Define sample for slide plots
slide_path = "/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Xenium_Prime_Human_Lung_Cancer_FFPE_outs/Xenium_Prime_Human_Lung_Cancer_FFPE_he_image_registered.ome.tif"
slide = openslide.open_slide(slide_path)

data_path = f"/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Xenium_Prime_Human_Lung_Cancer_FFPE_outs/preprocessed/fine_tune_refined_v2/processed_xenium_data_fine_tune_refined_v2_annotated.h5ad"
adata = sc.read_h5ad(data_path)
cell_df = adata.obs
# %% Regular patch for UNI2
idx=42247
# idx = 13920
# Extract MPP
mpp_x = float(slide.properties.get("openslide.comment").split('PhysicalSizeX="')[1].split('"')[0])
current_mpp = mpp_x
print("Slide MPP:", current_mpp)

x_centroid = cell_df.iloc[idx]['x_centroid']
y_centroid = cell_df.iloc[idx]['y_centroid']

# --- Extract current slide MPP ---
mpp_x = float(slide.properties.get("openslide.comment").split('PhysicalSizeX="')[1].split('"')[0])
current_mpp = mpp_x
print("Slide MPP:", current_mpp)

# --- Calculate scale ---
target_mpp = 0.5  # Target resolution
scale = target_mpp / current_mpp

# --- Extract the patch ---
patch_size = 224  # Final size after resizing
big = int(patch_size * scale)

tlx, tly = int(x_centroid - big / 2), int(y_centroid - big / 2)
patch = slide.read_region((tlx, tly), 0, (big, big)).convert("RGB")
patch = patch.resize((patch_size, patch_size), Image.LANCZOS)
patch_array = np.array(patch)  # (224, 224, 3)

# --- Add center marker ---
h, w, _ = patch_array.shape
center_x, center_y = w // 2, h // 2

# Draw a cross marker
color = (0, 255, 0)  # Red in BGR
marker_size = 1
thickness = 3

# Since OpenCV uses BGR, we use patch_array directly
cv2.line(patch_array,
         (center_x - marker_size, center_y),
         (center_x + marker_size, center_y),
         color,
         thickness)

cv2.line(patch_array,
         (center_x, center_y - marker_size),
         (center_x, center_y + marker_size),
         color,
         thickness)

# --- Show the result ---
plt.figure(figsize=(5, 5))
plt.imshow(patch_array)
plt.axis('off')
plt.show()


# %% Patch image for UNI2

# --- Split into subpatches ---

# Assume patch_array is (224, 224, 3)
n_splits = 14  # 14x14 grid

# Calculate size of each subpatch
h, w, c = patch_array.shape
sub_h = h // n_splits
sub_w = w // n_splits

# Create a blank canvas to arrange subpatches with padding
padding = 2  # white space between subpatches
canvas_h = n_splits * sub_h + (n_splits - 1) * padding
canvas_w = n_splits * sub_w + (n_splits - 1) * padding

# White background
canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

# Fill canvas with subpatches
for i in range(n_splits):
    for j in range(n_splits):
        y0 = i * (sub_h + padding)
        x0 = j * (sub_w + padding)
        
        subpatch = patch_array[
            i*sub_h:(i+1)*sub_h,
            j*sub_w:(j+1)*sub_w,
            :
        ]
        
        canvas[y0:y0+sub_h, x0:x0+sub_w, :] = subpatch

# --- Display the result ---
plt.figure(figsize=(8, 8))
plt.imshow(canvas)
plt.axis('off')
plt.show()

# %%
tlx, tly = int(x_centroid - big / 2), int(y_centroid - big / 2)
patch = slide.read_region((tlx, tly), 0, (big, big)).convert("RGB")
patch = patch.resize((patch_size, patch_size), Image.LANCZOS)
patch_array = np.array(patch)  # (224, 224, 3)

# --- Define 14x14 splitting parameters ---
n_splits = 14
h, w, c = patch_array.shape
sub_h = h // n_splits
sub_w = w // n_splits

# Center 6x6 block
# Center of 14 is 7
# We want blocks from 4 to 9 (inclusive of 4, exclusive of 10)
start_idx = 4
end_idx = 10  # because Python slicing is exclusive at the end

# Crop directly the big block (non-split version first)

crop_y0 = start_idx * sub_h
crop_y1 = end_idx * sub_h
crop_x0 = start_idx * sub_w
crop_x1 = end_idx * sub_w

center_block = patch_array[crop_y0:crop_y1, crop_x0:crop_x1, :]

# --- 1. Plot the non-split cropped 6x6 block ---
plt.figure(figsize=(6,6))
plt.imshow(center_block)
plt.axis('off')
plt.show()

# --- 2. Now create the split version with padding ---

# Create canvas
padding = 2
canvas_h = 6 * sub_h + (6-1) * padding
canvas_w = 6 * sub_w + (6-1) * padding
canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255  # white background

# Fill the canvas
for i in range(start_idx, end_idx):  # i = 4 to 9
    for j in range(start_idx, end_idx):  # j = 4 to 9
        row = i - start_idx  # 0 to 5
        col = j - start_idx  # 0 to 5

        y0 = row * (sub_h + padding)
        x0 = col * (sub_w + padding)

        subpatch = patch_array[
            i*sub_h:(i+1)*sub_h,
            j*sub_w:(j+1)*sub_w,
            :
        ]

        canvas[y0:y0+sub_h, x0:x0+sub_w, :] = subpatch

# Plot the 6x6 split subpatches
plt.figure(figsize=(8,8))
plt.imshow(canvas)
plt.axis('off')
plt.show()
# %% Performance bar plot - horizontal

# Define settings
tissues = ["Lung", "Breast", "Prostate"]
cell_types = ["Fibroblast", "Lymphocyte", "Tumor"]

# Reversed model order
models = ["Unimodal", "Spatial", "Dual-Modality", "Multi-Input"]
custom_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # blue, orange, green, red

# Corrected F1-scores from SingleR table
f1_scores_singler = {
    "Lung": [
        [0.79, 0.81, 0.91, 0.90],  # Fibroblast
        [0.50, 0.59, 0.87, 0.85],  # Lymphocyte (T cell)
        [0.82, 0.82, 0.91, 0.91],  # Tumor
    ],
    "Breast": [
        [0.68, 0.70, 0.90, 0.89],  # Fibroblast
        [0.60, 0.59, 0.88, 0.88],  # Lymphocyte (T cell)
        [0.85, 0.85, 0.93, 0.93],  # Tumor
    ],
    "Prostate": [
        [0.86, 0.86, 0.91, 0.90],  # Fibroblast
        [0.14, 0.14, 0.68, 0.44],  # Lymphocyte (T cell)
        [0.82, 0.82, 0.89, 0.88],  # Tumor
    ]
}

# Create subplots
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8), sharex=True)

for idx, tissue in enumerate(tissues):
    ax = axes[idx]
    scores = np.array(f1_scores_singler[tissue])  # shape: (3 cell types, 4 models)
    scores = scores[:, ::-1]  # reverse model order
    y_pos = np.arange(len(cell_types))
    bar_width = 0.2

    for i, (model, color) in enumerate(zip(models[::-1], custom_colors[::-1])):
        bars = ax.barh(
            y_pos + i * bar_width,
            scores[:, i],
            height=bar_width,
            color=color,
            label=model if idx == 0 else "",
            align='center'
        )
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{width:.2f}", va='center', ha='left', fontsize=8)

    ax.set_yticks(y_pos + bar_width * 1.5)
    ax.set_yticklabels(cell_types)
    ax.set_title(f"{tissue}")
    ax.set_xlim(0, 1.05)
    if idx == 1:
        ax.set_ylabel("Cell Type")
    if idx == 2:
        ax.set_xlabel("F1-score")

axes[0].legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")

plt.suptitle("Model Performance by Tissue and Cell Type (SingleR Labels)", fontsize=14)
plt.tight_layout()
plt.show()


# %% Performance plot - vertical AISTIL

# Define settings
tissues = ["Lung", "Breast", "Prostate"]
cell_types = ["Fibroblast", "Lymphocyte", "Tumor"]
models = ["Unimodal", "Spatial", "Dual-Modality", "Multi-Input"]
custom_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # blue, orange, green, red

# Corrected F1-scores from SingleR table
f1_scores_singler = {
    "Lung": [
        [0.79, 0.81, 0.91, 0.90],  # Fibroblast
        [0.50, 0.59, 0.87, 0.85],  # Lymphocyte (T cell)
        [0.82, 0.82, 0.91, 0.91],  # Tumor
    ],
    "Breast": [
        [0.68, 0.70, 0.90, 0.89],  # Fibroblast
        [0.60, 0.59, 0.88, 0.88],  # Lymphocyte (T cell)
        [0.85, 0.85, 0.93, 0.93],  # Tumor
    ],
    "Prostate": [
        [0.86, 0.86, 0.91, 0.90],  # Fibroblast
        [0.14, 0.14, 0.48, 0.44],  # Lymphocyte (T cell)
        [0.82, 0.82, 0.89, 0.88],  # Tumor
    ]
}

# Create subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 6), sharey=True)

bar_width = 0.2
x_pos = np.arange(len(cell_types))

# Custom legend for the star
star_legend = Line2D([0], [0], marker='*', color='w', label='Best Model',
                     markerfacecolor='black', markersize=14)

for idx, tissue in enumerate(tissues):
    ax = axes[idx]
    scores = np.array(f1_scores_singler[tissue])  # (3 cell types x 4 models)

    for i, model in enumerate(models):
        offset = -1.5 + i
        bar_positions = x_pos + offset * bar_width
        bars = ax.bar(bar_positions, scores[:, i], width=bar_width,
                      color=custom_colors[i], label=model if idx == 2 else "")

    # Highlight top performers
    for j, ct in enumerate(cell_types):
        values = scores[j]
        max_val = np.max(values)
        top_indices = [i for i, v in enumerate(values) if np.isclose(v, max_val)]

        if len(top_indices) == 1:
            i = top_indices[0]
            bar_x = x_pos[j] + (-1.5 + i) * bar_width
            ax.text(bar_x, max_val + 0.015, "★", ha='center', va='bottom', fontsize=14, color='black')
        else:
            x_start = x_pos[j] + (-1.5 + top_indices[0]) * bar_width
            x_end = x_pos[j] + (-1.5 + top_indices[-1]) * bar_width
            y = max_val + 0.015
            ax.plot([x_start, x_start, x_end, x_end], [y, y + 0.01, y + 0.01, y], lw=1.5, c='black')
            ax.text((x_start + x_end) / 2, y + 0.015, "★", ha='center', va='bottom', fontsize=14, color='black')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(cell_types, fontsize=20, rotation=45)
    ax.set_title(f"{tissue}", fontsize=20)
    ax.set_ylim(0, 1.12)
    ax.tick_params(axis='y', labelsize=18)

axes[1].set_xlabel("Cell Type", fontsize=14)
axes[0].set_ylabel("F1-score", fontsize=18)
axes[2].legend(title="Model", bbox_to_anchor=(1.1, 1.1), loc="upper left", fontsize=10, title_fontsize=12)
axes[2].legend(
    handles=[*[
        Line2D([0], [0], color=color, lw=10, label=label)
        for color, label in zip(custom_colors, models)
    ], star_legend],
    title="Model",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    fontsize=10,
    title_fontsize=12
)
# plt.suptitle("Model Performance by Tissue and Cell Type (SingleR Labels)", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "f1_scores_lung-breast-prostate_aistil.png"), dpi=300)
plt.show()
# %% Performance plot - vertical singleR

# Define settings
tissues = ["Lung", "Breast", "Prostate"]
cell_types = ["B Cell", "Endothelial", "Epithelial", "Fibroblast", "Macrophage", "T Cell"]
models = ["Unimodal", "Spatial", "Dual-Modality", "Multi-Input"]
custom_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # blue, orange, green, red

# F1-scores for AISTIL labels
f1_scores = {
    "Lung": [
        [0.90, 0.90, 0.91, 0.96],  # B Cell
        [0.94, 0.93, 0.94, 0.98],  # Endothelial
        [0.97, 0.97, 0.97, 0.99],  # Epithelial
        [0.92, 0.93, 0.93, 0.93],  # Fibroblast
        [0.92, 0.93, 0.93, 0.97],  # Macrophage
        [0.93, 0.93, 0.94, 0.97],  # T Cell
    ],
    "Breast": [
        [0.78, 0.78, 0.81, 0.80],
        [0.78, 0.78, 0.79, 0.75],
        [0.95, 0.95, 0.96, 0.96],
        [0.82, 0.82, 0.83, 0.82],
        [0.82, 0.82, 0.83, 0.82],
        [0.88, 0.88, 0.89, 0.88],
    ],
    "Prostate": [
        [0.55, 0.59, 0.60, 0.96],
        [0.80, 0.81, 0.83, 0.79],
        [0.94, 0.95, 0.95, 0.94],
        [0.87, 0.88, 0.89, 0.88],
        [0.84, 0.82, 0.86, 0.79],
        [0.81, 0.81, 0.82, 0.78],
    ]
}

# Create 2x2 subplots (last one reserved for legend)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
axes = axes.flatten()

bar_width = 0.18
x = np.arange(len(cell_types))

for idx, tissue in enumerate(tissues):
    ax = axes[idx]
    scores = np.array(f1_scores[tissue])  # shape: (6 cell types, 4 models)

    for i, model in enumerate(models):
        offset = -1.5 + i
        bars = ax.bar(x + offset * bar_width, scores[:, i], width=bar_width,
                      color=custom_colors[i], label=model if idx == 0 else "")
   
    # Highlight top performers with stars or brackets
    for j, ct in enumerate(cell_types):
        values = scores[j]
        max_val = np.max(values)
        top_indices = [i for i, v in enumerate(values) if np.isclose(v, max_val)]

        if len(top_indices) == 1:
            i = top_indices[0]
            xpos = j + (-1.5 + i) * bar_width
            ax.text(xpos, max_val + 0.01, "★", ha='center', va='bottom', fontsize=14, color='black')
        else:
            x_start = j + (-1.5 + top_indices[0]) * bar_width
            x_end = j + (-1.5 + top_indices[-1]) * bar_width
            ax.plot([x_start, x_start, x_end, x_end], [max_val+0.01, max_val+0.015, max_val+0.015, max_val+0.01], lw=1.5, c='black')
            ax.text((x_start + x_end)/2, max_val + 0.02, "★", ha='center', va='bottom', fontsize=14, color='black')

    ax.set_xticks(x)
    ax.set_xticklabels(cell_types, fontsize=16, rotation=45, ha='right')
    ax.set_title(f"{tissue}", fontsize=20)
    ax.set_ylim(0.5, 1.05)
    ax.tick_params(axis='y', labelsize=18)
    
# Add legend to the 4th subplot
axes[3].axis('off')
legend_handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in custom_colors]
legend_labels = models + ["Best Model"]
legend_handles.append(plt.Line2D([0], [0], marker='*', color='black', markersize=12, linestyle='None'))
axes[3].legend(legend_handles, legend_labels, loc='upper left', fontsize=12, title_fontsize=14)

# Common labels and layout
fig.supylabel("F1-score", fontsize=18, x=0.01, y=0.55)
fig.supxlabel("Cell Type", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "f1_scores_lung-breast-prostate_singleR.png"), dpi=300)
plt.show()

# %%
