#!/usr/bin/env python
# coding: utf-8

# In[1]:


#%%
import os
import numpy as np 
import pandas as pd 
import scanpy as sc 
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchmetrics import Accuracy

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import argparse 
# In[2]:

parser = argparse.ArgumentParser(description="Run training with specified parameters.")

parser.add_argument("--cancer", type=str, default="lung",
                    choices=["lung", "breast", "lymph_node", "prostate", "skin", "ovarian", "cervical"],
                    help="Cancer type to use")
parser.add_argument("--morph_version", type=str, default="v1",
                    help="Morphology feature variant for ground truth")
parser.add_argument("--label_source", type=str, default="singleR",
                    help="Source of labels for classificaiton")
parser.add_argument("--use_projections", type=int, default=0,
                    help="Use contrastive projections instead of zero-shot embeddings")
args = parser.parse_args()

cancer = args.cancer         # {lung, breast, …}
morph_version = args.morph_version     # dataset variant
label_source = args.label_source  # singleR, celltypist, aistil, combined
use_projections = args.use_projections # 0 or 1 

#%%
# Select platform
platform = "xenium" # xenium or visium 
ground_truth = "refined"  # refined or cellvit
level = 0
filtered_genes = False
# label_source = "aistil"  # singleR, celltypist, aistil, combined
use_qc = False
# morph_version = "v2"
limit_classes = True  # Set to False to use all classes


if platform == "xenium":
    xenium_folder_dict = {"lung": "Xenium_Prime_Human_Lung_Cancer_FFPE_outs",
                          "breast":"Xenium_Prime_Breast_Cancer_FFPE_outs",
                          "lymph_node": "Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_outs",
                          "prostate": "Xenium_Prime_Human_Prostate_FFPE_outs",
                          "skin": "Xenium_Prime_Human_Skin_FFPE_outs",
                          "ovarian": "Xenium_Prime_Ovarian_Cancer_FFPE_outs",
                          "cervical": "Xenium_Prime_Cervical_Cancer_FFPE_outs"
                          }

    xenium_folder = xenium_folder_dict[cancer]
    print("Processing:", xenium_folder)
    celltypist_data_path = f"/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/{xenium_folder}/preprocessed/fine_tune_{ground_truth}_v2/processed_xenium_data_fine_tune_{ground_truth}_ImmuneHigh_v2.h5ad"
    singleR_data_path = f"/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/{xenium_folder}/preprocessed/fine_tune_{ground_truth}_v2/processed_xenium_data_fine_tune_{ground_truth}_v2_annotated.h5ad"
    
    embedding_dir = f"/rsrch9/home/plm/idso_fa1_pathology/TIER2/paul-xenium/embeddings/public_data/{xenium_folder}"
    gene_emb_path = f"{embedding_dir}/scGPT/scGPT_CP.h5ad"

    if filtered_genes:
        gene_embedding_file = f"/rsrch9/home/plm/idso_fa1_pathology/TIER2/paul-xenium/embeddings/public_data/{xenium_folder}/processed_xenium_refined_clustering_filtered_v2.csv"
    else:
        gene_embedding_file = f"/rsrch9/home/plm/idso_fa1_pathology/TIER2/paul-xenium/embeddings/public_data/{xenium_folder}/processed_xenium_{ground_truth}_v2.csv"

    results_root = "/rsrch9/home/plm/idso_fa1_pathology/TIER2/paul-xenium/classification_results/public_data"
    results_dir = f"{results_root}/{xenium_folder}"
    os.makedirs(results_dir, exist_ok=True)
        
elif platform == "visium":
    data_path = "/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Visium_HD_Human_Lung_Cancer_post_Xenium_Prime_5k_Experiment2/binned_outputs/square_002um/preprocessed/bin2cell/to_tokenize/corrected_cells_matched_preprocessed_refined_v2.h5ad"

    gene_embedding_file = "/rsrch9/home/plm/idso_fa1_pathology/TIER2/paul-xenium/embeddings/public_data/Visium_HD_Human_Lung_Cancer_post_Xenium_Prime_5k_Experiment2/bin2cell/embeddings_output/processed_visium_hd_bin2cell.csv"
    morph_embedding_dir = f"/rsrch9/home/plm/idso_fa1_pathology/TIER2/paul-xenium/embeddings/public_data/Visium_HD_Human_Lung_Cancer_post_Xenium_Prime_5k_Experiment2"



# Load AnnData
# if label_source == "singleR":
data_path = singleR_data_path
adata = sc.read_h5ad(data_path)

if label_source == "celltypist":
    data_path = celltypist_data_path
    adata = sc.read_h5ad(data_path)
    
elif label_source == "combined":
    adata = sc.read_h5ad(singleR_data_path)
    bdata = sc.read_h5ad(celltypist_data_path)
    adata.obs["majority_voting"] = bdata.obs["majority_voting"]
    adata.obs["qc_celltypist"] = bdata.obs["qc_celltypist"]
    cell_data = adata.obs

cell_data = adata.obs

# Spatial Information 
spatial_coords = cell_data[['x_centroid', 'y_centroid']].rename(columns={'x_centroid': 'x', 'y_centroid': 'y'})

# Load gene Embeddings 
if not use_projections:
    print("Using zero-shot embeddings")
    gdata = sc.read_h5ad(gene_emb_path)
    gene_embeddings = pd.DataFrame(gdata.obsm["X_scGPT"])
    gene_embeddings.index = cell_data.index
    
    # Load Morphology Embeddings 
    if morph_version == "v1":
        morph_embedding_csv = os.path.join(embedding_dir, "UNI2_cell_representation",f"level_{level}","morphology_embeddings_v2.csv")  # morphology_embeddings_v2
    else:
        morph_embedding_csv = os.path.join(embedding_dir, "UNI2_cell_representation",f"level_{level}","uni2_pretrained_embeddings.csv")  # morphology_embeddings_v2
    morph_embeddings = pd.read_csv(morph_embedding_csv, index_col="Unnamed: 0")
else:
    print("Using contrastive projections")
    gene_embeddings = pd.read_csv(os.path.join(embedding_dir, "contrastive_learning", f"gene_projection_embeddings_{morph_version}.csv"), index_col=0)
    morph_embeddings = pd.read_csv(os.path.join(embedding_dir, "contrastive_learning", f"morph_projection_embeddings_{morph_version}.csv"), index_col=0)
    

if label_source=="singleR":
    print("Using labels from SingleR.")
    singleR_to_class_map = {
        "Smooth muscle": "fibroblast",
        "Fibroblasts": "fibroblast",
        "Endothelial cells": "endothelial",
        "CD4+ T-cells": "t_cell",
        "CD8+ T-cells": "t_cell",
        "B-cells": "b_cell",
        "Macrophages": "macrophage",
        "Epithelial cells": "epithelial",
    }
    
    target_classes = ["fibroblast", "endothelial",
                      "t_cell", "b_cell", "macrophage",
                      "epithelial"]
    
    # Map SingleR labels to 7-class system
    cell_data[label_source] = cell_data["singleR_class"].map(singleR_to_class_map)
    
    # Drop cells that are nan (if any)
    cell_data = cell_data.dropna(subset=[label_source])
    
    # Keep only those 7 classes
    cell_data = cell_data[cell_data[label_source].isin(target_classes)]
    
    if use_qc:
        cell_data = cell_data[cell_data["qc_singleR"]==1]

    
    # Reindex embeddings/coords
    gene_embeddings = gene_embeddings.reindex(cell_data.index)
    morph_embeddings = morph_embeddings.reindex(cell_data.index)
    spatial_coords = spatial_coords.reindex(cell_data.index)
    
    
elif label_source=="aistil":
    print("Using AISTIL labels")
    label_key = "class"
    target_classes = ["f", "l", "t"]  # Modify this list to restrict classification to specific classes
    if limit_classes:
        num_classes = len(target_classes)
        cell_data = cell_data[cell_data[label_key].isin(target_classes)]

        # Change index type for Visium data to match embeddings Idxs 
        if platform == "visium":
            morph_embeddings.index = morph_embeddings.index.astype(str)

        # Update corresponding embeddings and spatial coordinates
        gene_embeddings = gene_embeddings.reindex(cell_data.index)
        morph_embeddings = morph_embeddings.reindex(cell_data.index)
        spatial_coords = spatial_coords.reindex(cell_data.index)
    else:
        target_classes = ["f","l","o","t"]
        
elif label_source=="celltypist":
    print("Using CellTypist Labels")
    celltypist_to_class_map = {
        "Fibroblasts": "fibroblast",
        "Endothelial cells": "endothelial",
        "T cells": "t_cell",
        "B cells": "b_cell",
        "Macrophages": "macrophage",
        "Epithelial cells": "epithelial",
    }
    target_classes = ["fibroblast", "endothelial",
                      "t_cell", "b_cell", "macrophage",
                      "epithelial"]

    # Map SingleR labels to 7-class system
    cell_data[label_source] = cell_data["majority_voting"].map(celltypist_to_class_map)
    
    # Drop cells that are nan (if any)
    cell_data = cell_data.dropna(subset=[])
    
    # Keep only those 7 classes
    cell_data = cell_data[cell_data[label_source].isin(target_classes)]

    if use_qc:
        cell_data = cell_data[cell_data["qc_celltypist"]==1]

    # Reindex embeddings/coords
    gene_embeddings = gene_embeddings.reindex(cell_data.index)
    morph_embeddings = morph_embeddings.reindex(cell_data.index)
    spatial_coords = spatial_coords.reindex(cell_data.index)
    
elif label_source == "combined":
    print("Using combined SingleR and CellTypist labels (agreement only)")

    # Define the shared label map and target classes
    shared_class_map = {
        "Fibroblasts": "fibroblast",
        "Smooth muscle": "fibroblast",
        "Endothelial cells": "endothelial",
        "CD4+ T-cells": "t_cell",
        "CD8+ T-cells": "t_cell",
        "T cells": "t_cell",
        "B cells": "b_cell",
        "B-cells": "b_cell",
        "Macrophages": "macrophage",
        "Epithelial cells": "epithelial",
    }
    
    target_classes = ["fibroblast", "endothelial", "t_cell", "b_cell", "macrophage", "epithelial"]

    # First map the labels (these are safe)
    cell_data["singleR_mapped"] = cell_data["singleR_class"].map(shared_class_map)
    cell_data["celltypist_mapped"] = cell_data["majority_voting"].map(shared_class_map)
    
    # Then immediately filter with a properly aligned mask
    cell_data = cell_data[
        cell_data["singleR_mapped"].notnull() &
        cell_data["celltypist_mapped"].notnull() &
        (cell_data["singleR_mapped"] == cell_data["celltypist_mapped"])
    ].copy()
    

    # Rename the final label column
    cell_data["combined"] = cell_data["singleR_mapped"]
    
    if use_qc:
        qc_mask = cell_data["qc_singleR"] == 1
        if "qc_celltypist" in cell_data.columns:
            qc_mask &= cell_data["qc_celltypist"] == 1
        cell_data = cell_data[qc_mask]

    # Reindex everything to the filtered cells
    gene_embeddings = gene_embeddings.reindex(cell_data.index)
    morph_embeddings = morph_embeddings.reindex(cell_data.index)
    spatial_coords = spatial_coords.reindex(cell_data.index)



num_classes = len(target_classes)
label_mapping = {cls_name: i for i, cls_name in enumerate(target_classes)}
if label_source == "aistil":
    labels = pd.Series(cell_data[label_key].map(label_mapping))
else:
    labels = pd.Series(cell_data[label_source].map(label_mapping))


# In[3]:


print("Cell data index:", cell_data.index.tolist()[:5])
print("Morph embeddings index:", morph_embeddings.index.tolist()[:5])
print("Gene embeddings index:", gene_embeddings.index.tolist()[:5])
print("Spatial coords index:", spatial_coords.index.tolist()[:5])
cell_data


# In[ ]:


# Positional Encoding
class PositionalEncoding2D(nn.Module):
    """Sinusoidal positional encoding for spatial coordinates"""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(4, d_model)  # (sin(x), cos(x), sin(y), cos(y))
        
    def forward(self, coords):
        # Ensure coords is [B, 2]
        assert coords.ndim == 2, f"Expected coords shape [B, 2], but got {coords.shape}"
        x = coords[:, 0] * 2 * torch.pi
        y = coords[:, 1] * 2 * torch.pi
        
        pe = torch.stack([
            torch.sin(x), torch.cos(x),
            torch.sin(y), torch.cos(y)
        ], dim=-1)  # [B, 4]
        pe = self.proj(pe)  # [B, d_model]
        return pe

# Transformer Layer with Relative Position Attention
class RelativePositionTransformerLayer(nn.TransformerEncoderLayer):
    """Enhanced with relative position attention"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        # Initialize parent class with batch_first=True.
        super().__init__(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.pos_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.pos_norm = nn.LayerNorm(d_model)
        # Add a dropout for the feedforward branch
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, src, pos_emb):
        # Uncomment to debug
        # print(f"src shape before attn: {src.shape}")   # Expected [B, 2, d_model]
        # print(f"pos_emb shape before attn: {pos_emb.shape}")  # Expected [B, 1, d_model]

        # Ensure pos_emb has the same sequence length as src.
        pos_emb = pos_emb.expand(-1, src.shape[1], -1)  # Now [B, 2, d_model]
        
        # Uncomment to debug
        # print(f"pos_emb shape after expansion: {pos_emb.shape}")  # Should be [B, 2, d_model]

        # Standard self-attention
        src2 = self.self_attn(src, src, src, need_weights=False)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Position-aware attention
        src2 = self.pos_attn(src, pos_emb, pos_emb)[0]
        src = src + self.dropout2(src2)
        src = self.pos_norm(src)

        # Feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

# Cell Transformer
class CellTransformer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, num_classes=3):
        super().__init__()
        # Modality projections
        gene_input_dim = 128 if use_projections else 512
        self.gene_proj = nn.Linear(gene_input_dim, d_model)

        morph_input_dim = 128 if use_projections else 1536
        self.morph_proj = nn.Linear(morph_input_dim, d_model)

        self.spatial_pe = PositionalEncoding2D(d_model)

        # Modality type embeddings
        self.gene_type = nn.Parameter(torch.randn(1, d_model))
        self.morph_type = nn.Parameter(torch.randn(1, d_model))
        # Transformer layers
        self.layers = nn.ModuleList([
            RelativePositionTransformerLayer(d_model, num_heads)
            for _ in range(6)
        ])
        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, gene, morph, spatial):
        # Uncomment to debug
        # print(f"Input gene shape: {gene.shape}")      # Expected: [B, 512]
        # print(f"Input morph shape: {morph.shape}")      # Expected: [B, 1536]
        # print(f"Input spatial shape: {spatial.shape}")  # Expected: [B, 2]

        # Project each modality and add type embedding
        gene_emb = self.gene_proj(gene) + self.gene_type   # [B, d_model]
        morph_emb = self.morph_proj(morph) + self.morph_type  # [B, d_model]
        spatial_emb = self.spatial_pe(spatial)              # [B, d_model]
        
        # Uncomment to debug
        # print(f"Gene shape after projection: {gene_emb.shape}")    # Should be [B, d_model]
        # print(f"Morph shape after projection: {morph_emb.shape}")    # Should be [B, d_model]
        # print(f"Spatial shape after encoding: {spatial_emb.shape}")  # Should be [B, d_model]

        # Stack gene and morph embeddings into tokens
        tokens = torch.stack([gene_emb, morph_emb], dim=1)  # [B, 2, d_model]
        # Unsqueeze spatial embedding for attention (to be expanded in the layer)
        spatial_emb = spatial_emb.unsqueeze(1)  # [B, 1, d_model]

        # Pass tokens through transformer layers
        for layer in self.layers:
            tokens = layer(tokens, spatial_emb)  # Expected: [B, 2, d_model]

        # Pool over tokens and classify
        pooled = tokens.mean(dim=1)  # [B, d_model]
        return self.classifier(pooled)

# Data Handling
class CellDataset(Dataset):
    def __init__(self, gene, morph, spatial, labels):
        self.gene = torch.tensor(gene.values, dtype=torch.float32)      # [N, 512]
        self.morph = torch.tensor(morph.values, dtype=torch.float32)      # [N, 1536]
        self.spatial = torch.tensor(spatial.values, dtype=torch.float32)  # [N, 2]
        self.labels = torch.tensor(labels.values, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        gene_sample = self.gene[idx]     # [512]
        morph_sample = self.morph[idx]     # [1536]
        spatial_sample = self.spatial[idx] # [2]
        label = self.labels[idx]
        return gene_sample, morph_sample, spatial_sample, label

def compute_class_weights(labels):
    counts = np.bincount(labels.values.astype(int))
    weights = 1. / (counts + 1e-8)  # Prevent division by zero
    return torch.tensor(weights, dtype=torch.float32)



# Training and testing loop

num_epochs = 20 
batch_size = 64

# Initialize device and model
device = torch.device("cuda")
model = CellTransformer(d_model=512, num_heads=8, num_classes=num_classes).to(device)

# Optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
dataset = CellDataset(gene_embeddings, morph_embeddings, spatial_coords, labels)

# Split dataset into 80% train and 20% test (stratified by labels)
all_indices = np.arange(len(dataset))
train_idx, test_idx = train_test_split(
    all_indices, test_size=0.2, random_state=42, stratify=dataset.labels.numpy()
)
train_dataset = Subset(dataset, train_idx)
test_dataset = Subset(dataset, test_idx)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=3e-5, steps_per_epoch=len(train_loader), epochs=num_epochs
)
# Loss function with class weights moved to the proper device
criterion = nn.CrossEntropyLoss(weight=compute_class_weights(labels).to(device))

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    # Wrap the train_loader with tqdm for a progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
    for batch_idx, (gene, morph, spatial, lbls) in enumerate(progress_bar):
        gene, morph, spatial, lbls = (
            gene.to(device),
            morph.to(device),
            spatial.to(device),
            lbls.to(device)
        )

        optimizer.zero_grad()
        outputs = model(gene, morph, spatial)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item() * gene.size(0)
        _, predicted = torch.max(outputs, 1)
        total += lbls.size(0)
        correct += (predicted == lbls).sum().item()

        # Update progress bar with current loss
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / total
    accuracy = correct / total * 100 
    print(f"Epoch {epoch+1}: Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    # Evaluation on the test set
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for gene, morph, spatial, lbls in test_loader:
            gene, morph, spatial, lbls = (
                gene.to(device),
                morph.to(device),
                spatial.to(device),
                lbls.to(device)
            )
            outputs = model(gene, morph, spatial)
            loss = criterion(outputs, lbls)
            test_loss += loss.item() * gene.size(0)
            _, predicted = torch.max(outputs, 1)
            test_total += lbls.size(0)
            test_correct += (predicted == lbls).sum().item()
    avg_test_loss = test_loss / test_total
    test_accuracy = test_correct / test_total * 100
    print(f"Test: Loss: {avg_test_loss:.4f} | Accuracy: {test_accuracy:.2f}%")



# In[ ]:


# Eval on same data 
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for gene, morph, spatial, lbls in test_loader:
        gene = gene.to(device)
        morph = morph.to(device)
        spatial = spatial.to(device)
        lbls = lbls.to(device)

        outputs = model(gene, morph, spatial)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(lbls.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds, normalize="true")

# disp = ConfusionMatrixDisplay(confusion_matrix=cm,
#                               display_labels=target_classes)


# disp.plot(cmap='viridis', xticks_rotation='vertical')
# plt.title(f"Confusion Matrix: {platform}")
# plt.show()
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=target_classes))


# In[ ]:


report_dict = classification_report(all_labels, all_preds, target_names=target_classes, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

save_path = os.path.join(results_dir, "T4")
os.makedirs(save_path, exist_ok=True)
if not use_projections:
    report_df.to_csv(os.path.join(save_path, f"classification_report_{label_source}_{morph_version}.csv"))
else:
    report_df.to_csv(os.path.join(save_path, f"classification_report_{label_source}_{morph_version}_proj.csv"))
print("Results saved to:", save_path)


