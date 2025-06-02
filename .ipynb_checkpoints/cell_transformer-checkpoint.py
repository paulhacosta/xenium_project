#%%
import os
import numpy as np 
import pandas as pd 
import scanpy as sc 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchmetrics import Accuracy

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

#%%
#%% Define Class Limiting Parameters
limit_classes = True  # Set to False to use all classes
target_classes = ["f", "t"]  # Modify this list to restrict classification to specific classes

#%% Load Embedding data
root_dir = "/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Xenium_Prime_Human_Lung_Cancer_FFPE_outs"
embedding_dir = "/rsrch9/home/plm/idso_fa1_pathology/TIER2/paul-xenium/embeddings/public_data/Xenium_Prime_Human_Lung_Cancer_FFPE_outs"
adata = sc.read_h5ad(os.path.join(root_dir, "preprocessed","fine_tune_refined_v2", "processed_xenium_data_fine_tune_refined_v2.h5ad"))
cell_data = adata.obs

# Spatial Information 
spatial_coords = cell_data[['x_centroid', 'y_centroid']].rename(columns={'x_centroid': 'x', 'y_centroid': 'y'})

# Load Morphological Embeddings
morph_embedding_csv = os.path.join(embedding_dir, "UNI2_cell_representation","morphology_embeddings_v2.csv")
morph_embeddings = pd.read_csv(morph_embedding_csv, index_col="Unnamed: 0")

# Load Gene embeddings
experiment = "xenium_5k_paired"
embeddings_dir = "/rsrch9/home/plm/idso_fa1_pathology/TIER2/paul-xenium/embeddings/public_data"
visium_folder = f"{embeddings_dir}/Visium_HD_Human_Lung_Cancer_post_Xenium_Prime_5k_Experiment2"
xenium_folder = f"{embeddings_dir}/Xenium_Prime_Human_Lung_Cancer_FFPE_outs"


embedding_file_dict = {"xenium_5k": f"{xenium_folder}/processed_xenium.csv",
                      "xenium_5k_paired": f"{xenium_folder}/processed_xenium_refined_v2.csv",
                      "visium_hd_8um_bins": f"{visium_folder}/008um/embeddings_output/processed_visium_hd_008um.csv",
                      "visium_hd_bin2cell": f"{visium_folder}/bin2cell/embeddings_output/processed_visium_hd_bin2cell.csv",
                      }

gene_embeddings_file = embedding_file_dict[experiment]
gene_embeddings = pd.read_csv(gene_embeddings_file, index_col="Unnamed: 0")
gene_embeddings.index = morph_embeddings.index

if limit_classes:
    num_classes = len(target_classes)
    cell_data = cell_data[cell_data["class"].isin(target_classes)]
    
    # Update corresponding embeddings and spatial coordinates
    gene_embeddings = gene_embeddings.loc[cell_data.index]
    morph_embeddings = morph_embeddings.loc[cell_data.index]
    spatial_coords = spatial_coords.loc[cell_data.index]


#%% Core Functions and Classes 

# Create Dataset class
class CellDataset(Dataset):
    def __init__(self, gene, morph, spatial, labels):
        self.gene = gene
        self.morph = morph
        self.spatial = spatial
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            self.gene[idx],
            self.morph[idx],
            self.spatial[idx],
            self.labels[idx]
        )


# Model Architecture
class CellTransformer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, num_classes=3):
        super().__init__()
        
        # Projection layers
        self.gene_proj = nn.Linear(512, d_model)
        self.morph_proj = nn.Linear(1536, d_model)
        
        # Spatial processing
        self.spatial_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, d_model)
        )
        
        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=4*d_model,
                dropout=0.1,
                batch_first=True  # Critical for correct dimensions
            ),
            num_layers=4
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )
        
        # Initialize CLS token
        self.cls_token = nn.Parameter(torch.randn(1, d_model))

    def forward(self, gene, morph, spatial):
        # Project inputs [B, 512], [B, 1536], [B, 2]
        gene_proj = self.gene_proj(gene).unsqueeze(1)  # [B, 1, d_model]
        morph_proj = self.morph_proj(morph).unsqueeze(1)
        spatial_proj = self.spatial_encoder(spatial).unsqueeze(1)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(gene.size(0), 1, -1)  # [B, 1, d_model]
        
        # Concatenate all features
        x = torch.cat([cls_tokens, gene_proj, morph_proj, spatial_proj], dim=1)  # [B, 4, d_model]
        
        # Transformer processing
        x = self.transformer(x)  # [B, 4, d_model]
        
        # Use CLS token for classification
        cls_output = x[:, 0, :]  # [B, d_model]
        return self.classifier(cls_output)



#%% Convert to torch tensors

# Convert DataFrames to PyTorch tensors
gene_tensor = torch.tensor(gene_embeddings.values, dtype=torch.float32)
morph_tensor = torch.tensor(morph_embeddings.values, dtype=torch.float32)
spatial_tensor = torch.tensor(spatial_coords.values, dtype=torch.float32)

# Normalize spatial coordinates (critical for neural networks)
def min_max_scale(tensor):
    min_vals = tensor.min(dim=0, keepdim=True)[0]
    max_vals = tensor.max(dim=0, keepdim=True)[0]
    return (tensor - min_vals) / (max_vals - min_vals + 1e-8)  # Avoid division by zero


spatial_tensor = min_max_scale(spatial_tensor)

# Convert labels to numerical indices
label_mapping = {"f": 0, "l": 1, "t": 2, "o":3}
labels = torch.tensor([label_mapping[lbl] for lbl in cell_data["class"]], dtype=torch.long)

#%% Initialize dataset and dataloader
batch_size = 64  # Adjust based on GPU memory
dataset = CellDataset(gene_tensor, morph_tensor, spatial_tensor, labels)

# Train/test split
indices = np.arange(len(gene_tensor))
train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=labels.numpy(), random_state=42)
train_dataset = Subset(dataset, train_idx)
test_dataset = Subset(dataset, test_idx)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)


# Compute class weights to handle imbalance
class_weights_np = compute_class_weight('balanced', classes=np.unique(labels.numpy()), y=labels.numpy())
class_weights = torch.tensor(class_weights_np, dtype=torch.float32)


#%% Training and Testing 

num_epochs = 100

# Initialize model
device = torch.device("cuda:1")
model = CellTransformer(d_model=512, num_heads=8, num_classes=num_classes).to(device)

# Training setup
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    
    for batch in train_loader:
        gene, morph, spatial, labels = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        outputs = model(gene, morph, spatial)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_correct += accuracy_metric(outputs, labels)
        total_samples += labels.size(0)

    print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Train Accuracy: {total_correct/total_samples:.2%}")

# Evaluation
model.eval()
total_correct, total_samples = 0, 0

with torch.no_grad():
    for batch in test_loader:
        gene, morph, spatial, labels = [b.to(device) for b in batch]
        outputs = model(gene, morph, spatial)
        total_correct += accuracy_metric(outputs, labels)
        total_samples += labels.size(0)

print(f"Test Accuracy: {total_correct/total_samples:.2%}")

# Save model
# torch.save(model.state_dict(), "cell_classifier.pth")





# %%
