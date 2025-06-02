"""
End‑to‑end contrastive fine‑tuning of UNI2 with on‑the‑fly patch extraction
and DataLoader prefetching.

* Extracts cell‑centred patches directly from the WSI with OpenSlide.
* Uses a torch Dataset/DataLoader (multi‑worker, pinned memory) so JPEG
  decoding overlaps GPU compute.
* Drops CLS + REG tokens, selects centre spatial tokens (level 1 by default).
* Fine‑tunes UNI2 jointly with projection heads via InfoNCE loss against
  fixed scGPT gene embeddings.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import numpy as np
import pandas as pd
import openslide
from PIL import Image
from tqdm import tqdm
import scanpy as sc
import argparse 

# ---------------------------------------------------------------------
# Define parameters
# ---------------------------------------------------------------------
# cancer = "lung"          # {lung, breast, …}
# ground_truth = "refined"      # dataset variant
# level = 0               # centre‑token level (0 or 1 here)
# batch_size = 72
# num_workers = 8               # DataLoader workers (tune to CPU cores)
# proj_dim = 128             # dimension of joint embedding space
# lr = 1e-5
# epochs = 20

parser = argparse.ArgumentParser(description="Run training with specified parameters.")

parser.add_argument("--cancer", type=str, default="lung",
                    choices=["lung", "breast", "lymph_node", "prostate", "skin", "ovarian", "cervical"],
                    help="Cancer type to use")
parser.add_argument("--ground_truth", type=str, default="refined",
                    help="Dataset variant for ground truth")
parser.add_argument("--level", type=int, default=0,
                    choices=[0, 1],
                    help="Center-token level (0 or 1)")
parser.add_argument("--batch_size", type=int, default=72,
                    help="Batch size for training")
parser.add_argument("--num_workers", type=int, default=8,
                    help="Number of DataLoader workers")
parser.add_argument("--proj_dim", type=int, default=128,
                    help="Projection dimension of joint embedding space")
parser.add_argument("--lr", type=float, default=1e-5,
                    help="Learning rate")
parser.add_argument("--epochs", type=int, default=20,
                    help="Number of training epochs")

args = parser.parse_args()

cancer = args.cancer         # {lung, breast, …}
ground_truth = args.ground_truth     # dataset variant
level = args.level           # centre‑token level (0 or 1 here)
batch_size = args.batch_size
num_workers = args.num_workers               # DataLoader workers (tune to CPU cores)
proj_dim = args.proj_dim            # dimension of joint embedding space
lr = args.lr
epochs = args.epochs


#  ---- paths ---------------------------------------------------------
xenium_sample_dict = {
    "lung":       "Xenium_Prime_Human_Lung_Cancer_FFPE_outs",
    "breast":     "Xenium_Prime_Breast_Cancer_FFPE_outs",
    "lymph_node": "Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_outs",
    "prostate":   "Xenium_Prime_Human_Prostate_FFPE_outs",
    "skin":       "Xenium_Prime_Human_Skin_FFPE_outs",
    "ovarian":    "Xenium_Prime_Ovarian_Cancer_FFPE_outs",
    "cervical":   "Xenium_Prime_Cervical_Cancer_FFPE_outs",
}
root   = "/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics"
xenium_sample  = xenium_sample_dict[cancer]
adata_path = f"{root}/{xenium_sample}/preprocessed/fine_tune_{ground_truth}_v2/processed_xenium_data_fine_tune_{ground_truth}_v2_annotated.h5ad"
emb_path  = f"/rsrch9/home/plm/idso_fa1_pathology/TIER2/paul-xenium/embeddings/public_data/{xenium_sample}/scGPT_CP.h5ad"
# slide_path= f"{root}/{xenium_sample}/Xenium_Prime_Human_Lung_Cancer_FFPE_he_image_coregistered_pyramid.ome.tif"  # adjust for cancer type if needed
slide_path= f"{root}/{xenium_sample}/{xenium_sample.rsplit('_',1)[0]}_he_image_registered.ome.tif" 

# ---------------------------------------------------------------------
# Load cell metadata & gene embeddings (fixed)
# ---------------------------------------------------------------------
adata = sc.read_h5ad(adata_path)
cell_df = adata.obs                     # index = cell IDs
gdata = sc.read_h5ad(emb_path)
gene_emb = pd.DataFrame(gdata.obsm["X_scGPT"], index=cell_df.index)

print("Cells:", cell_df.shape[0])
print("Gene‑embedding dim:", gene_emb.shape[1])

# ---------------------------------------------------------------------
# Slide info (MPP)
# ---------------------------------------------------------------------
slide = openslide.open_slide(slide_path)
mpp_x = float(slide.properties.get("openslide.comment").split('PhysicalSizeX="')[1].split('"')[0])
current_mpp= mpp_x
print("Slide MPP:", current_mpp)

# ---------------------------------------------------------------------
# Torch vision transform for UNI2
# ---------------------------------------------------------------------
patch_size = 224
transform = transforms.Compose([
    transforms.Resize((patch_size, patch_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])

target_mpp = 0.5  # 20×
scale = target_mpp / current_mpp

# ---------------------------------------------------------------------
# Dataset with on‑the‑fly patch extraction
# ---------------------------------------------------------------------
class CellPatchDataset(Dataset):
    def __init__(self, slide, cell_df, transform, scale, patch_size):
        self.slide      = slide
        self.cells      = cell_df.reset_index(drop=False)  # keep cell IDs
        self.tfm        = transform
        self.scale      = scale
        self.patch_size = patch_size

    def __len__(self):
        return len(self.cells)

    def _read_patch(self, x, y):
        big = int(self.patch_size * self.scale)
        tlx, tly = int(x - big/2), int(y - big/2)
        patch = self.slide.read_region((tlx, tly), 0, (big, big)).convert("RGB")
        return patch.resize((self.patch_size, self.patch_size), Image.LANCZOS)

    def __getitem__(self, idx):
        row = self.cells.iloc[idx]
        patch = self._read_patch(row.x_centroid, row.y_centroid)
        img_t = self.tfm(patch)
        return img_t, idx

# ---------------------------------------------------------------------
# DataLoader with prefetching
# ---------------------------------------------------------------------
dataloader = DataLoader(
    CellPatchDataset(slide, cell_df, transform, scale, patch_size),
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)

# ---------------------------------------------------------------------
# UNI2 model (trainable)
# ---------------------------------------------------------------------
model_dir = "/rsrch5/home/plm/phacosta/models/public/UNI2-h"
uni2_cfg = {
    'model_name':'vit_giant_patch14_224','img_size':224,'patch_size':14,'depth':24,
    'num_heads':24,'init_values':1e-5,'embed_dim':1536,'mlp_ratio':2.66667*2,
    'num_classes':0,'no_embed_class':True,'mlp_layer':timm.layers.SwiGLUPacked,
    'act_layer':torch.nn.SiLU,'reg_tokens':8,'dynamic_img_size':True
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = timm.create_model(pretrained=False, **uni2_cfg)
model.load_state_dict(torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
model.to(device).train()
prefix_tokens = getattr(model, "num_prefix_tokens", 9)  # fallback reg_tokens + cls 

level_idx_map = {
    0: torch.tensor([119,120,135,136]),
    1: torch.tensor([102,103,104,105,118,119,120,121,134,135,136,137,150,151,152,153]),
}
idx_center = level_idx_map[level].to(device)

# ---------------------------------------------------------------------
# Projection heads & loss
# ---------------------------------------------------------------------
class Projection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim,256), nn.ReLU(), nn.Linear(256,out_dim))
    def forward(self,x):
        return self.mlp(x)

proj_gene = Projection(gene_emb.shape[1], proj_dim).to(device)
proj_morph = Projection(1536, proj_dim).to(device)

def info_nce(a, p, t=0.07):
    a, p = F.normalize(a, dim=1), F.normalize(p, dim=1)
    return F.cross_entropy(a @ p.T / t, torch.arange(a.size(0), device=a.device))

opt = optim.Adam(list(model.parameters()) + list(proj_gene.parameters()) + list(proj_morph.parameters()), lr=lr)

# ---------------------------------------------------------------------
# Checkpoint directory & training loop
# ---------------------------------------------------------------------
ckpt_dir = f"/rsrch5/home/plm/phacosta/models/fine_tuned/UNI2/finetuned_uni2_contrastive_{cancer}"
os.makedirs(ckpt_dir, exist_ok=True)

best_loss = float("inf")
for epoch in range(1, epochs + 1):
    running = 0.0
    for imgs, idx_batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
        imgs = imgs.to(device, non_blocking=True)
        idx_np = idx_batch.cpu().numpy()
        gene_batch = torch.as_tensor(
            gene_emb.iloc[idx_np].values, dtype=torch.float32,
        ).to(device, non_blocking=True)

        tokens  = model.forward_features(imgs)
        spatial = tokens[:, prefix_tokens:, :]
        center  = spatial[:, idx_center, :].mean(1)

        g_proj = proj_gene(gene_batch)
        m_proj = proj_morph(center)
        loss   = 0.5 * (info_nce(g_proj, m_proj) + info_nce(m_proj, g_proj))

        opt.zero_grad(); loss.backward(); opt.step()
        running += loss.item()

    avg_loss = running / len(dataloader)
    print(f"Epoch {epoch}/{epochs} | Avg loss: {avg_loss:.4f}")

    # ---- checkpoint every epoch ----
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "proj_gene": proj_gene.state_dict(),
        "proj_morph": proj_morph.state_dict(),
        "optimizer": opt.state_dict(),
        "avg_loss": avg_loss,
    }
    torch.save(ckpt, f"{ckpt_dir}/epoch_{epoch:03d}.pth")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(ckpt, f"{ckpt_dir}/best.pth")
        print("Saved new best checkpoint")

print("✓ Training complete. Best loss:", best_loss)
