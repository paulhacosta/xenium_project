#!/usr/bin/env python
# coding: utf-8

#%%

"""
BLEEP‑style contrastive training with UNI2 + scGPT
--------------------------------------------------
This script mirrors the original BLEEP training loop (DDP‑ready, `ProjectionHead`,
smoothed CLIP loss) but swaps in:
• **UNI2** as the image encoder (fine‑tuned)
• **Pre‑extracted scGPT gene embeddings** as the gene branch (optionally frozen)
• **Cell‑level patches** instead of Visium spots

Key difference from the first draft: **the explicit `F.normalize` calls on the
embeddings have been removed** to exactly match the original BLEEP loss.
"""

import os
import torch
import timm
import openslide
import scanpy as sc
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
import pandas as pd
from PIL import Image
import time
import argparse
import yaml

torch.backends.cuda.matmul.allow_tf32 = True           # ← optional speed


#%%

parser = argparse.ArgumentParser(
    description="BLEEP-UNI2 contrastive training"
)
parser.add_argument("--cancer",
                    type=str,
                    default="lung",
                    choices=["lung", "breast", "lymph_node",
                             "prostate", "skin", "ovarian", "cervical"],
                    help="Dataset (key of xenium_sample_dict)")
parser.add_argument("--epochs",
                    type=int,
                    default=10,
                    help="Training epochs")
parser.add_argument("--use_L1_reg",
                    action="store_true",
                    help="Enable L1 regularisation on projection heads")
parser.add_argument("--run_name",
                    type=str,
                    default="bleep_run",
                    help="Sub-folder inside fine_tuned/UNI2/ where checkpoints go")

args = parser.parse_args()            # ← parsed once

# -----------------------------------------------------------------------------
# Configuration (matching origianl BLEEP. Edit as needed)
# -----------------------------------------------------------------------------
class CFG:
    # data
    cancer = args.cancer          # {lung, breast, …}
    ground_truth = "refined"       # dataset variant
    level = 0              # UNI2 spatial‑token level
    batch_size = 72
    num_workers = 8

    # optimisation
    temperature = 1.0
    patience = 2.0
    projection_dim= 256
    lr = 1e-4
    weight_decay = 1e-3
    dropout = 0.1
    epochs = args.epochs

    # Embeddings
    morph_emb_dims = 1536
    gene_emb_dims = 512
    patch_size = 224

    # Regularization
    use_L1_reg = args.use_L1_reg
    l1_lambda = 1e-4
    
    # paths
    root = "/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics"
    xenium_sample_dict = {
        "lung":"Xenium_Prime_Human_Lung_Cancer_FFPE_outs",
        "breast": "Xenium_Prime_Breast_Cancer_FFPE_outs",
        "lymph_node": "Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_outs",
        "prostate": "Xenium_Prime_Human_Prostate_FFPE_outs",
        "skin": "Xenium_Prime_Human_Skin_FFPE_outs",
        "ovarian": "Xenium_Prime_Ovarian_Cancer_FFPE_outs",
        "cervical": "Xenium_Prime_Cervical_Cancer_FFPE_outs"
    }
    model_dir = "/rsrch5/home/plm/phacosta/models/public/UNI2-h"  # pretrained UNI2 weights
    ckpt_dir = f"/rsrch5/home/plm/phacosta/models/fine_tuned/UNI2/{args.run_name}"  # outputs


#%%

# -----------------------------------------------------------------------------
# Resolve dataset‑specific paths
# -----------------------------------------------------------------------------
sample  = CFG.xenium_sample_dict[CFG.cancer]
adata_path = f"{CFG.root}/{sample}/preprocessed/fine_tune_{CFG.ground_truth}_v2/processed_xenium_data_fine_tune_{CFG.ground_truth}_v2_annotated.h5ad"
emb_path  = f"/rsrch9/home/plm/idso_fa1_pathology/TIER2/paul-xenium/embeddings/public_data/{sample}/scGPT/scGPT_CP.h5ad"
# slide_path= f"{CFG.root}/{sample}/Xenium_Prime_Human_Lung_Cancer_FFPE_he_image_registered.ome.tif"
slide_path = f"{CFG.root}/{sample}/{sample.replace('outs','he_image_registered.ome.tif')}"


# -----------------------------------------------------------------------------
# Load metadata & gene embeddings (fixed / optionally frozen)
# -----------------------------------------------------------------------------
adata   = sc.read_h5ad(adata_path)
cell_df = adata.obs  # index = cell IDs (expects x_centroid / y_centroid cols)

gdata   = sc.read_h5ad(emb_path)
gene_emb = pd.DataFrame(gdata.obsm["X_scGPT"], index=cell_df.index)
print("Cells:", cell_df.shape[0], "| Gene‑embedding dim:", gene_emb.shape[1])

# -----------------------------------------------------------------------------
# OpenSlide & image resolution
# -----------------------------------------------------------------------------
slide = openslide.open_slide(slide_path)
MPP = float(slide.properties.get("openslide.comment").split('PhysicalSizeX="')[1].split('"')[0])
print("Slide MPP:", MPP)

# -----------------------------------------------------------------------------
# Torch transformation & patch parameters
# -----------------------------------------------------------------------------
transform = transforms.Compose([
    # transforms.Resize((CFG.patch_size, CFG.patch_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])
scale_factor = 0.5 / MPP  # rescale to ~20× (0.5µm/px)


#%%
# -----------------------------------------------------------------------------
# Dataset: on‑the‑fly cell‑centred patch extraction
# -----------------------------------------------------------------------------
class CellPatchDataset(Dataset):
    def __init__(self, slide, cell_df, gene_df, transform, scale, patch_size):
        self.slide = slide
        self.cells = cell_df.reset_index(drop=False)   # keeps cell IDs in col “index”
        self.gene_df = gene_df                          # gene_emb DataFrame
        self.tfm = transform
        self.scale = scale
        self.patch_size = patch_size

    def __len__(self):
        return len(self.cells)

    def _read_patch(self, x, y):
        big = int(self.patch_size * self.scale)
        tlx, tly = int(x - big/2), int(y - big/2)
        patch = self.slide.read_region((tlx, tly), 0, (big, big)).convert("RGB")
        return patch.resize((self.patch_size, self.patch_size), Image.LANCZOS)

    def __getitem__(self, idx):
        row   = self.cells.iloc[idx]
        patch = self._read_patch(row.x_centroid, row.y_centroid)
        img_t = self.tfm(patch)

        cell_id   = row["index"]                              # original cell ID
        gene_vec  = torch.tensor(
            self.gene_df.loc[cell_id].values, dtype=torch.float32
        )

        return {"image": img_t, "gene": gene_vec}

#%%
# -----------------------------------------------------------------------------
# Projection head (identical to original BLEEP)
# -----------------------------------------------------------------------------
class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim=CFG.projection_dim, dropout=CFG.dropout):
        super().__init__()
        self.proj = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc   = nn.Linear(projection_dim, projection_dim)
        self.do   = nn.Dropout(dropout)
        self.ln   = nn.LayerNorm(projection_dim)
    def forward(self, x):
        h = self.proj(x)
        x = self.gelu(h)
        x = self.fc(x)
        x = self.do(x)
        return self.ln(x + h)
#%%
# -----------------------------------------------------------------------------
# UNI2 image encoder (timm) – put into train mode for fine‑tuning
# -----------------------------------------------------------------------------
uni2_cfg = {
    'model_name':'vit_giant_patch14_224','img_size':224,'patch_size':14,'depth':24,
    'num_heads':24,'init_values':1e-5,'embed_dim':1536,'mlp_ratio':2.66667*2,
    'num_classes':0,'no_embed_class':True,'mlp_layer':timm.layers.SwiGLUPacked,
    'act_layer':torch.nn.SiLU,'reg_tokens':8,'dynamic_img_size':True,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
uni2 = timm.create_model(pretrained=False, **uni2_cfg)
uni2.load_state_dict(torch.load(os.path.join(CFG.model_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
uni2 = uni2.to(device).train()

# prefix_tokens = getattr(uni2, "num_prefix_tokens", 9)
# level_idx_map = {
#     0: torch.tensor([119,120,135,136]),
#     1: torch.tensor([102,103,104,105,118,119,120,121,134,135,136,137,150,151,152,153]),
# }
# center_idx = level_idx_map[CFG.level].to(device)

#%%
# -----------------------------------------------------------------------------
# Full BLEEP‑style model
# -----------------------------------------------------------------------------
class BLEEP_UNI2(nn.Module):
    def __init__(
        self,
        img_enc: nn.Module,
        gene_dim: int,
        morph_dim: int = CFG.morph_emb_dims,     # 1536
        init_temp: float = CFG.temperature       # 1.0 → logit_scale = 1
    ):
        super().__init__()
        # encoders + projection heads
        self.image_encoder = img_enc
        self.image_proj    = ProjectionHead(morph_dim)
        self.gene_proj     = ProjectionHead(gene_dim)

        # learnable logit-scale   (log(1/τ))
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / init_temp)))

        # register centre-token indices so they move with .to(device)
        self.prefix_tokens = 9                           # CLS + 8 REG
        self.register_buffer(
            "center_idx",
            torch.tensor([119, 120, 135, 136], dtype=torch.long),
            persistent=False
        )

    # ------------------------------------------------------------------
    def _encode_image(self, imgs: torch.Tensor) -> torch.Tensor:
        tok     = self.image_encoder.forward_features(imgs)        # (B, 265, 1536)
        spatial = tok[:, self.prefix_tokens:, :]                  # drop prefixes
        center  = spatial.index_select(1, self.center_idx).mean(1)
        return self.image_proj(center)                            # (B,256)

    # ------------------------------------------------------------------
    def forward(self, imgs: torch.Tensor, genes: torch.Tensor) -> torch.Tensor:
        #  embeddings (L2-normalised)
        img_vec  = F.normalize(self._encode_image(imgs),  dim=-1)  # (B,256)
        gene_vec = F.normalize(self.gene_proj(genes),     dim=-1)  # (B,256)

        # contrastive logits with learnable temperature
        scale  = self.logit_scale.exp()                            # scalar > 0
        logits = scale * (gene_vec @ img_vec.T)                    # (B,B)

        # smoothed intra-modal targets (no gradients needed)
        with torch.no_grad():
            sim_img  = scale * (img_vec  @ img_vec.T)
            sim_gene = scale * (gene_vec @ gene_vec.T)
            targets  = F.softmax(0.5 * (sim_img + sim_gene), dim=-1)  # (B,B)

        # cross-entropy, symmetrised
        loss_gene = F.cross_entropy(logits,   targets,   reduction='none')
        loss_img  = F.cross_entropy(logits.T, targets.T, reduction='none')
        return 0.5 * (loss_gene + loss_img).mean()

#%%
# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
loader = DataLoader(
    CellPatchDataset(slide, cell_df, gene_emb, transform, scale_factor, CFG.patch_size),
    batch_size=CFG.batch_size,
    shuffle=True,
    num_workers=CFG.num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)

# helper – returns a plain dict with only user-defined fields
def cfg_to_dict(cfg_cls):
    # vars(cfg_cls) is the writable view of the class namespace
    return {k: v for k, v in vars(cfg_cls).items() if not k.startswith("__")}

#%%
# -----------------------------------------------------------------------------
# Optimiser & training loop
# -----------------------------------------------------------------------------
model = BLEEP_UNI2(uni2, gene_emb.shape[1]).to(device)
opt   = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
scaler = GradScaler()                          

os.makedirs(CFG.ckpt_dir, exist_ok=True)


best = float("inf")
for epoch in range(1, CFG.epochs + 1):
    model.train()
    running = 0.0
    start   = time.time()

    # wrap loader in tqdm → shows progress & ETA
    prog = tqdm(loader, desc=f"Epoch {epoch}/{CFG.epochs}", unit="batch")
    for step, batch in enumerate(prog, 1):
        imgs  = batch["image"].to(device, non_blocking=True)
        genes = batch["gene"].to(device,  non_blocking=True)

        with autocast():    
            loss = model(imgs, genes)
            if CFG.use_L1_reg:
                l1_penalty = torch.zeros([], device=device)
                for name, p in model.named_parameters():
                    if p.requires_grad and ("image_proj" in name or "gene_proj" in name):
                        l1_penalty += p.abs().sum()
                total_loss = loss + CFG.l1_lambda * l1_penalty
            else:
                total_loss = loss

        # opt.zero_grad()
        # total_loss.backward()
        # opt.step()
        opt.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()    # backward (scaled)
        scaler.step(opt)                       # optimiser step (scaled)
        scaler.update()  
        running += total_loss.item()      # log what you optimise
        
        prog.set_postfix(loss=running / step, lr=opt.param_groups[0]["lr"])

    avg = running / len(loader)
    elapsed = time.time() - start
    print(f"Epoch {epoch}: avg loss {avg:.4f}  |  time {elapsed/60:.1f} min")

    torch.save(
        {"epoch": epoch,
         "model": model.state_dict(),
         "opt":   opt.state_dict(),
         "loss":  avg},
        f"{CFG.ckpt_dir}/epoch_{epoch:03d}.pth"
    )

    if avg < best:
        best = avg
        torch.save({"epoch": epoch, "model": model.state_dict()},
                   f"{CFG.ckpt_dir}/best.pth")
        print("✓ new best")

        cfg_path = os.path.join(CFG.ckpt_dir, "config.yaml")
        with open(cfg_path, "w") as fp:
            # vars(CFG) is the writable dict of the class namespace
            yaml.safe_dump(
                {k: v for k, v in vars(CFG).items() if not k.startswith("__")},
                fp,
                sort_keys=False)
        print(f"↳ wrote config to {cfg_path}")
print("Training complete. Best loss:", best)


