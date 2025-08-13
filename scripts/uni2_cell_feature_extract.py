# requirements: pip install pyyaml timm openslide-python pandas numpy tqdm scanpy torch torchvision pillow

import os, argparse, yaml
import torch, timm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np, pandas as pd
import openslide
from PIL import Image
from tqdm import tqdm
import scanpy as sc
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# Read parameters from a YAML file
# ──────────────────────────────────────────────────────────────────────────────
def get_config() -> dict:
    p = argparse.ArgumentParser(description="Extract token embeddings from WSIs")
    p.add_argument("--config", required=True, help="Path to YAML file with run parameters")
    cfg_path = p.parse_args().config

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # sensible defaults if optional keys are missing
    defaults = dict(
        slide_mpp = 0.25,  # µm / pixel
        level     = 0,
        batch_size= 128,
        num_workers = 8,
        patch_size  = 224,
        target_mpp  = 0.5,  # 20× for UNI2
    )
    for k, v in defaults.items():
        cfg.setdefault(k, v)

    # quick sanity-check
    required = ["slide_path", "save_path", "coordinate_path"]
    missing  = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing keys in YAML file: {missing}")

    return cfg

cfg = get_config()

# ──────────────────────────────────────────────────────────────────────────────
# Unpack variables
# ──────────────────────────────────────────────────────────────────────────────
slide_path      = cfg["slide_path"]
save_path       = cfg["save_path"]
coordinate_path = cfg["coordinate_path"]
slide_mpp       = cfg["slide_mpp"]
level           = cfg["level"]
batch_size      = cfg["batch_size"]
num_workers     = cfg["num_workers"]
patch_size      = cfg["patch_size"]
target_mpp      = cfg["target_mpp"]

# --------------------------------------------------------------------------- #
# Create save directory
sample_name   = os.path.splitext(os.path.basename(slide_path))[0]
emb_save_dir  = os.path.join(save_path, sample_name, f"level_{level}")
os.makedirs(emb_save_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Load cell centroids
cell_df = pd.read_csv(coordinate_path)                       # must have x_centroid / y_centroid
print("Cells:", cell_df.shape[0])

# ---------------------------------------------------------------------------
# OpenSlide & scale factor
slide = openslide.open_slide(slide_path)
scale = target_mpp / slide_mpp
print(f"Slide mpp: {slide_mpp} | Target mpp: {target_mpp} | Scale: {scale:.3f}")

# ---------------------------------------------------------------------------
# TorchVision preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])

scale = target_mpp / slide_mpp
print(f"Target mpp: {target_mpp} | Scale factor: {scale}")
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
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_dir = "/rsrch9/home/plm/idso_fa1_pathology/TIER2/paul-xenium/models/public/UNI2-h"

uni2_cfg = {
    'model_name':'vit_giant_patch14_224','img_size':224,'patch_size':14,'depth':24,
    'num_heads':24,'init_values':1e-5,'embed_dim':1536,'mlp_ratio':2.66667*2,
    'num_classes':0,'no_embed_class':True,'mlp_layer':timm.layers.SwiGLUPacked,
    'act_layer':torch.nn.SiLU,'reg_tokens':8,'dynamic_img_size':True
}

# load pretrained weights
model = timm.create_model(pretrained=False, **uni2_cfg).to(device)
model.load_state_dict(torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"), map_location="cuda"), strict=True)
model.eval()

prefix_tokens = getattr(model, "num_prefix_tokens", 9)  # fallback reg_tokens + cls 
level_idx_map = {
    0: torch.tensor([119,120,135,136]),
    1: torch.tensor([102,103,104,105,118,119,120,121,134,135,136,137,150,151,152,153])}

idx_center = level_idx_map[level].to(device)

morph_list  = []          # 1536‑D (UNI2 backbone)
cell_ids    = []

model.eval()
with torch.no_grad():
    for imgs, idx_batch in tqdm(dataloader, desc="Extract embeddings"):
        imgs   = imgs.to(device, non_blocking=True)
        idx_np = idx_batch.cpu().numpy()
        cell_ids.extend(cell_df.index[idx_np])

        # ---- UNI2 forward --------------------------------------------------
        tokens  = model.forward_features(imgs)                 # (B,265,1536)
        spatial = tokens[:, prefix_tokens:, :]                 # drop prefixes
        center  = spatial[:, idx_center, :].mean(1)            # (B,1536)

        # ---- morphology‑only embedding ------------------------------------
        morph_list.append(center.cpu())                        # keep 1536‑D
# --------------------------------------------------------------------------
# Concatenate and save
# --------------------------------------------------------------------------
morph_arr  = torch.cat(morph_list).numpy()     # (N_cells,1536)

pd.DataFrame(morph_arr, index=cell_ids).to_csv(os.path.join(emb_save_dir, "uni2_pretrained_embeddings.csv"))
print("Saved morphology‑only embeddings to:", emb_save_dir)