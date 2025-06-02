"""
Contrastive + Reconstruction fine‑tuning
=======================================

*   UNI2 backbone (ViT) + projection heads (InfoNCE, as before)
*   NEW: MLP regressor that tries to reconstruct the full Xenium 5 k
    log‑normalised expression profile from the centre‑token embedding.
*   Joint loss = InfoNCE + λ·MSE  (λ default = 0.1)
*   Checkpoints saved every epoch + best checkpoint on lowest joint loss
-----------------------------------------------------------------------
"""
# ---------------------------------------------------------------------
# 0. Imports
# ---------------------------------------------------------------------
import os, torch, timm, scanpy as sc
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd, numpy as np, openslide
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------
# 1. User params
# ---------------------------------------------------------------------
cancer = "lung"          # {lung, breast, …}
ground_truth = "refined"
batch_size = 72
num_workers = 4
proj_dim = 128
gene_dim = 5001            # Xenium Prime 5 k panel
lr = 1e-4
epochs = 10
lambda_mse = 0.1             # weight on reconstruction loss
level = 0               # centre‑token level (0 or 1)
freeze_uni2 = True           # set True for baseline
ckpt_dir = f"/rsrch5/home/plm/phacosta/models/fine_tuned/gene_reconstruction/ckpts_contrastive_recon_{cancer}"

# ---------------------------------------------------------------------
# 2. Paths
# ---------------------------------------------------------------------
xenium_sample_dict = {
    "lung":       "Xenium_Prime_Human_Lung_Cancer_FFPE_outs",
    "breast":     "Xenium_Prime_Breast_Cancer_FFPE_outs",
    "lymph_node": "Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_outs",
    "prostate":   "Xenium_Prime_Human_Prostate_FFPE_outs",
    "skin":       "Xenium_Prime_Human_Skin_FFPE_outs",
    "ovarian":    "Xenium_Prime_Ovarian_Cancer_FFPE_outs",
    "cervical":   "Xenium_Prime_Cervical_Cancer_FFPE_outs",
}
data_root   = "/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics"
embedding_root = "/rsrch9/home/plm/idso_fa1_pathology/TIER2/paul-xenium/embeddings"
xenium_sample  = xenium_sample_dict[cancer]

root = "/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics"
adata_path = f"{data_root}/{xenium_sample}/preprocessed/fine_tune_{ground_truth}_v2/processed_xenium_data_fine_tune_{ground_truth}_v2_annotated.h5ad"
emb_path = f"{embedding_root}/public_data/{xenium_sample}/scGPT_CP.h5ad"
slide_path = f"{data_root}/{xenium_sample}/{xenium_sample.rsplit('_',1)[0]}_he_image_registered.ome.tif" 

os.makedirs(ckpt_dir, exist_ok=True)

# ---------------------------------------------------------------------
# 3. Load data
# ---------------------------------------------------------------------
adata = sc.read_h5ad(adata_path)
cell_df = adata.obs                       # index = cell IDs
gdata = sc.read_h5ad(emb_path)
gene_emb = pd.DataFrame(gdata.obsm["X_scGPT"], index=cell_df.index)  # (N,512)

slide = openslide.open_slide(slide_path)
mpp_x = float(slide.properties.get("openslide.comment").split('PhysicalSizeX="')[1].split('"')[0])
current_mpp= mpp_x
target_mpp = 0.5           # 20×
scale = target_mpp / current_mpp
   
# ---------------------------------------------------------------------
# 4. Dataset & DataLoader
# ---------------------------------------------------------------------
patch_size = 224
tfm = transforms.Compose([
    transforms.Resize((patch_size, patch_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406),
                         std =(0.229,0.224,0.225)),
])

class CellPatchDS(Dataset):
    def __init__(self, slide, cells, tfm, scale, size):
        self.slide, self.cells, self.tfm = slide, cells.reset_index(drop=False), tfm
        self.scale, self.size = scale, size
    def __len__(self): return len(self.cells)
    def _patch(self, x,y):
        big = int(self.size*self.scale)
        tlx, tly = int(x-big/2), int(y-big/2)
        img = self.slide.read_region((tlx,tly),0,(big,big)).convert("RGB")
        return img.resize((self.size,self.size), Image.LANCZOS)
    def __getitem__(self, i):
        r = self.cells.iloc[i]
        return self.tfm(self._patch(r.x_centroid,r.y_centroid)), i

loader = DataLoader(
    CellPatchDS(slide, cell_df, tfm, scale, patch_size),
    batch_size, shuffle=True, num_workers=num_workers,
    pin_memory=True, persistent_workers=True, prefetch_factor=4
)

# ---------------------------------------------------------------------
# 5. Models
# ---------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

uni2_cfg = {
    'model_name':'vit_giant_patch14_224','img_size':224,'patch_size':14,'depth':24,
    'num_heads':24,'init_values':1e-5,'embed_dim':1536,'mlp_ratio':2.66667*2,
    'num_classes':0,'no_embed_class':True,'mlp_layer':timm.layers.SwiGLUPacked,
    'act_layer':torch.nn.SiLU,'reg_tokens':8,'dynamic_img_size':True
}
model = timm.create_model(pretrained=False, **uni2_cfg)
model.load_state_dict(torch.load("/rsrch5/home/plm/phacosta/models/public/UNI2-h/pytorch_model.bin", map_location="cpu"))
model.to(device).train()
if freeze_uni2:
    for p in model.parameters(): p.requires_grad = False
prefix_tokens = getattr(model, "num_prefix_tokens", 9)

level_idx = {
    0: torch.tensor([119,120,135,136], device=device),
    1: torch.tensor([102,103,104,105,118,119,120,121,
                     134,135,136,137,150,151,152,153], device=device)
}[level]

class Projection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim,256), nn.ReLU(), nn.Linear(256,out_dim))
    def forward(self,x): return self.mlp(x)

proj_gene  = Projection(gene_emb.shape[1], proj_dim).to(device)
proj_morph = Projection(1536, proj_dim).to(device)

reg_head = nn.Sequential(
    nn.Linear(1536,512), nn.ReLU(),
    nn.Linear(512,gene_dim)
).to(device)

def info_nce(a,p,t=0.07):
    a,p = F.normalize(a,dim=1), F.normalize(p,dim=1)
    return F.cross_entropy(a @ p.T / t, torch.arange(a.size(0), device=a.device))

opt = optim.Adam(
    list(filter(lambda p: p.requires_grad, model.parameters())) +
    list(proj_gene.parameters()) +
    list(proj_morph.parameters()) +
    list(reg_head.parameters()),
    lr=lr
)

# ---------------------------------------------------------------------
# 6. Training
# ---------------------------------------------------------------------
best_loss = float("inf")
for ep in range(1, epochs+1):
    run_info, run_mse = 0.0, 0.0
    model.train(); proj_gene.train(); proj_morph.train(); reg_head.train()
    for imgs, idx in tqdm(loader, desc=f"Epoch {ep}"):
        imgs = imgs.to(device, non_blocking=True)
        idx_np = idx.numpy()
        gene_batch = torch.as_tensor(
            gene_emb.iloc[idx_np].values, dtype=torch.float32
        ).to(device, non_blocking=True)




        # Forward
        tok   = model.forward_features(imgs)                 # (B,265,1536)
        spat  = tok[:, prefix_tokens:, :]
        center= spat[:, level_idx, :].mean(1)                # (B,1536)

        g_proj = proj_gene(gene_batch)                       # (B,128)
        m_proj = proj_morph(center)                          # (B,128)
        expr_pred = reg_head(center)                         # (B,5000)
        true_expr = torch.as_tensor(
            adata.X[idx_np].toarray(), dtype=torch.float32
        ).to(device, non_blocking=True)

        info = 0.5*(info_nce(g_proj,m_proj)+info_nce(m_proj,g_proj))
        mse  = F.mse_loss(expr_pred, true_expr)
        loss = info + lambda_mse*mse

        opt.zero_grad(); loss.backward(); opt.step()
        run_info += info.item(); run_mse += mse.item()

    info_avg = run_info/len(loader); mse_avg = run_mse/len(loader)
    print(f"Epoch {ep}/{epochs} | InfoNCE={info_avg:.4f} | MSE={mse_avg:.4f}")

    ckpt = {
        "epoch": ep, "model": model.state_dict(),
        "proj_gene": proj_gene.state_dict(),
        "proj_morph": proj_morph.state_dict(),
        "reg_head": reg_head.state_dict(),
        "opt": opt.state_dict(),
        "loss": float(loss)
    }
    torch.save(ckpt, f"{ckpt_dir}/epoch_{ep:03d}.pth")
    if loss < best_loss:
        best_loss = loss
        torch.save(ckpt, f"{ckpt_dir}/best.pth")
        print("✓ new best checkpoint")

print("Training done.  Best joint loss:", best_loss)