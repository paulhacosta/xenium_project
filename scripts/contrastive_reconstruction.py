"""
Contrastive + Reconstruction with an **internal 10 % hold‑out split**

•  Trains UNI2 + projection heads (InfoNCE) **and** a regressor that
   reconstructs Xenium‑5 k expression from the centre ViT embedding.
•  Cells are split 90 % / 10 % (stratified by pre‑computed Leiden clusters)
   before any optimisation.
•  After training, the script:
   – predicts expression on the unseen 10 % cells  
   – computes per‑gene Pearson R and R², saves them to CSV  
   – plots spatial scatter heatmaps (true vs. predicted) for marker genes
---------------------------------------------------------------------------
Adjust path strings, `marker_genes`, or `gene_dim` for your environment.
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
from sklearn.model_selection import train_test_split

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
freeze_uni2 = False           # set True for baseline
ckpt_dir = f"/rsrch5/home/plm/phacosta/models/fine_tuned/gene_reconstruction/ckpts_contrastive_recon_{cancer}_ss"
marker_genes = [
    # ‑‑ T‑cell markers
    "CD3D", "CD3E", "CD4", "CD8A", "TRBC1",
    # ‑‑ B‑cell markers
    "MS4A1",   # a.k.a. CD20
    "CD79A",
    "CD19",
    # ‑‑ Macrophage / myeloid markers
    "CD68",
    "CD163",
    "LYZ",
    # ‑‑ Example tumor / proliferation / stroma markers 
    "TP53", "MKI67", "COL1A1"
]

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

results_dir = f"/rsrch5/home/plm/phacosta/TIER2_Results/xenium_project/{xenium_sample}/"

os.makedirs(results_dir, exist_ok=True)
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



# --------------------------------------------------------------------- #
# 4  Hold‑out split (stratified by existing Leiden clusters)
# --------------------------------------------------------------------- #
# If 'leiden' labels are missing, make them from the scGPT embeddings
if "leiden" not in cell_df.columns:
    print("No 'leiden' column found – running a quick clustering on scGPT space.")
    ad_tmp = sc.AnnData(
        gene_emb.values,              # (N_cells, 512)
        obs=pd.DataFrame(index=cell_df.index)
    )
    sc.pp.neighbors(ad_tmp, use_rep="X", n_neighbors=15)
    sc.tl.leiden(ad_tmp, resolution=0.5)
    cell_df["leiden"] = ad_tmp.obs["leiden"].astype(str).values
    print("Leiden clusters added:", cell_df["leiden"].unique())



train_idx, test_idx = train_test_split(
    np.arange(len(cell_df)),
    test_size=0.10,
    stratify=cell_df['leiden'],
    random_state=42
)
print("Train:", len(train_idx), "| Test:", len(test_idx))

# --------------------------------------------------------------------- #
# 5  Slide & transform
# --------------------------------------------------------------------- #
slide         = openslide.open_slide(slide_path)
mpp_x         = float(slide.properties.get("openslide.comment").split('PhysicalSizeX=\"')[1].split('\"')[0])
scale         = 0.5 / mpp_x             # 20× target MPP
patch_size    = 224
tfm = transforms.Compose([
    transforms.Resize((patch_size, patch_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406),
                         std =(0.229,0.224,0.225)),
])

class CellPatchDS(Dataset):
    def __init__(self, idx_array):
        self.cells = cell_df.iloc[idx_array].reset_index(drop=False)
    def __len__(self): return len(self.cells)
    def __getitem__(self, i):
        r = self.cells.iloc[i]
        big = int(patch_size * scale)
        tlx, tly = int(r.x_centroid - big/2), int(r.y_centroid - big/2)
        img = slide.read_region((tlx,tly),0,(big,big)).convert("RGB")
        img = img.resize((patch_size,patch_size), Image.LANCZOS)
        return tfm(img), r.name  # r.name == original row index (cell ID)

def make_loader(idxs, shuffle, bs):
    return DataLoader(
        CellPatchDS(idxs), batch_size=bs, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, persistent_workers=True)

train_loader = make_loader(train_idx, shuffle=True,  bs=batch_size)
test_loader  = make_loader(test_idx,  shuffle=False, bs=batch_size)

# --------------------------------------------------------------------- #
# 6  Model definitions
# --------------------------------------------------------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

uni2_cfg = {
    'model_name':'vit_giant_patch14_224','img_size':224,'patch_size':14,'depth':24,
    'num_heads':24,'init_values':1e-5,'embed_dim':1536,'mlp_ratio':2.66667*2,
    'num_classes':0,'no_embed_class':True,'mlp_layer':timm.layers.SwiGLUPacked,
    'act_layer':torch.nn.SiLU,'reg_tokens':8,'dynamic_img_size':True
}
uni2_ckpt = "/rsrch5/home/plm/phacosta/models/public/UNI2-h/pytorch_model.bin"
model = timm.create_model(pretrained=False, **uni2_cfg)
model.load_state_dict(torch.load(uni2_ckpt, map_location="cpu"))
model.to(device).train()
if freeze_uni2:
    for p in model.parameters(): p.requires_grad = False
prefix_tokens = getattr(model, "num_prefix_tokens", 9)

level_idx = {
    0: torch.tensor([119,120,135,136], device=device),
    1: torch.tensor([
        102,103,104,105,118,119,120,121,
        134,135,136,137,150,151,152,153
    ], device=device)
}[level]

class Projection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim,256), nn.ReLU(), nn.Linear(256,out_dim))
    def forward(self,x): return self.mlp(x)

proj_gene  = Projection(gene_emb.shape[1], proj_dim).to(device)
proj_morph = Projection(1536, proj_dim).to(device)
reg_head   = nn.Sequential(
    nn.Linear(1536,512), nn.ReLU(),
    nn.Linear(512,gene_dim)
).to(device)

def info_nce(a,p,t=0.07):
    a,p = F.normalize(a,dim=1), F.normalize(p,dim=1)
    return F.cross_entropy(a @ p.T / t, torch.arange(a.size(0), device=a.device))

opt = optim.Adam(
    list(filter(lambda p: p.requires_grad, model.parameters())) +
    list(proj_gene.parameters()) + list(proj_morph.parameters()) +
    list(reg_head.parameters()),
    lr=lr
)

# --------------------------------------------------------------------- #
# 7  Training loop
# --------------------------------------------------------------------- #
best_loss = float("inf")
for ep in range(1, epochs+1):
    model.train(); proj_gene.train(); proj_morph.train(); reg_head.train()
    run_info = run_mse = 0.0
    for imgs, ids in tqdm(train_loader, desc=f"Epoch {ep}"):
        imgs = imgs.to(device, non_blocking=True)
        idx_np = ids.numpy()
        gene_batch = torch.as_tensor(
            gene_emb.iloc[idx_np].values, dtype=torch.float32
        ).to(device, non_blocking=True)

        tok    = model.forward_features(imgs)
        center = tok[:, prefix_tokens:, :][:, level_idx, :].mean(1)  # (B,1536)

        g_proj = proj_gene(gene_batch)
        m_proj = proj_morph(center)
        expr_pred = reg_head(center)

        true_expr = torch.as_tensor(
            adata.X[idx_np].toarray(), dtype=torch.float32
        ).to(device, non_blocking=True)

        info = 0.5*(info_nce(g_proj,m_proj)+info_nce(m_proj,g_proj))
        mse  = F.mse_loss(expr_pred, true_expr)
        loss = info + lambda_mse*mse

        opt.zero_grad(); loss.backward(); opt.step()
        run_info += info.item(); run_mse += mse.item()

    info_avg = run_info / len(train_loader)
    mse_avg  = run_mse  / len(train_loader)
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

print("Training done. Best joint loss:", best_loss)

# --------------------------------------------------------------------- #
# 8  Evaluation on the 10 % hold‑out
# --------------------------------------------------------------------- #
model.eval(); reg_head.eval()
preds, truths, ids_all = [], [], []
with torch.no_grad():
    for imgs, ids in tqdm(test_loader, desc="Predict hold‑out"):
        imgs = imgs.to(device, non_blocking=True)
        tok  = model.forward_features(imgs)
        center = tok[:, prefix_tokens:, :][:, level_idx, :].mean(1)
        preds.append(reg_head(center).cpu())
        truths.append(torch.as_tensor(
            adata.X[ids.numpy()].toarray(), dtype=torch.float32).cpu())
        ids_all.extend(ids.numpy())

pred_mat  = torch.cat(preds).numpy()   # (N_test,5k)
true_mat  = torch.cat(truths).numpy()

# per‑gene Pearson R & R²
R, R2 = [], []
for g in range(gene_dim):
    if np.std(true_mat[:, g]) == 0:
        R.append(np.nan); R2.append(np.nan); continue
    r = pearsonr(true_mat[:, g], pred_mat[:, g])[0]
    R.append(r); R2.append(r**2)

pd.Series(R,  name="PearsonR").to_csv(os.path.join(results_dir, "per_gene_R.csv"))
pd.Series(R2, name="R2").to_csv(os.path.join(results_dir, "per_gene_R2.csv"))
print("Median Pearson R :", np.nanmedian(R))
print("Median R²        :", np.nanmedian(R2))

# --------------------------------------------------------------------- #
# 9  Spatial heatmaps for marker genes
# --------------------------------------------------------------------- #
gene_to_idx = {g:i for i,g in enumerate(adata.var_names)}
test_cells  = cell_df.iloc[test_idx].copy()

# Create a subdirectory for plots if desired
plot_dir = os.path.join(results_dir, "gene_plots")
os.makedirs(plot_dir, exist_ok=True)

for g in marker_genes:
    gi = gene_to_idx[g]
    test_cells[f"{g}_true"] = true_mat[:, gi]
    test_cells[f"{g}_pred"] = pred_mat[:, gi]

    for kind in ["true", "pred"]:
        plt.figure(figsize=(6, 5))
        plt.scatter(
            test_cells['x_centroid'], test_cells['y_centroid'],
            c=test_cells[f"{g}_{kind}"], cmap="viridis", s=2
        )
        plt.gca().invert_yaxis()
        plt.axis('equal')
        plt.title(f"{g} ({kind})")
        plt.colorbar(label="log1p counts")
        plt.tight_layout()

        # Save the figure
        save_path = os.path.join(plot_dir, f"{g}_{kind}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
