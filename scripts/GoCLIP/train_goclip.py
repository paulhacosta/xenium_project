#!/usr/bin/env python3
"""
Contrastive training (CLIP-style) with UNI2 (vision) + CoCa (text)
- Pools 4 center tokens from UNI2
- Uses CoCa text encoder from OpenCLIP (OmiCLIP-style)
- Projection heads to a shared space and symmetric InfoNCE loss

Job-friendly:
- Argparse + CFG-style wiring
- Optional AMP/ BF16 mixed precision with safe grad clipping
- Progress bars, CSV logs, best/last checkpoints, YAML config dump
"""

from __future__ import annotations

import os, glob, csv, time, math, argparse, yaml
from contextlib import nullcontext
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import timm
import open_clip
import torchvision.transforms as T
import scanpy as sc
import openslide
from PIL import Image


# ──────────────────────────────
# Args → CFG
# ──────────────────────────────
parser = argparse.ArgumentParser(description="CLIP-GO contrastive training (UNI2 + CoCa)")

# data
parser.add_argument("--cancer", type=str, default="lung",
                    choices=["lung","breast","lymph_node","prostate","skin","ovarian","cervical"])
parser.add_argument("--ground_truth", type=str, default="refined")

# optimization
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=72)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-3)
parser.add_argument("--proj_dim", type=int, default=256)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--grad_clip", type=float, default=1.0)

# precision toggle
parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="fp32",
                    help="Numerics for training: fp32 (no AMP), fp16 (AMP), or bf16 (autocast if supported).")

# text
parser.add_argument("--context_len", type=int, default=76)
parser.add_argument("--freeze_text", action="store_true", help="Freeze CoCa text encoder (recommended for warmup)")

# run id / paths
parser.add_argument("--run_name", type=str, default="run1", help="Subfolder for checkpoints/logs")
parser.add_argument("--root", type=str, default="/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics")
parser.add_argument("--go_dir", type=str, default="/rsrch9/home/plm/idso_fa1_pathology/TIER2/paul-xenium/gene_ontology")
parser.add_argument("--model_dir", type=str, default="/rsrch9/home/plm/idso_fa1_pathology/TIER2/paul-xenium/models/public/UNI2-h")
parser.add_argument("--ckpt_root", type=str, default="/rsrch9/home/plm/idso_fa1_pathology/TIER2/paul-xenium/models/fine_tuned/GoCLIP")

# misc
parser.add_argument("--target_mpp", type=float, default=0.5, help="Target µm/px (≈20×)")
parser.add_argument("--seed", type=int, default=1337)

args = parser.parse_args()


class CFG:
    # data
    cancer = args.cancer
    ground_truth = args.ground_truth
    level = 0
    batch_size = args.batch_size
    num_workers = args.num_workers

    # optimisation
    temperature = 1.0
    projection_dim = args.proj_dim
    lr = args.lr
    weight_decay = args.weight_decay
    dropout = args.dropout
    epochs = args.epochs
    grad_clip = args.grad_clip

    # Embeddings
    morph_emb_dims = 1536
    patch_size = 224

    # Text / CoCa
    coca_model = "coca_ViT-L-14"
    coca_pretrain = "laion2B-s13b-b90k"
    context_len = args.context_len
    freeze_text = args.freeze_text

    # paths
    root = args.root
    xenium_sample_dict = {
        "lung":"Xenium_Prime_Human_Lung_Cancer_FFPE_outs",
        "breast": "Xenium_Prime_Breast_Cancer_FFPE_outs",
        "lymph_node": "Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_outs",
        "prostate": "Xenium_Prime_Human_Prostate_FFPE_outs",
        "skin": "Xenium_Prime_Human_Skin_FFPE_outs",
        "ovarian": "Xenium_Prime_Ovarian_Cancer_FFPE_outs",
        "cervical": "Xenium_Prime_Cervical_Cancer_FFPE_outs",
    }
    go_dir = args.go_dir
    model_dir = args.model_dir
    ckpt_dir = os.path.join(args.ckpt_root, args.cancer, args.run_name)
    target_mpp = args.target_mpp

    # precision
    precision = args.precision  # "fp32" | "fp16" | "bf16"

    # logging
    log_csv = os.path.join(ckpt_dir, "train_log.csv")
    cfg_yaml = os.path.join(ckpt_dir, "config.yaml")


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(args.seed)


# ──────────────────────────────
# Projection head
# ──────────────────────────────
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = CFG.projection_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, proj_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(proj_dim, proj_dim)
        self.ln  = nn.LayerNorm(proj_dim)
    def forward(self, x):
        h = self.fc1(x)
        x = self.act(h)
        x = self.fc2(x)
        return self.ln(x + h)


# ──────────────────────────────
# UNI2 wrapper (4-center tokens)
# ──────────────────────────────
class UNI2Wrapper(nn.Module):
    def __init__(self, uni2: nn.Module, centre_idx: List[int]):
        super().__init__()
        self.uni2 = uni2
        self.centre_idx = torch.tensor(centre_idx, dtype=torch.long)
        self.prefix_tokens = 9
    def forward(self, x: torch.Tensor):
        tok = self.uni2.forward_features(x)               # (B, 265, 1536)
        spatial = tok[:, self.prefix_tokens:, :]
        centre  = spatial.index_select(1, self.centre_idx.to(x.device)).mean(1)
        return centre                                     # (B, 1536)


# ──────────────────────────────
# Dataset: patch + GO sentence
# ──────────────────────────────
class CellPatchTextDataset(Dataset):
    def __init__(self, slide, cell_df: pd.DataFrame, sentences: pd.Series,
                 transform, scale: float, patch_size: int = CFG.patch_size):
        self.slide = slide
        self.cells = cell_df.reset_index(drop=False)   # keep cell_id in 'index'
        self.sentences = sentences
        self.tfm = transform
        self.scale = scale
        self.patch_size = patch_size

    def __len__(self): return len(self.cells)

    def _read_patch(self, x, y):
        big = int(self.patch_size * self.scale)
        tlx, tly = int(x - big/2), int(y - big/2)
        patch = self.slide.read_region((tlx, tly), 0, (big, big)).convert("RGB")
        return patch.resize((self.patch_size, self.patch_size), Image.LANCZOS)

    def __getitem__(self, idx):
        row   = self.cells.iloc[idx]
        patch = self._read_patch(row.x_centroid, row.y_centroid)
        img_t = self.tfm(patch)
        cell_id = row["index"]
        sent = self.sentences.loc[cell_id]
        sent = "" if pd.isna(sent) else str(sent)
        return {"image": img_t, "text": sent, "cell_id": cell_id}


# ──────────────────────────────
# CLIP-GO model
# ──────────────────────────────
class CLIPGO(nn.Module):
    def __init__(self, vision_backbone: nn.Module):
        super().__init__()
        self.context_len = CFG.context_len
        self.freeze_text = CFG.freeze_text

        # Vision
        self.vision_encoder = vision_backbone
        vision_dim = vision_backbone.uni2.embed_dim  # 1536
        self.vision_proj  = ProjectionHead(vision_dim, CFG.projection_dim)

        # Text (CoCa)
        self.text_encoder, _, _ = open_clip.create_model_and_transforms(
            CFG.coca_model, pretrained=CFG.coca_pretrain,
            cache_dir=os.path.expanduser("~/.cache/open_clip")
        )
        self.tokenizer = open_clip.get_tokenizer(CFG.coca_model)
        if self.freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

        # Determine text width dynamically
        with torch.no_grad():
            dummy = self.tokenizer(["dummy"], context_length=self.context_len)
            txt_feat = self.text_encoder.encode_text(dummy)
        text_dim = txt_feat.shape[-1]
        self.text_proj = ProjectionHead(text_dim, CFG.projection_dim)

        # Learnable temperature (logit scale)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/0.07)))

    def encode_image(self, imgs):
        feats = self.vision_encoder(imgs)
        return self.vision_proj(feats)

    def encode_text(self, sentences: List[str]):
        tokens = self.tokenizer(sentences, context_length=self.context_len).to(next(self.parameters()).device)
        if self.freeze_text:
            with torch.no_grad():
                feats = self.text_encoder.encode_text(tokens)
        else:
            feats = self.text_encoder.encode_text(tokens)
        return self.text_proj(feats)

    def forward(self, imgs, sentences):
        img_emb = F.normalize(self.encode_image(imgs), dim=-1)
        txt_emb = F.normalize(self.encode_text(sentences), dim=-1)
        scale   = self.logit_scale.exp()
        logits  = scale * img_emb @ txt_emb.T
        targets = torch.arange(img_emb.size(0), device=img_emb.device)
        loss_i = F.cross_entropy(logits, targets)
        loss_t = F.cross_entropy(logits.T, targets)
        return 0.5 * (loss_i + loss_t), logits, logits.T


# ──────────────────────────────
# Helpers: slide/MPP, transforms, cfg dump
# ──────────────────────────────
def get_slide_and_mpp(slide_dir: str):
    tifs = sorted(glob.glob(os.path.join(slide_dir, "**", "*he_image_registered*.ome.tif"), recursive=True))
    if not tifs:
        tifs = sorted(glob.glob(os.path.join(slide_dir, "**", "*.tif"), recursive=True))
        assert tifs, f"No slide tif found under {slide_dir}"
    slide_path = tifs[0]
    slide = openslide.open_slide(slide_path)
    # robust MPP parse
    props = slide.properties
    mpp = None
    for key in ("openslide.mpp-x", "aperio.MPP", "tiff.XResolution"):
        if key in props:
            try:
                mpp = float(props[key]); break
            except Exception:
                pass
    if mpp is None:
        mpp = CFG.target_mpp
    return slide, mpp, slide_path

def build_transforms():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])

def dump_cfg(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump({k: v for k, v in vars(CFG).items() if not k.startswith("__")},
                       f, sort_keys=False)


# ──────────────────────────────
# Train utilities
# ──────────────────────────────
@torch.inference_mode()
def batch_acc(logits):
    target = torch.arange(logits.size(0), device=logits.device)
    pred_i = logits.max(dim=1).indices
    pred_t = logits.max(dim=0).indices
    acc_i = (pred_i == target).float().mean()
    acc_t = (pred_t == target).float().mean()
    return acc_i.item(), acc_t.item()


def train_clipgo(model: CLIPGO, loader: DataLoader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

    # ── precision selection ───────────────────────────────────────────────────
    use_cuda = (device.type == "cuda")
    req = CFG.precision.lower()

    if not use_cuda:
        mode = "fp32"  # CPU → fp32
    elif req == "bf16":
        if torch.cuda.is_bf16_supported():
            mode = "bf16"
        else:
            print("[WARN] bf16 not supported on this GPU → falling back to fp16 AMP")
            mode = "fp16"
    else:
        mode = req  # "fp16" or "fp32"

    if mode == "bf16":
        autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
        scaler = torch.cuda.amp.GradScaler(enabled=False)   # not needed for bf16
    elif mode == "fp16":
        autocast_ctx = torch.cuda.amp.autocast(enabled=True)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:  # "fp32"
        # Strict FP32: disable TF32 to make comparisons fair
        if use_cuda:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        autocast_ctx = nullcontext()
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    print(f"[precision] mode={mode} | device={device}")
    if use_cuda:
        print(f"[cuda] GPU: {torch.cuda.get_device_name(0)}")

    # ── logging setup ─────────────────────────────────────────────────────────
    os.makedirs(CFG.ckpt_dir, exist_ok=True)
    if not os.path.isfile(CFG.log_csv):
        with open(CFG.log_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch","step","loss","acc_i","acc_t","lr","logit_scale","imgs_per_sec"])

    best = math.inf
    for epoch in range(1, CFG.epochs + 1):
        model.train()
        running, ema_loss = 0.0, None
        step_start = time.time()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{CFG.epochs}", dynamic_ncols=True)

        for step, batch in enumerate(pbar, 1):
            imgs  = batch["image"].to(device, non_blocking=True)
            texts = batch["text"]
            B = imgs.size(0)

            opt.zero_grad(set_to_none=True)
            try:
                with autocast_ctx:
                    loss, logits_it, logits_ti = model(imgs, texts)

                if scaler.is_enabled():       # fp16 path
                    scaler.scale(loss).backward()
                    if CFG.grad_clip is not None:
                        scaler.unscale_(opt)  # unscale before clipping
                        nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
                    scaler.step(opt)
                    scaler.update()
                else:                         # fp32 / bf16 path
                    loss.backward()
                    if CFG.grad_clip is not None:
                        nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
                    opt.step()

            except torch.cuda.OutOfMemoryError:
                if use_cuda:
                    torch.cuda.empty_cache()
                print("[WARN] CUDA OOM: skipping batch")
                continue

            running += loss.item()
            ema_loss = loss.item() if ema_loss is None else 0.9*ema_loss + 0.1*loss.item()
            acc_i, acc_t = batch_acc(logits_it)

            # throughput
            dt = max(time.time() - step_start, 1e-6)
            ips = B / dt
            step_start = time.time()

            lr = opt.param_groups[0]["lr"]
            logit_scale = float(model.logit_scale.exp().detach().cpu())
            pbar.set_postfix(loss=f"{ema_loss:.4f}", acc_i=f"{acc_i:.3f}",
                             acc_t=f"{acc_t:.3f}", lr=f"{lr:.1e}",
                             τ=f"{1.0/logit_scale:.3f}", ips=f"{ips:.0f}")

            if step % 50 == 0 or step == len(loader):
                with open(CFG.log_csv, "a", newline="") as f:
                    csv.writer(f).writerow([epoch, step, loss.item(), acc_i, acc_t, lr, logit_scale, ips])

        avg = running / max(len(loader), 1)
        print(f"Epoch {epoch:03d} | loss {avg:.4f} | logit_scale {logit_scale:.3f} (τ≈{1.0/logit_scale:.3f})")

        torch.save({"epoch": epoch, "model": model.state_dict()}, os.path.join(CFG.ckpt_dir, f"epoch_{epoch:03d}.pth"))
        torch.save({"epoch": epoch, "model": model.state_dict()}, os.path.join(CFG.ckpt_dir, "last.pth"))
        if avg < best:
            best = avg
            torch.save({"epoch": epoch, "model": model.state_dict()}, os.path.join(CFG.ckpt_dir, "best.pth"))
            print("✓ new best")

    dump_cfg(CFG.cfg_yaml)
    print("Training complete. Best loss:", best)


# ──────────────────────────────
# Main wiring
# ──────────────────────────────
def main():
    # Resolve dataset-specific paths
    sample  = CFG.xenium_sample_dict[CFG.cancer]
    sample_dir = os.path.join(CFG.root, sample)

    # Load AnnData (expects x_centroid / y_centroid in .obs)
    adata_path = os.path.join(
        sample_dir, "preprocessed", f"fine_tune_{CFG.ground_truth}_v2",
        f"processed_xenium_data_fine_tune_{CFG.ground_truth}_v2_annotated.h5ad",
    )
    adata = sc.read_h5ad(adata_path)
    cell_df = adata.obs.copy()  # index = cell_id

    # Sentences CSV at go_dir/<sample_with_outs_replaced>.csv
    sent_path = f"{CFG.go_dir}/{sample.replace('outs', 'GO.csv')}"
    assert os.path.isfile(sent_path), f"Missing sentences file: {sent_path}"
    sentences = pd.read_csv(sent_path, index_col="cell_id")["go_sentences"].astype(str)
    sentences = sentences.reindex(cell_df.index)
    missing = sentences.isna().sum()
    if missing > 0:
        print(f"[WARN] {missing} cells missing sentences; filling with empty strings.")
        sentences = sentences.fillna("")

    # Slide + scale
    slide, mpp, slide_path = get_slide_and_mpp(sample_dir)
    scale_factor = max(CFG.target_mpp / float(mpp), 1e-6)

    # Vision backbone (UNI2)
    uni2_cfg = {
        'model_name':'vit_giant_patch14_224','img_size':CFG.patch_size,'patch_size':14,'depth':24,
        'num_heads':24,'init_values':1e-5,'embed_dim':CFG.morph_emb_dims,'mlp_ratio':2.66667*2,
        'num_classes':0,'no_embed_class':True,'mlp_layer':timm.layers.SwiGLUPacked,
        'act_layer':torch.nn.SiLU,'reg_tokens':8,'dynamic_img_size':True,
    }
    uni2 = timm.create_model(pretrained=False, **uni2_cfg)
    uni2_weights = os.path.join(CFG.model_dir, "pytorch_model.bin")
    if os.path.isfile(uni2_weights):
        uni2.load_state_dict(torch.load(uni2_weights, map_location="cpu"), strict=True)

    centre_idx = [119, 120, 135, 136] if CFG.level == 0 else [
        102,103,104,105,118,119,120,121,134,135,136,137,150,151,152,153
    ]
    vision_backbone = UNI2Wrapper(uni2, centre_idx)

    # DataLoader
    transform = build_transforms()
    dataset = CellPatchTextDataset(slide, cell_df, sentences, transform,
                                   scale=scale_factor, patch_size=CFG.patch_size)
    loader  = DataLoader(dataset,
                         batch_size=CFG.batch_size,
                         shuffle=True,
                         num_workers=CFG.num_workers,
                         pin_memory=True,
                         persistent_workers=True,
                         prefetch_factor=4,
                         drop_last=True)  # strict B×B matching for contrastive

    # Model + train
    os.makedirs(CFG.ckpt_dir, exist_ok=True)
    model = CLIPGO(vision_backbone)

    # Training uses CFG.precision internally (fp32 | fp16 | bf16)
    train_clipgo(model, loader)


if __name__ == "__main__":
    main()
