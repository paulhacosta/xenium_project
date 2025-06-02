#%%
import os
import numpy as np 
import pandas as pd 
import scanpy as sc 
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle 

from matplotlib.colors import ListedColormap

from matplotlib.lines import Line2D

import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import seaborn as sns
from IPython.display import display  # for pretty DataFrame output

from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

#%%
cancer = "lung"
xenium_folder_dict = {
    "lung": "Xenium_Prime_Human_Lung_Cancer_FFPE_outs",
    "breast": "Xenium_Prime_Breast_Cancer_FFPE_outs",
    "lymph_node": "Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_outs",
    "prostate": "Xenium_Prime_Human_Prostate_FFPE_outs",
    "skin": "Xenium_Prime_Human_Skin_FFPE_outs",
    "ovarian": "Xenium_Prime_Ovarian_Cancer_FFPE_outs",
    "cervical": "Xenium_Prime_Cervical_Cancer_FFPE_outs"
}

xenium_folder = xenium_folder_dict[cancer]
enrich_dir = f"/rsrch9/home/plm/idso_fa1_pathology/TIER2/paul-xenium/enrichment/public_data/{xenium_folder}"
label_data_path = f"/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/{xenium_folder}/preprocessed/fine_tune_refined_v2/processed_xenium_data_fine_tune_refined_v2_annotated.h5ad"

gene_emb = sc.read_h5ad(os.path.join(enrich_dir, "gene_emb_adata.h5ad"))
gene_proj = sc.read_h5ad(os.path.join(enrich_dir,"gene_proj_adata.h5ad"))
morph_emb = sc.read_h5ad(os.path.join(enrich_dir,"morph_emb_adata.h5ad"))
morph_proj = sc.read_h5ad(os.path.join(enrich_dir,"morph_proj_adata.h5ad"))

label_adata = sc.read_h5ad(label_data_path)
gene_emb.obs["label_singleR"] = label_adata.obs["singleR_class"]
gene_proj.obs["label_singleR"] = label_adata.obs["singleR_class"]
morph_emb.obs["label_singleR"] = label_adata.obs["singleR_class"]
morph_proj.obs["label_singleR"] = label_adata.obs["singleR_class"]

spatial_coords = label_adata.obs[['x_centroid', 'y_centroid']].rename(columns={'x_centroid': 'x', 'y_centroid': 'y'})

#%%
def plot_cluster_label_dotplot(
    adata,
    cluster_key="leiden",
    label_key="label_singleR",
    target_classes=None,
    figsize=(10, 6),
    cmap="tab20",
    title="Cluster vs Cell Type Dot Plot",
    cluster_color_map=None,
):
    if target_classes is not None:
        adata = adata[adata.obs[label_key].isin(target_classes)].copy()

    count_matrix = pd.crosstab(adata.obs[label_key], adata.obs[cluster_key])
    max_val = count_matrix.values.max()
    clusters = count_matrix.columns.astype(str)

    if cluster_color_map is None:
        colormap = plt.get_cmap(cmap, len(clusters))
        cluster_color_map = {cl: colormap(i) for i, cl in enumerate(clusters)}

    fig, ax = plt.subplots(figsize=figsize)
    for i, label in enumerate(count_matrix.index):
        for j, cluster in enumerate(clusters):
            value = count_matrix.loc[label, cluster]
            size = (value / max_val) * 500
            ax.scatter(j, i, s=size, color=cluster_color_map[cluster], alpha=0.8, edgecolors='black')

    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels(clusters, rotation=45)
    ax.set_yticks(range(len(count_matrix.index)))
    ax.set_yticklabels(count_matrix.index)
    ax.set_xlabel("Leiden Clusters")
    ax.set_ylabel("Cell Type Labels")
    ax.set_title(title)

    example_sizes = [100, 300, 500]
    size_labels = ["", "", "More Cells"]
    size_handles = [
        Line2D([], [], marker='o', linestyle='None',
               markersize=np.sqrt(s), color='gray', alpha=0.5, markeredgecolor='black',
               label=lab)
        for s, lab in zip(example_sizes, size_labels)
    ]
    ax.legend(handles=size_handles, title="Legend", frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left', labelspacing=1.5)

    plt.tight_layout()
    plt.show()



def plot_spatial_clusters(
    adata,
    spatial_coords,
    cluster_key="leiden",
    cmap="tab20",
    s=1,
    figsize=(8, 6),
    title="Spatial Visualization of Clusters",
    cluster_color_map=None,
):
    coords = spatial_coords.loc[adata.obs_names]
    adata.obs['x'] = coords['x']
    adata.obs['y'] = coords['y']

    if cluster_color_map is None:
        clusters = sorted(adata.obs[cluster_key].astype(int).unique())
        colormap = plt.get_cmap(cmap, len(clusters))
        cluster_color_map = {str(cl): colormap(i) for i, cl in enumerate(clusters)}

    cluster_colors = adata.obs[cluster_key].astype(str).map(cluster_color_map)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        adata.obs['x'],
        adata.obs['y'],
        c=cluster_colors,
        s=s,
        alpha=0.8,
        edgecolors='none'
    )
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title(title)
    plt.tight_layout()

    plt.show()

#%%
target_singleR = ['CD4+ T-cells',
                'CD8+ T-cells',
                'B-cells',
                'Epithelial cells',
                'Macrophages',
                'Fibroblasts',
                'Endothelial cells',
                "other"
                ]

target_aistil = [
    "t",         # tumor
    "f",         # fibroblast
    "l",         # lymphocyte
    "o",         # other
]

label_dict = {"label_singleR": target_singleR, "label_aistil":target_aistil}

target_singleR = ['CD4+ T-cells',
                'CD8+ T-cells',
                'B-cells',
                'Epithelial cells',
                'Macrophages',
                'Fibroblasts',
                'Endothelial cells',
                "other"
                ]

target_aistil = [
    "t",         # tumor
    "f",         # fibroblast
    "l",         # lymphocyte
    "o",         # other
]

label_dict = {"label_singleR": target_singleR, "label_aistil":target_aistil}

# %%
label_name = "label_aistil"

umap_legend = "right margin"
fig = sc.pl.umap(
    gene_emb,
    color=['leiden', label_name],
    size=20,
    legend_loc=umap_legend,
    frameon=False,
    show=True,  # <-- important for saving
    return_fig=False
)

leiden_palette = gene_emb.uns["leiden_colors"]
leiden_ids = sorted(gene_emb.obs["leiden"].astype(int).unique())
cluster_color_map = {str(i): c for i, c in zip(leiden_ids, leiden_palette)}




#%%

def cluster_enrichment(adata, label_key, cluster_key="leiden",
                       fdr=0.05, min_cells=10):
    """
    Fisher-exact enrichment of each label in every cluster.

    Returns a tidy DataFrame:
        cluster | cell_type |  n_in |  n_out |  exp_in |  FE |  pval |  qval
    """

    # Contingency table (cell_type × cluster)
    ct = pd.crosstab(adata.obs[label_key], adata.obs[cluster_key])

    results = []
    for cell_type, row in ct.iterrows():
        total_type = row.sum()           # all cells of this type
        total_other = ct.values.sum() - total_type

        for cluster, n_in in row.items():
            # Skip very small clusters (optional)
            if n_in < min_cells:
                continue

            n_cluster = ct[cluster].sum()          # cluster size
            n_out = total_type - n_in              # same type, outside cluster
            n_not_in = n_cluster - n_in            # other types in cluster
            n_other_out = total_other - n_not_in   # other types outside

            table = [[n_in, n_out],
                     [n_not_in, n_other_out]]

            _, p = fisher_exact(table, alternative="greater")
            expected_in = total_type * n_cluster / ct.values.sum()
            fe = (n_in / expected_in) if expected_in else np.nan

            results.append([cluster, cell_type, n_in, n_out,
                            expected_in, fe, p])

    res_df = pd.DataFrame(results, columns=[
        "cluster", "cell_type", "n_in", "n_out",
        "exp_in", "fold_enrichment", "pval"
    ])
    if len(res_df):
        res_df["qval"] = multipletests(res_df.pval, method="fdr_bh")[1]
    return res_df.sort_values(["cluster", "qval"])



# %%
emb_dict = {
    "gene_proj":  gene_proj,
    "morph_proj": morph_proj,
    "gene_emb":   gene_emb,
    "morph_emb":  morph_emb,
}

label_keys = ["label_singleR", "label_aistil"]

enrichment_tables = {}
for feat_name, ad in emb_dict.items():
    for lbl in label_keys:
        key = f"{feat_name} | {lbl}"
        enrich_df = cluster_enrichment(ad, label_key=lbl)
        enrichment_tables[key] = enrich_df
        print(f"\n=== {key} ===")
        display(enrich_df.query("qval < 0.05")
                         .head(10))          # show top hits
#%%

# %%

def enrichment_heatmap(enrich_df, vmin=0, vmax=5, figsize=(12,4)):
    """
    enrich_df: output of cluster_enrichment()
               must contain 'cluster', 'cell_type', 'qval'
    """
    # pivot to matrix: rows=cell_type, cols=cluster, values=-log10(q)
    mat = (enrich_df
           .assign(logq = -np.log10(enrich_df['qval'].replace(0, 1e-300)))
           .pivot(index='cell_type', columns='cluster', values='logq')
           .fillna(0))
    
    plt.figure(figsize=figsize)
    sns.heatmap(mat, cmap='viridis', vmin=vmin, vmax=vmax,
                cbar_kws={"label": "-log₁₀(q-value)"})
    plt.xlabel("Leiden cluster")
    plt.ylabel("Cell type")
    plt.title("Enrichment heat-map")
    plt.tight_layout()
    plt.show()

# example usage
enrichment_heatmap(enrichment_tables["gene_proj | label_singleR"])

plot_cluster_label_dotplot(gene_proj,
                           cluster_key="leiden",
                           label_key=label_name,
                           target_classes=label_dict[label_name],
                           title=f"gene_proj | {label_name}", 
                           cluster_color_map=cluster_color_map)

# %%
def summary_for_label(enrich_df, cell_type, q_thresh=0.05):
    df = enrich_df.query("cell_type == @cell_type")
    sig  = df.query("qval <= @q_thresh")
    best = df.loc[df['fold_enrichment'].idxmax()]
    return {
        "n_sig_clusters": len(sig),
        "best_cluster":   best.cluster,
        "best_FE":        best.fold_enrichment,
        "best_q":         best.qval,
    }

gene_sets   = {"gene_emb": gene_emb,   "gene_proj": gene_proj}
morph_sets  = {"morph_emb": morph_emb, "morph_proj": morph_proj}

# 3-A  ─ tumour enrichment (label 't' in label_aistil)
tumour_stats = {}
for name, ad in gene_sets.items():
    e = cluster_enrichment(ad, label_key="label_aistil")      # from earlier
    tumour_stats[name] = summary_for_label(e, cell_type="t")

# 3-B  ─ TME enrichment (fibro, T-cell, macrophage, B-cell)
tme_labels = ["Fibroblasts", "CD4+ T-cells", "CD8+ T-cells",
              "Macrophages", "B-cells"]
tme_stats = {}
for name, ad in morph_sets.items():
    e = cluster_enrichment(ad, label_key="label_singleR")
    s = [summary_for_label(e, ct) for ct in tme_labels]
    tme_stats[name] = {
        "total_sig_clusters": sum(r["n_sig_clusters"] for r in s),
        "median_best_FE":     np.median([r["best_FE"] for r in s]),
    }

print("Tumour (label_aistil - 't')\n", pd.DataFrame(tumour_stats).T)
print("\nTME cells (label_singleR)\n",   pd.DataFrame(tme_stats).T)

# show tumour sub-clusters
plot_cluster_label_dotplot(gene_proj, "leiden", "label_aistil",
                           target_classes=["t"], title="gene_proj – tumour")
plot_spatial_clusters(gene_proj, spatial_coords, "leiden",
                      title="gene_proj – tumour clusters")

# heat-map for TME labels in morph_proj
e_morph_proj = cluster_enrichment(morph_proj, "label_singleR")
tme_heatmap  = e_morph_proj.query("cell_type in @tme_labels")
enrichment_heatmap(tme_heatmap, vmax=5)   # helper from earlier

# %%
def cluster_purity(adata, label_key, cluster_key="leiden"):
    # majority label fraction for each cluster
    ctab = pd.crosstab(adata.obs[cluster_key], adata.obs[label_key])
    majority = ctab.max(axis=1)
    purity = (majority / ctab.sum(axis=1)).rename("purity")
    return purity

# Tumour example
p_emb  = cluster_purity(morph_emb,  label_key="label_aistil")
p_proj = cluster_purity(morph_proj, label_key="label_aistil")

print("Median purity  (embeddings) :", p_emb.median())
print("Median purity  (projections):", p_proj.median())


# ------------------------------------------------------------------
def cluster_purity(adata, label_key, cluster_key="leiden"):
    """Return a Series indexed by cluster with majority-label purity."""
    ctab      = pd.crosstab(adata.obs[cluster_key], adata.obs[label_key])
    majority  = ctab.max(axis=1)           # highest count per cluster
    purity    = (majority / ctab.sum(axis=1)).rename("purity")
    purity.index = purity.index.astype(str)
    return purity

# ------------------------------------------------------------------
def purity_violinplot(morph_emb, morph_proj,
                      label_key="label_singleR", cluster_key="leiden",
                      cmap=("steelblue", "firebrick"), figsize=(6,4)):
    """Violin + box comparing purity distributions."""
    
    # 1 -- compute purity for each feature set
    pur_emb  = cluster_purity(morph_emb,  label_key, cluster_key)
    pur_proj = cluster_purity(morph_proj, label_key, cluster_key)
    
    df = pd.concat([
            pd.DataFrame({"purity": pur_emb,  "feature_set": "morph_emb"}),
            pd.DataFrame({"purity": pur_proj, "feature_set": "morph_proj"})
         ], ignore_index=True)

    # 2 -- violin plot
    plt.figure(figsize=figsize)
    ax = sns.violinplot(x="feature_set", y="purity",
                        data=df, palette=cmap, inner=None, cut=0)
    sns.boxplot(x="feature_set", y="purity",
                data=df, width=0.2, palette=cmap, showcaps=False,
                boxprops={'zorder': 2}, medianprops={'color':'black'})
    ax.set_ylabel("Cluster purity (majority-label fraction)")
    ax.set_xlabel("")
    ax.set_title("Morphology: purity distribution per cluster")
    plt.ylim(0,1.05)
    plt.tight_layout()
    plt.show()
    
    return df  # handy if you want to inspect numbers programmatically

# ------------------------------------------------------------------
# compute the two purity Series (index = cluster id)
pur_emb  = cluster_purity(morph_emb,  "label_singleR")   # Series
pur_proj = cluster_purity(morph_proj, "label_singleR")

# 1 ▸ find clusters present in both
common = pur_emb.index.intersection(pur_proj.index)

# 2 ▸ run Wilcoxon on matched values
from scipy.stats import wilcoxon
stat, p = wilcoxon(
    pur_emb.loc[common],
    pur_proj.loc[common],
    alternative="less"    # projections expected to be higher
)

print(f"Wilcoxon (emb < proj): p = {p:.3e}")



# %%
