#%%
import os 
import scanpy as sc 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 

#%%
adataWH = sc.read_h5ad("/rsrch9/home/plm/idso_fa1_pathology/TIER2/paul-xenium/embeddings/public_data/Xenium_Prime_Human_Lung_Cancer_FFPE_outs/scGPT_WH.h5ad")
adataCP = sc.read_h5ad("/rsrch9/home/plm/idso_fa1_pathology/TIER2/paul-xenium/embeddings/public_data/Xenium_Prime_Human_Lung_Cancer_FFPE_outs/scGPT_CP.h5ad")
adataB2C_WH = sc.read_h5ad("/rsrch9/home/plm/idso_fa1_pathology/TIER2/paul-xenium/embeddings/public_data/Visium_HD_Human_Lung_Cancer_post_Xenium_Prime_5k_Experiment2/b2c_scGPT_WH.h5ad")
adataB2C_CP = sc.read_h5ad("/rsrch9/home/plm/idso_fa1_pathology/TIER2/paul-xenium/embeddings/public_data/Visium_HD_Human_Lung_Cancer_post_Xenium_Prime_5k_Experiment2/b2c_scGPT_CP.h5ad")

# %%

# List of AnnData objects
# adata_list = {"adataWH": adataWH, "adataCP": adataCP, "adataB2C_WH": adataB2C_WH, "adataB2C_CP": adataB2C_CP}
adata_list = {"adataB2C_CP": adataB2C_CP}

# Step 1: Clustering using scGPT embeddings
for name, adata in adata_list.items():
    print(f"Processing {name} using scGPT embeddings...")

    # Perform PCA directly on scGPT embeddings
    sc.pp.pca(adata, n_comps=50, use_highly_variable=False)

    # Build neighbors graph using scGPT embeddings
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30, use_rep="X_scGPT")

    # Leiden clustering
    sc.tl.leiden(adata, resolution=0.25, key_added="leiden")

    # UMAP for dimensionality reduction
    sc.tl.umap(adata)

    # UMAP Plot
    sc.pl.umap(adata, color=["leiden"], show=False)

    # Step 2: Spatial Mapping of Clusters

    # Create a color palette for clusters
    unique_clusters = sorted(adata.obs["leiden"].unique())  # Ensure sorted order
    cluster_palette = sns.color_palette("tab20", n_colors=len(unique_clusters))

    cluster_colors = {str(cluster): cluster_palette[i] for i, cluster in enumerate(unique_clusters)}

    # Assign colors to each cell according to its cluster
    adata.obs["cluster_color"] = adata.obs["leiden"].astype(str).map(cluster_colors)

    # Plot spatial distribution
    plt.figure(figsize=(14, 10))
    if 'B2C' in name:
        cell_coords = np.array(adataB2C_CP.obsm["spatial"])  # Shape: (N, 2)
        adataB2C_CP.obs["y_centroid"] = cell_coords[:,0]
        adataB2C_CP.obs["x_centroid"] = cell_coords[:,1]


    plt.scatter(
        adata.obs["x_centroid"],
        adata.obs["y_centroid"],
        c=adata.obs["cluster_color"].tolist(),
        alpha=0.6,
        s=5
    )

    plt.xlabel("X Centroid")
    plt.ylabel("Y Centroid")
    plt.title(f"Spatial Distribution of Clusters - {name} (scGPT)")

    # Invert y-axis if necessary
    if 'B2C' not in name:
        plt.gca().invert_yaxis()

    # Construct legend for clusters
    legend_elements = [
        plt.Line2D(
            [0], [0],
            marker='o',
            color='w',
            markerfacecolor=color,
            markersize=8,
            label=f"Cluster {cluster}"
        )
        for cluster, color in cluster_colors.items()
    ]
    plt.legend(
        handles=legend_elements,
        title="Cluster",
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )

    plt.axis("off")
    plt.show()


# %%
