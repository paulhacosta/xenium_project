#%%
import os 
import scanpy as sc 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 

#%%
sdata = sc.read_h5ad("/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Visium_HD_Human_Lung_Cancer_post_Xenium_Prime_5k_Experiment2/binned_outputs/square_002um/preprocessed/bin2cell/to_tokenize/corrected_cells_matched_preprocessed_refined_v2.h5ad")
cdata = sc.read_h5ad("/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Visium_HD_Human_Lung_Cancer_post_Xenium_Prime_5k_Experiment2/binned_outputs/square_002um/b2c_cellvit_preprocessed_refined_fullres_v2.h5ad")
xdata = sc.read_h5ad("/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Xenium_Prime_Human_Lung_Cancer_FFPE_outs/preprocessed/fine_tune_refined_v2/processed_xenium_data_fine_tune_refined_v2.h5ad")

# cdata.obs['x_centroid'] = cdata.obs['array_row']
# cdata.obs['y_centroid'] = cdata.obs['array_col']

cell_coords = np.array(cdata.obsm["spatial"])  # Shape: (N, 2)
cdata.obs["x_centroid"] = cell_coords[:,0]
cdata.obs["y_centroid"] = cell_coords[:,1]

# List of AnnData objects
adata_list = {"sdata": sdata, "cdata": cdata, "xdata": xdata}

for name, adata in adata_list.items():
    print(f"Processing {name}...")

    print(adata.shape)
    # Normalize total counts
    # sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
    # Log-transform the data
    # sc.pp.log1p(adata)

    # PCA
    sc.pp.pca(adata, n_comps=50)

    # Build neighbors graph
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)

    # Leiden clustering
    sc.tl.leiden(adata, resolution=0.25, key_added="leiden")

    # UMAP
    sc.tl.umap(adata)

    # UMAP Plot
    sc.pl.umap(adata, color=["leiden"], show=False)

    # Check for x_centroid and y_centroid in adata.obs
    if "x_centroid" in adata.obs and "y_centroid" in adata.obs:
        # Create a color palette for clusters
        unique_clusters = sorted(adata.obs["leiden"].unique())  # Ensure sorted order
        cluster_palette = sns.color_palette("tab20", n_colors=len(unique_clusters))

        cluster_colors = {str(cluster): cluster_palette[i] for i, cluster in enumerate(unique_clusters)}

        # Assign colors to each cell according to its cluster
        adata.obs["cluster_color"] = adata.obs["leiden"].astype(str).map(cluster_colors)

        # Plot spatial distribution
        plt.figure(figsize=(14, 10))
        plt.scatter(
            adata.obs["x_centroid"],
            adata.obs["y_centroid"],
            c=adata.obs["cluster_color"].tolist(),
            alpha=0.6,
            s=5
        )

        plt.xlabel("X Centroid")
        plt.ylabel("Y Centroid")
        plt.title(f"Spatial Distribution of Clusters - {name}")

        # Invert y-axis if necessary
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

# %% Xenium Genes only
sdata = sc.read_h5ad("/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Visium_HD_Human_Lung_Cancer_post_Xenium_Prime_5k_Experiment2/binned_outputs/square_002um/preprocessed/bin2cell/to_tokenize/corrected_cells_matched_preprocessed_refined_v2.h5ad")
cdata = sc.read_h5ad("/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Visium_HD_Human_Lung_Cancer_post_Xenium_Prime_5k_Experiment2/binned_outputs/square_002um/b2c_cellvit_preprocessed_refined_fullres_v2.h5ad")
xdata = sc.read_h5ad("/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Xenium_Prime_Human_Lung_Cancer_FFPE_outs/preprocessed/fine_tune_refined_v2/processed_xenium_data_fine_tune_refined_v2.h5ad")

cdata.obs['x_centroid'] = cdata.obs['array_row']
cdata.obs['y_centroid'] = cdata.obs['array_col']

# List of AnnData objects

# Extract the list of genes in xdata
xdata_genes = xdata.var_names.intersection(sdata.var_names).intersection(cdata.var_names)

# Filter sdata and cdata to keep only genes present in xdata
sdata = sdata[:, xdata_genes].copy()
cdata = cdata[:, xdata_genes].copy()
adata_list = {"sdata": sdata, "cdata": cdata, "xdata": xdata}

# Re-run the clustering pipeline for all datasets
for name, adata in adata_list.items():
    print(f"Processing {name}...")
    print(adata.shape)
    # Normalize total counts
    sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
    # Log-transform the data
    sc.pp.log1p(adata)

    # PCA
    sc.pp.pca(adata, n_comps=50)

    # Build neighbors graph
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)

    # Leiden clustering
    sc.tl.leiden(adata, resolution=0.25, key_added="leiden")

    # UMAP
    sc.tl.umap(adata)

    # UMAP Plot
    sc.pl.umap(adata, color=["leiden"], show=False)

    # Check for x_centroid and y_centroid in adata.obs
    if "x_centroid" in adata.obs and "y_centroid" in adata.obs:
        # Create a color palette for clusters
        unique_clusters = sorted(adata.obs["leiden"].unique())  # Ensure sorted order
        cluster_palette = sns.color_palette("tab20", n_colors=len(unique_clusters))

        cluster_colors = {str(cluster): cluster_palette[i] for i, cluster in enumerate(unique_clusters)}

        # Assign colors to each cell according to its cluster
        adata.obs["cluster_color"] = adata.obs["leiden"].astype(str).map(cluster_colors)

        # Plot spatial distribution
        plt.figure(figsize=(14, 10))
        plt.scatter(
            adata.obs["x_centroid"],
            adata.obs["y_centroid"],
            c=adata.obs["cluster_color"].tolist(),
            alpha=0.6,
            s=5
        )

        plt.xlabel("X Centroid")
        plt.ylabel("Y Centroid")
        plt.title(f"Spatial Distribution of Clusters - {name} | Xenium Genes Only")

        # Invert y-axis if necessary
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

# %% HVG - 2000
sdata = sc.read_h5ad("/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Visium_HD_Human_Lung_Cancer_post_Xenium_Prime_5k_Experiment2/binned_outputs/square_002um/preprocessed/bin2cell/to_tokenize/corrected_cells_matched_preprocessed_refined_v2.h5ad")
cdata = sc.read_h5ad("/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Visium_HD_Human_Lung_Cancer_post_Xenium_Prime_5k_Experiment2/binned_outputs/square_002um/b2c_cellvit_preprocessed_refined_fullres_v2.h5ad")
xdata = sc.read_h5ad("/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Xenium_Prime_Human_Lung_Cancer_FFPE_outs/preprocessed/fine_tune_refined_v2/processed_xenium_data_fine_tune_refined_v2.h5ad")

cdata.obs['x_centroid'] = cdata.obs['array_row']
cdata.obs['y_centroid'] = cdata.obs['array_col']

# List of AnnData objects
adata_list = {"sdata": sdata, "cdata": cdata, "xdata": xdata}

# Step 1: Identify the top 2000 highly variable genes (HVGs) for each dataset
for name, adata in adata_list.items():
    print(f"Selecting HVGs for {name}...")
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3", inplace=True)

    # Filter dataset to keep only HVGs
    adata_list[name] = adata[:, adata.var["highly_variable"]].copy()

# Step 2: Re-run the clustering pipeline for HVG-filtered datasets
for name, adata in adata_list.items():
    print(f"Processing {name} (HVG-filtered)...")
    print(adata.shape)
    # Normalize total counts
    sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
    # Log-transform the data
    sc.pp.log1p(adata)

    # PCA
    sc.pp.pca(adata, n_comps=50)

    # Build neighbors graph
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)

    # Leiden clustering
    sc.tl.leiden(adata, resolution=0.25, key_added="leiden")

    # UMAP
    sc.tl.umap(adata)

    # UMAP Plot
    sc.pl.umap(adata, color=["leiden"], show=False)

    # Check for x_centroid and y_centroid in adata.obs
    if "x_centroid" in adata.obs and "y_centroid" in adata.obs:
        # Create a color palette for clusters
        unique_clusters = sorted(adata.obs["leiden"].unique())  # Ensure sorted order
        cluster_palette = sns.color_palette("tab20", n_colors=len(unique_clusters))

        cluster_colors = {str(cluster): cluster_palette[i] for i, cluster in enumerate(unique_clusters)}

        # Assign colors to each cell according to its cluster
        adata.obs["cluster_color"] = adata.obs["leiden"].astype(str).map(cluster_colors)

        # Plot spatial distribution
        plt.figure(figsize=(14, 10))
        plt.scatter(
            adata.obs["x_centroid"],
            adata.obs["y_centroid"],
            c=adata.obs["cluster_color"].tolist(),
            alpha=0.6,
            s=5
        )

        plt.xlabel("X Centroid")
        plt.ylabel("Y Centroid")
        plt.title(f"Spatial Distribution of Clusters - {name} | HVG-filtered")

        # Invert y-axis if necessary
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
