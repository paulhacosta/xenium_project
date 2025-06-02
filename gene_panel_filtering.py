
#%%
import os
import pandas as pd 
import scanpy as sc 
import matplotlib.pyplot as plt
import seaborn as sns
#%%
prime_5k_metadata = "/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/XeniumPrimeHuman5Kpan_tissue_pathways_metadata.csv"
metadata_df = pd.read_csv(prime_5k_metadata)

# Define keywords for filtering
keywords = [
    'fibroblast', 'stromal cell', 'mesenchymal cell', 'connective tissue cell', 'fibroblast-like', 'ECM', 'collagen', 'myofibroblast', 'muscle',
    'lymphocyte', 'T cell', 'B cell', 'NK cell', 'leukocyte', 'immune cell', 'CD4', 'CD8', 'plasma cell', 'lymphoid', 'cytotoxic', 'helper', 'regulatory', 'macrophage',
    'tumor', 'cancer cell', 'malignant cell', 'neoplasm', 'neoplastic', 'carcinoma', 'sarcoma', 'adenoma', 'oncogenic', 'transformed cell', 'metastasis'
]

# Additional marker genes identified by Luisa
marker_genes = [
    'PTPRC', 'ACTA2', 'KRT', 'CD163', 'CD19', 'CD34', 'CD3E', 'CD3G', 'CD4', 'CD5', 'CD68', 'CD79A', 'CD8A', 'CDH1', 'CDX2', 'COL4A1', 'COL4A2', 
    'COL4A4', 'COL5A1', 'COL5A2', 'CTLA4', 'CXCL13', 'EPCAM', 'GZMA', 'GZMB', 'GZMH', 'GZMK', 'ICOS', 'IDO1', 'IFNG', 'IL10', 'IL6', 'KRT20', 'LAG3', 
    'MARCO', 'MKI67', 'MPO', 'MS4A1', 'MSLN', 'MUC1', 'MUC16', 'MUC4', 'MUC5AC', 'MUC5B', 'NCAM1', 'PAX5', 'PDCD1', 'PECAM1', 'SOX2'
]

# Filter genes based on expanded keyword list and additional marker genes
filtered_metadata_expanded_df = metadata_df[
    metadata_df['cell_type'].str.contains('|'.join(keywords), case=False, na=False) |
    metadata_df['gene_name'].isin(marker_genes)
]

# Identify excluded genes
excluded_genes_df = metadata_df[~metadata_df['gene_name'].isin(filtered_metadata_expanded_df['gene_name'])]

# Save the filtered and excluded gene lists
filtered_metadata_expanded_df.to_csv(os.path.join(os.path.dirname(prime_5k_metadata), "xenium5k_filtered_panel-tum-lym-fibro_updated.csv"), index=False)
excluded_genes_df.to_csv(os.path.join(os.path.dirname(prime_5k_metadata), "xenium5k_filtered_panel-other_updated.csv"), index=False)


#%%
# Load Xenium AnnData object
adata_root = "/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Xenium_Prime_Human_Lung_Cancer_FFPE_outs/preprocessed/fine_tune_refined_v2"
adata_file = "processed_xenium_data_fine_tune_refined_v2.h5ad"

adata = sc.read_h5ad(os.path.join(adata_root, adata_file))

# Inspect cell classifications from H&E segmentation
print(adata.obs["class"].value_counts())

# Perform clustering on the ST data
sc.pp.pca(adata, n_comps=50)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)
sc.tl.leiden(adata, resolution=0.5)
sc.tl.umap(adata)
sc.pl.umap(adata, color=["leiden", "class"])

# Create a table of cluster vs. H&E cell classifications
cluster_vs_class = pd.crosstab(adata.obs["leiden"], adata.obs["class"])
print(cluster_vs_class)

# Systematically assign clusters to cell types
cluster_assignments = cluster_vs_class.idxmax(axis=1)  # Assign each cluster to the most common cell type
adata.obs["assigned_class"] = adata.obs["leiden"].map(cluster_assignments)

# Identify marker genes for each cluster
sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon")
sc.pl.rank_genes_groups(adata, n_genes=10, sharey=False)

# Extract marker genes based on the systematic cluster assignments
tumor_clusters = cluster_assignments[cluster_assignments == "t"].index.tolist()
fibroblast_clusters = cluster_assignments[cluster_assignments == "f"].index.tolist()
lymphocyte_clusters = cluster_assignments[cluster_assignments == "l"].index.tolist()

# Extract marker genes for each mapped group
tumor_markers = {cluster: adata.uns["rank_genes_groups"]["names"][cluster] for cluster in tumor_clusters}
fibroblast_markers = {cluster: adata.uns["rank_genes_groups"]["names"][cluster] for cluster in fibroblast_clusters}
lymphocyte_markers = {cluster: adata.uns["rank_genes_groups"]["names"][cluster] for cluster in lymphocyte_clusters}

# Convert to DataFrames and save marker genes to CSV
# pd.DataFrame.from_dict(tumor_markers, orient='index').T.to_csv("tumor_markers.csv", index=False)
# pd.DataFrame.from_dict(fibroblast_markers, orient='index').T.to_csv("fibroblast_markers.csv", index=False)
# pd.DataFrame.from_dict(lymphocyte_markers, orient='index').T.to_csv("lymphocyte_markers.csv", index=False)

#%%
# Plot spatial distribution of classes
# Define class color mapping
class_colors = {"l": "green", "t": "red", "f": "blue", "o": "black"}
adata.obs["class_color"] = adata.obs["class"].map(class_colors)

#  Define unique colors for each cluster
unique_clusters = adata.obs["leiden"].unique().tolist()
cluster_palette = sns.color_palette("bright", n_colors=len(unique_clusters))
cluster_colors = {str(cluster): cluster_palette[i] for i, cluster in enumerate(unique_clusters)}
# Map clusters to colors
adata.obs["cluster_color"] = adata.obs["leiden"].astype(str).map(cluster_colors)


# Plot spatial distribution of classes with legend outside
plt.figure(figsize=(14.481632215915177, 10))
scatter = plt.scatter(adata.obs["x_centroid"], adata.obs["y_centroid"], c=adata.obs["class_color"], alpha=0.6, s=5)
plt.xlabel("X Centroid")
plt.ylabel("Y Centroid")
plt.title("Spatial Distribution of Cell Classes")
plt.gca().invert_yaxis()

# Add legend for class colors
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=cls) for cls, color in class_colors.items()]
plt.legend(handles=legend_elements, title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.axis("off")

plt.show()

# Plot spatial distribution of clusters with legend outside
plt.figure(figsize=(14.481632215915177, 10))
scatter = plt.scatter(adata.obs["x_centroid"], adata.obs["y_centroid"], c=adata.obs["cluster_color"].tolist(), alpha=0.6, s=5)
plt.xlabel("X Centroid")
plt.ylabel("Y Centroid")
plt.title("Spatial Distribution of Clusters")
plt.gca().invert_yaxis()

# Add legend for cluster colors
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=f"Cluster {cluster}") for cluster, color in cluster_colors.items()]
plt.legend(handles=legend_elements, title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.axis("off")
plt.show()


# %% Filter and save genes based on top 200 differential expression
num_genes = 200
tumor_clusters = cluster_assignments[cluster_assignments == "t"].index.tolist()
fibroblast_clusters = cluster_assignments[cluster_assignments == "f"].index.tolist()
lymphocyte_clusters = cluster_assignments[cluster_assignments == "l"].index.tolist()

# Ensure unique genes across all clusters before comparing
tumor_genes = set(g for cluster in tumor_clusters for g in adata.uns["rank_genes_groups"]["names"][cluster][:num_genes])
fibroblast_genes = set(g for cluster in fibroblast_clusters for g in adata.uns["rank_genes_groups"]["names"][cluster][:num_genes])
lymphocyte_genes = set(g for cluster in lymphocyte_clusters for g in adata.uns["rank_genes_groups"]["names"][cluster][:num_genes])

# Extract gene names from metadata filtering
filtered_genes = set(filtered_metadata_expanded_df["gene_name"])

# Compare overlaps
tumor_overlap = filtered_genes.intersection(tumor_genes)
fibroblast_overlap = filtered_genes.intersection(fibroblast_genes)
lymphocyte_overlap = filtered_genes.intersection(lymphocyte_genes)

# Unique genes in each set
tumor_unique = tumor_genes - filtered_genes
fibroblast_unique = fibroblast_genes - filtered_genes
lymphocyte_unique = lymphocyte_genes - filtered_genes

# Compute genes that exist in metadata but were NOT found in clustering
metadata_unique = filtered_genes - (tumor_genes | fibroblast_genes | lymphocyte_genes)

print(f"Number of Genes Unique to Metadata: {len(metadata_unique)}")


# Print refined comparison results
print(f"Refined Tumor Genes Overlapping with Metadata Filter: {len(tumor_overlap)}")
print(f"Refined Fibroblast Genes Overlapping with Metadata Filter: {len(fibroblast_overlap)}")
print(f"Refined Lymphocyte Genes Overlapping with Metadata Filter: {len(lymphocyte_overlap)}")

print(f"Refined Unique Tumor Marker Genes: {len(tumor_unique)}")
print(f"Refined Unique Fibroblast Marker Genes: {len(fibroblast_unique)}")
print(f"Refined Unique Lymphocyte Marker Genes: {len(lymphocyte_unique)}")

# Convert to DataFrame using metadata information
selected_genes = set(tumor_genes) | set(fibroblast_genes) | set(lymphocyte_genes)
genes_df = metadata_df[metadata_df['gene_name'].isin(selected_genes)].copy()

# Add a column for the assigned cell type
genes_df["assigned_cell_type"] = genes_df["gene_name"].apply(lambda x: 
    "Tumor" if x in tumor_genes else 
    "Fibroblast" if x in fibroblast_genes else 
    "Lymphocyte" if x in lymphocyte_genes else "Unknown")

# Save the final dataframe
genes_df.to_csv(os.path.join(os.path.dirname(prime_5k_metadata), "xenium5k_filtered_panel-deg_lung.csv"), index=False)

# Check output
print(genes_df.head())
# %% Filter adata and save as new AnnData files 

# Filter adata by clustering-derived genes and metadata-filtered genes
adata_clustering_filtered = adata[:, list(selected_genes)].copy()
adata_metadata_filtered = adata[:, list(filtered_genes)].copy()

# Save filtered datasets
# Ensure all categorical/object columns are stored as strings before saving
for df in [adata_clustering_filtered.obs, adata_clustering_filtered.var, 
           adata_metadata_filtered.obs, adata_metadata_filtered.var]:
    for col in df.select_dtypes(include=['category', 'object']).columns:
        df[col] = df[col].astype(str)

# Save filtered datasets
adata_clustering_filtered.write_h5ad(os.path.join(adata_root, adata_file.replace("v2.h5ad", "clustering_filtered_v2.h5ad")))
# adata_metadata_filtered.write_h5ad(os.path.join(adata_root.replace("v2", "metadata_filtered_v2"), adata_file.replace("v2.h5ad", "metadata_filtered_v2.h5ad")))

print("Filtered AnnData files saved.")

# %%
