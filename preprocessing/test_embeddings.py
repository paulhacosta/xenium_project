#%%
import os
import pandas as pd 
import umap
import matplotlib.pyplot as plt
import scanpy as sc
from datasets import load_from_disk
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm 
import numpy as np 
from matplotlib.colors import ListedColormap

#%%
platform = "visium"  # visium or xenium
cancer = "cervical"

if platform=="xenium":
    xenium_folder_dict = {"lung": "Xenium_Prime_Human_Lung_Cancer_FFPE_outs",
                        "breast":"Xenium_Prime_Breast_Cancer_FFPE_outs",
                        "lymph_node": "Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_outs",
                        "prostate": "Xenium_Prime_Human_Prostate_FFPE_outs",
                        "skin": "Xenium_Prime_Human_Skin_FFPE_outs",
                        "ovarian": "Xenium_Prime_Ovarian_Cancer_FFPE_outs",
                        "cervical": "Xenium_Prime_Cervical_Cancer_FFPE_outs"
                        }

    # for cancer in xenium_folder_dict:
    xenium_folder = xenium_folder_dict[cancer]

    data_path = f"/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/{xenium_folder}"
    preprocessed_file = os.path.join(data_path, f"processed_{platform}_data.h5ad")
    spot_size=30
else:
    xenium_folder = "Human_Lung_Cancer_(FFPE)_lusc_preprocessed"
    data_path = "/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/TEST"
    preprocessed_file = os.path.join(data_path, f"{xenium_folder}.h5ad")
    spot_size=200

embeddings_file = os.path.join(data_path, f"embeddings_output", f"processed_{platform}.csv")
embeddings = pd.read_csv(embeddings_file, index_col="Unnamed: 0")



adata = sc.read_h5ad(preprocessed_file)

#%
# Assuming embeddings are in a numpy array called `embeddings`
# Replace 'embeddings' with the variable containing your embeddings
umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
embeddings_2d = umap_reducer.fit_transform(embeddings)

# Plot the reduced embeddings
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=5, alpha=0.7)
plt.title("UMAP Visualization of Geneformer Embeddings")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.show()


#% Load tokenized dataset
# Path to your .dataset folder
# dataset_path = f"{data_path}/tokenize_output/processed_xenium.dataset"

# Load the dataset
# tokenized_dataset = load_from_disk(dataset_path)

## The dataset contains the following:
# Dataset({
#     features: ['input_ids', 'x_scaled', 'y_scaled', 'total_counts', 'length'],
#     num_rows: 136329
# })

#%
# Determine the optimal number of clusters (optional)
range_n_clusters = [3, 4, 5, 6, 7, 8, 9, 10]
silhouette_avg_scores = []

for n_clusters in tqdm(range_n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    silhouette_avg = silhouette_score(embeddings, cluster_labels)
    silhouette_avg_scores.append(silhouette_avg)

# Plot silhouette scores
plt.figure()
plt.plot(range_n_clusters, silhouette_avg_scores, marker='o')
plt.title("Silhouette Scores for Different Numbers of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Average Silhouette Score")
plt.savefig(f"figures/{xenium_folder}_silhouette_scores.png", dpi=300)
plt.show()

# Choose the optimal number of clusters based on the highest silhouette score
optimal_clusters = range_n_clusters[silhouette_avg_scores.index(max(silhouette_avg_scores))]
print(f"Optimal number of clusters: {optimal_clusters}")

# Perform clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)

# Add cluster labels to the embeddings DataFrame
embeddings['cluster'] = cluster_labels

#%

# Plot UMAP with cluster labels
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_labels, cmap='tab10', s=5)
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.title("UMAP Visualization with Cluster Labels")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.savefig(f"figures/{xenium_folder}_clusters.png", dpi=300)
plt.show()


#%

# Merge cluster labels with spatial coordinates
adata.obs['cluster'] = embeddings['cluster'].values.astype(str)
adata.obs['cluster'] = adata.obs['cluster'].astype('category')
adata.obsm['spatial'] = np.array(adata.obs[['x_centroid', 'y_centroid']])

# Use a subset of 'tab10' colormap with the number of clusters
num_clusters = len(adata.obs['cluster'].unique())  # Now works
cmap = ListedColormap(plt.get_cmap('tab10').colors[:num_clusters])

# Plot spatial distribution with the adjusted colormap
plt.rcParams["figure.figsize"] = [24, 16]  # Width, Height in inches
sc.pl.spatial(
    adata,
    color='cluster',
    spot_size=spot_size,
    cmap=cmap,
    title='Spatial Distribution of Clusters',
    save=f"{xenium_folder}_spatial_cluster.png"  # Save as PNG

)

#%%