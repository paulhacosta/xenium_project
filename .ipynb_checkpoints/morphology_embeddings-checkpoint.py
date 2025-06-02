#%% 
import os 
import numpy as np 
import scanpy as sc 
import pandas as pd 
from glob import glob
from tqdm import tqdm
import umap

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist
import geojson

#%%
# Select platform
platform = "visium" # xenium or visium 
ground_truth = "refined"  # refined or cellvit
level = 0

#%% Define Class Limiting Parameters
limit_classes = True  # Set to False to use all classes

if platform == "xenium":
    cancer = "lung"
    xenium_folder_dict = {"lung": "Xenium_Prime_Human_Lung_Cancer_FFPE_outs",
                          "breast":"Xenium_Prime_Breast_Cancer_FFPE_outs",
                          "lymph_node": "Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_outs",
                          "prostate": "Xenium_Prime_Human_Prostate_FFPE_outs",
                          "skin": "Xenium_Prime_Human_Skin_FFPE_outs",
                          "ovarian": "Xenium_Prime_Ovarian_Cancer_FFPE_outs",
                          "cervical": "Xenium_Prime_Cervical_Cancer_FFPE_outs"
                          }

    xenium_folder = xenium_folder_dict[cancer]
    
    data_path = f"/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/{xenium_folder}/preprocessed/fine_tune_{ground_truth}_v2/processed_xenium_data_fine_tune_{ground_truth}_v2.h5ad"
    gene_embedding_file = f"/rsrch9/home/plm/idso_fa1_pathology/TIER2/paul-xenium/embeddings/public_data/{xenium_folder}/processed_xenium_{ground_truth}_v2.csv"
    
    # Load Morphological Embeddings
    morph_embedding_dir = "/rsrch9/home/plm/idso_fa1_pathology/TIER2/paul-xenium/embeddings/public_data/Xenium_Prime_Human_Lung_Cancer_FFPE_outs"

elif platform == "visium":
    data_path = "/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/Visium_HD_Human_Lung_Cancer_post_Xenium_Prime_5k_Experiment2/binned_outputs/square_002um/preprocessed/bin2cell/to_tokenize/corrected_cells_matched_preprocessed_refined_v2.h5ad"

    gene_embedding_file = "/rsrch9/home/plm/idso_fa1_pathology/TIER2/paul-xenium/embeddings/public_data/Visium_HD_Human_Lung_Cancer_post_Xenium_Prime_5k_Experiment2/bin2cell/embeddings_output/processed_visium_hd_bin2cell.csv"
    morph_embedding_dir = f"/rsrch9/home/plm/idso_fa1_pathology/TIER2/paul-xenium/embeddings/public_data/Visium_HD_Human_Lung_Cancer_post_Xenium_Prime_5k_Experiment2"

# Load AnnData
adata = sc.read_h5ad(data_path)
cell_data = adata.obs

# Spatial Information 
spatial_coords = cell_data[['x_centroid', 'y_centroid']].rename(columns={'x_centroid': 'x', 'y_centroid': 'y'})

# Load gene Embeddings 
gene_embeddings = pd.read_csv(gene_embedding_file, index_col="Unnamed: 0")
gene_embeddings.index = cell_data.index

# Load Morphology Embeddings 
morph_embedding_csv = os.path.join(morph_embedding_dir, "UNI2_cell_representation",f"level_{level}","morphology_embeddings_v2.csv")
morph_embeddings = pd.read_csv(morph_embedding_csv, index_col="Unnamed: 0")

if platform == "visium":
    morph_embeddings.index = morph_embeddings.index.astype(str)

#%% Cluster features 

# Choose dimensionality reduction method
use_umap = True  # Set to False to use PCA

# Dim reduction variables
k=5
variable_components = False
pca_components = 50

# Add spatial coordinates to embeddings df 
morph_embeddings["x"] = cell_data['x_centroid']
morph_embeddings["y"] = cell_data['y_centroid']

# Filter Embeddings to remove dropped cells
morph_embeddings = morph_embeddings.dropna(subset=['x', 'y'])

# Standardize embeddings
scaler = StandardScaler()
scaled_embeddings = scaler.fit_transform(morph_embeddings.iloc[:, :-2])  # Exclude 'x', 'y'

# Apply KMeans clustering on raw embeddings
kmeans_raw = KMeans(n_clusters=k, random_state=42, n_init=10)
morph_embeddings['Cluster_raw'] = kmeans_raw.fit_predict(scaled_embeddings)

# Visualize Clusters in Raw Embedding Space 
plt.figure(figsize=(8, 6))
scatter = sns.scatterplot(data=morph_embeddings, x=morph_embeddings.columns[0], y=morph_embeddings.columns[1], 
                          hue='Cluster_raw', palette='Set1', alpha=0.7, s=5)
plt.title('Cluster Visualization in Raw Embedding Space (First Two Features)')
handles, labels = scatter.get_legend_handles_labels()
plt.legend(handles=handles, labels=labels, title='Cluster', markerscale=2)
plt.show()

#  Dimensionality Reduction (PCA or UMAP) 
if use_umap:
    print("Using UMAP for dimensionality reduction...")
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    reduced_embeddings = umap_model.fit_transform(scaled_embeddings)
    morph_embeddings["Dim1"] = reduced_embeddings[:, 0]
    morph_embeddings["Dim2"] = reduced_embeddings[:, 1]
else:
    print("Using PCA for dimensionality reduction...")
    pca = PCA()
    pca.fit(scaled_embeddings)
    explained_variance = np.cumsum(pca.explained_variance_ratio_)

    # Plot explained variance to decide # of components
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='-')
    plt.axhline(y=0.95, color='r', linestyle='--')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.show()

    # Keep enough components to explain 95% variance
    if variable_components:
        n_components = np.argmax(explained_variance >= 0.95) + 1
        print(f"Using {n_components} PCA components to preserve 95% variance.")
    else: 
        n_components = pca_components
        print(f"Using {n_components} PCA components.")

    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(scaled_embeddings)
    morph_embeddings["Dim1"] = reduced_embeddings[:, 0]
    morph_embeddings["Dim2"] = reduced_embeddings[:, 1]

# Cluster in Reduced Space 
kmeans_reduced = KMeans(n_clusters=5, random_state=42, n_init=10)
morph_embeddings['Cluster_reduced'] = kmeans_reduced.fit_predict(reduced_embeddings)

#  Plot Clusters in UMAP/PCA Space Before Spatial Plot 
plt.figure(figsize=(8, 6))
scatter = sns.scatterplot(data=morph_embeddings, x='Dim1', y='Dim2', hue='Cluster_reduced', palette='Set1', alpha=0.7, s=5)
plt.title(f'Cluster Visualization in {"UMAP" if use_umap else "PCA"} Space')
handles, labels = scatter.get_legend_handles_labels()
plt.legend(handles=handles, labels=labels, title='Cluster', markerscale=2)
plt.show()

#  Plot Spatial Distribution of Clusters 
plt.figure(figsize=(8, 6))
scatter = sns.scatterplot(data=morph_embeddings, x='x', y='y', hue='Cluster_reduced', palette='Set1', alpha=0.7, s=1)
plt.title(f'Spatial Cluster Distribution ({"UMAP" if use_umap else "PCA"})')
plt.gca().invert_yaxis()
if platform == "visium":
    plt.gca().invert_xaxis()
handles, labels = scatter.get_legend_handles_labels()
plt.legend(handles=handles, labels=labels, title='Cluster', markerscale=5)
plt.show()

# Optional: Compute Spatial Compactness 
# def cluster_spatial_compactness(df, cluster_col):
#     compactness = []
#     for cluster in df[cluster_col].unique():
#         cluster_points = df[df[cluster_col] == cluster][['x', 'y']]
#         if len(cluster_points) > 1:  # Ensure at least two points
#             compactness.append(np.mean(pdist(cluster_points)))
#     return np.mean(compactness) if compactness else np.nan  # Return NaN if no valid clusters

# compactness_raw = cluster_spatial_compactness(morph_embeddings, 'Cluster_raw')
# compactness_reduced = cluster_spatial_compactness(morph_embeddings, 'Cluster_reduced')

# print(f"Spatial Compactness (Raw Embeddings): {compactness_raw:.3f}")
# print(f"Spatial Compactness (Reduced Embeddings - {'UMAP' if use_umap else 'PCA'}): {compactness_reduced:.3f}")

# Optional: Compute Silhouette Scores 
# silhouette_raw = silhouette_score(scaled_embeddings, kmeans_raw.labels_)
# silhouette_reduced = silhouette_score(reduced_embeddings, kmeans_reduced.labels_)

# print(f"Silhouette Score (Raw Embeddings): {silhouette_raw:.3f}")
# print(f"Silhouette Score (Reduced Embeddings - {'UMAP' if use_umap else 'PCA'}): {silhouette_reduced:.3f}")

# %% Subcluster 


# Select Cluster to Subcluster (e.g., Cluster 0) 
target_cluster = 0  # Change this to any cluster you want to subcluster
subcluster_k = 3  # Set the number of subclusters

# Extract only cells from the target cluster
subcluster_df = morph_embeddings[morph_embeddings['Cluster_reduced'] == target_cluster].copy()

print(f"Subclustering Cluster {target_cluster} with {subcluster_k} subclusters...")
print(f"Selected {len(subcluster_df)} cells for subclustering.")

#  Apply K-Means to Subcluster 
kmeans_sub = KMeans(n_clusters=subcluster_k, random_state=42, n_init=10)
subcluster_df['Subcluster'] = kmeans_sub.fit_predict(subcluster_df[['Dim1', 'Dim2']])  # Use reduced space (UMAP/PCA)

# Merge results back into the main dataframe
morph_embeddings.loc[subcluster_df.index, 'Subcluster'] = subcluster_df['Subcluster']

# Plot Subclusters in UMAP/PCA Space 
plt.figure(figsize=(8, 6))
scatter = sns.scatterplot(data=subcluster_df, x='Dim1', y='Dim2', hue='Subcluster', palette='Set2', alpha=0.7, s=5)
plt.title(f'Subclustering of Cluster {target_cluster} in {"UMAP" if use_umap else "PCA"} Space')
handles, labels = scatter.get_legend_handles_labels()
plt.legend(handles=handles, labels=labels, title='Subcluster', markerscale=2)
plt.show()

#  Plot Subclusters in Spatial Domain 
plt.figure(figsize=(8, 6))
scatter = sns.scatterplot(data=subcluster_df, x='x', y='y', hue='Subcluster', palette='Set2', alpha=0.7, s=1)
plt.title(f'Spatial Subclustering of Cluster {target_cluster}')
plt.gca().invert_yaxis()
handles, labels = scatter.get_legend_handles_labels()
plt.legend(handles=handles, labels=labels, title='Subcluster', markerscale=5)
plt.show()

# %% Export as Geojson 

# Define color mapping for each cluster (extend as needed)
cluster_color_mapping = {
    0: [31, 119, 180],  # Blue
    1: [255, 127, 14],  # Orange
    2: [44, 160, 44],   # Green
    3: [214, 39, 40],   # Red
    4: [148, 103, 189], # Purple
    5: [140, 86, 75],   # Brown
    6: [227, 119, 194], # Pink
    7: [127, 127, 127], # Gray
    8: [188, 189, 34],  # Yellow-green
    9: [23, 190, 207]   # Cyan
}

# Define Function to Export GeoJSON 
def export_geojson(embeddings_df, cluster_col, output_file):
    """
    Convert clusters into a GeoJSON file for QuPath overlay.

    Parameters:
        embeddings_df (pd.DataFrame): DataFrame containing x, y coordinates and clusters.
        cluster_col (str): Column name for cluster assignment (e.g., 'Cluster_reduced' or 'Subcluster').
        output_file (str): Path to save the GeoJSON file.
    """
    geojson_features = []

    for _, row in embeddings_df.iterrows():
        cluster_id = row[cluster_col]

        if pd.isna(cluster_id):  # Skip NaN values
            continue

        cluster_id = int(cluster_id)  # Convert to integer
        color = cluster_color_mapping.get(cluster_id, [0, 0, 0])  # Default to black if missing

        feature = geojson.Feature(
            geometry=geojson.Point((row['x'], row['y'])),
            properties={
                "cluster": cluster_id,
                "color": color  # Color stored as RGB for QuPath compatibility
            }
        )
        geojson_features.append(feature)

    geojson_data = geojson.FeatureCollection(geojson_features)

    with open(output_file, "w") as f:
        geojson.dump(geojson_data, f)

    print(f"GeoJSON file saved: {output_file}")
    return geojson_data


# Define Save Directory 
save_dir = "/rsrch9/home/plm/idso_fa1_pathology/TIER2/paul-xenium/embeddings/public_data/Xenium_Prime_Human_Lung_Cancer_FFPE_outs"
os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

# Generate GeoJSON for Clusters 
cluster_geojson = export_geojson(morph_embeddings, "Cluster_reduced", os.path.join(save_dir, "morph_clusters_k10.geojson"))

#  Generate GeoJSON for Subclusters (if available) 
if "Subcluster" in morph_embeddings.columns:
    subcluster_geojson = export_geojson(morph_embeddings.dropna(subset=["Subcluster"]), "Subcluster", os.path.join(save_dir, "morph_subclusters_k3.geojson"))

# %%
