#%%
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# %%


def cluster_embedding(
    file_path,
    cluster_list=[5, 10],
    standardize=True,
    random_state=42
):
    """
    Loads CSV embedding data, performs K-means for each k in cluster_list,
    computes silhouette scores, and returns a dict of DataFrames with cluster labels.
    
    Parameters:
    - file_path (str): Path to CSV file containing embeddings.
    - cluster_list (list): List of k values for KMeans.
    - standardize (bool): Whether to apply standard scaling.
    - random_state (int): Random seed for reproducibility.
    """
    
    print(f"\n--- Clustering file: {file_path} ---")
    
    # Load the embeddings from CSV
    data = pd.read_csv(file_path, header=0, index_col=None)
    
    # Optional standardization
    if standardize:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        data_for_clustering = data_scaled
    else:
        data_for_clustering = data.values  # (rows x features)
    
    results = {}  # to store results for each k
    
    for k in cluster_list:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        labels = kmeans.fit_predict(data_for_clustering)
        
        # Silhouette Score (higher is better, up to 1.0)
        sil_score = silhouette_score(data_for_clustering, labels)
        print(f"K={k} -> Silhouette Score: {sil_score:.4f}")
        
        # Store the cluster labels
        cluster_df = pd.DataFrame(index=data.index)
        cluster_df[f"Cluster_k{k}"] = labels
        
        results[k] = cluster_df
    
    return results

#%%
experiment = "visium_hd_8um_bins"
embeddings_dir = "/rsrch9/home/plm/idso_fa1_pathology/TIER2/paul-xenium/embeddings/public_data"
visium_folder = f"{embeddings_dir}/Visium_HD_Human_Lung_Cancer_post_Xenium_Prime_5k_Experiment2"
xenium_folder = f"{embeddings_dir}/Xenium_Prime_Human_Lung_Cancer_FFPE_outs"


embedding_file_dict = {"xenium_5k": f"{xenium_folder}/processed_xenium.csv",
                      "xenium_5k_paired": f"{xenium_folder}/processed_xenium_refined_v2.csv",
                      "visium_hd_8um_bins": f"{visium_folder}/008um/embeddings_output/processed_visium_hd_008um.csv",
                      "visium_hd_bin2cell": f"{visium_folder}/bin2cell/embeddings_output/processed_visium_hd_bin2cell.csv",
                      }


embeddings_file = embedding_file_dict[experiment]
embeddings = pd.read_csv(embeddings_file)


# %%
file_paths = list(embedding_file_dict.values())
k_values = [5, 10]
all_results = {}  # Dictionary to hold results for each file

for fp in file_paths:
    if os.path.exists(fp):
        # Perform clustering
        results_dict = cluster_embedding(
            file_path=fp,
            cluster_list=k_values,
            standardize=True,
            random_state=42
        )
        
        # Save cluster labels to CSV
        for k, df_clusters in results_dict.items():
            out_path = f"{os.path.splitext(fp)[0]}_clusters_k{k}.csv"
            df_clusters.to_csv(out_path, index=True)
            print(f"Saved cluster labels to: {out_path}")
        
        # Store in memory
        all_results[fp] = results_dict
    else:
        print(f"File not found: {fp}")


#%%

cluster_list = [5]

for experiment_name, embedding_fp in embedding_file_dict.items():
    print(f"\n--- Plotting PCA for: {experiment_name} ---")
    
    if not os.path.exists(embedding_fp):
        print(f"Embedding file not found: {embedding_fp}")
        continue
    
    # Step A: Load the original embedding
    data = pd.read_csv(embedding_fp, header=0, index_col=None)
    
    # (Optional) standardize in the same way as in cluster_embedding()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Step B: PCA to 2D
    pca = PCA(n_components=2, random_state=42)
    embedding_pca = pca.fit_transform(data_scaled)
    
    # Step C: For each k, find the cluster file, load, and plot
    for k in cluster_list:
        cluster_csv = f"{os.path.splitext(embedding_fp)[0]}_clusters_k{k}.csv"
        
        if not os.path.exists(cluster_csv):
            print(f"Cluster file not found: {cluster_csv}")
            continue
        
        # Load cluster CSV
        cluster_df = pd.read_csv(cluster_csv, header=0, index_col=0)
        # cluster_df has a column named "Cluster_k{k}"
        
        # Ensure the row order matches the embedding DataFrame
        # If the indexes align, we can just do:
        labels = cluster_df[f"Cluster_k{k}"].values
        
        # If there's any mismatch in row order, consider reindexing cluster_df by some ID column 
        # or verifying that the indexes are the same.

        # Plot
        plt.figure(figsize=(6,5))
        sc = plt.scatter(
            embedding_pca[:, 0],
            embedding_pca[:, 1],
            c=labels,
            cmap='tab10',
            s=5
        )
        plt.title(f"{experiment_name}\nPCA with K={k} clusters")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        
        cbar = plt.colorbar(sc)
        cbar.set_label("Cluster ID", rotation=270, labelpad=15)
        
        # Save or show the plot
        plot_name = f"{experiment_name}_pca_k{k}.png"
        # plot_path = os.path.join(plot_dir, plot_name)
        # plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        # print(f"Saved PCA plot for {experiment_name}, k={k} to: {plot_path}")
# %%
adata_dir = "/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics"
visium_folder = f"{adata_dir}/Visium_HD_Human_Lung_Cancer_post_Xenium_Prime_5k_Experiment2/binned_outputs"
xenium_folder = f"{adata_dir}/Xenium_Prime_Human_Lung_Cancer_FFPE_outs/preprocessed"


adata_file_dict = {"xenium_5k": f"{xenium_folder}/pretrained/processed_xenium_data.h5ad",
                      "xenium_5k_paired": f"{xenium_folder}/fine_tune_refined_v2/processed_xenium_data_fine_tune_refined_v2.h5ad",
                      "visium_hd_8um_bins": f"{visium_folder}/square_008um/preprocessed/008um/filtered_feature_bc_matrix_preprocessed.h5ad",
                      "visium_hd_bin2cell": f"{visium_folder}/square_002um/preprocessed/bin2cell/preprocessed_bin2cell.h5ad",
                    }

for key in adata_file_dict:
    adata = sc.read_h5ad(adata_file_dict[key])
    print(adata)

# %%
for experiment_name, adata_fp in adata_file_dict.items():
    if not os.path.exists(adata_fp):
        print(f"[WARNING] Anndata file not found for {experiment_name}: {adata_fp}")
        continue
    
    print(f"\n--- Loading Anndata for: {experiment_name} ---")
    adata = sc.read_h5ad(adata_fp)
    print(adata)  # Show basic info
    
    # Check we have a corresponding embedding base file to build cluster paths
    if experiment_name not in embedding_file_dict:
        print(f"[WARNING] No embedding file for {experiment_name}, skipping cluster merge.")
        continue
    
    embedding_fp = embedding_file_dict[experiment_name]
    base_name = os.path.splitext(embedding_fp)[0]
    
    # For each k in cluster_list, load cluster CSV, merge, and plot
    for k in cluster_list:
        cluster_csv = f"{base_name}_clusters_k{k}.csv"
        if not os.path.exists(cluster_csv):
            print(f"[WARNING] Cluster file not found: {cluster_csv} (skipping).")
            continue
        
        clusters_df = pd.read_csv(cluster_csv, header=0, index_col=0)
        cluster_col_name = f"Cluster_k{k}"
        
        if cluster_col_name not in clusters_df.columns:
            print(f"[WARNING] Column '{cluster_col_name}' not in {cluster_csv}. Skipping.")
            continue
        
        # Ensure row alignment (by index). If your indexes match exactly:
        adata.obs[cluster_col_name] = clusters_df[cluster_col_name].values
        
        # Differentiate coordinate column names 
        # 1) Xenium: x_centroid, y_centroid
        # 2) Visium: pxl_row_in_fullres, pxl_col_in_fullres
        if all(col in adata.obs.columns for col in ["x_centroid", "y_centroid"]):
            # Xenium data
            x_coords = adata.obs["x_centroid"].values
            y_coords = adata.obs["y_centroid"].values
            
        elif all(col in adata.obs.columns for col in ["pxl_row_in_fullres", "pxl_col_in_fullres"]):
            # Visium data
            # Note: row_in_fullres typically corresponds to y, col_in_fullres corresponds to x
            x_coords = adata.obs["pxl_col_in_fullres"].values
            y_coords = adata.obs["pxl_row_in_fullres"].values
            
        else:
            print(f"[WARNING] No known coordinate columns found in adata.obs for {experiment_name}.")
            continue
        
        # Get cluster labels
        labels = adata.obs[cluster_col_name].values
        
        # Plot
        plt.figure(figsize=(15,10))
        scatt = plt.scatter(x_coords, y_coords, c=labels, cmap="tab10", s=5)
        
        # Typically for image-based coords, we might invert_yaxis
        plt.gca().invert_yaxis()
        
        plt.title(f"{experiment_name} | k={k} clusters")
        plt.xlabel("x")
        plt.ylabel("y")
        
        # Add colorbar
        # cbar = plt.colorbar(scatt)
        # cbar.set_label(cluster_col_name, rotation=270, labelpad=15)
        
        # Save figure
        out_name = f"{experiment_name}_k{k}_spatial.png"
        # out_path = os.path.join(plot_dir, out_name)
        # plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        # print(f"Saved spatial cluster plot for {experiment_name} (k={k}) -> {out_path}")


# %%
