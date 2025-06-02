#%%
import os

#%%
ground_truth = "refined"
level = 0  # assuming level is defined

xenium_folder_dict = {
    "lung": "Xenium_Prime_Human_Lung_Cancer_FFPE_outs",
    "breast": "Xenium_Prime_Breast_Cancer_FFPE_outs",
    "lymph_node": "Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_outs",
    "prostate": "Xenium_Prime_Human_Prostate_FFPE_outs",
    "skin": "Xenium_Prime_Human_Skin_FFPE_outs",
    "ovarian": "Xenium_Prime_Ovarian_Cancer_FFPE_outs",
    "cervical": "Xenium_Prime_Cervical_Cancer_FFPE_outs"
}

missing_files = {}

for cancer, folder_name in xenium_folder_dict.items():
    base_dir = f"/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/{folder_name}"
    embedding_dir = f"/rsrch9/home/plm/idso_fa1_pathology/TIER2/paul-xenium/embeddings/public_data/{folder_name}"
    
    singleR_data_path = os.path.join(base_dir, "preprocessed", f"fine_tune_{ground_truth}_v2", f"processed_xenium_data_fine_tune_{ground_truth}_v2_annotated.h5ad")
    gene_emb_path = os.path.join(embedding_dir, "scGPT", "scGPT_CP.h5ad")
    morph_embedding_v1 = os.path.join(embedding_dir, "UNI2_cell_representation", f"level_{level}", "morphology_embeddings_v2.csv")
    morph_embedding_v2 = os.path.join(embedding_dir, "UNI2_cell_representation", f"level_{level}", "uni2_pretrained_embeddings.csv")

    files = {
        "singleR_data_path": singleR_data_path,
        "gene_emb_path": gene_emb_path,
        "morph_embedding_v1": morph_embedding_v1,
        "morph_embedding_v2": morph_embedding_v2
    }

    missing = [name for name, path in files.items() if not os.path.exists(path)]
    if missing:
        missing_files[cancer] = missing

# Report
if missing_files:
    print("Missing files by cancer type:")
    for cancer, files in missing_files.items():
        print(f"- {cancer}: {', '.join(files)}")
else:
    print("✅ All files found for all cancer types.")


#%%

results_dir = "/rsrch9/home/plm/idso_fa1_pathology/TIER2/paul-xenium/classification_results/public_data"

xenium_folder_dict = {
    "lung": "Xenium_Prime_Human_Lung_Cancer_FFPE_outs",
    "breast": "Xenium_Prime_Breast_Cancer_FFPE_outs",
    "lymph_node": "Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_outs",
    "prostate": "Xenium_Prime_Human_Prostate_FFPE_outs",
    "skin": "Xenium_Prime_Human_Skin_FFPE_outs",
    "ovarian": "Xenium_Prime_Ovarian_Cancer_FFPE_outs",
    "cervical": "Xenium_Prime_Cervical_Cancer_FFPE_outs"
}

required_subfolders = ["T1", "T2", "T3", "T4"]
required_files = [
    "classification_report_aistil_v2.csv",
    "classification_report_singleR_v2.csv"
]

missing_info = {}

for cancer, folder_name in xenium_folder_dict.items():
    cancer_path = os.path.join(results_dir, folder_name)
    cancer_missing = []

    if not os.path.exists(cancer_path):
        cancer_missing.append("Missing folder")
    else:
        for sub in required_subfolders:
            sub_path = os.path.join(cancer_path, sub)
            if not os.path.exists(sub_path):
                cancer_missing.append(f"Missing subfolder: {sub}")
            else:
                for file in required_files:
                    file_path = os.path.join(sub_path, file)
                    if not os.path.exists(file_path):
                        cancer_missing.append(f"Missing file in {sub}: {file}")

    if cancer_missing:
        missing_info[cancer] = cancer_missing

# Report
if missing_info:
    print("Missing files or folders:")
    for cancer, items in missing_info.items():
        print(f"- {cancer}:")
        for item in items:
            print(f"    - {item}")
else:
    print("✅ All expected subfolders and files found.")


# %%

results_dir = "/rsrch9/home/plm/idso_fa1_pathology/TIER2/paul-xenium/classification_results/public_data"

xenium_folder_dict = {
    "lung": "Xenium_Prime_Human_Lung_Cancer_FFPE_outs",
    "breast": "Xenium_Prime_Breast_Cancer_FFPE_outs",
    "lymph_node": "Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_outs",
    "prostate": "Xenium_Prime_Human_Prostate_FFPE_outs",
    "skin": "Xenium_Prime_Human_Skin_FFPE_outs",
    "ovarian": "Xenium_Prime_Ovarian_Cancer_FFPE_outs",
    "cervical": "Xenium_Prime_Cervical_Cancer_FFPE_outs"
}

for folder_name in xenium_folder_dict.values():
    t3_path = os.path.join(results_dir, folder_name, "T3")

    if not os.path.exists(t3_path):
        print(f"❌ T3 folder missing for: {folder_name}")
        continue

    for fname in os.listdir(t3_path):
        full_path = os.path.join(t3_path, fname)

        if os.path.isfile(full_path):
            if not (fname.endswith("v1.csv") or fname.endswith("v2.csv")):
                base, ext = os.path.splitext(fname)
                new_name = f"{base}_v2.csv"
                new_path = os.path.join(t3_path, new_name)
                os.rename(full_path, new_path)
                print(f"✅ Renamed {fname} → {new_name}")

# %%
