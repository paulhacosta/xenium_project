# %%
import pandas as pd
import os

#%%
# Define root and folders
results_root = "/rsrch9/home/plm/idso_fa1_pathology/TIER2/paul-xenium/classification_results/public_data"
xenium_folders = [
    "Xenium_Prime_Human_Lung_Cancer_FFPE_outs",
    "Xenium_Prime_Breast_Cancer_FFPE_outs",
    "Xenium_Prime_Human_Prostate_FFPE_outs",
]

label_sources = ["aistil", "singleR"]
model_structure = ["T1", "T2", "T3", "T4"]

# Simplified names
tissue_name_map = {
    "Xenium_Prime_Human_Lung_Cancer_FFPE_outs": "Lung",
    "Xenium_Prime_Breast_Cancer_FFPE_outs": "Breast",
    "Xenium_Prime_Human_Prostate_FFPE_outs": "Prostate",
}

# Collect data
records = []

for folder in xenium_folders:
    for T in model_structure:
        for label_source in label_sources:
            path = os.path.join(results_root, folder, T, f"classification_report_{label_source}.csv")
            if os.path.exists(path):
                df = pd.read_csv(path, index_col=0)
                for label in df.index:
                    if label not in ["accuracy", "macro avg", "weighted avg"]:
                        records.append({
                            "Tissue": tissue_name_map[folder],
                            "Model": T,
                            "Label Source": label_source,
                            "Class": label,
                            "Precision": df.loc[label, "precision"],
                            "Recall": df.loc[label, "recall"],
                            "F1-Score": df.loc[label, "f1-score"],
                            "Support": int(df.loc[label, "support"]),
                        })

# Convert to DataFrame
results_df = pd.DataFrame(records)

# %%
# Clean and standardize
results_df["Tissue"] = results_df["Tissue"].str.strip().str.capitalize()
results_df["Model"] = results_df["Model"].str.strip()
results_df["Label Source"] = results_df["Label Source"].str.strip().str.lower()
results_df["Class"] = results_df["Class"].astype(str).str.strip().str.lower()

# Map AISTIL labels
aistil_class_map = {"f": "fibroblast", "l": "lymphocyte", "t": "tumor"}
results_df["Class"] = results_df.apply(
    lambda row: aistil_class_map.get(row["Class"], row["Class"])
    if row["Label Source"] == "aistil" else row["Class"],
    axis=1
)

# ==========================================
# AISTIL SECTION
# ==========================================
aistil_df = results_df[results_df["Label Source"] == "aistil"].copy()

aistil_classes = ["fibroblast", "lymphocyte", "tumor"]
metrics = ["Precision", "Recall", "F1-Score"]

pivot_aistil = aistil_df.pivot_table(
    index=["Tissue", "Model"],
    columns="Class",
    values=metrics
).reset_index()

# Force all expected columns
from itertools import product
full_col_index = pd.MultiIndex.from_product([metrics, aistil_classes])
non_metric_cols = pivot_aistil.columns[pivot_aistil.columns.get_level_values(0).isin(["Tissue", "Model"])]
pivot_aistil = pivot_aistil.reindex(columns=[*non_metric_cols, *full_col_index], fill_value=None)

# Generate AISTIL LaTeX table (revised)
# Abbreviated metric and class labels
short_class_map = {"fibroblast": "fib", "lymphocyte": "lym", "tumor": "tum"}
short_metric_map = {"Precision": "P", "Recall": "R", "F1-Score": "F1"}

latex_lines = []
col_format = "l" + "c" * len(full_col_index)
latex_lines.append("\\begin{table*}[H]")
latex_lines.append("\\centering")
latex_lines.append("\\scriptsize")
latex_lines.append(f"\\begin{{tabular}}{{{col_format}}}")
latex_lines.append("\\toprule")

# Shortened headers like F1_{tum}
headers = [
    f"${short_metric_map[m]}_{{{short_class_map[c]}}}$"
    for c in aistil_classes
    for m in metrics
]
latex_lines.append("Tissue & " + " & ".join(headers) + " \\\\")
latex_lines.append("\\midrule")

for model in model_structure:
    latex_lines.append("\\addlinespace")
    latex_lines.append(f"\\multicolumn{{{1 + len(headers)}}}{{l}}{{\\textit{{{model}}}}} \\\\")
    latex_lines.append("\\midrule")

    for tissue in ["Breast", "Lung", "Prostate"]:
        row = pivot_aistil[
            (pivot_aistil["Model"] == model) & (pivot_aistil["Tissue"] == tissue)
        ]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        row_values = [tissue]

        for cls in aistil_classes:
            for metric in metrics:
                val = row.get((metric, cls), None)
                row_values.append(f"{val:.2f}" if pd.notna(val) else "--")

        latex_lines.append(" & ".join(row_values) + " \\\\")

latex_lines.append("\\bottomrule")
latex_lines.append("\\end{tabular}")
latex_lines.append("\\caption{Performance metrics for AISTIL predictions across tissues.}")
latex_lines.append("\\label{tab:aistil_results}")
latex_lines.append("\\end{table*}")

print("\n% ===== AISTIL Table =====")
print("\n".join(latex_lines))


#%%
# ==========================================
# SINGLE-R SECTION
# ==========================================
singler_df = results_df[results_df["Label Source"] == "singler"].copy()

# Get all classes present in singleR
singler_classes = sorted(singler_df["Class"].unique())
metrics = ["Precision", "Recall", "F1-Score"]

# Pivot to wide format
pivot_singler = singler_df.pivot_table(
    index=["Tissue", "Model"],
    columns="Class",
    values=metrics
).reset_index()

# Force all expected columns
from itertools import product
full_col_index_singler = pd.MultiIndex.from_product([metrics, singler_classes])
non_metric_cols = pivot_singler.columns[pivot_singler.columns.get_level_values(0).isin(["Tissue", "Model"])]
pivot_singler = pivot_singler.reindex(columns=[*non_metric_cols, *full_col_index_singler], fill_value=None)

# Shorten class and metric names for the header
short_class_map_singler = {c: c[:3].lower() for c in singler_classes}
short_metric_map = {"Precision": "P", "Recall": "R", "F1-Score": "F1"}

latex_lines = []
col_format = "l" + "c" * len(full_col_index_singler)
latex_lines.append("\\begin{table*}[H]")
latex_lines.append("\\centering")
latex_lines.append("\\scriptsize")
latex_lines.append(f"\\begin{{tabular}}{{{col_format}}}")
latex_lines.append("\\toprule")

# Build header row
headers = [
    f"${short_metric_map[m]}_{{{short_class_map_singler[c]}}}$"
    for c in singler_classes
    for m in metrics
]
latex_lines.append("Tissue & " + " & ".join(headers) + " \\\\")
latex_lines.append("\\midrule")

# Populate rows
for model in model_structure:
    latex_lines.append("\\addlinespace")
    latex_lines.append(f"\\multicolumn{{{1 + len(headers)}}}{{l}}{{\\textit{{{model}}}}} \\\\")
    latex_lines.append("\\midrule")

    for tissue in ["Breast", "Lung", "Prostate"]:
        row = pivot_singler[
            (pivot_singler["Model"] == model) & (pivot_singler["Tissue"] == tissue)
        ]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        row_values = [tissue]

        for cls in singler_classes:
            for metric in metrics:
                val = row.get((metric, cls), None)
                row_values.append(f"{val:.2f}" if pd.notna(val) else "--")

        latex_lines.append(" & ".join(row_values) + " \\\\")

latex_lines.append("\\bottomrule")
latex_lines.append("\\end{tabular}")
latex_lines.append("\\caption{Performance metrics for SingleR predictions across tissues.}")
latex_lines.append("\\label{tab:singler_results}")
latex_lines.append("\\end{table*}")

print("\n% ===== SINGLE-R Table =====")
print("\n".join(latex_lines))


# %%
############################################################################################
# Comparing projections to embeddings
############################################################################################

# Define root and folders
results_root = "/rsrch9/home/plm/idso_fa1_pathology/TIER2/paul-xenium/classification_results/public_data"
xenium_folders = [
    "Xenium_Prime_Human_Lung_Cancer_FFPE_outs",
    "Xenium_Prime_Breast_Cancer_FFPE_outs",
    "Xenium_Prime_Human_Prostate_FFPE_outs",
]

label_sources = ["aistil", "singleR"]
model_structure = ["T1", "T2", "T3", "T4"]

tissue_name_map = {
    "Xenium_Prime_Human_Lung_Cancer_FFPE_outs": "Lung",
    "Xenium_Prime_Breast_Cancer_FFPE_outs": "Breast",
    "Xenium_Prime_Human_Prostate_FFPE_outs": "Prostate",
}

# ======================
# Collect data
# ======================
records = []

for folder in xenium_folders:
    for T in model_structure:
        for label_source in label_sources:
            emb_path = os.path.join(results_root, folder, T, f"classification_report_{label_source}_v2.csv")
            proj_path = os.path.join(results_root, folder, T, f"classification_report_{label_source}_v2_proj.csv")
            
            if not os.path.exists(emb_path) or not os.path.exists(proj_path):
                continue
            
            df_emb = pd.read_csv(emb_path, index_col=0)
            df_proj = pd.read_csv(proj_path, index_col=0)

            for label in df_emb.index:
                if label in ["accuracy", "macro avg", "weighted avg"]:
                    continue

                records.append({
                    "Tissue": tissue_name_map[folder],
                    "Model": T,
                    "Label Source": label_source,
                    "Class": label,
                    "F1_emb": df_emb.loc[label, "f1-score"],
                    "F1_proj": df_proj.loc[label, "f1-score"],
                })

# Combine into a dataframe
results_df = pd.DataFrame(records)

# %%
# Clean and standardize
results_df["Tissue"] = results_df["Tissue"].str.strip().str.capitalize()
results_df["Model"] = results_df["Model"].str.strip()
results_df["Label Source"] = results_df["Label Source"].str.strip().str.lower()
results_df["Class"] = results_df["Class"].astype(str).str.strip().str.lower()

# Map AISTIL class short names
aistil_class_map = {"f": "fibroblast", "l": "lymphocyte", "t": "tumor"}
results_df["Class"] = results_df.apply(
    lambda row: aistil_class_map.get(row["Class"], row["Class"])
    if row["Label Source"] == "aistil" else row["Class"],
    axis=1
)

# ==========================================
# AISTIL formatted pivot
# ==========================================
aistil_df = results_df[results_df["Label Source"] == "aistil"].copy()
aistil_classes = ["fibroblast", "lymphocyte", "tumor"]

pivot_aistil = aistil_df.pivot_table(
    index=["Tissue", "Model"],
    columns="Class",
    values=["F1_emb", "F1_proj"]
).reset_index()


# ==========================================
# LaTeX export for AISTIL
# ==========================================
from itertools import product

# Short class names
short_class_map = {"fibroblast": "fib", "lymphocyte": "lym", "tumor": "tum"}

# Header names like F1_fib,emb and F1_fib,proj
headers = []
for cls in aistil_classes:
    headers.append(f"$F1_{{{short_class_map[cls]},emb}}$")
    headers.append(f"$F1_{{{short_class_map[cls]},proj}}$")

col_format = "l" + "c" * len(headers)

latex_lines = []
latex_lines.append("\\begin{table*}[H]")
latex_lines.append("\\centering")
latex_lines.append("\\scriptsize")
latex_lines.append(f"\\begin{{tabular}}{{{col_format}}}")
latex_lines.append("\\toprule")
latex_lines.append("Tissue & " + " & ".join(headers) + " \\\\")
latex_lines.append("\\midrule")

for model in model_structure:
    latex_lines.append("\\addlinespace")
    latex_lines.append(f"\\multicolumn{{{1 + len(headers)}}}{{l}}{{\\textit{{{model}}}}} \\\\")
    latex_lines.append("\\midrule")

    for tissue in ["Breast", "Lung", "Prostate"]:
        row = pivot_aistil[
            (pivot_aistil["Model"] == model) & (pivot_aistil["Tissue"] == tissue)
        ]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        row_values = [tissue]

        for cls in aistil_classes:
            for ftype in ["F1_emb", "F1_proj"]:
                val = row.get((ftype, cls), None)
                row_values.append(f"{val:.2f}" if pd.notna(val) else "--")

        latex_lines.append(" & ".join(row_values) + " \\\\")

latex_lines.append("\\bottomrule")
latex_lines.append("\\end{tabular}")
latex_lines.append("\\caption{F1-score comparison between embeddings and projections for AISTIL cell types.}")
latex_lines.append("\\label{tab:aistil_proj_results}")
latex_lines.append("\\end{table*}")

print("\n% ===== AISTIL Table =====")
print("\n".join(latex_lines))




# %%

# ==========================================
# singleR formatted pivot
# ==========================================
singler_df = results_df[results_df["Label Source"] == "singler"].copy()
singler_classes = sorted(singler_df["Class"].unique())

pivot_singler = singler_df.pivot_table(
    index=["Tissue", "Model"],
    columns="Class",
    values=["F1_emb", "F1_proj"]
).reset_index()


# ==========================================
# LaTeX export for SingleR
# ==========================================


# Shorten class names for LaTeX labels
short_class_map_singler = {cls: cls[:3].lower() for cls in singler_classes}

# Header names like F1_b_c,emb and F1_b_c,proj
headers = []
for cls in singler_classes:
    headers.append(f"$F1_{{{short_class_map_singler[cls]},emb}}$")
    headers.append(f"$F1_{{{short_class_map_singler[cls]},proj}}$")

col_format = "l" + "c" * len(headers)

latex_lines = []
latex_lines.append("\\begin{table*}[H]")
latex_lines.append("\\centering")
latex_lines.append("\\scriptsize")
latex_lines.append(f"\\begin{{tabular}}{{{col_format}}}")
latex_lines.append("\\toprule")
latex_lines.append("Tissue & " + " & ".join(headers) + " \\\\")
latex_lines.append("\\midrule")

for model in model_structure:
    latex_lines.append("\\addlinespace")
    latex_lines.append(f"\\multicolumn{{{1 + len(headers)}}}{{l}}{{\\textit{{{model}}}}} \\\\")
    latex_lines.append("\\midrule")

    for tissue in ["Breast", "Lung", "Prostate"]:
        row = pivot_singler[
            (pivot_singler["Model"] == model) & (pivot_singler["Tissue"] == tissue)
        ]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        row_values = [tissue]

        for cls in singler_classes:
            for ftype in ["F1_emb", "F1_proj"]:
                val = row.get((ftype, cls), None)
                row_values.append(f"{val:.2f}" if pd.notna(val) else "--")

        latex_lines.append(" & ".join(row_values) + " \\\\")

latex_lines.append("\\bottomrule")
latex_lines.append("\\end{tabular}")
latex_lines.append("\\caption{F1-score comparison between embeddings and projections for SingleR cell types.}")
latex_lines.append("\\label{tab:singler_proj_results}")
latex_lines.append("\\end{table*}")

print("\n% ===== SINGLE-R Table =====")
print("\n".join(latex_lines))
# %%
