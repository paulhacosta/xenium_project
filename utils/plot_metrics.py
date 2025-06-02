#%%
import matplotlib.pyplot as plt
import numpy as np

#%%
# Define model names and order
models = ["MLP", "GNN (Cell Similarity)", "Transformer"]
platforms = ["scGPT", "Geneformer"]
# x_labels = [f"{p}\n{m}" for m in models for p in platforms]
# Correct label order
x_labels = [f"{p}\n{m}" for p in platforms for m in models]


# Define F1 scores for Xenium All Genes (solid bars) and Xenium Filtered Genes (dashed bars)
f1_scores_first = {
    "Fibroblast": [0.78, 0.82, 0.87],
    "Tumor": [0.83, 0.84, 0.89],
    "Lymphocytes": [0.58, 0.71, 0.80]
}

f1_scores_second = {
    "Fibroblast": [0.62, 0.75, 0.86],
    "Tumor": [0.36, 0.73, 0.87],
    "Lymphocytes": [0.34, 0.61, 0.79]
}

# Flatten data for plotting
f1_fibroblast = f1_scores_first["Fibroblast"] + f1_scores_second["Fibroblast"]
f1_tumor = f1_scores_first["Tumor"] + f1_scores_second["Tumor"]
f1_lymphocytes = f1_scores_first["Lymphocytes"] + f1_scores_second["Lymphocytes"]

# Set up bar positions
x = np.arange(len(x_labels))  # X locations for the groups
width = 0.25  # Width of the bars

# Create the bar chart
fig, ax = plt.subplots(figsize=(10, 6))

# Iterate over bars and apply different styles for All Genes (solid) and Filtered Genes (dashed)
for i in range(len(x_labels)):
    pattern = "" if "Geneformer" in x_labels[i] else "//"  # Solid for All Genes, Dashed for Filtered Genes
    
    # Fibroblast Bar
    ax.bar(x[i] - width, f1_fibroblast[i], width, label="Fibroblast" if i == 0 else "", color='blue', hatch=pattern, edgecolor='black')
    # Tumor Bar
    ax.bar(x[i], f1_tumor[i], width, label="Tumor" if i == 0 else "", color='red', hatch=pattern, edgecolor='black')
    # Lymphocytes Bar
    ax.bar(x[i] + width, f1_lymphocytes[i], width, label="Lymphocytes" if i == 0 else "", color='green', hatch=pattern, edgecolor='black')

    # Add raw numbers to each bar
    ax.text(x[i] - width, f1_fibroblast[i] + 0.02, f"{f1_fibroblast[i]:.2f}", ha='center', fontsize=10)
    ax.text(x[i], f1_tumor[i] + 0.02, f"{f1_tumor[i]:.2f}", ha='center', fontsize=10)
    ax.text(x[i] + width, f1_lymphocytes[i] + 0.02, f"{f1_lymphocytes[i]:.2f}", ha='center', fontsize=10)

# Labels and title
ax.set_xlabel("Platform and Model")
ax.set_ylabel("F1 Score")
ax.set_title("Model Performance: scGPT vs Geneformer (Xenium)", fontsize=12)  # Updated title
ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=45, ha="right")  # Apply text wrapping
ax.set_ylim(0, 1.1)
ax.legend(title="Cell Type")

plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np

# Define model names and order
models = ["MLP", "GNN (Cell Similarity)", "Transformer"]
platforms = ["scGPT", "Geneformer"]

# New label order: group by model type
x_labels = [f"{m}\n{p}" for m in models for p in platforms]

# F1 scores need to follow this new order
# Format: [scGPT MLP, Geneformer MLP, scGPT GNN, Geneformer GNN, scGPT Transformer, Geneformer Transformer]
f1_fibroblast = [0.78, 0.62, 0.82, 0.75, 0.87, 0.86]
f1_tumor =      [0.83, 0.36, 0.84, 0.73, 0.89, 0.87]
f1_lymphocytes =[0.58, 0.34, 0.71, 0.61, 0.80, 0.79]

# Set up bar positions
x = np.arange(len(x_labels))  # X locations for the groups
width = 0.25  # Width of the bars

# Create the bar chart
fig, ax = plt.subplots(figsize=(10, 6))

# Apply hatch pattern to scGPT (first of each pair)
for i, label in enumerate(x_labels):
    pattern = "//" if "scGPT" in label else ""  # Hatch for scGPT, solid for Geneformer
    
    # Fibroblast Bar
    ax.bar(x[i] - width, f1_fibroblast[i], width, label="Fibroblast" if i == 0 else "", color='blue', hatch=pattern, edgecolor='black')
    # Tumor Bar
    ax.bar(x[i], f1_tumor[i], width, label="Tumor" if i == 0 else "", color='red', hatch=pattern, edgecolor='black')
    # Lymphocytes Bar
    ax.bar(x[i] + width, f1_lymphocytes[i], width, label="Lymphocytes" if i == 0 else "", color='green', hatch=pattern, edgecolor='black')

    # Add raw numbers to each bar
    ax.text(x[i] - width, f1_fibroblast[i] + 0.02, f"{f1_fibroblast[i]:.2f}", ha='center', fontsize=10)
    ax.text(x[i], f1_tumor[i] + 0.02, f"{f1_tumor[i]:.2f}", ha='center', fontsize=10)
    ax.text(x[i] + width, f1_lymphocytes[i] + 0.02, f"{f1_lymphocytes[i]:.2f}", ha='center', fontsize=10)

# Labels and title
ax.set_xlabel("Model and Architecture")
ax.set_ylabel("F1 Score")
ax.set_title("Model Performance: scGPT vs Geneformer", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=45, ha="right")
ax.set_ylim(0, 1.2)
ax.legend(title="Cell Type",loc = "upper left")
ax.yaxis.grid(True, linestyle="--", color="gray", alpha=0.7)

plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Define cell types
categories = ["Fibroblast", "T Cell", "B Cell", "Macrophage", "Tumor"]

# F1-scores for each method
f1_scgpt = [0.95, 0.93, 0.89, 0.91, 0.87]        # hypothetical scGPT values
f1_geneformer = [0.76, 0.78, 0.59, 0.67, 0.89]   # your current values

# Define bar positions
x = np.arange(len(categories))  # [0, 1, 2, 3, 4]
width = 0.35  # width of each bar

# Define base colors (same as before)
colors = {
    "Fibroblast": "blue",
    "T Cell": "orange",
    "B Cell": "purple",
    "Macrophage": "green",
    "Tumor": "red"
}
bar_colors = [colors[cat] for cat in categories]

# Create the grouped bar plot
fig, ax = plt.subplots(figsize=(9, 5))

# Plot bars for scGPT (left bars)
bars1 = ax.bar(x - width/2, f1_scgpt, width, label="scGPT", color=bar_colors, edgecolor="black", hatch="//")
# Plot bars for Geneformer (right bars)
bars2 = ax.bar(x + width/2, f1_geneformer, width, label="Geneformer", color=bar_colors, edgecolor="black")

# Add text labels above bars
for i in range(len(categories)):
    ax.text(x[i] - width/2, f1_scgpt[i] + 0.02, f"{f1_scgpt[i]:.2f}", ha='center', fontsize=10)
    ax.text(x[i] + width/2, f1_geneformer[i] + 0.02, f"{f1_geneformer[i]:.2f}", ha='center', fontsize=10)

# Labels and formatting
ax.set_xlabel("Cell Type")
ax.set_ylabel("F1 Score")
ax.set_title("scGPT vs Geneformer F1 Scores by Cell Type", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylim(0, 1.2)


# Custom legend with white fill and colored edges
custom_legend = [
    Patch(facecolor='white', edgecolor='black', hatch='//', label='scGPT'),
    Patch(facecolor='white', edgecolor='black', label='Geneformer')
]
ax.legend(handles=custom_legend, title="Model")


# ax.legend()
ax.yaxis.grid(True, linestyle="--", color="gray", alpha=0.7)

plt.tight_layout()
plt.show()

# %%
