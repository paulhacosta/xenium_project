#%% 
import os 
import pandas as pd 
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
import math


#%%

sample_ids = [
    "1495773-2",
    "1535564-2",
    "1466459-1",
    "1438089-1",
    "1481840-1",
    "1458677-2",
    "1476760-2",
    "1470095-1",
    "1500745-2",
    "1479372-2",
    "1493626-1",
    "1534673-1",
    "1522909-2",
    "1587007-1",
    "1540049-1",
    "1534912-2"]
aistil_root = "/rsrch9/home/plm/idso_fa1_pathology/TIER2/utuc_aistil"

spatial_dir = f"{aistil_root}/7_spatial" 
cell_score_dir = f"{aistil_root}/4_cell_class/CellScore"


# %%
all_spatial = [os.path.join(spatial_dir, f+".csv") for f in sample_ids]
all_cellScore = [os.path.join(cell_score_dir, f+".svs_cellScore.csv") for f in sample_ids]

spatial_csv_files, cell_score_csv_files = [],[]
no_file = []
for c, f in enumerate(all_spatial):
    if os.path.exists(f) :
        spatial_csv_files.append(f)
        cell_score_csv_files.append(all_cellScore[c])
    else: 
        no_file.append(f)
# %%
# Function to read and concatenate CSV files into a single DataFrame
def concatenate_csv_files(file_list):
    df_list = [pd.read_csv(file) for file in file_list]
    concatenated_df = pd.concat(df_list, ignore_index=True)
    return concatenated_df

# Concatenate CSV files for spatial data and cell scores
spatial_df = concatenate_csv_files(spatial_csv_files)
cell_score_df = concatenate_csv_files(cell_score_csv_files)

spatial_df.rename(columns={"file_name": "tid_surrogate_id"}, inplace=True)

# Display the resulting DataFrames
print("Spatial DataFrame:")
print(spatial_df.head())

print("\nCell Score DataFrame:")
print(cell_score_df.head())

# %%

# Define the figure and axes
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot the counts distribution
spatial_df[['count_stromal', 'count_lymphocyte', 'count_cancer', 'count_othercells']].plot(
    kind='box', ax=axes[0], title='Counts Distribution', grid=False)
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')


# Plot the percentages distribution
spatial_df[['percentage_stromal', 'percentage_lymphocyte', 'percentage_cancer', 'percentage_othercells']].plot(
    kind='box', ax=axes[1], title='Percentages Distribution', grid=False)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

# Display the plots
plt.tight_layout()
plt.show()


# %%
meta_data = "/rsrch5/home/plm/phacosta/UTUC/UTUC_SWoodman_cases_PA.xlsx"

df = pd.read_excel(meta_data, sheet_name="Clean_Data")

# Filter df to only include rows where 'tid_surrogate_id' matches 'file_name' in spatial_df
clinical_df = df[df["tid_surrogate_id"].isin(spatial_df["tid_surrogate_id"])]

clinical_df.set_index("tid_surrogate_id", inplace=True)
spatial_df.set_index("tid_surrogate_id", inplace=True)

merged_df = clinical_df.merge(spatial_df, left_index=True, right_index=True)



# %% Stat test
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu

# Suppose your dataframe is `df`
# And let's say the numeric columns you want to test are:
numeric_cols = [
    'count_stromal', 'count_lymphocyte', 'count_cancer',
    'percentage_stromal', 'percentage_lymphocyte', 'percentage_cancer'
]

# Define your two groups:
group_no_tumor = merged_df[merged_df["Clinical response (SW)"] == "no tumor response"]
group_tumor_resp = merged_df[merged_df["Clinical response (SW)"] == "tumor response"]

# For each numeric feature, run a t-test:
for col in numeric_cols:
    # Drop any NaNs
    g1 = group_no_tumor[col].dropna()
    g2 = group_tumor_resp[col].dropna()
    
    # Perform two-sample t-test
    t_stat, p_val = ttest_ind(g1, g2, equal_var=False)  
    # If you suspect unequal variances, set equal_var=False (Welch's t-test).
    
    print(f"=== {col} ===")
    print(f"T-statistic: {t_stat:.3f}, P-value: {p_val:.3e}")
    print()


#%% Overall cell comparison 

column_comparison = "Clinical response (SW)"  # or "PFS (SW)"

# Your numeric columns of interest (already specified)
numeric_cols = [
    # Top row (stroma):
    'count_stromal',       # fibroblasts in stroma
    'count_lymphocyte',    # lymphocytes in stroma
    'count_cancer',
    'count_othercells',
    # Bottom row (percentages):
    'percentage_stromal',
    'percentage_lymphocyte',
    'percentage_cancer',
    'percentage_othercells'
]

# Map cell type substrings to more descriptive names
cell_dict = {
    "stromal": "fibroblasts",
    "lymphocyte": "lymphocytes",
    "cancer": "cancer cells",
    "othercells": "other cells"
}

categories = {
    "Clinical response (SW)": ["no tumor response", "tumor response"],
    "PFS (SW)": ["short PFS", "long PFS"]
}

# 1. Filter the data to include only the two groups of interest
df_filtered = merged_df[
    merged_df[column_comparison].isin(categories[column_comparison])
]

# 2. Prepare subplots
num_plots = len(numeric_cols)
ncols = 4
nrows = int(math.ceil(num_plots / ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 4*nrows), sharey=False)
axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]

# Map the x-axis label to remove "(SW)"
pretty_xlabel_map = {
    "Clinical response (SW)": "Clinical response",
    "PFS (SW)": "PFS"
}
x_label = pretty_xlabel_map.get(column_comparison, column_comparison)

# 3. Loop through numeric columns to plot
for i, col in enumerate(numeric_cols):
    ax = axes[i]
    
    # Parse the column to determine measure type and cell type
    # e.g. "count_stromal" -> measure_type="count", cell_name="stromal"
    #      "percentage_cancer" -> measure_type="percentage", cell_name="cancer"
    measure_type, cell_name = col.split("_", 1)
    
    # Decide y-axis label based on whether it's "count" or "percentage"
    if measure_type == "count":
        y_label = "Cell counts"
    elif measure_type == "percentage":
        y_label = "Cell percentages"
    else:
        y_label = "Value"  # fallback if there's another prefix
    
    # Get a nicer title for the cell type
    cell_type_label = cell_dict.get(cell_name, cell_name)
    
    # Subset the two groups for the statistical test
    group_no_tumor = df_filtered.loc[
        df_filtered[column_comparison] == categories[column_comparison][0],
        col
    ].dropna()
    group_tumor_resp = df_filtered.loc[
        df_filtered[column_comparison] == categories[column_comparison][1],
        col
    ].dropna()
    
    # Perform Welch’s t-test (equal_var=False)
    t_stat, p_val = ttest_ind(group_no_tumor, group_tumor_resp, equal_var=False)
    
    # Plot box + strip in the same subplot
    sns.boxplot(
        x=column_comparison, 
        y=col, 
        data=df_filtered, 
        palette="Set3",  # or any other palette
        ax=ax
    )
    sns.stripplot(
        x=column_comparison, 
        y=col, 
        data=df_filtered, 
        color="black", 
        alpha=0.6,
        ax=ax
    )
    
    # Update x-label, y-label, and subplot title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    ax.set_title(
        f"{cell_type_label}\n p={p_val:.2e}",
        fontsize=11
    )

# 4. Remove any unused subplot axes if numeric_cols < nrows*ncols
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])  # remove empty axes

plt.tight_layout()
plt.show()



# %% Comparison of cells within tissue - tumor vs stroma

# Configuration Flags
column_comparison = "Clinical response (SW)"   # "Clinical response (SW)" or "PFS (SW)"
test = "t"                           # "t" (Welch's t-test) or "u" (Mann–Whitney U test)
plot_type = "density"               # "counts", "percentages", or "density"

# Dictionary to specify which categories we want to compare
categories = {
    "Clinical response (SW)": ["no tumor response", "tumor response"],
    "PFS (SW)": ["short PFS", "long PFS"]
}

# Nicer names for cell letters
cell_dict = {"l": "lymphocytes", "t": "tumor cells", "f": "fibroblasts", "o": "other cells"}

# A small helper dict to make x-labels look nicer (remove " (SW)" part)
pretty_xlabel_map = {
    "Clinical response (SW)": "Clinical response",
    "PFS (SW)": "PFS"
}

# Prepare the Data
# Suppose your merged_df is the main dataframe
df = merged_df.copy()

# Create columns for total stroma and total tumor (for 'percentages')
df['stroma_total'] = df['count_f_stroma'] + df['count_l_stroma'] + df['count_o_stroma']
df['tumor_total']  = df['count_l_tumor']  + df['count_t_tumor']  + df['count_o_tumor']

# We'll map region -> name of the total column and area column
region_totals = {
    'stroma': 'stroma_total',
    'tumor': 'tumor_total'
}
region_area = {
    'stroma': 'stroma_mm2',
    'tumor': 'tumor_mm2'
}

# Reorder columns if you want a specific layout
numeric_cols = [
    # Top row (stroma):
    'count_f_stroma',  # fibroblasts in stroma
    'count_l_stroma',  # lymphocytes in stroma
    'count_o_stroma',  # other cells in stroma
    # Bottom row (tumor):
    'count_t_tumor',   # tumor cells in tumor
    'count_l_tumor',   # lymphocytes in tumor
    'count_o_tumor',   # other cells in tumor
]

# Filter the data to only the two categories of interest
df_filtered = df[df[column_comparison].isin(categories[column_comparison])]

# Decide on subplot layout
num_plots = len(numeric_cols)
ncols = 3
nrows = int(np.ceil(num_plots / ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 4*nrows))
axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]

# Choose the color palette based on column_comparison
palette_map = {
    "Clinical response (SW)": "Set2",
    "PFS (SW)": "Set3"
}
chosen_palette = palette_map.get(column_comparison, "Set2")

# Determine y-axis label based on plot_type
if plot_type == "counts":
    y_label = "Cell counts"
elif plot_type == "percentages":
    y_label = "Cell percentages"
elif plot_type == "density":
    y_label = "Cell density (counts per mm²)"
else:
    y_label = "Value"  # fallback

# Determine which x label to use
x_label = pretty_xlabel_map.get(column_comparison, column_comparison)

# Main Loop: Plot Each Column
for i, col in enumerate(numeric_cols):
    ax = axes[i]

    # Identify which tissue region this column belongs to ("stroma" or "tumor")
    region = col.rsplit('_', 1)[-1]  # e.g., 'stroma' or 'tumor'

    # Decide how to compute the y_col to plot
    if plot_type == "counts":
        # Raw counts
        y_col = col

    elif plot_type == "percentages":
        # col / total cells in that region
        pct_col = col + "_pct"
        df_filtered[pct_col] = df_filtered[col] / df_filtered[region_totals[region]]
        y_col = pct_col

    elif plot_type == "density":
        # col / area (mm^2) in that region
        dens_col = col.replace("count", "density")
        df_filtered[dens_col] = df_filtered[col] / df_filtered[region_area[region]]
        y_col = dens_col

    else:
        # Fallback: just use raw counts
        y_col = col

    # Subset the two groups for the statistical test
    group_1 = df_filtered.loc[
        df_filtered[column_comparison] == categories[column_comparison][0], 
        y_col
    ].dropna()
    group_2 = df_filtered.loc[
        df_filtered[column_comparison] == categories[column_comparison][1], 
        y_col
    ].dropna()

    # Perform chosen test
    if test == "t":
        # Welch's t-test
        stat, p_val = ttest_ind(group_1, group_2, equal_var=False)
        test_name = "Welch's t-test"
    else:
        # Mann–Whitney U test
        stat, p_val = mannwhitneyu(group_1, group_2, alternative='two-sided')
        test_name = "Mann–Whitney U"

    # Plot box + strip
    sns.boxplot(
        x=column_comparison,
        y=y_col,
        data=df_filtered,
        palette=chosen_palette,
        ax=ax
    )
    sns.stripplot(
        x=column_comparison,
        y=y_col,
        data=df_filtered,
        color="black",
        alpha=0.6,
        ax=ax
    )

    # Build a nicer title
    title_elements = col.rsplit("_")  # e.g., ['count', 'l', 'stroma']
    cell_type_letter = title_elements[1]  # f / l / t / o
    tissue_region = title_elements[2]     # stroma / tumor
    cell_type_name = cell_dict.get(cell_type_letter, cell_type_letter)

    ax.set_title(
        f"{cell_type_name} in {tissue_region}\n {test_name} p={p_val:.2e}",
        fontsize=10
    )

    # Update the x and y labels for each subplot
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

# Remove extra axes if numeric_cols < nrows*ncols
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# %% Regression model 

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score

# Prepare the data

df["density_l_tumor"] = df["count_l_tumor"] / df["tumor_mm2"]

# Encode the target
df["response_binary"] = df["Clinical response (SW)"].map({
    "no tumor response": 0,
    "tumor response": 1
})

# Suppose you've already computed: df["count_l_tumor_density"]
X = df[["density_l_tumor"]]  # or multiple features
y = df["response_binary"]

# Define pipeline
pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("logreg", LogisticRegression())
])

# Leave-One-Out CV
loo = LeaveOneOut()
scores = cross_val_score(pipeline, X, y, cv=loo, scoring='accuracy')
print("LOOCV accuracy for each split:", scores)
print("Mean accuracy:", np.mean(scores), "±", np.std(scores))

# Detailed classification results (optional)
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report

y_pred_cv = cross_val_predict(pipeline, X, y, cv=loo)
print("Confusion Matrix:\n", confusion_matrix(y, y_pred_cv))
print("Classification Report:\n", classification_report(y, y_pred_cv))

# %%
