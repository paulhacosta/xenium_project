#%%
import os 
from datasets import load_from_disk, load_dataset
import pandas as pd
from tqdm import tqdm
import pickle
from tqdm import tqdm
from multiprocessing import Pool
import pickle
import math
#%%
root_path = "/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data"


# Paths to required files
genecorpus_path = f"{root_path}/Genecorpus-30M/genecorpus_30M_2048.dataset"
xenium_path = f"{root_path}/10x_genomics/Xenium_Prime_Human_Lung_Cancer_FFPE_outs/tokenize_output/processed_xenium_fine_tune_refined.dataset"  
token_dict_path = f"{root_path}/Genecorpus-30M/token_dictionary.pkl"


#%%

# Decode input_ids to gene names
def decode_input_ids(input_ids):
    return [id_to_gene[token_id] for token_id in input_ids]

# Process a chunk of data
def process_chunk(chunk):
    unique_genes = set()
    for sample in chunk:
        decoded_genes = decode_input_ids(sample['input_ids'])
        unique_genes.update(decoded_genes)
    return unique_genes

# Split dataset into chunks
def split_into_chunks(dataset, chunk_size):
    num_chunks = math.ceil(len(dataset) / chunk_size)
    return [dataset[i * chunk_size: (i + 1) * chunk_size] for i in range(num_chunks)]

#%%
# Load the Genecorpus dataset
genecorpus_data = load_from_disk(genecorpus_path)

# Load the token dictionary
with open(token_dict_path, "rb") as f:
    token_dict = pickle.load(f)

# Reverse the mapping: token ID -> Ensembl gene ID
id_to_gene = {v: k for k, v in token_dict.items()}


#%% Define chunk size and split the dataset
chunk_size = 100000  # Adjust based on your memory and processing power
chunks = split_into_chunks(genecorpus_data, chunk_size)

# Process chunks in parallel
with Pool() as pool:
    results = list(tqdm(pool.imap(process_chunk, chunks), total=len(chunks), desc="Processing chunks"))

# Combine results from all chunks
unique_genes = set().union(*results)
print(f"Number of unique genes from Genecorpus: {len(unique_genes)}")

#%%
# Load Xenium dataset (assuming it contains gene names in Ensembl format)
xenium_data = load_from_disk(xenium_path)  # Adjust this path as needed
xenium_genes = set(xenium_data['gene_names'])  # Replace 'gene_names' with actual column name

#%%
# Find overlapping genes
overlapping_genes = unique_genes.intersection(xenium_genes)
print(f"Number of overlapping genes: {len(overlapping_genes)}")

# Optionally save results
with open("unique_genes_genecorpus.pkl", "wb") as f:
    pickle.dump(unique_genes, f)

with open("overlapping_genes.pkl", "wb") as f:
    pickle.dump(overlapping_genes, f)


# %%
