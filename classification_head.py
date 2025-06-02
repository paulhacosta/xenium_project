#%%
import os 
import pandas as pd 
import scanpy as sc
import numpy as np
import scanpy as sc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %%
cancer = "lung"
ground_truth = "cellvit"  # refined or cellvit
xenium_folder_dict = {"lung": "Xenium_Prime_Human_Lung_Cancer_FFPE_outs",
                      "breast":"Xenium_Prime_Breast_Cancer_FFPE_outs",
                      "lymph_node": "Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_outs",
                      "prostate": "Xenium_Prime_Human_Prostate_FFPE_outs",
                      "skin": "Xenium_Prime_Human_Skin_FFPE_outs",
                      "ovarian": "Xenium_Prime_Ovarian_Cancer_FFPE_outs",
                      "cervical": "Xenium_Prime_Cervical_Cancer_FFPE_outs"
                      }

xenium_folder = xenium_folder_dict[cancer]

embedding_path = f"/rsrch9/home/plm/idso_fa1_pathology/TIER2/paul-xenium/embeddings/public_data/{xenium_folder}/processed_xenium_{ground_truth}.csv"
embeddings = pd.read_csv(embedding_path)

# Drop 'Unnamed: 0' if it exists in the columns
if 'Unnamed: 0' in embeddings.columns:
    embeddings = embeddings.drop(columns='Unnamed: 0')

data_path = f"/rsrch9/home/plm/idso_fa1_pathology/TIER1/paul-xenium/public_data/10x_genomics/{xenium_folder}/preprocessed/fine_tune_{ground_truth}/processed_xenium_data_fine_tune_{ground_truth}.h5ad"

adata = sc.read_h5ad(data_path)
# %%

# Load embeddings and labels
labels = adata.obs['class'].values

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels_encoded, test_size=0.2, stratify=labels_encoded, random_state=42
)

# # Standardize features for SVM
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # ---- Train SVM ----
# print("Training SVM...")
# svm_clf = SVC(kernel='linear', probability=True, random_state=42)
# svm_clf.fit(X_train_scaled, y_train)
# y_pred_svm = svm_clf.predict(X_test_scaled)

# # SVM metrics
# print("SVM Classification Report:")
# print(classification_report(y_test, y_pred_svm, target_names=label_encoder.classes_))

device = torch.device('cuda')

# # Define MLP model
# class MLPClassifier(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(MLPClassifier, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(input_size, 256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, num_classes)
#         )

#     def forward(self, x):
#         return self.network(x)

class MLPClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLPClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# Hyperparameters
input_size = embeddings.shape[1]
num_classes = len(label_encoder.classes_)
batch_size = 64
num_epochs = 100
learning_rate = 0.001

# Cross-validation setup
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

# 5-fold cross-validation
for fold, (train_idx, test_idx) in enumerate(kf.split(embeddings, labels_encoded)):
    print(f"\nFold {fold + 1} / 5")

    # Split data
    X_train, X_test = embeddings.iloc[train_idx].values, embeddings.iloc[test_idx].values
    y_train, y_test = labels_encoded[train_idx], labels_encoded[test_idx]

    # Create DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss, optimizer
    model = MLPClassifier(input_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")

    # Evaluate the model
    model.eval()
    y_pred = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())

    # Compute fold metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Fold {fold + 1} Accuracy: {accuracy:.4f}")
    fold_results.append(accuracy)

# Report average performance across all folds
print("\nCross-Validation Results:")
print(f"  Average Accuracy: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")



# %%
