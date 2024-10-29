import os
import torch
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
import copy
import random
import multiprocessing as mp
from torch_geometric.data import HeteroData
import torch
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Enable CUDA debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


# Load node features
s_df = pd.read_csv('/data/servilla/DT_HGNN/Model/Nodes/s_emb_full_183.csv', index_col=0)  # Substrates CSV file
proteins_df = pd.read_csv('/data/servilla/DT_HGNN/Model/Nodes/p_emb_full_237197.csv', index_col=0)  # Combined proteins CSV file

# # Load edges
# tp_s_df = pd.read_csv('/data/servilla/DT_HGNN/Model/Strict_testing_datasets/STD_Edges/filtered_train_tp_s_edges.csv', index_col=0)
# ppi_df = pd.read_csv('/data/servilla/DT_HGNN/Model/Strict_testing_datasets/STD_Edges/filtered_ppi_edges.csv', index_col=0)
# ssi_df = pd.read_csv('/data/servilla/DT_HGNN/Model/Strict_testing_datasets/STD_Edges/filtered_ssi_edges.csv', index_col=0)

# Inspect and clean the data
def inspect_and_clean(df):
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
    print(f"Non-numeric columns: {non_numeric_columns}")
    if len(non_numeric_columns) > 0:
        df[non_numeric_columns] = df[non_numeric_columns].apply(pd.to_numeric, errors='coerce')
    df = df.fillna(0)
    return df

s_df = inspect_and_clean(s_df)
proteins_df = inspect_and_clean(proteins_df)

# Convert features to numpy arrays
s_features = s_df.values
p_features = proteins_df.values

# Check shapes to ensure correct dimensions
print(f"s_features shape: {s_features.shape}")  # Expected (183, 1536), no KD (212, 768)
print(f"p_features shape: {p_features.shape}")  # Expected (some number, 2048), no KD (571609, 1280)

# Normalize features (normalizes columns to have mean 0 and variance 1)
s_features = (s_features - np.mean(s_features, axis=0)) / np.std(s_features, axis=0)
p_features = (p_features - np.mean(p_features, axis=0)) / np.std(p_features, axis=0)

# Define the transformation layers, changes the number of features 1536 -> 2048
# for substrates and 2048 -> 2048 for proteins. The transform_p layer is useful 
# for transforming the feature representation within the same dimensional space,
#  y = Wx + b.

device = torch.device('cpu')  # Temporarily switch to CPU


transform_s = Linear(1536, 2048).to(device) # Change depending on the number of features
transform_p = Linear(2048, 2048).to(device)

# Apply transformations in batches, this can be useful when dealing with large 
# datasets that may not fit into memory or GPU all at once. 
def transform_in_batches(features, transform_layer, batch_size=10000):
    num_samples = features.shape[0]
    print(f"Number of samples: {num_samples}")
    transformed_features = []
    for i in range(0, num_samples, batch_size):
        batch = features[i:i + batch_size]
        batch_tensor = torch.tensor(batch, dtype=torch.float).to(device)
        transformed_batch = transform_layer(batch_tensor)
        transformed_features.append(transformed_batch.detach().cpu().numpy())  # Use detach() before numpy()
    return np.vstack(transformed_features) # Stack arrays in sequence vertically (row wise)

s_features_transformed = transform_in_batches(s_features, transform_s)
p_features_transformed = transform_in_batches(p_features, transform_p)

# Convert back to tensors
s_features_tensor = torch.tensor(s_features_transformed, dtype=torch.float).to(device)
p_features_tensor = torch.tensor(p_features_transformed, dtype=torch.float).to(device)

# Combine features, vertically stacks features (dim=0) to create a single tensor
all_features = torch.cat([p_features_tensor, s_features_tensor], dim=0)

protein_ids = set(proteins_df.index)
substrate_ids = set(s_df.index)

# # Load negative edges
# ppi_neg_df = pd.read_csv('/data/servilla/DT_HGNN/Model/Edges/Negative_Edges/negative_ppi_edges.csv', index_col=0)
# ssi_neg_df = pd.read_csv('/data/servilla/DT_HGNN/Model/Edges/Negative_Edges/negative_ssi_edges.csv', index_col=0)
# tp_s_neg_df = pd.read_csv('/data/servilla/DT_HGNN/Model/Edges/Negative_Edges/negative_tp_s_edges.csv', index_col=0)

# def concatenate_edges(pos_df, neg_df):
#     # Add a label column to indicate positive (1) or negative (0) edges
#     pos_df['label'] = 1
#     neg_df['label'] = 0

#     # Concatenate the positive and negative edges
#     combined_df = pd.concat([pos_df, neg_df], ignore_index=True)
#     return combined_df

# Load the combined edges
ppi_combined_df = pd.read_csv('/data/servilla/DT_HGNN/Model/Edges/combined_ppi_edges_full.csv')
ssi_combined_df = pd.read_csv('/data/servilla/DT_HGNN/Model/Edges/combined_ssi_edges_full.csv')
tp_s_combined_df = pd.read_csv('/data/servilla/DT_HGNN/Model/Edges/combined_tp_s_edges_full.csv')

def split_data(df, train_size=0.8, val_size=0.1, test_size=0.1):
    # Split into train and temp (80% train, 20% temp)
    train_df, temp_df = train_test_split(df, train_size=train_size, random_state=42)
    
    # Calculate the size for validation and test splits
    val_test_ratio = val_size / (val_size + test_size)  # 50% of temp goes to validation and 50% to test

    # Split temp into validation and test (10% each)
    val_df, test_df = train_test_split(temp_df, train_size=val_test_ratio, random_state=42)
    
    return train_df, val_df, test_df

# Split data for each edge type
ppi_train_df, ppi_val_df, ppi_test_df = split_data(ppi_combined_df)
ssi_train_df, ssi_val_df, ssi_test_df = split_data(ssi_combined_df)
tp_s_train_df, tp_s_val_df, tp_s_test_df = split_data(tp_s_combined_df)

# Create separate mappings
protein_mapping = {node_id: i for i, node_id in enumerate(proteins_df.index)}
substrate_mapping = {node_id: i for i, node_id in enumerate(s_df.index)}

# Helper function to apply the correct mapping
def apply_correct_mapping(df, source_mapping, target_mapping):
    df['source'] = df['source'].map(source_mapping)
    df['target'] = df['target'].map(target_mapping)
    df.dropna(inplace=True)
    return df

# Apply the correct mappings
tp_s_train_df = apply_correct_mapping(tp_s_train_df, protein_mapping, substrate_mapping)
tp_s_val_df = apply_correct_mapping(tp_s_val_df, protein_mapping, substrate_mapping)
tp_s_test_df = apply_correct_mapping(tp_s_test_df, protein_mapping, substrate_mapping)

ppi_train_df = apply_correct_mapping(ppi_train_df, protein_mapping, protein_mapping)
ppi_val_df = apply_correct_mapping(ppi_val_df, protein_mapping, protein_mapping)
ppi_test_df = apply_correct_mapping(ppi_test_df, protein_mapping, protein_mapping)

ssi_train_df = apply_correct_mapping(ssi_train_df, substrate_mapping, substrate_mapping)
ssi_val_df = apply_correct_mapping(ssi_val_df, substrate_mapping, substrate_mapping)
ssi_test_df = apply_correct_mapping(ssi_test_df, substrate_mapping, substrate_mapping)

# Create edge index tensors
train_edges_tp_s = torch.tensor(tp_s_train_df[['source', 'target']].values.T, dtype=torch.long)
val_edges_tp_s = torch.tensor(tp_s_val_df[['source', 'target']].values.T, dtype=torch.long)
test_edges_tp_s = torch.tensor(tp_s_test_df[['source', 'target']].values.T, dtype=torch.long)

train_edges_ppi = torch.tensor(ppi_train_df[['source', 'target']].values.T, dtype=torch.long)
val_edges_ppi = torch.tensor(ppi_val_df[['source', 'target']].values.T, dtype=torch.long)
test_edges_ppi = torch.tensor(ppi_test_df[['source', 'target']].values.T, dtype=torch.long)

train_edges_ssi = torch.tensor(ssi_train_df[['source', 'target']].values.T, dtype=torch.long)
val_edges_ssi = torch.tensor(ssi_val_df[['source', 'target']].values.T, dtype=torch.long)
test_edges_ssi = torch.tensor(ssi_test_df[['source', 'target']].values.T, dtype=torch.long)

# Convert the labels to tensors
train_labels_tp_s = torch.tensor(tp_s_train_df['label'].values, dtype=torch.float)
val_labels_tp_s = torch.tensor(tp_s_val_df['label'].values, dtype=torch.float)
test_labels_tp_s = torch.tensor(tp_s_test_df['label'].values, dtype=torch.float)

train_labels_ppi = torch.tensor(ppi_train_df['label'].values, dtype=torch.float)
val_labels_ppi = torch.tensor(ppi_val_df['label'].values, dtype=torch.float)
test_labels_ppi = torch.tensor(ppi_test_df['label'].values, dtype=torch.float)

train_labels_ssi = torch.tensor(ssi_train_df['label'].values, dtype=torch.float)
val_labels_ssi = torch.tensor(ssi_val_df['label'].values, dtype=torch.float)
test_labels_ssi = torch.tensor(ssi_test_df['label'].values, dtype=torch.float)

data = HeteroData()

# Assign node features
data['protein'].x = p_features_tensor
data['substrate'].x = s_features_tensor

# Assign training edges
data['protein', 'interacts_with', 'substrate'].edge_index = train_edges_tp_s
data['protein', 'interacts_with', 'protein'].edge_index = train_edges_ppi
data['substrate', 'interacts_with', 'substrate'].edge_index = train_edges_ssi



# Initialize the model
class GCNLinkPredictor(nn.Module):
    def __init__(self, protein_dim, substrate_dim, hidden_channels):
        super(GCNLinkPredictor, self).__init__()
        self.protein_conv1 = GCNConv(protein_dim, hidden_channels)
        self.substrate_conv1 = GCNConv(substrate_dim, hidden_channels)
        self.protein_conv2 = GCNConv(hidden_channels, hidden_channels)
        self.substrate_conv2 = GCNConv(hidden_channels, hidden_channels)
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def encode(self, x_dict, edge_index_dict):
        z_protein = self.protein_conv1(x_dict['protein'], edge_index_dict[('protein', 'interacts_with', 'protein')])
        z_substrate = self.substrate_conv1(x_dict['substrate'], edge_index_dict[('substrate', 'interacts_with', 'substrate')])
        return z_protein, z_substrate

    def forward(self, x_dict, edge_index_dict, edges):
        z_protein, z_substrate = self.encode(x_dict, edge_index_dict)
        z_combined = torch.cat([z_protein[edges[0]], z_substrate[edges[1]]], dim=-1)
        return self.link_predictor(z_combined).squeeze()

# Initialize the model
model = GCNLinkPredictor(protein_dim=2048, substrate_dim=2048, hidden_channels=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.BCEWithLogitsLoss()

# Early stopping parameters
patience = 10  # Number of epochs to wait before stopping if no improvement
best_val_loss = float('inf')
epochs_without_improvement = 0

# Learning Rate Scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4)


# Assuming `data` contains x_dict and edge_index_dict
x_dict = data.x_dict
edge_index_dict = data.edge_index_dict


# Update the train, validate, and test functions to return predictions
def train(x_dict, edge_index_dict, train_edges_tp_s, train_labels_tp_s): 
    model.train()
    optimizer.zero_grad()
    out = model(x_dict, edge_index_dict, train_edges_tp_s)
    loss = criterion(out, train_labels_tp_s)
    loss.backward()
    optimizer.step()

    return loss.item(), out.detach()

def validate():
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict, val_edges_tp_s)
        loss = criterion(out, val_labels_tp_s)
    return loss.item(), out

def test():
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict, test_edges_tp_s)
        loss = criterion(out, test_labels_tp_s)
    return loss.item(), out

# Calculate additional metrics
def calculate_metrics(labels, preds):
    preds = torch.sigmoid(preds).cpu().numpy()
    preds_binary = (preds > 0.5).astype(int)
    labels = labels.cpu().numpy()

    accuracy = accuracy_score(labels, preds_binary)
    precision = precision_score(labels, preds_binary)
    recall = recall_score(labels, preds_binary)
    f1 = f1_score(labels, preds_binary)
    auc = roc_auc_score(labels, preds)

    return accuracy, precision, recall, f1, auc

# Modify the training loop to include metric calculation and visualization
train_losses = []
val_losses = []
val_accuracies = []

# Training loop
epochs = 800
for epoch in range(epochs):
    # Training step
    train_loss, train_preds = train(x_dict, edge_index_dict, train_edges_tp_s, train_labels_tp_s)
    # Validation step
    val_loss, val_preds = validate()

    # Store losses
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Calculate validation metrics
    accuracy, precision, recall, f1, auc = calculate_metrics(val_labels_tp_s, val_preds)
    val_accuracies.append(accuracy)

    # Print metrics
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, "
          f"Val Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
          f"F1: {f1:.4f}, AUC: {auc:.4f}, LR: {scheduler.get_last_lr()[0]}")

    # Step the LR scheduler
    scheduler.step(val_loss)

    # Check for early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), '/data/servilla/DT_HGNN/data/Models_saves/best_model.pth')  # Save the best model
    else:
        epochs_without_improvement += 1
    
    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Load the best model
model.load_state_dict(torch.load('/data/servilla/DT_HGNN/data/Models_saves/best_model.pth'))

# Testing
test_loss, test_preds = test()
test_accuracy, test_precision, test_recall, test_f1, test_auc = calculate_metrics(test_labels_tp_s, test_preds)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")

# Step 4: Plot loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curves")
plt.legend()
plt.show()

# Step 5: Plot validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.legend()
plt.show()
