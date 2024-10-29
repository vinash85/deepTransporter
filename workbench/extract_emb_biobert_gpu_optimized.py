import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# Load the DataFrame
df = pd.read_csv(
    '/data_link/servilla/DT_HGNN/data/substrates/ChEBI_SMILES_Definition_filtered.csv')

# Load the BioBERT model and tokenizer
model_name = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Check if GPU is available and move model to GPU if it is
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Function to get embeddings for a given text


# def get_embeddings(texts):
#     inputs = tokenizer(texts, return_tensors='pt',
#                        truncation=True, padding=True, max_length=512)
#     inputs = {key: val.to(device)
#               for key, val in inputs.items()}  # Move inputs to GPU
#     with torch.no_grad():
#         outputs = model(**inputs)
#     # Move output to CPU
#     return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

def get_embeddings(texts):
    # Ensure input is in the correct format
    if isinstance(texts, str):
        texts = [texts]  # Wrap a single string in a list
    elif not isinstance(texts, (list, tuple)):
        raise ValueError(
            "Input should be a string or a list/tuple of strings.")

    inputs = tokenizer(texts, return_tensors='pt',
                       truncation=True, padding=True, max_length=512)
    inputs = {key: val.to(device)
              for key, val in inputs.items()}  # Move inputs to GPU
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()


# Batch processing
batch_size = 32  # Adjust batch size based on GPU memory
all_embeddings = []

for i in tqdm(range(0, len(df), batch_size)):
    batch_texts = df['Definition'][i:i+batch_size].tolist()
    batch_embeddings = get_embeddings(batch_texts)
    all_embeddings.append(batch_embeddings)

# Convert embeddings to a DataFrame with 768 columns
all_embeddings = np.vstack(all_embeddings)
embeddings_df = pd.DataFrame(all_embeddings, columns=[
                             f'{i}' for i in range(768)])

# Add the 'Entry' column to the embeddings DataFrame
embeddings_df.insert(0, 'ChEBI ID', df['ChEBI ID'])

# Save the embeddings DataFrame to a new CSV file
output_path = '/data_link/servilla/DT_HGNN/data/Embeddings/ChEBI_KD_embeddings.csv'
embeddings_df.to_csv(output_path, index=False)

# Display the first few rows of the final DataFrame
print(embeddings_df.head())
