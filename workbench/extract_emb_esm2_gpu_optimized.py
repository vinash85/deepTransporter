import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm

df = pd.read_csv('/home/miservilla/esm2/571609_FASTA_sequences.tsv', sep='\t')

model_name = "facebook/esm2_t33_650M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def tokenize_sequence(sequence):
    tokens = tokenizer(sequence, return_tensors='pt',
                       padding=True, truncation=True, max_length=1024)
    return tokens


def generate_embeddings(sequence):
    tokens = tokenize_sequence(sequence)
    tokens = {key: val.to(device)
              for key, val in tokens.items()}  # Move tokens to GPU
    with torch.no_grad():
        outputs = model(**tokens)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings


batch_size = 32  # Adjust based on GPU memory
all_embeddings = []

for i in tqdm(range(0, len(df), batch_size)):
    batch_sequences = df['Sequence'][i:i+batch_size].tolist()
    batch_embeddings = [generate_embeddings(seq) for seq in batch_sequences]
    all_embeddings.extend(batch_embeddings)

embeddings_2d = np.vstack(all_embeddings)

embeddings_df = pd.DataFrame(embeddings_2d, index=df['ID'])

embeddings_df.to_csv('FASTA_emb_571609.csv')

print(embeddings_df)
