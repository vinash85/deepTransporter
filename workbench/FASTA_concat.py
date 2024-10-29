import pandas as pd

FASTA_df = pd.read_csv(
    '/data_link/servilla/DT_HGNN/data/Embeddings/FASTA_emb_571609_CF.csv')
UniProt_KD_df = pd.read_csv(
    '/data_link/servilla/DT_HGNN/data/Embeddings/UniProt_KD_emb_571609.csv')

concatenated_embeddings = pd.concat(
    [FASTA_df, UniProt_KD_df], axis=1)
concatenated_embeddings.to_csv(
    '/data_link/servilla/DT_HGNN/Nodes/UniProt_embeddings.csv', index=True)
print(concatenated_embeddings)
