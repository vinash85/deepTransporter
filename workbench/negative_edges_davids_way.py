import pandas as pd
import numpy as np


edges = pd.read_csv(
    '/data/servilla/DT_HGNN/Model/Edges/Dusty_Edges/tp_s_edges_13340.csv')

# substrates
substrate = edges['target'].value_counts()
substrate_counts = substrate.values
substrate_probs = substrate_counts/substrate_counts.sum()
substrates = substrate.index.values

# proteins
protein = edges['source'].value_counts()
protein_counts = protein.values
protein_probs = protein_counts/protein_counts.sum()
proteins = protein.index.values

# set
positive_pairs = set(zip(edges['source'], edges['target']))


def choose_sample(proteins=proteins,
                  protein_probs=protein_probs,
                  substrates=substrates,
                  substrate_probs=substrate_probs):
    p = np.random.choice(proteins, p=protein_probs)
    s = np.random.choice(substrates, p=substrate_probs)
    return p, s


num_samples = len(edges)
negative_pairs = set()
for i in range(num_samples):
    p, s = choose_sample()
    while ((p, s) in positive_pairs) or ((p, s) in negative_pairs):
        p, s = choose_sample()
    negative_pairs.add((p, s))

negative_pairs = list(negative_pairs)
negative_pairs = pd.DataFrame(negative_pairs, columns=['source', 'target'])
print(negative_pairs)
negative_pairs.to_csv(
    '/data/servilla/DT_HGNN/Model/Edges/Negative_Edges/distributed_negative_tp_s_edges.csv')
