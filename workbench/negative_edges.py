import pandas as pd
import random
import multiprocessing as mp

# Load node features
s_df = pd.read_csv('/data/servilla/DT_HGNN/Nodes/s_emb_183.csv', index_col=0)
proteins_df = pd.read_csv('/data/servilla/DT_HGNN/Nodes/p_emb_filtered.csv', index_col=0)

# Load edges
ppi_df = pd.read_csv('/data/servilla/DT_HGNN/Edges/ppi_edges_6663523.csv', index_col=0)
ssi_df = pd.read_csv('/data/servilla/DT_HGNN/Edges/rhea_edges_2177.csv', index_col=0)
tp_s_df = pd.read_csv('/data/servilla/DT_HGNN/Edges/trans_sub_edges_13340.csv', index_col=0)

protein_ids = list(proteins_df.index)
substrate_ids = list(s_df.index)

def generate_negative_edges_chunk(chunk, possible_sources, possible_targets, existing_edges):
    negative_edges = []
    for _ in chunk:
        while True:
            source = random.choice(possible_sources)
            target = random.choice(possible_targets)
            if (source, target) not in existing_edges and (target, source) not in existing_edges:
                negative_edges.append((source, target))
                break
    return negative_edges

def generate_negative_edges(df, possible_sources, possible_targets, num_cores, chunk_size=10000):
    existing_edges = set(zip(df['source'], df['target']))

    # Create chunks
    chunks = [range(i, min(i + chunk_size, len(df))) for i in range(0, len(df), chunk_size)]
    
    with mp.Pool(num_cores) as pool:
        results = pool.starmap(
            generate_negative_edges_chunk,
            [(chunk, possible_sources, possible_targets, existing_edges) for chunk in chunks]
        )

    # Flatten the list of lists
    negative_edges = [edge for sublist in results for edge in sublist]
    negative_df = pd.DataFrame(negative_edges, columns=['source', 'target'])
    return negative_df

# Set number of cores
num_cores = 64

# Generate negative edges for PPI
ppi_neg_df = generate_negative_edges(ppi_df, protein_ids, protein_ids, num_cores)
ppi_neg_df.to_csv('/data/servilla/DT_HGNN/Edges/negative_ppi_edges.csv')

# Generate negative edges for SSI
ssi_neg_df = generate_negative_edges(ssi_df, substrate_ids, substrate_ids, num_cores)
ssi_neg_df.to_csv('/data/servilla/DT_HGNN/Edges/negative_ssi_edges.csv')

# Generate negative edges for TP_S
tp_s_neg_df = generate_negative_edges(tp_s_df, protein_ids, substrate_ids, num_cores)
tp_s_neg_df.to_csv('/data/servilla/DT_HGNN/Edges/negative_tp_s_edges.csv')
