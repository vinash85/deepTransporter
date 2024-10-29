import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
from stellargraph import StellarGraph

# Function to parse the RHEA data


def parse_rhea_data(rhea_data):
    reactions = []
    current_entry = None
    for line in rhea_data.split('\n'):
        line = line.strip()
        if line.startswith('ENTRY'):
            current_entry = line.split()[1]
        elif line.startswith('EQUATION'):
            equation = line.split('EQUATION')[1].strip()
            reactions.append((current_entry, equation))
    return reactions

# Function to create edges from RHEA reactions


def create_edges_from_reactions(reactions):
    edges = []
    for entry, equation in reactions:
        if '=' in equation:
            substrates, products = equation.split('=')
            substrates = substrates.strip().split(' + ')
            products = products.strip().split(' + ')
            for s in substrates:
                for p in products:
                    edges.append((s.strip(), p.strip()))
                    edges.append((p.strip(), s.strip()))
        elif '=>' in equation:
            substrates, products = equation.split('=>')
            substrates = substrates.strip().split(' + ')
            products = products.strip().split(' + ')
            for s in substrates:
                for p in products:
                    edges.append((s.strip(), p.strip()))
        elif '<=>' in equation:
            substrates, products = equation.split('<=>')
            substrates = substrates.strip().split(' + ')
            products = products.strip().split(' + ')
            for s in substrates:
                for p in products:
                    edges.append((s.strip(), p.strip()))
                    edges.append((p.strip(), s.strip()))
    return edges


def remove_leading_number(input_string):
    parts = input_string.split(' ', 1)
    if parts[0].isdigit():
        return parts[1]
    return input_string


# Function to clean node IDs
def clean_node_ids(edges):
    cleaned_edges = []
    for source, target in edges:
        cleaned_source = source.replace('>', '').replace('<', '').strip()
        cleaned_source = remove_leading_number(cleaned_source)
        cleaned_target = target.replace('>', '').replace('<', '').strip()
        cleaned_target = remove_leading_number(cleaned_target)
        cleaned_edges.append((cleaned_source, cleaned_target))
        # print(cleaned_source, cleaned_target)
    return cleaned_edges


# Load RHEA data from a text file
with open('/data_link/servilla/DT_HGNN/Edges/rhea-reactions.txt', 'r') as file:
    rhea_data = file.read()

# Parse the RHEA data
rhea_reactions = parse_rhea_data(rhea_data)

# Create edges from parsed RHEA reactions
rhea_edges = create_edges_from_reactions(rhea_reactions)

# Clean the node IDs in the edges
cleaned_rhea_edges = clean_node_ids(rhea_edges)

# Convert edges to DataFrame
df_rhea_edges = pd.DataFrame(cleaned_rhea_edges, columns=['source', 'target'])

# Load ChEBI data from a CSV file
chebi_data = pd.read_csv(
    '/data_link/servilla/DT_HGNN/Nodes/ChEBI_embeddings.csv', index_col=0)

# Ensure the data types are correct
df_node_data = chebi_data.apply(pd.to_numeric, errors='coerce')

# Check if there are any NaN values and handle them
if df_node_data.isnull().values.any():
    df_node_data = df_node_data.fillna(0)

# Filter edges to include only those with source and target nodes present in chebi_data
valid_chebi_ids = set(df_node_data.index)
filtered_edges = df_rhea_edges[
    (df_rhea_edges['source'].isin(valid_chebi_ids)) &
    (df_rhea_edges['target'].isin(valid_chebi_ids))
]

# Create a StellarGraph object
graph = StellarGraph(nodes=df_node_data, edges=filtered_edges)

# Display the graph information
print(graph.info())

print(filtered_edges)


# # To visualize the graph (optional)
# nx_graph = graph.to_networkx()
# pos = nx.spring_layout(nx_graph)
# nx.draw(nx_graph, pos, with_labels=True, node_size=700,
#         node_color='lightblue', font_size=10, font_weight='bold')
# plt.show()
