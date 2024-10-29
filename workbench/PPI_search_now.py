from tqdm.notebook import tqdm
import pandas as pd
import numpy as np

# Define file paths
uniprot_file = "/data_link/servilla/DT_HGNN/data/PPI_data/Chunks/UniProtID_to_STRINGID.tsv"
ppi_file = "/data_link/servilla/DT_HGNN/data/PPI_data/PPI_results.csv"
output_file = "/data_link/servilla/DT_HGNN/data/PPI_data/PPI_UniProt.csv"

# Load the TSV and CSV files into DataFrames
uniprot_df = pd.read_csv(uniprot_file, sep='\t')
ppi_df = pd.read_csv(ppi_file, sep=',')
ppi_uniprot_df = pd.read_csv(output_file)

# Ensure column names are correct
uniprot_to_column = 'To'
ppi_protein1_column = 'protein1'
ppi_protein2_column = 'protein2'
uniprot_from_column = 'From'

# Rename column in 'ppi_uniprot_df', add column 'protein2'
ppi_uniprot_df = ppi_uniprot_df.rename(columns={'From': 'protein1'})
ppi_uniprot_df['protein2'] = np.nan
# print(ppi_uniprot_df)

# # Save the resulting DataFrame to a new CSV file
# ppi_df.to_csv(output_file, index=False)

# print(f"Matching entities saved to {output_file}")


for i, row in tqdm(ppi_df.iterrows(), total=ppi_df.shape[0]):
    if row[1] in uniprot_df['To'].values:
        ppi_uniprot_df.at[i, 'protein2'] = uniprot_df.loc[uniprot_df['To']
                                                          == row[1], 'From'].values[0]

ppi_uniprot_df.to_csv(output_file, index=False)
