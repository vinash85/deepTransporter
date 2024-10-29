import requests
import csv
import pandas as pd

# Read the list of UniProt IDs from the input CSV file
input_file = '/data_link/servilla/DT_HGNN/workbench/Entry_571609.csv'
output_file = '/data_link/servilla/DT_HGNN/workbench/PPI_list.csv'

# Load the UniProt IDs into a list
with open(input_file, mode='r') as file:
    csv_reader = csv.reader(file)
    list_of_UniProt = list(csv_reader)

# Define the base URL for the STRING API
base_url = 'https://string-db.org/api/tsv/interaction_partners?'

# Initialize a list to store the PPI data
PPI_list = []

# Loop through each protein in the list
for entry in list_of_UniProt:
    UniProt = entry[0]
    # Construct the complete URL with the protein ID
    url = base_url + f'identifiers={UniProt}'

    # Send a GET request using the requests library
    response = requests.get(url)

    # Check for a successful response (status code 200)
    if response.status_code == 200:
        # Access the data as a string
        data = response.text

        # Split the data into lines and prepend each line with the UniProt ID
        lines = data.strip().split('\n')
        for line in lines[1:]:  # Skip the header line from the API response
            PPI_list.append(f'{UniProt}\t{line}')
    else:
        PPI_list.append(UniProt)

# Add the header line at the beginning
header = "UniProt\tstringId_A\tstringId_B\tpreferredName_A\tpreferredName_B\tncbiTaxonId\tscore\tnscore\tfscore\tpscore\tascore\tescore\tdscore\ttscore"
PPI_list.insert(0, header)

# Write the PPI data to the output CSV file
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')
    for line in PPI_list:
        writer.writerow(line.split('\t'))

print("PPI list has been saved to", output_file)
