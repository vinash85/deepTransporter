import requests

# Replace 'your_protein_list' with a list of protein identifiers (e.g., UniProt IDs)
protein_list = ['A0A044RE18']

# Choose the desired API endpoint (refer to STRING API documentation for details)
base_url = 'https://string-db.org/api/tsv/interaction_partners?'

# Loop through each protein in the list
for protein in protein_list:
    # Construct the complete URL with the protein ID
    url = base_url + f'identifiers={protein}'

    # Send a GET request using the requests library
    response = requests.get(url)
    print(url)

    # Check for successful response (status code 200)
    if response.status_code == 200:
        # Access the data as a string
        data = response.text
        print(f"Data for protein {protein}:")
        print(data)
    else:
        print(
            f"Error retrieving data for protein {protein}: {response.status_code}")
