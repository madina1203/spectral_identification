import pandas as pd
import os

def load_tsv(tsv_path):
    # Load the TSV and prepare for matching
    tsv_data = pd.read_csv(tsv_path, sep='\t', usecols=['#Scan#', 'full_CCMS_path'])
    tsv_data['#Scan#'] = tsv_data['#Scan#'].astype(int)  # Ensure scan numbers are integers
    return tsv_data

# def process_csv(csv_path, tsv_data):
#     # Extract the base name of the CSV file to match with 'full_CCMS_path'
#     base_name = os.path.basename(csv_path).replace('.csv', '')
#     # Find matching entries in the TSV 'full_CCMS_path' and get corresponding scan numbers
#     matching_scans = tsv_data[tsv_data['full_CCMS_path'].str.contains(base_name)]['#Scan#']
#     matching_scans_set = set(matching_scans)  # Convert to set for faster lookup
#
#     # Initialize a list to collect processed dataframes
#     processed_chunks = []
#
#     # Read the CSV file in chunks
#     chunksize = 10000  # You can adjust the chunksize based on your memory availability
#     for chunk in pd.read_csv(csv_path, chunksize=chunksize):
#         chunk['Scan'] = chunk['Scan'].astype(int)  # Convert scan numbers in CSV to integers
#         # Label the scans based on whether they match the scans from the TSV
#         chunk['label'] = chunk['Scan'].apply(lambda x: 1 if x in matching_scans_set else 0)
#         processed_chunks.append(chunk)
#
#     # Concatenate all processed chunks into one dataframe
#     final_df = pd.concat(processed_chunks, ignore_index=True)
#
#     # Save the annotated CSV to a new file
#     new_csv_path = csv_path.replace('.csv', '_annotated.csv')
#     final_df.to_csv(new_csv_path, index=False)
#     return new_csv_path

# new function that chooses only ccms_peak scan matches for the cases when there are both ccms_peak and peak
def process_csv(csv_path, tsv_data):
    # Read the CSV file
    csv_data = pd.read_csv(csv_path)
    csv_data['Scan'] = csv_data['Scan'].astype(int)  # Convert scan numbers in CSV to integers

    # Extract the base name of the CSV file to match with 'full_CCMS_path'
    base_name = os.path.basename(csv_path).replace('_processed.csv', '')

    # Define specific MSV folders and their required subpaths for the 'ccms_peak' condition
    required_subpaths = {
        'MSV000087935': 'ccms_peak',
        'MSV000088023': 'ccms_peak',
        'MSV000088226': 'ccms_peak'
    }

    # Check if the CSV path contains any specific MSV folder and adjust the matching criteria accordingly
    match_string = None
    for msv_key, subpath in required_subpaths.items():
        if msv_key in csv_path:
            # Create a match string that specifically looks for the 'ccms_peak' subdirectory
            match_string = f"{msv_key}/{subpath}/.*/{base_name}"
            break

    # If no specific match string was set and the file belongs to other MSVs, use a general match
    if not match_string:
        match_string = f"{base_name}"

    # Find matching entries in the TSV 'full_CCMS_path' using the constructed match string
    matching_scans = tsv_data[tsv_data['full_CCMS_path'].str.contains(match_string)]['#Scan#']

    # Label the scans in the CSV data based on whether they match the scans from the TSV
    csv_data['label'] = csv_data['Scan'].apply(lambda x: 1 if x in matching_scans.values else 0)

    # Save the annotated CSV to a new file
    new_csv_path = csv_path.replace('.csv', '_annotated.csv')
    csv_data.to_csv(new_csv_path, index=False)
    return new_csv_path

def explore_directories(root_path, tsv_data):
    # Explore directories and process each CSV file
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith('.csv'):
                csv_path = os.path.join(root, file)
                process_csv(csv_path, tsv_data)

# Assuming 'tsv_path' and 'root_path' are defined elsewhere
tsv_data = load_tsv(tsv_path)
explore_directories(root_path, tsv_data)
