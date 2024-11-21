import pandas as pd
import os

def load_tsv(tsv_path):
    # Load the TSV and prepare for matching
    tsv_data = pd.read_csv(tsv_path, sep='\t', usecols=['#Scan#', 'full_CCMS_path'])
    tsv_data['#Scan#'] = tsv_data['#Scan#'].astype(int)  # Ensure scan numbers are integers
    return tsv_data

# def process_csv(csv_path, tsv_data):
#     # Read the CSV file
#     csv_data = pd.read_csv(csv_path)
#     csv_data['Scan'] = csv_data['Scan'].astype(int)  # Convert scan numbers in CSV to integers
#
#     # Extract the base name of the CSV file to match with 'full_CCMS_path'
#     base_name = os.path.basename(csv_path).replace('_processed.csv', '')
#
#     # Find matching entries in the TSV 'full_CCMS_path' and get corresponding scan numbers
#     matching_scans = tsv_data[tsv_data['full_CCMS_path'].str.contains(base_name)]['#Scan#']
#
#     # Label the scans in the CSV data based on whether they match the scans from the TSV
#     csv_data['label'] = csv_data['Scan'].apply(lambda x: 1 if x in matching_scans.values else 0)
#
#     # Save the annotated CSV to a new file
#     new_csv_path = csv_path.replace('.csv', '_annotated.csv')
#     csv_data.to_csv(new_csv_path, index=False)
#     return new_csv_path

def process_csv(csv_path, tsv_data):
    # Read the CSV file
    csv_data = pd.read_csv(csv_path)
    csv_data['Scan'] = csv_data['Scan'].astype(int)  # Convert scan numbers in CSV to integers

    # Extract the base name of the CSV file
    base_name = os.path.basename(csv_path).replace('_processed.csv', '')

    # Build regex to match both 'ccms_peak' and 'peak' paths including the base name
    regex_pattern = f"/(ccms_peak|peak)/.*{base_name}.mzML"

    # Find all matching entries in the TSV 'full_CCMS_path'
    all_matches = tsv_data[tsv_data['full_CCMS_path'].str.contains(regex_pattern, regex=True)]

    # Determine if 'ccms_peak' entries exist
    ccms_peak_matches = all_matches[all_matches['full_CCMS_path'].str.contains('ccms_peak')]

    # Select the appropriate scans based on the available paths
    if not ccms_peak_matches.empty:
        matching_scans = ccms_peak_matches  # Use ccms_peak matches if available
    else:
        # If no ccms_peak matches, fall back to peak or any matches (as a last resort)
        peak_matches = all_matches[all_matches['full_CCMS_path'].str.contains('peak')]
        if not peak_matches.empty:
            matching_scans = peak_matches
        else:
            matching_scans = all_matches  # Use any matches if no specific subpaths are found

    # Extract scan numbers for matching
    matching_scans = matching_scans['#Scan#']

    # Label the scans in the CSV data
    csv_data['label'] = csv_data['Scan'].apply(lambda x: 1 if x in matching_scans.values else 0)

    # Save the annotated CSV to a new file
    new_csv_path = csv_path.replace('.csv', '_annotated.csv')
    csv_data.to_csv(new_csv_path, index=False)
    return new_csv_path


def explore_directories(root_path, tsv_data):
    # Explore directories and process each CSV file
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith('_processed.csv'):
                csv_path = os.path.join(root, file)
                process_csv(csv_path, tsv_data)


# Load the TSV data
tsv_path = '/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/Library_search_results/merged__library_file.tsv'
tsv_data = load_tsv(tsv_path)

# Start exploring the directories
root_path = '/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/MSV_Files'
explore_directories(root_path, tsv_data)
