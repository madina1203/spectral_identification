import pandas as pd
import os


def load_tsv(tsv_path):
    """
    Loads the TSV file and prepares it for matching.
    It now includes the 'Compound_Name' column.
    """
    # Define the columns to use from the TSV file.
    columns_to_use = ['#Scan#', 'full_CCMS_path', 'Compound_Name']

    # Load the TSV data using only the specified columns.
    tsv_data = pd.read_csv(tsv_path, sep='\t', usecols=columns_to_use)

    # Ensure scan numbers are integers for accurate matching.
    tsv_data['#Scan#'] = tsv_data['#Scan#'].astype(int)

    # Drop rows where Compound_Name is missing, as they cannot be used for annotation.
    tsv_data.dropna(subset=['Compound_Name'], inplace=True)

    return tsv_data


def process_csv(csv_path, tsv_data):
    """
    Processes a single CSV file, matches scans with the TSV data,
    and adds the corresponding 'Compound_Name'. If no match is found,
    it adds an empty 'Compound_Name' column.
    """
    print(f"Processing file: {csv_path}")

    # Read the CSV file into a pandas DataFrame.
    csv_data = pd.read_csv(csv_path)

    # NEW: Define the new file path for the output.
    new_csv_path = csv_path.replace('_processed_annotated.csv', '_annotated_compound.csv')

    # NEW: Check if the 'Compound_Name' column already exists in the source file.
    if 'Compound_Name' in csv_data.columns:
        print(f"  'Compound_Name' column already exists. Saving file as is.")
        # If it exists, just save the file with the new name and stop processing.
        csv_data.to_csv(new_csv_path, index=False)
        print(f"  Saved file to: {new_csv_path}\n")
        return new_csv_path

    # Ensure the 'Scan' column in the CSV is of integer type.
    csv_data['Scan'] = csv_data['Scan'].astype(int)

    # Extract the base name of the CSV file (e.g., 'sample_name') to find its
    # corresponding path in the TSV file.
    base_name = os.path.basename(csv_path).replace('_processed_annotated.csv', '')

    # Construct a regular expression to find matching paths in the TSV.
    # This pattern looks for paths containing '/ccms_peak/' or '/peak/'
    # followed by the base name of the file.
    regex_pattern = f"/(ccms_peak|peak)/.*{base_name}.mzML"

    # Filter the TSV data to find all rows with a path matching the pattern.
    all_matches = tsv_data[tsv_data['full_CCMS_path'].str.contains(regex_pattern, regex=True, na=False)]

    # --- Logic to select the best matching data from the TSV ---
    # We prioritize 'ccms_peak' paths as they are considered more specific.
    ccms_peak_matches = all_matches[all_matches['full_CCMS_path'].str.contains('ccms_peak')]

    if not ccms_peak_matches.empty:
        # If 'ccms_peak' matches exist, use them.
        final_matches = ccms_peak_matches
        print(f"  Found {len(final_matches)} matches in 'ccms_peak' for {base_name}")
    else:
        # If not, fall back to any path containing 'peak'.
        peak_matches = all_matches[all_matches['full_CCMS_path'].str.contains('peak')]
        if not peak_matches.empty:
            final_matches = peak_matches
            print(f"  No 'ccms_peak' matches. Found {len(final_matches)} matches in 'peak' for {base_name}")
        else:
            # If neither is found, use any match found by the initial regex.
            final_matches = all_matches
            print(f"  No specific path matches. Found {len(final_matches)} general matches for {base_name}")

    # NEW: Modified logic to handle cases with and without matches.
    if final_matches.empty:
        print(f"  No matching scans found in TSV for {base_name}. Adding empty 'Compound_Name' column.")
        # If no matches are found, create the 'Compound_Name' column and fill it with empty values.
        csv_data['Compound_Name'] = pd.NA
        annotated_csv = csv_data
    else:
        # If matches are found, proceed with the original merge logic.
        # We only need the scan number and compound name for the mapping.
        # We also drop duplicates to ensure each scan number maps to one compound name.
        scan_to_compound_map = final_matches[['#Scan#', 'Compound_Name']].drop_duplicates(subset=['#Scan#'])

        # Merge the original CSV data with our scan-to-compound map.
        # A 'left' merge ensures all rows from the original CSV are kept.
        annotated_csv = pd.merge(
            csv_data,
            scan_to_compound_map,
            left_on='Scan',
            right_on='#Scan#',
            how='left'
        )
        # The merge operation adds the '#Scan#' column, which we can drop.
        annotated_csv = annotated_csv.drop(columns=['#Scan#'])

    # Save the newly annotated DataFrame to a new CSV file.
    annotated_csv.to_csv(new_csv_path, index=False)
    print(f"  Saved annotated file to: {new_csv_path}\n")

    return new_csv_path


def explore_directories(root_path, tsv_data):
    """
    Recursively walks through directories, finds unprocessed CSV files,
    and processes them. It skips files that have already been annotated.
    """
    # os.walk generates the file names in a directory tree.
    for root, dirs, files in os.walk(root_path):
        for file in files:
            # Check if the file is a processed CSV we need to handle.
            if file.endswith('_processed_annotated.csv'):
                csv_path = os.path.join(root, file)

                # NEW: Define the expected output path.
                new_csv_path = csv_path.replace('_processed_annotated.csv', '_annotated_compound.csv')

                # NEW: Check if the output file already exists.
                if os.path.exists(new_csv_path):
                    print(f"Skipping because output already exists: {new_csv_path}")
                    continue  # If it exists, skip to the next file.

                # If the output does not exist, process the file.
                process_csv(csv_path, tsv_data)


# --- Main execution block ---

# IMPORTANT: Replace these paths with the actual paths on your system.

tsv_path = '/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/Library_search_results/merged_library_file.tsv'
root_path = '/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/MSV_Files'

# Check if the provided paths exist before starting.
if not os.path.isfile(tsv_path):
    print(f"Error: TSV file not found at '{tsv_path}'")
elif not os.path.isdir(root_path):
    print(f"Error: Root directory not found at '{root_path}'")
else:
    # Load the main TSV data once.
    print("Loading TSV data...")
    tsv_data = load_tsv(tsv_path)
    print("TSV data loaded. Starting directory exploration...")

    # Start the process of exploring directories and annotating files.
    explore_directories(root_path, tsv_data)

    print("Processing complete.")
