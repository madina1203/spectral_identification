import pandas as pd
import os

def process_csv_files(root_directory):
    # Standardized column names
    column_mappings = {
        "FT Resolution": "Orbitrap Resolution",
        "Orbitrap Resolution": "Orbitrap Resolution",
        "AGC Target": "AGC Target",
        "TargetAGC": "AGC Target",
        "HCD Energy [eV]": "HCD Energy V",
        "HCD Energy V": "HCD Energy V",
        "HCD Energy eV": "HCD Energy V",
        "NrLockMasses": "Number of Lock Masses",
        "Number of Lock Masses": "Number of Lock Masses",
        "LM Correction (ppm)": "LM m/z-Correction (ppm)",
        "LM m/z-Correction (ppm)": "LM m/z-Correction (ppm)"
    }

    # Columns that are always required (without alternatives)
    base_columns = [
        "Scan", "MSOrder", "Polarity", "RT [min]", "LowMass", "HighMass",
        "TIC", "BasePeakPosition", "BasePeakIntensity", "Charge State",  "Monoisotopic M/Z",
        "Ion Injection Time (ms)",  "MS2 Isolation Width",  "Conversion Parameter C",
         "LM Search Window (ppm)", "Number of LM Found", "Mild Trapping Mode",
        "Source CID eV", "SelectedMass1", "Activation1", "Energy1", "Orbitrap Resolution", "AGC Target",  "HCD Energy V",
        "Number of Lock Masses", "LM m/z-Correction (ppm)", "label"



    ]

    # Traverse all directories and subdirectories
    for root, dirs, files in os.walk(root_directory):
        for filename in files:
            if filename.endswith('.csv'):
                csv_path = os.path.join(root, filename)
                df = pd.read_csv(csv_path)

                # Check if 'Ionization' column contains 'ESI'
                if 'Ionization' in df.columns and df['Ionization'].eq('ESI').any():
                    new_df = pd.DataFrame()

                    # Handle base columns
                    for col in base_columns:
                        if col in df.columns:
                            new_df[col] = df[col]

                    # Handle alternative columns
                    for original, standardized in column_mappings.items():
                        if original in df.columns:
                            new_df[standardized] = df[original]

                    # Save the new DataFrame to a CSV file
                    new_csv_path = csv_path.replace('.csv', '_processed.csv')
                    new_df.to_csv(new_csv_path, index=False)
                    # print(f"Processed and saved: {new_csv_path}")
                else:
                    # Print details if Ionization is not 'ESI'
                    unique_ionization = df['Ionization'].unique()
                    print(f"File {csv_path} skipped. Ionization types found: {unique_ionization}")

#merging all csvs into 1
def merge_csv_files(msv_dir, output_csv):
    # List to store DataFrames from each CSV
    all_dataframes = []

    # Walk through the directory recursively
    for root, dirs, files in os.walk(msv_dir):
        for file in files:
            # Only consider CSV files ending with '_processed_annotated.csv'
            if file.endswith('_processed_annotated.csv'):
                csv_path = os.path.join(root, file)
                try:
                    # Read the CSV file
                    df = pd.read_csv(csv_path)

                    # Only process CSV files with more than 1 row
                    if len(df) > 1:
                        all_dataframes.append(df)
                        print(f"Added: {csv_path}")

                except Exception as e:
                    print(f"Error processing file {csv_path}: {e}")

    # Merge all DataFrames into one
    if all_dataframes:
        merged_df = pd.concat(all_dataframes, ignore_index=True)
        num_rows, num_columns = merged_df.shape
        # Save the merged DataFrame into a new CSV file
        merged_df.to_csv(output_csv, index=False)
        print(f"Merged CSV saved to: {output_csv}")
        print(f"Number of rows: {num_rows}")
        print(f"Number of columns: {num_columns}")
    else:
        print("No CSV files with more than 1 row found.")


#transforming merged csv to parquet
def convert_csv_to_parquet(csv_file, parquet_file):
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_file, low_memory=False)

        # Save the DataFrame to a Parquet file
        df.to_parquet(parquet_file, index=False)

        print(f"Successfully converted {csv_file} to {parquet_file}")
    except Exception as e:
        print(f"Error converting file: {e}")

# Assuming 'directory_path' is defined
directory_path = "/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/MSV_Files"
output_file = os.path.join(directory_path, "merged.csv")  # Specify the output file path
csv_file = "/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/MSV_Files/merged.csv"  # Replace with the actual CSV path
parquet_file = "/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/MSV_Files/output_file.parquet"  # Replace with the desired Parquet path
convert_csv_to_parquet(csv_file, parquet_file)
#process_csv_files(directory_path)
#merge_csv_files(directory_path, output_file)