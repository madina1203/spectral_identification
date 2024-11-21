import pandas as pd
import os

# def check_csv_columns(csv_path, required_columns, grouped_columns):
#     # Try to read the CSV file
#     try:
#         data = pd.read_csv(csv_path)
#         # List to store messages about missing columns
#         missing_columns = []
#         # Check for required columns
#         for col in required_columns:
#             if col not in data.columns:
#                 missing_columns.append(col)
#         # Check for grouped columns (where at least one in each group is required)
#         for group in grouped_columns:
#             if not any(col in data.columns for col in group):
#                 missing_columns.append(f"One of {', '.join(group)}")
#         return missing_columns if missing_columns else None
#     except Exception as e:
#         print(f"Error reading {csv_path}: {str(e)}")
#         return None
#
# def explore_and_check_directories(root_path, required_columns, grouped_columns):
#     # Walk through all directories and subdirectories starting from root_path
#     for root, dirs, files in os.walk(root_path):
#         for file in files:
#             if file.endswith('.csv'):
#                 csv_path = os.path.join(root, file)
#                 missing_columns = check_csv_columns(csv_path, required_columns, grouped_columns)
#                 if missing_columns:
#                     print(f"For CSV file {csv_path}, the following columns are not present: {', '.join(missing_columns)}")

# List of required columns
# required_columns = [
#     "TIC", "BasePeakIntensity", "Analyzer", "Conversion Parameter C", "BasePeakPosition",
#     "Polarity", "Charge State", "MSOrder", "Energy1", "Ionization", "SelectedMass1", "Activation1",
#     "Monoisotopic M/Z", "Ion Injection Time (ms)", "RT [min]", "MS2 Isolation Width", "AGC",
#      "Mild Trapping Mode",
#     "Source CID eV", "LM Search Window (ppm)", "LowMass", "HighMass"
# ]

# required_columns = ["HCD Energy [eV]", "HCD Energy V", "HCD Energy", "HCD Energy eV"]
# # Grouped columns where at least one of each group must be present
# grouped_columns = [
#     ["NrLockMasses", "Number of Lock Masses"],
#     ["FT Resolution", "Orbitrap Resolution"],
#      ["AGC Target", "TargetAGC"],
#     ["LM Correction (ppm)", "LM m/z-Correction (ppm)"],
# ]
#
import os
import pandas as pd

import os
import pandas as pd


import os
import pandas as pd

def process_csv_files(msv_dir):
    # Walk through the directory recursively
    for root, dirs, files in os.walk(msv_dir):
        for file in files:
            # Only consider CSV files ending with '_processed_annotated.csv'
            if file.endswith('_processed_annotated.csv'):
                csv_path = os.path.join(root, file)
                try:
                    # Read the CSV file
                    df = pd.read_csv(csv_path)

                    # Check if the required columns are present before making changes
                    columns_to_remove = ['Analyzer', 'Ionization', 'AGC', 'HCD Energy']
                    for col in columns_to_remove:
                        if col in df.columns:
                            df.drop(columns=col, inplace=True)

                    # Replace values in 'Polarity' column
                    if 'Polarity' in df.columns:
                        df['Polarity'] = df['Polarity'].map({'Positive': 1, 'Negative': 0})

                    # Replace values in 'Mild Trapping Mode' column
                    if 'Mild Trapping Mode' in df.columns:
                        df['Mild Trapping Mode'] = df['Mild Trapping Mode'].map({'On': 1, 'Off': 0})

                    # Replace values in 'Activation1' column where the value is 'HCD'
                    if 'Activation1' in df.columns:
                        df['Activation1'] = df['Activation1'].replace('HCD', 1)

                    # Save the modified CSV file back (overwrites the original file)
                    df.to_csv(csv_path, index=False)
                    print(f"Processed file: {csv_path}")

                except Exception as e:
                    print(f"Error processing file {csv_path}: {e}")






#checking activation values
def check_activation_column(msv_dir):
    # Walk through the directory recursively
    for root, dirs, files in os.walk(msv_dir):
        for file in files:
            # Only consider CSV files ending with '_processed_annotated.csv'
            if file.endswith('_processed_annotated.csv'):
                csv_path = os.path.join(root, file)
                try:
                    # Read the CSV file
                    df = pd.read_csv(csv_path)

                    # Check if the 'Activation1' column is present
                    if 'Activation1' in df.columns:
                        # Filter rows where 'Activation1' is not 'HCD'
                        non_hcd_rows = df[df['Activation1'] != 'HCD']

                        # If there are any non-HCD values, print the file name and the corresponding values
                        if not non_hcd_rows.empty:
                            print(f"File: {csv_path}")
                            print(non_hcd_rows[['Activation1']])

                except Exception as e:
                    print(f"Error processing file {csv_path}: {e}")



required_columns = ["Scan", "MSOrder", "Polarity", "RT [min]", "LowMass", "HighMass",
                    "TIC", "BasePeakPosition", "BasePeakIntensity", "Charge State",
                    "Monoisotopic M/Z", "Ion Injection Time (ms)", "MS2 Isolation Width",
                    "Conversion Parameter C", "LM Search Window (ppm)", "Number of LM Found",
                    "Mild Trapping Mode", "Source CID eV", "SelectedMass1", "Activation1",
                    "Energy1", "Orbitrap Resolution", "AGC Target", "HCD Energy V",
                    "Number of Lock Masses", "LM m/z-Correction (ppm)", "label"]

#counting the total number of rows and adding empty columns that are absent to csv files
def count_rows(msv_dir):
    csv_count = 0
    total_rows = 0

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
                        csv_count += 1
                        total_rows += len(df)

                        # Check if any required columns are missing
                        missing_columns = [col for col in required_columns if col not in df.columns]

                        if missing_columns:
                            # Add missing columns in the correct position with empty values
                            for col in missing_columns:
                                insert_index = required_columns.index(col)
                                df.insert(insert_index, col, "")

                            # Save the modified CSV file back (overwrites the original file)
                            df.to_csv(csv_path, index=False)
                            print(f"Modified and processed file: {csv_path}")

                except Exception as e:
                    print(f"Error processing file {csv_path}: {e}")

    # Summary of results
    print(f"Number of CSV files with >1 row: {csv_count}")
    print(f"Total number of rows across all CSV files with >1 row: {total_rows}")




import os
import pandas as pd
#adding file path to the csvs to get the corresponding scans from mzml (with the same name) later
def add_filepath_column(msv_dir):
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
                        # Find the position of the "MSV" part of the path
                        start_index = csv_path.lower().find("msv")

                        if start_index != -1:
                            relative_path = csv_path[start_index:]

                            # Add the file path as a new column at the left
                            df.insert(0, 'Filepath', relative_path)

                            # Save the modified CSV file back (overwrites the original file)
                            df.to_csv(csv_path, index=False)
                            print(f"Processed and added file path for file: {csv_path}")
                        else:
                            print(f"'MSV' not found in path: {csv_path}")

                except Exception as e:
                    print(f"Error processing file {csv_path}: {e}")

root_path = '/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/MSV_Files'
add_filepath_column(root_path)
#count_rows(root_path)
# explore_and_check_directories(root_path, required_columns, grouped_columns)
#process_csv_files(root_path)

#check_activation_column(root_path)