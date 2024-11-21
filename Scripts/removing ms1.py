import os
import pandas as pd


def remove_msorder_rows(root_path):
    # Traverse the directory tree starting from the root_path
    for root, dirs, files in os.walk(root_path):
        for file in files:
            # Check if the file is a CSV and ends with '_processed_annotated.csv'
            if file.endswith('_processed_annotated.csv'):
                csv_path = os.path.join(root, file)

                # Load the CSV file into a pandas DataFrame
                csv_data = pd.read_csv(csv_path)

                # Check if 'MSOrder' column exists
                if 'MSOrder' in csv_data.columns:
                    # Remove rows where MSOrder is 1
                    csv_data_filtered = csv_data[csv_data['MSOrder'] != 1]

                    # Save the updated DataFrame back to the same CSV file (overwrite)
                    csv_data_filtered.to_csv(csv_path, index=False)
                    print(f"Updated file: {csv_path}")
                else:
                    print(f"MSOrder column not found in: {csv_path}")


# Example usage: provide the root directory containing MSV folders
remove_msorder_rows('/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/MSV_Files')
