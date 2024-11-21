import pandas as pd
import os

def merge_tsv_files(folder_path, output_file):
    # List to store dataframes
    dfs = []

    # Loop through each file in the directory
    for file in sorted(os.listdir(folder_path)):
        if file.endswith('.tsv'):
            # Full path to the file
            file_path = os.path.join(folder_path, file)
            # Read the TSV file and append to the list
            df = pd.read_csv(file_path, sep='\t')
            dfs.append(df)

    # Concatenate all DataFrames, assuming the same headers/columns
    merged_df = pd.concat(dfs, ignore_index=True)

    # Save the concatenated dataframe to a TSV file, without header duplication
    merged_df.to_csv(output_file, sep='\t', index=False)
    print(f"Merged file saved as {output_file}")

# Specify the folder containing TSV files and the output file name
folder_path = '/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/Library_search_results'
output_file = '/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/Library_search_results/merged__library_file.tsv'

merge_tsv_files(folder_path, output_file)
