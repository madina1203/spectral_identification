import pandas as pd
from collections import defaultdict
import os

def find_duplicate_paths(tsv_path):
    # Load the TSV file
    data = pd.read_csv(tsv_path, sep='\t', usecols=['full_CCMS_path'])

    # Dictionary to hold the paths for each file name
    file_paths = defaultdict(set)

    # Extract file name and path from each entry and store in the dictionary
    for full_path in data['full_CCMS_path']:
        path, filename = os.path.split(full_path)
        file_paths[filename].add(path)

    # Check for files with more than one distinct path and print them
    for filename, paths in file_paths.items():
        if len(paths) > 1:
            print(f"File Name: {filename}")
            for path in paths:
                print(f"\tPath: {path}")

# Path to the TSV file
tsv_path = '/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/Library_search_results/merged__library_file.tsv'
find_duplicate_paths(tsv_path)
