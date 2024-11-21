import os
import pandas as pd


def analyze_labels(folder_path):
    # Prepare a list to collect data for the output CSV
    results = []

    # Walk through the folder and subfolders
    for root, _, files in os.walk(folder_path):
        for file in files:
            # Check for CSV files ending with '_processed_annotated.csv'
            if file.endswith('_processed_annotated.csv'):
                file_path = os.path.join(root, file)

                # Read the CSV file
                df = pd.read_csv(file_path)

                # Check if the 'label' column exists
                if 'label' in df.columns:
                    # Count label values
                    label_1_count = (df['label'] == 1).sum()
                    total_count = len(df['label'])
                    label_0_count = total_count - label_1_count
                    label_1_percentage = (label_1_count / total_count) * 100 if total_count > 0 else 0

                    # Append the results
                    results.append({
                        'File Name': file,
                        'Labeled as 1 Count': label_1_count,
                        'Labeled as 1 Percentage': f"{label_1_percentage:.2f}%",
                        'Labeled as 0 Count': label_0_count
                    })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Save the results to a new CSV file in the same folder
    output_file = os.path.join(folder_path, 'label_analysis_summary.csv')
    results_df.to_csv(output_file, index=False)

    print(f"Analysis complete. Summary saved to: {output_file}")

# Example usage:
analyze_labels('/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/MSV_Files/MSV000095364/raw')
