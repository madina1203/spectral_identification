import os
import pandas as pd


def process_hcd_energy_column(mdv_folder_path):
    # Traverse the directory structure to locate each CSV file
    for root, _, files in os.walk(mdv_folder_path):
        for file in files:
            if file.endswith('_processed_annotated.csv'):
                file_path = os.path.join(root, file)
                # Load the CSV file
                df = pd.read_csv(file_path)

                # Check if the file has more than one row
                if len(df) > 1 and 'HCD Energy V' in df.columns:
                    # Get the index of 'HCD Energy V' column
                    col_index = df.columns.get_loc('HCD Energy V')

                    # Split the values in 'HCD Energy V' into three columns
                    hcd_values = df['HCD Energy V'].apply(
                        lambda x: [float(i) for i in str(x).split(',')] if ',' in str(x) else [float(x)] * 3
                    )
                    # Expand the list into separate columns
                    hcd_df = pd.DataFrame(hcd_values.tolist(),
                                          columns=['HCD Energy V(1)', 'HCD Energy V(2)', 'HCD Energy V(3)'])

                    # Drop the original 'HCD Energy V' column
                    df = df.drop(columns=['HCD Energy V'])

                    # Insert new columns at the original position
                    for i, col_name in enumerate(['HCD Energy V(1)', 'HCD Energy V(2)', 'HCD Energy V(3)']):
                        df.insert(col_index + i, col_name, hcd_df[col_name])

                    # Save the updated dataframe back to the file
                    df.to_csv(file_path, index=False)
                    print(f"Processed {file_path}")
                else:
                    print(f"Skipped {file_path} (only one row or missing 'HCD Energy V')")



# Use the function
process_hcd_energy_column('/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/MSV_Files')
