from src.transformers.CustomDataset import MassSpecDataset

# Instantiate the dataset
dataset = MassSpecDataset(
    mzml_file='/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/MSV_Files/MSV000095364/raw/preserv_etoh_blank_ID_03.mzML',
    csv_file='/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/MSV_Files/MSV000095364/raw/preserv_etoh_blank_ID_03_processed_annotated.csv'
)

# Check the number of MS1 and MS2 scans loaded
print("Number of scans loaded from mzML file:", len(dataset.scan_list))
print("Number of MS2 entries loaded from CSV file:", len(dataset.ms2_data))
# Print sample scan numbers from mzML file
print("Sample scan numbers from mzML file:")
for scan in dataset.scan_list[118:129]:
    print("mzML scan number:", scan['scan_number'])

# Print sample scan numbers from CSV file
print("\nSample scan numbers from CSV file:")
print(dataset.ms2_data['Scan'].head(10).values)
# Check if any pairs were created
if len(dataset.data_pairs) == 0:
    print("No data pairs found. Check if the MS2 scan numbers match between the mzML and CSV files.")
else:
    # Print the first 5 data pairs if available
    print("First 5 data pairs in the dataset:")
    for i in range(min(5, len(dataset.data_pairs))):
        data_pair = dataset[i]
        print(f"Data Pair {i + 1}:")
        print("MS1 Scan Number:", data_pair['ms1_scan_number'])
        print("MS2 Scan Number:", data_pair['ms2_scan_number'])
        print("m/z Array:", data_pair['mz_array'])
        print("Intensity Array:", data_pair['intensity_array'])
        print("Number of peaks:", len(data_pair['mz_array']))
        print("Instrument Settings:", data_pair['instrument_settings'])
        print("Label:", data_pair['label'])
        print()  # Blank line for readability
