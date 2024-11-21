#extraction of MS1 scan numbers which have consecutive MS2
import csv

# # Replace 'your_file.csv' with the path to your CSV file
# csv_file = '/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/MSV_Files/MSV000087935/NEG_MSMS_raw/DOM_Interlab-LCMS_Lab26_A_Neg_MS2_rep1_processed_annotated.csv'
#
# # Read the Scan column from the CSV file
# scan_numbers = []
# with open(csv_file, 'r') as csvfile:
#     reader = csv.DictReader(csvfile)
#     for row in reader:
#         # Assuming the 'Scan' column contains integer values
#         scan_numbers.append(int(row['Scan']))
#
# scan_numbers_set = set(scan_numbers)
# min_scan_in_data = min(scan_numbers)
#
# # Adjust min_scan to include the very first possible scan number
# min_scan = max(1, min_scan_in_data - 1)
# max_scan = max(scan_numbers)
#
# # Generate the full range of scan numbers
# all_scan_numbers = set(range(min_scan, max_scan + 1))
#
# # Find the missing scan numbers
# missing_scans = all_scan_numbers - scan_numbers_set
#
# # Find missing scans where the next scan is present
# output = []
# for n in missing_scans:
#     if (n + 1) in scan_numbers_set:
#         output.append(n)
#
# # Sort the output list
# output.sort()
#
# print("Missing Scan Numbers with Next Scan Present:", output)
import re
from pyteomics import mzml

# Replace 'path_to_your_mzml_file.mzML' with the actual path to your mzML file
file_mzml = '/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/MSV_Files/MSV000087935/NEG_MSMS_raw/DOM_Interlab-LCMS_Lab26_A_Neg_MS2_rep1.mzML'


# Function to extract scan number from the 'id' string
def get_scan_number(id_string):
    match = re.search(r'scan=(\d+)', id_string)
    if match:
        return int(match.group(1))
    else:
        # If 'scan=' pattern is not found, try to extract digits at the end of the string
        match = re.search(r'(\d+)$', id_string)
        if match:
            return int(match.group(1))
        else:
            return None  # Unable to extract scan number

# Initialize variables
previous_spectrum = None

# List to store MS1 spectra with consecutive MS2 scans
ms1_spectra_with_consecutive_ms2 = []

with mzml.read(file_mzml) as reader:
    for spectrum in reader:
        ms_level = spectrum.get('ms level')
        id_string = spectrum.get('id')
        scan_number = get_scan_number(id_string)

        # Skip if scan number or ms level is not available
        if scan_number is None or ms_level is None:
            continue

        # Create a dictionary for the current spectrum
        current_spectrum = {
            'ms_level': ms_level,
            'scan_number': scan_number,
            'mz_array': spectrum.get('m/z array'),
            'intensity_array': spectrum.get('intensity array')
        }

        if previous_spectrum is not None:
            # Check if previous spectrum is MS1 and current spectrum is MS2
            if previous_spectrum['ms_level'] == 1 and current_spectrum['ms_level'] == 2:
                # Check if scan numbers are consecutive
                if current_spectrum['scan_number'] == previous_spectrum['scan_number'] + 1:
                    # Found an MS1 scan with a consecutive MS2 scan
                    print(f"MS1 scan number {previous_spectrum['scan_number']} has consecutive MS2 scan number {current_spectrum['scan_number']}")

                    # Access the m/z and intensity arrays from the MS1 spectrum
                    mz_array = previous_spectrum['mz_array']
                    intensity_array = previous_spectrum['intensity_array']

                    # Store the MS1 spectrum data as needed
                    ms1_spectra_with_consecutive_ms2.append({
                        'scan_number': previous_spectrum['scan_number'],
                        'mz_array': mz_array,
                        'intensity_array': intensity_array
                    })

        # Update previous_spectrum for the next iteration
        previous_spectrum = current_spectrum

# Now you have a list of MS1 spectra with consecutive MS2 scans
# You can further process or analyze them as needed
print(f"\nTotal MS1 spectra with consecutive MS2 scans: {len(ms1_spectra_with_consecutive_ms2)}")

# Example: Accessing the m/z and intensity arrays of the first MS1 spectrum
if ms1_spectra_with_consecutive_ms2:
    first_ms1 = ms1_spectra_with_consecutive_ms2[0]
    print(f"\nFirst MS1 Spectrum Scan Number: {first_ms1['scan_number']}")
    print("m/z array:", first_ms1['mz_array'])
    print("Intensity array:", first_ms1['intensity_array'])
