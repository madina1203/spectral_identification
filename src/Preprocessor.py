from pyteomics import mzml
import spectrum_utils.spectrum as sus
import numpy as np
import re


# Define a function to process the mzML files and apply filter_intensity
def process_mzml_with_filter(mzml_files, min_intensity=0.01, max_num_peaks=400):
    processed_spectra = []

    for file_mzml in mzml_files:
        with mzml.read(file_mzml) as reader:
            for spectrum in reader:
                ms_level = spectrum.get('ms level')

                # Only process MS1 spectra (ms level = 1)
                if ms_level == 1:
                    mz_array = spectrum.get('m/z array')
                    intensity_array = spectrum.get('intensity array')
                    spectrum_id = spectrum.get('id', 'unknown_spectrum')

                    # Print each spectrum ID to confirm its format
                    print(f"Processing Spectrum ID: {spectrum_id}")

                    # Set precursor_mz and precursor_charge to np.nan if not available
                    precursor_mz = spectrum.get('precursor m/z', np.nan)
                    precursor_charge = spectrum.get('precursor charge', np.nan)

                    # Create an MsmsSpectrum object from the m/z and intensity arrays
                    mz_spectrum = sus.MsmsSpectrum(
                        identifier=spectrum_id,
                        precursor_mz=precursor_mz,
                        precursor_charge=precursor_charge,
                        mz=mz_array,
                        intensity=intensity_array,
                        retention_time=spectrum.get('scan start time', 0)
                    )

                    # Apply the filter_intensity function
                    mz_spectrum = mz_spectrum.filter_intensity(
                        min_intensity=min_intensity, max_num_peaks=max_num_peaks
                    )

                    # Scale intensities using the square root
                    mz_spectrum = mz_spectrum.scale_intensity(scaling="root")
                    # Collect processed spectra
                    processed_spectra.append(mz_spectrum)

    return processed_spectra


# Example usage:
mzml_files = ["/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/MSV_Files/MSV000095364/raw/preserv_etoh_blank_ID_03.mzML"]
processed_spectra = process_mzml_with_filter(mzml_files)
# Print only the spectrum with scan number 119
for spectrum in processed_spectra:
    # Extract the scan number from the identifier using a regular expression
    match = re.search(r'scan=(\d+)', spectrum.identifier)
    if match:
        print(f"Extracted Scan Number: {match.group(1)} from Identifier: {spectrum.identifier}")
    if match and match.group(1) == "119":
        print(f"Spectrum with Scan Number 119:")
        print("m/z values:", spectrum.mz)
        print("Intensity values:", spectrum.intensity)
        print("Number of peaks:", len(spectrum.mz))
        print()  # Blank line for readability