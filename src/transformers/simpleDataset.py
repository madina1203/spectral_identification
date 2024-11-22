from torch.utils.data import Dataset
from pyteomics import mzml
import pandas as pd
import re
import numpy as np
import spectrum_utils.spectrum as sus

class SimpleMassSpecDataset(Dataset):
    def __init__(self, mzml_file, csv_file):
        self.mzml_file = mzml_file
        self.csv_file = csv_file

        # Load scans and MS2 data
        self.scan_list = self.load_scans()
        self.ms2_data = self.load_ms2_data()

        # Align MS2 scans with their preceding processed MS1 scans
        self.data_pairs = self.align_scans()

    def get_scan_number(self, id_string):
        match = re.search(r'scan=(\d+)', id_string)
        return int(match.group(1)) if match else None

    def load_scans(self):
        scan_list = []
        with mzml.read(self.mzml_file) as reader:
            for spectrum in reader:
                ms_level = spectrum.get('ms level')
                id_string = spectrum.get('id')
                scan_number = self.get_scan_number(id_string)


                if scan_number is None or ms_level is None:
                    continue

                mz_array = spectrum.get('m/z array')
                intensity_array = spectrum.get('intensity array')

                # Only preprocess MS1 spectra
                if ms_level == 1:
                    # Create an MsmsSpectrum object for preprocessing
                    mz_spectrum = sus.MsmsSpectrum(
                        identifier=str(scan_number),
                        precursor_mz=np.nan,
                        precursor_charge=np.nan,
                        mz=mz_array,
                        intensity=intensity_array,
                        retention_time=spectrum.get('scan start time', 0)
                    )

                    # Apply preprocessing: filter_intensity and scale_intensity
                    mz_spectrum = mz_spectrum.filter_intensity(min_intensity=0.01, max_num_peaks=400)
                    mz_spectrum = mz_spectrum.scale_intensity(scaling="root")

                    # Store the preprocessed data
                scan_list.append({
                        'scan_number': scan_number,
                        'ms_level': ms_level,
                        'mz_array': mz_spectrum.mz,
                        'intensity_array': mz_spectrum.intensity
                    })

        # Sort the scan list by scan number to ensure correct order
        scan_list.sort(key=lambda x: x['scan_number'])
        return scan_list

    def load_ms2_data(self):
        ms2_df = pd.read_csv(self.csv_file)
        ms2_df['Scan'] = ms2_df['Scan'].astype(int)  # Ensure 'Scan' column is integer
        return ms2_df

    # def get_instrument_settings_columns(self):
    #     return ["Scan", "MSOrder", "Polarity", "RT [min]", "LowMass", "HighMass",
    #             "TIC", "BasePeakPosition", "BasePeakIntensity", "Charge State",
    #             "Monoisotopic M/Z", "Ion Injection Time (ms)", "MS2 Isolation Width",
    #             "Conversion Parameter C", "LM Search Window (ppm)", "Number of LM Found",
    #             "Mild Trapping Mode", "Source CID eV", "SelectedMass1", "Activation1",
    #             "Energy1", "Orbitrap Resolution", "AGC Target", "HCD Energy V(1)", "HCD Energy V(2)",
    #             "HCD Energy V(3)", "Number of Lock Masses", "LM m/z-Correction (ppm)"]

    def align_scans(self):
        data_pairs = []
        ms2_scan_info = self.ms2_data.set_index('Scan').to_dict('index')
        # instrument_settings_cols = self.get_instrument_settings_columns()

        current_ms1_data = None
        current_ms1_scan_number = None

        for scan in self.scan_list:
            scan_number = scan['scan_number']

            ms_level = scan['ms_level']

            if ms_level == 1:
                # Update the current processed MS1 scan data
                current_ms1_data = {
                    'mz_array': scan['mz_array'],
                    'intensity_array': scan['intensity_array']
                }
                current_ms1_scan_number = scan_number

            elif ms_level == 2:
                # Check if the MS2 scan is in the CSV data
                if scan_number in ms2_scan_info and current_ms1_data is not None:
                    ms2_info = ms2_scan_info[scan_number]
                    # instrument_settings = [float(ms2_info[col]) for col in instrument_settings_cols if col in ms2_info]
                    # instrument_settings = np.array(instrument_settings, dtype=float)  # Convert to NumPy array
                    selected_mass = ms2_info.get('SelectedMass1', None)
                    label = ms2_info['label']  # Adjust if your label column has a different name

                    # Append the data pair with preprocessed MS1 data and MS2 data
                    data_pairs.append({
                        'ms1_scan_number': current_ms1_scan_number,
                        'ms2_scan_number': scan_number,
                        'mz_array': current_ms1_data['mz_array'],
                        'intensity_array': current_ms1_data['intensity_array'],
                        # 'instrument_settings': instrument_settings,
                        'precursor_mz': selected_mass,
                        'label': label
                    })
                else:
                    continue  # Skip if MS2 scan is not in CSV or no preceding MS1 data

        return data_pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        return self.data_pairs[idx]
