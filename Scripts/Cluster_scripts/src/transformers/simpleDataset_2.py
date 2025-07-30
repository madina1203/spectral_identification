from torch.utils.data import Dataset
from pyteomics import mzml
import pandas as pd
import re
import numpy as np
import spectrum_utils.spectrum as sus
import random


class SimpleMassSpecDataset(Dataset):
    def __init__(self, mzml_file, csv_file, scaling="standardize"):
        """
        Initialize the dataset with feature scaling, excluding certain columns.
        :param mzml_file: Path to the mzML file.
        :param csv_file: Path to the CSV file with compound names.
        :param scaling: Type of scaling ('standardize').
        """
        self.mzml_file = mzml_file
        self.csv_file = csv_file
        self.scaling = scaling
        self.feature_stats = None  # To store mean and std for standardization
        self.excluded_columns = {"Scan"}  # Columns to exclude from scaling

        # Load scans and MS2 data
        self.scan_list = self.load_scans()
        self.ms2_data = self.load_ms2_data()

        # Compute statistics (mean and std) for instrument settings
        if self.scaling == "standardize":
            self.compute_stats()

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
        # Now expects a CSV that includes the 'Compound_Name' column
        ms2_df = pd.read_csv(self.csv_file)
        ms2_df['Scan'] = ms2_df['Scan'].astype(int)  # Ensure 'Scan' column is integer
        return ms2_df

    def get_instrument_settings_columns(self):
        # Returns the list of columns to be used as instrument settings features
        return ["Scan", "RT [min]", "LowMass", "HighMass",
                "TIC", "BasePeakPosition", "BasePeakIntensity", "Charge State",
                "Monoisotopic M/Z", "Ion Injection Time (ms)",
                "Conversion Parameter C", "LM Search Window (ppm)", "Number of LM Found",
                "Mild Trapping Mode",
                "Energy1", "Orbitrap Resolution", "HCD Energy V(1)", "HCD Energy V(2)",
                "HCD Energy V(3)", "Number of Lock Masses", "LM m/z-Correction (ppm)"]

    def compute_stats(self):
        """
        Compute mean and std for each instrument settings column, excluding certain columns.
        """
        all_instrument_settings = []
        instrument_settings_cols = [
            col for col in self.get_instrument_settings_columns() if col not in self.excluded_columns
        ]

        for _, ms2_info in self.ms2_data.iterrows():
            instrument_settings = [
                float(ms2_info[col]) for col in instrument_settings_cols if col in ms2_info
            ]
            all_instrument_settings.append(instrument_settings)

        all_instrument_settings = np.array(all_instrument_settings)
        self.feature_stats = {
            i: {"mean": np.mean(all_instrument_settings[:, i]), "std": np.std(all_instrument_settings[:, i])}
            for i in range(all_instrument_settings.shape[1])
        }

    def scale_features(self, instrument_settings):
        """
        Standardize the instrument settings using internally computed mean and std.
        """
        scaled_settings = []
        stats_index = 0
        for col_index, value in enumerate(instrument_settings):
            if col_index in self.excluded_columns or col_index >= len(self.feature_stats):
                scaled_settings.append(value)
            else:
                mean_val = self.feature_stats[stats_index]["mean"]
                std_val = self.feature_stats[stats_index]["std"]
                scaled_value = (value - mean_val) / std_val if std_val > 0 else 0
                scaled_settings.append(scaled_value)
                stats_index += 1
        return np.array(scaled_settings, dtype=np.float32)

    def align_scans(self):
        data_pairs = []
        ms2_scan_info = self.ms2_data.set_index('Scan').to_dict('index')
        instrument_settings_cols = self.get_instrument_settings_columns()

        current_ms1_data = None
        current_ms1_scan_number = None

        print(f"align_scans: {len(self.scan_list)} scans in mzML, {len(self.ms2_data)} rows in CSV")
        found_ms2 = 0
        found_in_csv = 0
        found_with_compound = 0

        for scan in self.scan_list:
            scan_number = scan['scan_number']
            ms_level = scan['ms_level']

            if ms_level == 1:
                current_ms1_data = {
                    'mz_array': scan['mz_array'],
                    'intensity_array': scan['intensity_array']
                }
                current_ms1_scan_number = scan_number

            elif ms_level == 2:
                found_ms2 += 1
                if scan_number in ms2_scan_info and current_ms1_data is not None:
                    found_in_csv += 1
                    ms2_info = ms2_scan_info[scan_number]
                    compound_name = ms2_info.get('Compound_Name', None)
                    if compound_name is not None and str(compound_name).strip() != '':
                        found_with_compound += 1
                    else:
                        print(f"Skipping scan {scan_number}: missing compound name")
                        continue
                    instrument_settings = [float(ms2_info[col]) for col in instrument_settings_cols if col in ms2_info]
                    instrument_settings = self.scale_features(instrument_settings)
                    selected_mass = ms2_info.get('SelectedMass1', None)
                    label = ms2_info['label']
                    data_pairs.append({
                        'ms1_scan_number': current_ms1_scan_number,
                        'ms2_scan_number': scan_number,
                        'mz_array': current_ms1_data['mz_array'],
                        'intensity_array': current_ms1_data['intensity_array'],
                        'instrument_settings': instrument_settings,
                        'precursor_mz': selected_mass,
                        'label': label,
                        'compound_name': compound_name
                    })
                else:
                    print(f"Skipping scan {scan_number}: not in ms2_scan_info or no current MS1 data")
                    continue
        print(f"Total MS2 scans: {found_ms2}, in CSV: {found_in_csv}, with compound: {found_with_compound}, data pairs: {len(data_pairs)}")
        return data_pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        return self.data_pairs[idx]
