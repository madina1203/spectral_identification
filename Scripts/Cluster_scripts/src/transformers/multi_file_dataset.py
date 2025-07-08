from torch.utils.data import Dataset
from pyteomics import mzml
import pandas as pd
import re
import numpy as np
import spectrum_utils.spectrum as sus
import time
import os
import pyarrow as pa
import lancedb
from typing import List, Dict, Any, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleMassSpecDatasetProcessor:
    """
    Class for processing a single pair of mzml and csv files.
    Based on the logic from SimpleMassSpecDataset.
    """

    def __init__(self, mzml_file: str, csv_file: str, scaling: str = "standardize"):
        """
        Initialize processor for a single file pair.

        Args:
            mzml_file: Path to mzML file.
            csv_file: Path to CSV file.
            scaling: Scaling type ('standardize').
        """
        self.mzml_file = mzml_file
        self.csv_file = csv_file
        self.scaling = scaling
        self.feature_stats = None  # For storing mean and standard deviation
        self.excluded_columns = {"Scan"}  # Columns excluded from scaling

        logger.info(f"Processing files: {mzml_file} and {csv_file}")

        # Load scans and MS2 data
        self.scan_list = self.load_scans()
        self.ms2_data = self.load_ms2_data()

        # Calculate statistics (mean and standard deviation) for instrument settings
        if self.scaling == "standardize":
            self.compute_stats()

        # Match MS2 scans with their preceding processed MS1 scans
        self.data_pairs = self.align_scans()

    def get_scan_number(self, id_string: str) -> Optional[int]:
        """
        Extract scan number from identifier string.

        Args:
            id_string: Identifier string.

        Returns:
            Scan number or None if not found.
        """
        match = re.search(r'scan=(\d+)', id_string)
        return int(match.group(1)) if match else None

    def load_scans(self) -> List[Dict[str, Any]]:
        """
        Load and preprocess scans from mzML file.

        Returns:
            List of processed scans.
        """
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

                # Preprocess only MS1 spectra
                if ms_level == 1:
                    # Create MsmsSpectrum object for preprocessing
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

                    # Save processed data
                scan_list.append({
                    'scan_number': scan_number,
                    'ms_level': ms_level,
                    'mz_array': mz_spectrum.mz if ms_level == 1 else mz_array,
                    'intensity_array': mz_spectrum.intensity if ms_level == 1 else intensity_array
                })

        # Sort scan list by scan number to ensure correct order
        scan_list.sort(key=lambda x: x['scan_number'])
        return scan_list

    def load_ms2_data(self) -> pd.DataFrame:
        """
        Load MS2 data from CSV file.

        Returns:
            DataFrame with MS2 data.
        """
        ms2_df = pd.read_csv(self.csv_file)
        ms2_df['Scan'] = ms2_df['Scan'].astype(int)  # Ensure 'Scan' column has integer type
        return ms2_df

    def get_instrument_settings_columns(self) -> List[str]:
        """
        Get list of instrument settings columns.

        Returns:
            List of column names.
        """
        return ["Scan", "RT [min]", "LowMass", "HighMass",
                "TIC", "BasePeakPosition", "BasePeakIntensity", "Charge State",
                "Monoisotopic M/Z", "Ion Injection Time (ms)",
                "Conversion Parameter C", "LM Search Window (ppm)", "Number of LM Found",
                "Mild Trapping Mode",
                "Energy1", "Orbitrap Resolution", "HCD Energy V(1)", "HCD Energy V(2)",
                "HCD Energy V(3)", "Number of Lock Masses", "LM m/z-Correction (ppm)"]

    def compute_stats(self) -> None:
        """
        Calculate mean and standard deviation for each instrument settings column,
        excluding certain columns.
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

    def scale_features(self, instrument_settings: List[float]) -> np.ndarray:
        """
        Standardize instrument settings using internally calculated mean and standard deviation.
        Skip excluded columns.

        Args:
            instrument_settings: Array of raw instrument settings.

        Returns:
            Standardized instrument settings.
        """
        scaled_settings = []
        stats_index = 0  # Index for statistics, skips excluded columns

        # Итерация по настройкам инструмента
        for col_index, value in enumerate(instrument_settings):
            # Check if current column is excluded or should not be scaled
            if col_index in self.excluded_columns or col_index >= len(self.feature_stats):
                # Leave excluded columns unchanged
                scaled_settings.append(value)
            else:
                # Apply standardization
                mean_val = self.feature_stats[stats_index]["mean"]
                std_val = self.feature_stats[stats_index]["std"]
                scaled_value = (value - mean_val) / std_val if std_val > 0 else 0
                scaled_settings.append(scaled_value)
                stats_index += 1  # Increment stats_index only for scalable columns

        return np.array(scaled_settings, dtype=np.float32)

    def align_scans(self) -> List[Dict[str, Any]]:
        """
        Match MS2 scans with their preceding processed MS1 scans.

        Returns:
            List of data pairs.
        """
        data_pairs = []
        ms2_scan_info = self.ms2_data.set_index('Scan').to_dict('index')
        instrument_settings_cols = self.get_instrument_settings_columns()

        current_ms1_data = None
        current_ms1_scan_number = None

        for scan in self.scan_list:
            scan_number = scan['scan_number']
            ms_level = scan['ms_level']

            if ms_level == 1:
                # Update current processed MS1 scan data
                current_ms1_data = {
                    'mz_array': scan['mz_array'],
                    'intensity_array': scan['intensity_array']
                }
                current_ms1_scan_number = scan_number

            elif ms_level == 2:
                # Check if MS2 scan exists in CSV data
                if scan_number in ms2_scan_info and current_ms1_data is not None:
                    ms2_info = ms2_scan_info[scan_number]

                    selected_mass = ms2_info.get('SelectedMass1', None)
                    label = ms2_info['label']  # Configure if label column has a different name

                    # Get and scale instrument settings
                    instrument_settings = self.scale_features([
                        ms2_info[col] for col in instrument_settings_cols if col in ms2_info
                    ])

                    # Add data pair with processed MS1 data and MS2 data
                    data_pairs.append({
                        'ms1_scan_number': current_ms1_scan_number,
                        'ms2_scan_number': scan_number,
                        'mz_array': current_ms1_data['mz_array'],
                        'intensity_array': current_ms1_data['intensity_array'],
                        'instrument_settings': instrument_settings,
                        'precursor_mz': selected_mass,
                        'label': label,
                        'source_mzml': os.path.basename(self.mzml_file),
                        'source_csv': os.path.basename(self.csv_file)
                    })
                else:
                    continue  # Skip if MS2 scan is not in CSV or no preceding MS1 data

        return data_pairs


class MultiFileMassSpecDataset:
    """
    Class for processing multiple pairs of mzml and csv files and saving results in lance format.
    """

    def __init__(self, mzml_files: List[str], csv_files: List[str], scaling: str = "standardize"):
        """
        Initialize dataset with multiple files.

        Args:
            mzml_files: List of paths to mzML files.
            csv_files: List of paths to CSV files.
            scaling: Scaling type ('standardize').
        """
        if len(mzml_files) != len(csv_files):
            raise ValueError("Number of mzML and CSV files must match")

        self.mzml_files = mzml_files
        self.csv_files = csv_files
        self.scaling = scaling
        self.data_pairs = []

        logger.info(f"Initializing MultiFileMassSpecDataset with {len(mzml_files)} file pairs")

    def process_files(self) -> None:
        """
        Process all file pairs and combine results.
        """
        all_data_pairs = []

        for i, (mzml_file, csv_file) in enumerate(zip(self.mzml_files, self.csv_files)):
            logger.info(f"Processing file pair {i + 1}/{len(self.mzml_files)}: {mzml_file}, {csv_file}")

            # Process one file pair
            processor = SimpleMassSpecDatasetProcessor(mzml_file, csv_file, self.scaling)

            # Add results to the overall list
            all_data_pairs.extend(processor.data_pairs)

            logger.info(f"Obtained {len(processor.data_pairs)} data pairs from files {mzml_file}, {csv_file}")

        self.data_pairs = all_data_pairs
        logger.info(f"Total of {len(self.data_pairs)} data pairs obtained from {len(self.mzml_files)} file pairs")

    def save_to_lance(self, output_file: str) -> None:
        """
        Save processed data in lance format.

        Args:
            output_file: Path to output lance file.
        """
        if not self.data_pairs:
            logger.warning("No data to save. Run process_files() first.")
            return

        logger.info(f"Saving data in lance format: {output_file}")

        # Create directory for output file if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

        # Prepare data for saving
        records = []

        for pair in self.data_pairs:
            # Convert numpy arrays to lists for serialization
            record = {
                'ms1_scan_number': pair['ms1_scan_number'],
                'ms2_scan_number': pair['ms2_scan_number'],
                'mz_array': pair['mz_array'].tolist(),
                'intensity_array': pair['intensity_array'].tolist(),
                'instrument_settings': pair['instrument_settings'].tolist(),
                'label': pair['label'],
                'source_mzml': pair['source_mzml'],
                'source_csv': pair['source_csv']
            }

            # Add precursor_mz if it exists
            if 'precursor_mz' in pair and pair['precursor_mz'] is not None:
                record['precursor_mz'] = pair['precursor_mz']

            records.append(record)

        # Create PyArrow table
        df = pd.DataFrame(records)
        table = pa.Table.from_pandas(df)

        # Save in lance format using lance.write_dataset (compatible with lance.dataset and depthcharge)
        import lance
        lance.write_dataset(table, output_file, mode="overwrite")
        logger.info(f"Data successfully saved to {output_file} using lance.write_dataset")

    def get_data_pairs(self) -> List[Dict[str, Any]]:
        """
        Get all data pairs.

        Returns:
            List of data pairs.
        """
        return self.data_pairs


def read_file_paths_from_txt(file_path: str) -> Tuple[List[str], List[str]]:
    """
    Read mzML and CSV file paths from a text file.

    The text file should have one pair per line in the format:
    /path/to/mzml/file.mzML, /path/to/csv/file.csv

    Args:
        file_path: Path to the text file containing file paths.

    Returns:
        Tuple of (mzml_files, csv_files) lists.
    """
    mzml_files = []
    csv_files = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(',')
            if len(parts) != 2:
                logger.warning(f"Skipping invalid line: {line}")
                continue

            mzml_path = parts[0].strip()
            csv_path = parts[1].strip()

            if not mzml_path.endswith('.mzML'):
                logger.warning(f"Skipping non-mzML file: {mzml_path}")
                continue

            if not csv_path.endswith('.csv'):
                logger.warning(f"Skipping non-CSV file: {csv_path}")
                continue

            mzml_files.append(mzml_path)
            csv_files.append(csv_path)

    return mzml_files, csv_files


def process_multiple_files(mzml_files: List[str], csv_files: List[str], output_file: str,
                           scaling: str = "standardize") -> None:
    """
    Wrapper function for processing multiple files and saving results in lance format.

    Args:
        mzml_files: List of paths to mzML files.
        csv_files: List of paths to CSV files.
        output_file: Path to output lance file.
        scaling: Scaling type ('standardize').
    """
    dataset = MultiFileMassSpecDataset(mzml_files, csv_files, scaling)
    dataset.process_files()
    dataset.save_to_lance(output_file)

    logger.info(f"Processing complete. Data saved to {output_file}")
    return dataset.get_data_pairs()


def process_from_file_list(file_list_path: str, output_file: str, scaling: str = "standardize") -> None:
    """
    Process multiple files from a text file containing file paths and save results in lance format.

    Args:
        file_list_path: Path to text file containing mzML and CSV file paths.
        output_file: Path to output lance file.
        scaling: Scaling type ('standardize').
    """
    mzml_files, csv_files = read_file_paths_from_txt(file_list_path)

    if not mzml_files or not csv_files:
        logger.error("No valid file paths found in the input file.")
        return

    logger.info(f"Found {len(mzml_files)} file pairs in {file_list_path}")
    process_multiple_files(mzml_files, csv_files, output_file, scaling)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process multiple mzML and CSV files and save in lance format.')

    # Create mutually exclusive group for input methods
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--file-list', help='Path to text file containing mzML and CSV file paths')
    input_group.add_argument('--mzml', nargs='+', help='Paths to mzML files')

    parser.add_argument('--csv', nargs='+', help='Paths to CSV files (required if --mzml is used)')
    parser.add_argument('--output', required=True, help='Path to output lance file')
    parser.add_argument('--scaling', default='standardize', choices=['standardize', 'none'],
                        help='Scaling type (standardize or none)')

    args = parser.parse_args()
    # Process based on input method
    if args.file_list:
        process_from_file_list(args.file_list, args.output, args.scaling)
    else:
        if not args.csv:
            parser.error("--csv is required when --mzml is used")
        if len(args.mzml) != len(args.csv):
            parser.error("Number of mzML and CSV files must match")
        process_multiple_files(args.mzml, args.csv, args.output, args.scaling)