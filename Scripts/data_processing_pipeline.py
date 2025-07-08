#!/usr/bin/env python3
# Mass Spectrometry Data Processing Pipeline
# This script automates the processing of MS data files for model training preparation

import os
import sys
import pandas as pd
import logging
import argparse
import shutil
import re
from datetime import datetime
from typing import Set, List, Dict, Tuple, Optional, Union


# Configure logging
def setup_logging(log_dir: str) -> logging.Logger:
    """Set up logging to both console and file."""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"ms_pipeline_{timestamp}.log")

    # Create logger
    logger = logging.getLogger("ms_pipeline")
    logger.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Step 1: Delete incomplete MSOrder files
def delete_incomplete_msorder_files(root_path: str, logger: logging.Logger) -> int:
    """
    Recursively finds .csv files under root_path that do not contain '2' in the 'MSOrder' column
    and deletes both the .csv and the corresponding .raw file with the same base name.

    Returns:
        int: Number of deleted file pairs
    """
    deleted_count = 0
    logger.info("Starting deletion of incomplete MSOrder files...")

    for dirpath, _, filenames in os.walk(root_path):
        for file in filenames:
            if file.endswith(".csv"):
                csv_path = os.path.join(dirpath, file)
                base_name = os.path.splitext(file)[0]
                raw_path = os.path.join(dirpath, base_name + ".raw")

                try:
                    df = pd.read_csv(csv_path)
                except Exception as e:
                    logger.error(f"Could not read {csv_path}: {e}")
                    continue

                # Check if 'MSOrder' column exists and contains value 2
                if "MSOrder" in df.columns:
                    msorder_values = set(df["MSOrder"].dropna().unique())
                    if 2 not in msorder_values:
                        # Delete CSV
                        os.remove(csv_path)
                        logger.info(f"Deleted CSV: {csv_path}")

                        # Delete corresponding RAW file if it exists
                        if os.path.exists(raw_path):
                            os.remove(raw_path)
                            logger.info(f"Deleted RAW: {raw_path}")
                        else:
                            logger.warning(f"RAW file not found for: {csv_path}")

                        deleted_count += 1
                else:
                    # If MSOrder column is missing, treat as invalid and delete
                    os.remove(csv_path)
                    logger.info(f"Deleted CSV (missing MSOrder): {csv_path}")
                    if os.path.exists(raw_path):
                        os.remove(raw_path)
                        logger.info(f"Deleted RAW: {raw_path}")
                    else:
                        logger.warning(f"RAW file not found for: {csv_path}")
                    deleted_count += 1

    logger.info(f"Finished deletion. Total files removed (CSV + RAW pairs): {deleted_count}")
    return deleted_count


# Step 2: Check CSV files for required columns and proper MSOrder values
def check_csv_files(root_path: str, required_columns: List[str], logger: logging.Logger) -> Tuple[List[str], List[str]]:
    """
    Recursively checks CSV files under the root_path for:
    1. MSOrder values (must contain both 1 and 2).
    2. Presence of required columns.

    Returns:
        Tuple containing:
            - List of files missing MSOrder values
            - List of files missing required columns
    """
    missing_msorder_files = []
    missing_columns_files = []
    msorder_incomplete_count = 0

    logger.info("Checking CSV files for required columns and MSOrder values...")

    for dirpath, _, filenames in os.walk(root_path):
        for file in filenames:
            if file.endswith(".csv"):
                file_path = os.path.join(dirpath, file)

                try:
                    df = pd.read_csv(file_path)
                except Exception as e:
                    logger.error(f"Could not read {file_path}: {e}")
                    continue

                # Check if 'MSOrder' exists and has required values
                if "MSOrder" in df.columns:
                    unique_orders = set(df["MSOrder"].dropna().unique())
                    if not ({1, 2}.issubset(unique_orders)):
                        logger.warning(f"MSOrder incomplete in file: {file_path} — found: {unique_orders}")
                        missing_msorder_files.append(file_path)
                        msorder_incomplete_count += 1
                else:
                    logger.warning(f"Missing 'MSOrder' column in: {file_path}")
                    missing_msorder_files.append(file_path)

                # Check for missing columns
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    logger.warning(f"{file_path} is missing columns: {missing_columns}")
                    missing_columns_files.append((file_path, missing_columns))

    logger.info(f"Number of files that do not have both MS1 and MS2 scans: {msorder_incomplete_count}")
    logger.info(f"Number of files missing required columns: {len(missing_columns_files)}")

    return missing_msorder_files, missing_columns_files


# Step 3: Standardize columns in CSV files
def standardize_columns(root_path: str, logger: logging.Logger) -> Dict[str, List[str]]:
    """
    Process CSV files to standardize column names and ensure all required columns exist.
    Handles different naming conventions for the same data fields.

    Returns:
        Dict containing lists of files with missing essential columns
    """
    # Standardized column mappings
    column_mappings = {
        "FT Resolution": "Orbitrap Resolution",
        "Orbitrap Resolution": "Orbitrap Resolution",
        "AGC Target": "AGC Target",
        "TargetAGC": "AGC Target",
        "HCD Energy [eV]": "HCD Energy V",
        "HCD Energy V": "HCD Energy V",
        "HCD Energy eV": "HCD Energy V",
        "NrLockMasses": "Number of Lock Masses",
        "Number of Lock Masses": "Number of Lock Masses",
        "LM Correction (ppm)": "LM m/z-Correction (ppm)",
        "LM m/z-Correction (ppm)": "LM m/z-Correction (ppm)"
    }

    # Columns that are always required
    base_columns = [
        "Scan", "MSOrder", "Polarity", "RT [min]", "LowMass", "HighMass",
        "TIC", "BasePeakPosition", "BasePeakIntensity", "Charge State", "Monoisotopic M/Z",
        "Ion Injection Time (ms)", "MS2 Isolation Width", "Conversion Parameter C",
        "LM Search Window (ppm)", "Number of LM Found", "Mild Trapping Mode",
        "Source CID eV", "SelectedMass1", "Activation1", "Energy1", "Orbitrap Resolution", "AGC Target",
        "HCD Energy V", "Number of Lock Masses", "LM m/z-Correction (ppm)", "label"
    ]

    # Track files with completely missing columns
    missing_columns_report = {col: [] for col in base_columns}
    processed_count = 0
    skipped_count = 0

    logger.info("Starting column standardization process...")

    for dirpath, _, filenames in os.walk(root_path):
        for file in filenames:
            if file.endswith('.csv') and not file.endswith(('_processed.csv', '_annotated.csv')):
                csv_path = os.path.join(dirpath, file)

                try:
                    df = pd.read_csv(csv_path)

                    # Skip if DataFrame is empty
                    if df.empty:
                        logger.warning(f"Skipping empty CSV: {csv_path}")
                        skipped_count += 1
                        continue

                except Exception as e:
                    logger.error(f"Error reading {csv_path}: {e}")
                    skipped_count += 1
                    continue

                # Initialize with empty dummy label column if needed
                if 'label' not in df.columns:
                    df['label'] = pd.NA

                # Create a new DataFrame with standardized columns
                new_df = pd.DataFrame()

                # Ensure all base columns are present
                for col in base_columns:
                    # Try to find the column using the mapping
                    found = False
                    for original, standardized in column_mappings.items():
                        if standardized == col and original in df.columns:
                            new_df[col] = df[original]
                            found = True
                            break

                    # If not found through mapping, check if it exists directly
                    if not found and col in df.columns:
                        new_df[col] = df[col]
                        found = True

                    # If the column is still not found, track it and add empty column
                    if not found:
                        missing_columns_report[col].append(csv_path)
                        new_df[col] = pd.NA

                # Save the processed CSV
                processed_csv_path = os.path.join(dirpath, os.path.splitext(file)[0] + '_processed.csv')
                new_df.to_csv(processed_csv_path, index=False)
                logger.info(f"Processed and standardized columns: {processed_csv_path}")
                processed_count += 1

    # Log summary of column standardization
    logger.info(f"Column standardization complete. Processed {processed_count} files, skipped {skipped_count} files.")

    # Log completely missing columns
    for col, files in missing_columns_report.items():
        if files:
            logger.warning(f"Column '{col}' missing in {len(files)} files")
            for file_path in files[:5]:  # Log first 5 files only to avoid excessive logging
                logger.debug(f"  - {file_path}")
            if len(files) > 5:
                logger.debug(f"  - ... and {len(files) - 5} more files")

    return missing_columns_report


# Step 4: Label data using TSV information
def label_data(root_path: str, tsv_path: str, logger: logging.Logger) -> int:
    """
    Process CSV files to add labels based on TSV data matching.

    Returns:
        int: Number of files labeled
    """
    if not os.path.exists(tsv_path):
        logger.error(f"TSV file not found: {tsv_path}")
        return 0

    logger.info(f"Loading TSV data from {tsv_path}")
    try:
        # Load the TSV data
        tsv_data = pd.read_csv(tsv_path, sep='\t', usecols=['#Scan#', 'full_CCMS_path'])
        tsv_data['#Scan#'] = tsv_data['#Scan#'].astype(int)  # Ensure scan numbers are integers
        logger.info(f"Loaded {len(tsv_data)} entries from TSV file")
    except Exception as e:
        logger.error(f"Failed to load TSV file: {e}")
        return 0

    labeled_count = 0

    # Find all processed CSV files and label them
    for dirpath, _, filenames in os.walk(root_path):
        for file in filenames:
            if file.endswith('_processed.csv'):
                csv_path = os.path.join(dirpath, file)

                try:
                    # Read the CSV file
                    csv_data = pd.read_csv(csv_path)

                    if 'Scan' not in csv_data.columns:
                        logger.warning(f"Skipping labeling for {csv_path}: 'Scan' column not found")
                        continue

                    csv_data['Scan'] = csv_data['Scan'].astype(int)  # Convert scan numbers to integers

                    # Extract the base name of the CSV file
                    base_name = os.path.basename(csv_path).replace('_processed.csv', '')

                    # Build regex to match both 'ccms_peak' and 'peak' paths including the base name
                    regex_pattern = f"/(ccms_peak|peak)/.*{base_name}.mzML"

                    # Find all matching entries in the TSV 'full_CCMS_path'
                    all_matches = tsv_data[tsv_data['full_CCMS_path'].str.contains(regex_pattern, regex=True)]

                    # Determine if 'ccms_peak' entries exist
                    ccms_peak_matches = all_matches[all_matches['full_CCMS_path'].str.contains('ccms_peak')]

                    # Select the appropriate scans based on the available paths
                    if not ccms_peak_matches.empty:
                        matching_scans = ccms_peak_matches  # Use ccms_peak matches if available
                    else:
                        # If no ccms_peak matches, fall back to peak or any matches (as a last resort)
                        peak_matches = all_matches[all_matches['full_CCMS_path'].str.contains('peak')]
                        if not peak_matches.empty:
                            matching_scans = peak_matches
                        else:
                            matching_scans = all_matches  # Use any matches if no specific subpaths are found

                    # Extract scan numbers for matching
                    matching_scan_values = matching_scans['#Scan#'].values if not matching_scans.empty else []

                    # Label the scans in the CSV data
                    csv_data['label'] = csv_data['Scan'].apply(lambda x: 1 if x in matching_scan_values else 0)

                    # Save the annotated CSV
                    annotated_csv_path = csv_path.replace('_processed.csv', '_annotated.csv')
                    csv_data.to_csv(annotated_csv_path, index=False)
                    logger.info(f"Labeled data saved to: {annotated_csv_path}")
                    labeled_count += 1

                except Exception as e:
                    logger.error(f"Error labeling {csv_path}: {e}")

    logger.info(f"Labeling complete. Successfully labeled {labeled_count} files.")
    return labeled_count


# Step 5: Organize processed files
def organize_processed_files(root_path: str, output_dir: str, logger: logging.Logger) -> int:
    """
    Copies processed and annotated files to the output directory,
    preserving the folder structure.

    Returns:
        int: Number of files copied
    """
    copied_count = 0

    logger.info(f"Organizing processed files to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Create a mapping of source to destination paths
    copy_tasks = []

    for dirpath, _, filenames in os.walk(root_path):
        for file in filenames:
            # Copy final annotated files and their raw data
            if file.endswith('_annotated.csv'):
                source_path = os.path.join(dirpath, file)

                # Calculate the relative path to maintain folder structure
                rel_path = os.path.relpath(dirpath, root_path)
                dest_dir = os.path.join(output_dir, rel_path)
                os.makedirs(dest_dir, exist_ok=True)

                dest_path = os.path.join(dest_dir, file)
                copy_tasks.append((source_path, dest_path))

                # Also copy the corresponding RAW file if it exists
                base_name = os.path.splitext(file.replace('_annotated', ''))[0]
                raw_path = os.path.join(dirpath, base_name + ".raw")
                if os.path.exists(raw_path):
                    raw_dest_path = os.path.join(dest_dir, base_name + ".raw")
                    copy_tasks.append((raw_path, raw_dest_path))

    # Perform the copying
    for src, dst in copy_tasks:
        try:
            shutil.copy2(src, dst)
            copied_count += 1
            logger.debug(f"Copied: {src} -> {dst}")
        except Exception as e:
            logger.error(f"Failed to copy {src}: {e}")

    logger.info(f"Successfully copied {copied_count} processed files to {output_dir}")
    return copied_count


# Step 6: Generate summary report
def generate_summary(root_path: str, output_dir: str, deleted_count: int,
                     missing_msorder: List[str], missing_columns_report: Dict[str, List[str]],
                     processed_count: int, labeled_count: int, copied_count: int,
                     logger: logging.Logger) -> None:
    """Generate a summary report of the pipeline execution."""

    # Count files with missing columns
    missing_cols_count = {}
    for col, files in missing_columns_report.items():
        missing_cols_count[col] = len(files)

    # Create a detailed summary
    summary = {
        "Files Processed": {
            "Total CSV files found": sum(1 for _ in find_files_by_extension(root_path, ".csv")),
            "Files deleted (incomplete MSOrder)": deleted_count,
            "Files with missing MSOrder values": len(missing_msorder),
            "Files with standardized columns": processed_count,
            "Files labeled with TSV data": labeled_count,
            "Final files copied to output directory": copied_count
        },
        "Missing Columns Summary": missing_cols_count,
        "Pipeline Execution": {
            "Input directory": os.path.abspath(root_path),
            "Output directory": os.path.abspath(output_dir),
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }

    # Log the summary
    logger.info("Pipeline Execution Summary:")
    for section, items in summary.items():
        logger.info(f"--- {section} ---")
        for key, value in items.items():
            logger.info(f"  {key}: {value}")


# Utility function to find files by extension
def find_files_by_extension(root_path: str, extension: str) -> List[str]:
    """Find all files with the given extension in the root_path directory tree."""
    result = []
    for dirpath, _, filenames in os.walk(root_path):
        for file in filenames:
            if file.endswith(extension):
                result.append(os.path.join(dirpath, file))
    return result


# Main pipeline function
def run_pipeline(root_path: str, output_dir: str, tsv_path: str, required_columns: List[str],
                 logger: logging.Logger) -> None:
    """Run the complete MS data processing pipeline."""
    logger.info(f"Starting MS data processing pipeline on {root_path}")
    logger.info(f"Output directory: {output_dir}")

    # Step 1: Delete incomplete MSOrder files
    deleted_count = delete_incomplete_msorder_files(root_path, logger)

    # Step 2: Check remaining CSV files for quality
    missing_msorder, missing_columns_list = check_csv_files(root_path, required_columns, logger)

    # Step 3: Standardize columns in CSV files
    missing_columns_report = standardize_columns(root_path, logger)

    # Step 4: Label data using TSV information
    labeled_count = label_data(root_path, tsv_path, logger)

    # Step 5: Organize processed files
    copied_count = organize_processed_files(root_path, output_dir, logger)

    # Step 6: Generate summary
    generate_summary(
        root_path=root_path,
        output_dir=output_dir,
        deleted_count=deleted_count,
        missing_msorder=missing_msorder,
        missing_columns_report=missing_columns_report,
        processed_count=sum(1 for _ in find_files_by_extension(root_path, "_processed.csv")),
        labeled_count=labeled_count,
        copied_count=copied_count,
        logger=logger
    )

    logger.info("Pipeline execution completed successfully!")


def main():
    """Main entry point of the script."""
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="MS Data Processing Pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input directory containing MS data files")
    parser.add_argument("--output", "-o", required=True, help="Output directory for processed files")
    parser.add_argument("--tsv", "-t", required=True, help="Path to the TSV file for labeling")
    parser.add_argument("--log-dir", "-l", default="logs", help="Directory for log files")
    args = parser.parse_args()

    # Define required columns for MS data
    required_columns = [
        "Scan", "MSOrder", "Polarity", "RT [min]", "LowMass", "HighMass",
        "TIC", "BasePeakPosition", "BasePeakIntensity", "Charge State",
        "Monoisotopic M/Z", "Ion Injection Time (ms)", "MS2 Isolation Width",
        "Conversion Parameter C", "LM Search Window (ppm)", "Number of LM Found",
        "Mild Trapping Mode", "Source CID eV", "SelectedMass1", "Activation1",
        "Energy1", "Orbitrap Resolution", "AGC Target", "HCD Energy V",
        "Number of Lock Masses", "LM m/z-Correction (ppm)"
    ]

    # Set up logging
    logger = setup_logging(args.log_dir)

    try:
        # Run the pipeline
        run_pipeline(
            root_path=args.input,
            output_dir=args.output,
            tsv_path=args.tsv,
            required_columns=required_columns,
            logger=logger
        )
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()