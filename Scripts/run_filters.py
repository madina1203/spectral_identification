import argparse
from glob import glob
import os
import sys
sys.path.append('../src')  # Ensure the src module is accessible
from src.custom_utilities import load_spectra, apply_filters, save_spectra


def process_file(input_file, output_dir):
    """Process a single MGF file, apply filters, and save the filtered output."""
    # Load spectra
    spectra = load_spectra(input_file)
    print(f"Loaded {len(spectra)} spectra from {input_file}.")

    # Apply filters
    filtered_spectra = apply_filters(spectra)
    print(f"Filtered down to {len(filtered_spectra)} spectra.")

    # Construct the output file path
    base_name = os.path.basename(input_file)
    filtered_file_name = f"{os.path.splitext(base_name)[0]}_filtered.mgf"
    output_file_path = os.path.join(output_dir, filtered_file_name)

    save_spectra(filtered_spectra, output_file_path)
    print(f"Filtered spectra saved to {output_file_path}.")


def main(input_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Find all .mgf files in the input directory
    mgf_files = glob(os.path.join(input_dir, '*.mgf'))

    for mgf_file in mgf_files:
        process_file(mgf_file, output_dir)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Process MGF files by applying spectral filters.")
#     # parser.add_argument('input_dir', type=str, help='Directory containing MGF files to process.')
#     # parser.add_argument('output_dir', type=str, help='Directory where filtered MGF files will be saved.')
#     #
#     # args = parser.parse_args()
#     #
#     # main(args.input_dir, args.output_dir)
    input_dir = '/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/spectral_libraries_for_cleaning'  # Set the path to your input directory
    output_dir = '/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Filtered_spectral_libraries'  # Set the path to your output directory
    main(input_dir, output_dir)