import os
import subprocess

# Path to the folder containing ThermoRawFileParser.exe
thermo_parser_dir = "/Users/madinabekbergenova/Downloads/ThermoRawFileParser1.4.4"

# Base folder containing MSV files
base_folder = "/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/MSV_Files"

# Function to recursively find subfolders containing .raw files and run the command
def convert_raw_to_mzml(base_folder, thermo_parser_dir):
    # Change directory to where ThermoRawFileParser.exe is located
    os.chdir(thermo_parser_dir)

    # Walk through the directory structure
    for root, dirs, files in os.walk(base_folder):
        # Check if any .raw files exist in the current folder
        if any(file.endswith('.raw') for file in files):
            # Run the ThermoRawFileParser command for the current folder
            command = f"mono ThermoRawFileParser.exe -d={root}"
            print(f"Running command: {command}")
            # Execute the command
            subprocess.run(command, shell=True)

# Run the conversion
convert_raw_to_mzml(base_folder, thermo_parser_dir)
