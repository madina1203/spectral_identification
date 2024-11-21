import pandas as pd


def collect_scan_numbers(tsv_path):
    # Load the TSV file
    tsv_data = pd.read_csv(tsv_path, sep='\t', usecols=['#Scan#', 'full_CCMS_path'])

    # Define the specific paths to search for
    path_peak = 'MSV000087935/peak/mzml/POS_MSMS_mzml/DOM_Interlab-LCMS_Lab26_M_Pos_MS2_rep2.mzML'
    path_ccms_peak = 'MSV000087935/ccms_peak/raw/POS_MSMS_raw/DOM_Interlab-LCMS_Lab26_M_Pos_MS2_rep2.mzML'

    # Filter data to get scan numbers for the specific paths
    scans_peak = tsv_data[tsv_data['full_CCMS_path'].str.contains(path_peak)]['#Scan#'].tolist()
    scans_ccms_peak = tsv_data[tsv_data['full_CCMS_path'].str.contains(path_ccms_peak)]['#Scan#'].tolist()

    return scans_peak, scans_ccms_peak


# Path to your TSV file
tsv_path = '/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/Library_search_results/MOLECULAR-LIBRARYSEARCH-V2-51f844bb-download_all_identifications-main.tsv'

# Get the scan numbers
scans_peak, scans_ccms_peak = collect_scan_numbers(tsv_path)

print("Scans for 'peak':", scans_peak)
print("Scans for 'ccms_peak':", scans_ccms_peak)
