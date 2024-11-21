import os
from matchms.importing import load_from_mgf
from matchms.similarity import CosineGreedy
from matchms import calculate_scores
import numpy as np


def process_file(file_path, reference_spectra, similarity_measure):
    """Process each individual .mgf file to calculate and save similarity scores."""
    spectra = list(load_from_mgf(file_path))
    scores = calculate_scores(reference_spectra, spectra[:100], similarity_measure, is_symmetric=False)
    scores_array = scores.scores.to_array()  # Convert scores to structured array    np.save(file_path.replace('.mgf', '_scores.npy'), scores_array)  # Save scores as NumPy array
    np.save(file_path.replace('.mgf', '_full_scores_2.npy'), scores_array)
    print("Scores are calculated")

def main():
    spectra_dir = '/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/temporary_for_similarity_check'
    reference_spectra_file = '/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Filtered_spectral_libraries/filtered_libraries_merged.mgf'
    reference_spectra = list(load_from_mgf(reference_spectra_file))

    similarity_measure = CosineGreedy(tolerance=0.005)

    for filename in os.listdir(spectra_dir):
        if filename.endswith(".mgf"):
            file_path = os.path.join(spectra_dir, filename)
            process_file(file_path, reference_spectra, similarity_measure)


if __name__ == '__main__':
    main()
