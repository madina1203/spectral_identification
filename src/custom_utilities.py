from matchms.importing import load_from_mgf
from matchms.exporting import save_as_mgf
from matchms.filtering import normalize_intensities, remove_peaks_outside_top_k, remove_peaks_around_precursor_mz, select_by_mz
from custom_filters import custom_filter_peaks_below_1_percent
from scale_intensity import scale_intensities_by_sqrt

def load_spectra(file_path):
    """Load spectra from a given MGF file."""
    return list(load_from_mgf(file_path))

def apply_filters(spectra):
    """Apply a series of filters to the spectra."""
    filtered_spectra = []
    for spectrum in spectra:
        precursor_mz = spectrum.get("precursor_mz")
        #commented normalize intensities to not to apply to gnps library (its already applied)
        #spectrum = normalize_intensities(spectrum)
        # Apply select_by_mz function based on precursor_mz value
        if precursor_mz <= 50:
            spectrum = select_by_mz(spectrum, 50, 1000)
        else:
            spectrum = select_by_mz(spectrum, 50, precursor_mz)

        # Further processing of the spectrum
        spectrum = remove_peaks_around_precursor_mz(spectrum)
        spectrum = remove_peaks_outside_top_k(spectrum, k=150)
        spectrum = custom_filter_peaks_below_1_percent(spectrum)
        spectrum = scale_intensities_by_sqrt(spectrum)
        if spectrum is not None:
            filtered_spectra.append(spectrum)
    return filtered_spectra

def save_spectra(spectra, output_path):
    """Save processed spectra to an MGF file."""
    save_as_mgf(spectra, output_path)
