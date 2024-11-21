import logging
import numpy as np
from matchms.typing import SpectrumType
from matchms.Fragments import Fragments  # Adjusted relative import to absolute for clarity
from matchms import Spectrum
# Set up logging
logger = logging.getLogger(__name__)


def scale_intensities_by_sqrt(spectrum_in: SpectrumType) -> SpectrumType:
    """Scale the intensities of peaks by taking the square root.

    Args:
        spectrum_in (SpectrumType): The input spectrum.

    Returns:
        SpectrumType: The spectrum with scaled intensities.
    """
    if spectrum_in is None:
        return None

    # Clone the spectrum to avoid modifying the original object
    spectrum = spectrum_in.clone()

    if len(spectrum.peaks) == 0:
        return spectrum

    # Apply square root to each intensity
    new_intensities = np.sqrt(spectrum.peaks.intensities)

    # Update the peaks of the cloned spectrum
    spectrum.peaks = Fragments(mz=spectrum.peaks.mz, intensities=new_intensities)

    return spectrum

def custom_filter_peaks_below_1_percent(spectrum_in: SpectrumType) -> (SpectrumType):
    """Remove peaks below 1% of the base peak intensity."""
    if spectrum_in is None:
        logger.info("Received None spectrum, returning None.")
        return None

    # Clone the spectrum to avoid modifying the original object
    spectrum = spectrum_in.clone()

    # Proceed only if there are peaks in the spectrum
    if len(spectrum.peaks) == 0:
        logger.info("No peaks in spectrum, returning cloned spectrum.")
        return spectrum

    # Find the base peak intensity
    base_peak_intensity = np.max(spectrum.peaks.intensities)

    # Define the intensity threshold as 1% of the base peak intensity
    intensity_threshold = base_peak_intensity * 0.01

    # Filter peaks that are above the intensity threshold
    above_threshold_indices = spectrum.peaks.intensities >= intensity_threshold
    filtered_mzs = spectrum.peaks.mz[above_threshold_indices]
    filtered_intensities = spectrum.peaks.intensities[above_threshold_indices]

    # Check if any peaks remain after filtering
    if len(filtered_mzs) == 0:
        logger.warning("All peaks are below the 1% threshold, returning None.")
        return None


    # Update the peaks of the cloned spectrum
    spectrum.peaks = Fragments(mz=filtered_mzs, intensities=filtered_intensities)

    return spectrum







