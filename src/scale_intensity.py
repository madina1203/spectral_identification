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
