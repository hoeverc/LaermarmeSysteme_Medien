"""Look up real third-octave band middle frequencies."""

import numpy as np
from numpy.typing import NDArray


def lookup_real_third_oct_middle_freq(
  tmf: NDArray[np.float64],
) -> NDArray[np.float64]:
  """Look up real middle frequencies for third-octave bands.

  For any given preferred third-octave band middle frequency,
  returns the corresponding real (exact) middle frequency.

  Parameters
  ----------
  tmf : array_like
      Preferred third-octave band middle frequencies.

  Returns
  -------
  NDArray[np.float64]
      Real (exact) third-octave band middle frequencies.
  """
  tmf_pref=np.array([
    1, 1.25, 1.6, 2, 2.5, 3.15, 4, 5, 6.3, 8,
    10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80,
    100, 125, 160, 200, 250, 315, 400, 500, 630,
    800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
    5000, 6300, 8000, 10000, 12500, 16000, 20000
  ])
  tmf_real=np.array([
    0.98, 1.23, 1.55, 1.95, 2.46, 3.1, 3.91, 4.92,
    6.2, 7.81, 9.84, 12.4, 15.6, 19.7, 24.8, 31.3,
    39.4, 49.6, 62.5, 78.7, 99.2, 125, 157.5,
    198.4, 250, 315, 396.9, 500, 630, 793.7, 1000,
    1259.9, 1587.4, 2000, 2519.8, 3174.8, 4000,
    5039.7, 6349.6, 8000, 10079.4, 12699.2, 16000,
    20158.7
  ])

  _, _, ind_pref=np.intersect1d(
    np.asarray(tmf), tmf_pref, return_indices=True
  )
  return tmf_real[ind_pref]
