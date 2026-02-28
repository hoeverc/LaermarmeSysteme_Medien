"""Compute third-octave band spectrum from narrowband data."""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple


def get_third_octave_band_spectrum(
  f: NDArray[np.float64],
  lp: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
  """Compute third-octave band spectrum from narrowband levels.

  Groups narrowband sound pressure levels into third-octave
  bands by energetic summation within each band.

  Parameters
  ----------
  f : array_like
      Frequency vector in Hz (1-D, sorted ascending).
  lp : array_like
      Narrowband sound pressure levels in dB (1-D).

  Returns
  -------
  tmf : NDArray[np.float64]
      Third-octave band centre frequencies in Hz.
  lp_third : NDArray[np.float64]
      Third-octave band levels in dB.
  """
  f=np.asarray(f, dtype=float)
  lp=np.asarray(lp, dtype=float)
  assert f.ndim==1 and lp.ndim==1 and len(f)==len(lp), \
    "f and lp must be 1-D vectors of equal length"

  # Preferred and real third-octave band middle frequencies
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

  # Band edge frequencies (lower/upper)
  tlf_real=tmf_real/2**(1/6)
  tuf_real=tmf_real*2**(1/6)

  # Determine which bands fall within the data range
  # First band whose lower edge <= f[0]
  mask_low=f[0]>tlf_real
  ind_low=(np.where(mask_low)[0][-1]+1
    if np.any(mask_low) else 0)

  # Last band whose upper edge >= f[-1]
  mask_high=f[-1]<tuf_real
  ind_high=(np.where(mask_high)[0][-1]
    if np.any(mask_high) else len(tmf_pref)-1)

  # Select bands within range
  tmf=tmf_pref[ind_low:ind_high+1]
  tlf_r=tlf_real[ind_low:ind_high+1]
  tuf_r=tuf_real[ind_low:ind_high+1]

  # Energetic summation within each third-octave band
  lp_third=np.zeros(len(tmf))
  for nn in range(len(tmf)):
    mask=(f>=tlf_r[nn]) & (f<=tuf_r[nn])
    if np.any(mask):
      lp_third[nn]=10*np.log10(
        np.sum(10**(lp[mask]/10))
      )
    else:
      lp_third[nn]=-np.inf

  return tmf, lp_third
