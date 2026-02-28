"""A-weighting for octave and third-octave band data."""

import numpy as np
from numpy.typing import NDArray


def aweight(
  f: NDArray[np.float64],
  lp: NDArray[np.float64],
) -> NDArray[np.float64]:
  """Apply A-weighting to octave or third-octave band levels.

  Parameters
  ----------
  f : array_like
      Centre frequencies (octave or third-octave bands).
  lp : array_like
      Sound pressure levels in dB.

  Returns
  -------
  NDArray[np.float64]
      A-weighted sound pressure levels in dB(A).
  """
  # Reference centre frequencies
  fref=np.array([
    10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80,
    100, 125, 160, 200, 250, 315, 400, 500, 630,
    800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
    5000, 6300, 8000, 10000, 12500, 16000, 20000
  ])
  # A-weighting corrections in dB
  aw_ref=np.array([
    -70.4, -63.4, -56.7, -50.5, -44.7, -39.4,
    -34.6, -30.2, -26.2, -22.5, -19.1, -16.1,
    -13.4, -10.9, -8.6, -6.6, -4.8, -3.2, -1.9,
    -0.8, 0, 0.6, 1.0, 1.2, 1.3, 1.2, 1.0, 0.5,
    -0.1, -1.1, -2.5, -4.3, -6.6, -9.3
  ])

  f_arr=np.asarray(f, dtype=float)
  lp_arr=np.asarray(lp, dtype=float)

  # Validate frequencies
  if not np.all(np.isin(f_arr, fref)):
    raise ValueError(
      "One or more entries in f are not valid "
      "centre frequencies"
    )

  # Look up A-weighting corrections via dict
  aw_dict=dict(zip(fref, aw_ref))
  aw=np.array([aw_dict[fi] for fi in f_arr])

  return lp_arr+aw
