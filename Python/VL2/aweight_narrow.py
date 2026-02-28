"""A-weighting for narrowband frequency data."""

import numpy as np
from numpy.typing import NDArray


def aweight_narrow(
  f: NDArray[np.float64],
  lp: NDArray[np.float64],
) -> NDArray[np.float64]:
  """Apply A-weighting to narrowband frequency data.

  Uses the analytical A-weighting transfer function.

  Parameters
  ----------
  f : array_like
      Frequency vector in Hz.
  lp : array_like
      Sound pressure levels in dB.

  Returns
  -------
  NDArray[np.float64]
      A-weighted sound pressure levels in dB(A).
  """
  f=np.asarray(f, dtype=float)
  lp=np.asarray(lp, dtype=float)
  assert len(f)==len(lp), \
    "f and lp must be vectors of same length"

  # A-weighting transfer function (numerator / denominator)
  numer=12200**2*f**4
  denom=((f**2+20.6**2)
    *np.sqrt((f**2+107.7**2)*(f**2+737.9**2))
    *(f**2+12200**2))
  r_a=numer/denom

  # Offset in dB and apply
  aw=2+20*np.log10(r_a)
  return lp+aw
