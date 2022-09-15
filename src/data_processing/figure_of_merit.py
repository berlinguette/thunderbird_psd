import numpy as np
from scipy import signal
from scipy.optimize import curve_fit


def gaussian(x: list, mu: float, sigma: float, A: float) -> list:
    """Returns a Gaussian distribution"""
    return A * np.exp(-((x - mu)**2 / (2 * (sigma**2))))


def bimodal(
    x: list,
    mu1: float,
    sigma1: float,
    A1: float,
    mu2: float,
    sigma2: float,
    A2: float
) -> list:
    """Returns a bimodal Gaussian distribution"""
    return gaussian(x, mu1, sigma1, A1) + gaussian(x, mu2, sigma2, A2)


def FWHM(sigma: float) -> float:
    """Returns Full Width Half Maximum of a Gaussian distribution"""
    return 2 * np.sqrt(2 * np.log(2)) * sigma


def FOM(mu1: float, sigma1: float, mu2: float, sigma2: float) -> float:
    return abs(mu2 - mu1) / (FWHM(sigma1) + FWHM(sigma2))


def guess_bimodal_params(
    counts: np.ndarray,
    bins: np.ndarray,
    widths: list = [3, 5, 10, 20]
) -> float:
    """Returns a guess starting condition for :func:`scipy.optimize.curvefit`"""
    idxs = signal.find_peaks_cwt(counts, widths)  # May need adjusting?

    if len(idxs) > 2:
        pk_idx1 = counts[idxs].argmax()
        pk_idx2 = np.delete(counts, pk_idx1).argmax()
    else:
        pk_idx1, pk_idx2 = idxs

    sigma1, sigma2 = 0.02, 0.02

    return bins[:-1][pk_idx1], sigma1, counts[pk_idx1], bins[:-1][pk_idx2], sigma2, counts[pk_idx2]


def fit_fom(
    counts: list,
    bins: list,
    guesses: tuple = None
) -> tuple[tuple, float]:
    """Attempts to fit two Gaussians to the Figure of Merit"""

    if guesses is None:
        try:
            starting_guesses = guess_bimodal_params(counts, bins)
        except ValueError:
            starting_guesses = (0.2, 0.01, 800, 0.4, 0.01, 200)
    else:
        starting_gueses = guesses

    params, cov = curve_fit(bimodal, bins[:-1], counts, starting_guesses)

    return abs(params), cov
