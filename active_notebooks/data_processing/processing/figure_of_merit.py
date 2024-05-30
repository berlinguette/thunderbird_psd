import numpy as np
import pandas as pd
from scipy import interpolate, signal
from scipy.optimize import curve_fit
from bimodal_fitting import BimodalParams


def gaussian(x: np.ndarray, mu: float, sigma: float, A: float) -> np.ndarray:
    """Returns a Gaussian distribution"""
    return A * np.exp(-((x - mu) ** 2 / (2 * (sigma**2))))


def bimodal(
    x: np.ndarray, 
    mu1: float, sigma1: float, A1: float, 
    mu2: float, sigma2: float, A2: float
) -> np.ndarray:
    """Returns a bimodal Gaussian distribution"""
    return gaussian(x, mu1, sigma1, A1) + gaussian(x, mu2, sigma2, A2)


def FWHM(sigma: float) -> float:
    """Returns Full Width Half Maximum of a Gaussian distribution"""
    return 2 * np.sqrt(2 * np.log(2)) * sigma


def FOM(mu1: float, sigma1: float, mu2: float, sigma2: float) -> float:
    return abs(mu2 - mu1) / (FWHM(sigma1) + FWHM(sigma2))


def guess_bimodal_params(
    counts: np.ndarray, bins: np.ndarray, widths: list = [3, 5, 10, 20]
) -> float:
    """Returns a guess starting condition for :func:`scipy.optimize.curvefit`"""
    idxs = signal.find_peaks_cwt(counts, widths)  # May need adjusting?

    if len(idxs) > 2:
        pk_idx1 = counts[idxs].argmax()
        pk_idx2 = np.delete(counts, pk_idx1).argmax()
    else:
        pk_idx1, pk_idx2 = idxs

    sigma1, sigma2 = 0.02, 0.02

    return BimodalParams(
        bins[:-1][pk_idx1],
        sigma1,
        counts[pk_idx1],
        bins[:-1][pk_idx2],
        sigma2,
        counts[pk_idx2],
    )


def fit_fom(counts: np.ndarray, bins: np.ndarray, guesses: BimodalParams|None = None) -> tuple[tuple, float]:
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


def n_sigma_classifier_OLD(
    psd: pd.DataFrame, gauss_params: tuple, n: float, n_bins: int = 100
) -> tuple[pd.DataFrame, pd.DataFrame, interpolate.interp1d, interpolate.interp1d]:
    """Classifies the neutrons given a value :param:`n` sigma above the mean + stdev
    of the gamma distribution
    """
    _, bins = np.histogram(psd.loc["tail / total"], n_bins)

    mu, sigma, A = gauss_params
    gamma_gauss = gaussian(bins[:-1], mu, sigma, A)
    gate_gauss = gaussian(bins[:-1], mu + n * sigma, sigma, A)

    y_gate = gate_gauss[gate_gauss.argmax() :]
    y_gate_bins = bins[:-1][gate_gauss.argmax() :]
    y_gamma = gamma_gauss[gamma_gauss.argmax() :]
    y_gamma_bins = bins[:-1][gamma_gauss.argmax() :]

    f_gate = interpolate.interp1d(
        y_gate_bins, y_gate, kind="linear", fill_value="extrapolate"
    )
    f_gamma = interpolate.interp1d(
        y_gamma_bins, y_gamma, kind="linear", fill_value="extrapolate"
    )

    def neutron_filter(signal):
        # amplitude and tail/total swapped because axis are swapped during graphing
        gate_cond = signal["amplitude"] <= f_gate(signal["tail / total"])
        gamma_cond = signal["amplitude"] > f_gamma(signal["tail / total"])
        return gate_cond and gamma_cond

    filt = psd.apply(neutron_filter)

    neutrons = psd.T[filt].T
    gammas = psd.T[~filt].T

    return neutrons, gammas, f_gate, f_gamma

def n_sigma_classifier(
    psd: pd.DataFrame, gauss_params: tuple, n: float
) -> tuple:

    mu, sigma, A = gauss_params

    def neutron_filter(signal):
        # amplitude and tail/total swapped because axis are swapped during graphing
        gate_cond = signal["amplitude"] <= gaussian(signal["tail / total"], mu + n*sigma, sigma, A)
        gamma_cond = signal["amplitude"] > gaussian(signal["tail / total"], *gauss_params)
        return gate_cond and gamma_cond

    filt = psd.apply(neutron_filter)

    neutrons = psd.T[filt].T
    gammas = psd.T[~filt].T

    return neutrons, gammas
