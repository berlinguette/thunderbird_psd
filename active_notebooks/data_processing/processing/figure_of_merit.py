"""This module is responsible for figure of merit calculation."""
import numpy as np
import pandas as pd
from data_processing.types import BimodalParams
from scipy import interpolate, signal
from scipy.optimize import curve_fit


def gaussian(x: np.ndarray, mu: float, sigma: float, A: float) -> np.ndarray:
    """Gaussian distribution function.

    :param x: Array of random variable values
    :type x: np.ndarray
    :param mu: Mean value of Gaussian distribution (or position of peak)
    :type mu: float
    :param sigma: Standard deviation of Gaussian distribution (controlling width of peak)
    :type sigma: float
    :param A: Vertical scaling factor (or height of peak)
    :type A: float
    :return: Result of Gaussian distribution function for given x values
    :rtype: np.ndarray
    """    
    """Returns a Gaussian distribution"""
    return A * np.exp(-((x - mu) ** 2 / (2 * (sigma**2))))


def bimodal(
    x: np.ndarray,
    mu1: float,
    sigma1: float,
    A1: float,
    mu2: float,
    sigma2: float,
    A2: float,
) -> np.ndarray:
    """Bimodal distribution function.
    The bimodal distribution is the sum of 2 independent Gaussian distributions.

    :param x: Array of random variable values
    :type x: np.ndarray
    :param mu1: Mu for the first Gaussian distribution
    :type mu1: float
    :param sigma1: Sigma for the first Gaussian distribution
    :type sigma1: float
    :param A1: Vertical scaling factor for the first Gaussian distribution
    :type A1: float
    :param mu2: Mu for the second Gaussian distribution
    :type mu2: float
    :param sigma2: Sigma for the second Gaussian distribution
    :type sigma2: float
    :param A2: Vertical scaling factor for the second Gaussian distribution
    :type A2: float
    :return: Result of bimodal distribution function for given x values
    :rtype: np.ndarray
    """    
    """Returns a bimodal Gaussian distribution"""
    return gaussian(x, mu1, sigma1, A1) + gaussian(x, mu2, sigma2, A2)


def FWHM(sigma: float) -> float:
    """Returns "full width half maximum", or FWHM, of a Gaussian distribution.
    This is the width of the Gaussian "peak" at half its maximum height.

    :param sigma: Sigma of a Gaussian distribution
    :type sigma: float
    :return: FWHM for the given Gaussian distribution parameters
    :rtype: float
    """    
    return 2 * np.sqrt(2 * np.log(2)) * sigma


def FOM(mu1: float, sigma1: float, mu2: float, sigma2: float) -> float:
    """Returns "figure of merit" value, or FOM, for a given bimodal distribution.
    This value represents the separation between peaks in the bimodal distribution.

    :param mu1: Mu value of the first Gaussian in the bimodal distribution
    :type mu1: float
    :param sigma1: Sigma value of the first Gaussian in the bimodal distribution
    :type sigma1: float
    :param mu2: Mu value of the second Gaussian in the bimodal distribution
    :type mu2: float
    :param sigma2: Sigma value of the second Gaussian in the bimodal distribution
    :type sigma2: float
    :return: FOM value for given bimodal distribution parameters
    :rtype: float
    """
    return abs(mu2 - mu1) / (FWHM(sigma1) + FWHM(sigma2))


def guess_bimodal_params(
    counts: np.ndarray, bins: np.ndarray, widths: list = [3, 5, 10, 20]
) -> BimodalParams:
    """Returns a guess of bimodal parameters.
    This guess can be used with the curvefit() function from scipy.optimize

    :param counts: Histogram of neutron detector counts vs PSD 
    :type counts: np.ndarray
    :param bins: PSD bin edges
    :type bins: np.ndarray
    :param widths: Array of widths for CWT matrix, see scipy.signal.find_peaks_cwt, defaults to [3, 5, 10, 20]
    :type widths: list, optional
    :return: Guess of bimodal distribution parameters
    :rtype: BimodalParams
    """    
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


def fit_fom(
    counts: np.ndarray, bins: np.ndarray, guesses: BimodalParams | None = None
) -> tuple[tuple, float]:
    """Atempts to fit two Gaussian distributions to the Figure of Merit value

    :param counts: Histogram of neutron detector counts vs PSD 
    :type counts: np.ndarray
    :param bins: PSD bin edges
    :type bins: np.ndarray
    :param guesses: Guess of bimodal fit parameters for the given PSD histogram, defaults to None
    :type guesses: BimodalParams | None, optional
    :return: Curve fit parameters, curve fit covariance
    :rtype: tuple[tuple, float]
    """
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
    """[DEPRECATED]Classifies the neutrons given a value 'n' sigma above the sum of 
    mean and standard deviation of the gamma ray distribution

    :param psd: Dataframe with PSD values
    :type psd: pd.DataFrame
    :param gauss_params: Mu, sigma and A of a Gaussian distribution
    :type gauss_params: tuple
    :param n: Number of sigmas above gamma Gaussian distribution to count as neutrons
    :type n: float
    :param n_bins: number of PSD bins to use, defaults to 100
    :type n_bins: int, optional
    :return: Neutron dataframe, gamma ray dataframe, gate function, gamma function
    :rtype: tuple[pd.DataFrame, pd.DataFrame, interpolate.interp1d, interpolate.interp1d]
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
        y_gate_bins, y_gate, kind="linear", fill_value="extrapolate"  # type: ignore
    )
    f_gamma = interpolate.interp1d(
        y_gamma_bins, y_gamma, kind="linear", fill_value="extrapolate"  # type: ignore
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


def n_sigma_classifier(psd: pd.DataFrame, gauss_params: tuple, n: float) -> tuple:
    """Classifies the neutrons given a value 'n' sigma above the sum of 
    mean and standard deviation of the gamma ray distribution

    :param psd: Dataframe with PSD values
    :type psd: pd.DataFrame
    :param gauss_params: Mu, sigma and A of a Gaussian distribution
    :type gauss_params: tuple
    :rtype: tuple[pd.DataFrame, pd.DataFrame
    :return: Neutron dataframe, gamma ray dataframe
    """
    mu, sigma, A = gauss_params

    def neutron_filter(signal):
        # amplitude and tail/total swapped because axis are swapped during graphing
        gate_cond = signal["amplitude"] <= gaussian(
            signal["tail / total"], mu + n * sigma, sigma, A
        )
        gamma_cond = signal["amplitude"] > gaussian(
            signal["tail / total"], *gauss_params
        )
        return gate_cond and gamma_cond

    filt = psd.apply(neutron_filter)

    neutrons = psd.T[filt].T
    gammas = psd.T[~filt].T

    return neutrons, gammas
