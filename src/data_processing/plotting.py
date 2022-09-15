import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_processing.figure_of_merit import fit_fom, gaussian, FOM


def plot_signal(
    data: pd.DataFrame,
    sample_interval: float = 2
) -> tuple[plt.Figure, plt.Axes]:
    """Plots signals
    """
    if data is None:
        return None

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(
        np.arange(0, data.shape[0] * sample_interval, sample_interval),
        data
    )

    ax.grid(visible=True)
    ax.tick_params(axis='both', labelsize=14)

    ax.set_xlabel("time (ns)", fontsize=18)
    ax.set_ylabel("amplitude ($V$)", fontsize=18)

    return fig, ax


def plot_fom(
    psd: list,
    unimodal: bool = False,
    n_bins: int = 100,
    guesses: tuple = None
) -> tuple[plt.Figure, plt.Axes]:
    """"""
    counts, bins = np.histogram(psd, n_bins)

    params, cov = fit_fom(counts, bins, n_bins=n_bins, guesses=guesses)

    fig, ax = plt.subplots(figsize=(12, 9))

    ax.plot(
        bins[:-1],
        counts,
        "--",
        label="Original PSD"
    )

    if not unimodal:
        ax.plot(
            bins[:-1],
            gaussian(bins[:-1], *params[0:3]),
            label="Neutron",
            c="orange"
        )

    ax.plot(
        bins[:-1],
        gaussian(bins[:-1], *params[3:]),
        label="$\gamma$",
        c="indigo"
    )

    fom_val = "N/A" if unimodal else f"{FOM(params[0], params[1], params[3], params[4]):.3f}"
    ax.set_xlim(-0.1, 0.8)
    ax.set_title(f"FoM: {fom_val}")
    ax.legend()

    ax.set_ylabel("counts", fontsize=14)
    ax.set_xlabel("tail/total (a.u.)", fontsize=14)

    params_dict = {"mu1": params[0], "sigma1": params[1], "A1": params[2],
                   "mu2": params[3], "sigma2": params[4], "A2": params[5]}

    return fig, ax, params_dict, cov
