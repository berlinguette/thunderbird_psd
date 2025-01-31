import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from data_processing.processing.spectrum_unfolding import NDHistogram


filestem_pattern = re.compile(r"neutron_(\d\.\d{3})_MeV")


def load_neutron_response_matrix(
    dir: Path,
    min_L: float = 0,
    max_L: float = 7,
    L_bin_widths: float = 0.005,
    format: Literal["fwf", "npy"] | None = None,
) -> NDHistogram:
    """Loads a neutron response matrix from simulation result files.

    Files can be in MCNP fixed width format or Numpy ndarray format.

    Simulated neutron light output (L) values will be used to create a 2D NDHistogram,
    where axis 0 represents L, and axis 1 represents neutron energy (E).

    :param dir: Folder holding all response matrix files
    :type dir: Path
    :param min_L: minimum L value used in histogram, defaults to 0
    :type min_L: float, optional
    :param max_L: maximum L value used in histogram, defaults to 7
    :type max_L: float, optional
    :param L_bin_widths: width of L bins used in histogram, defaults to 0.005
    :type L_bin_widths: float, optional
    :param format: File format, defaults to None (i.e. infer from filenames)
    :type format: Literal['fwf', 'npy'] | None, optional
    :raises ValueError: if folder contains no data files matching the format
    :raises ValueError: if format cannot be inferred
    :return: Neutron response matrix
    :rtype: NDHistogram
    """
    files = [file for file in dir.iterdir() if file.is_file()]
    if len(files) <= 0:
        raise ValueError("Provided folder contains no data files")

    if format is None:
        format = _infer_format(files)

    if format == "fwf":
        return _load_R_from_fwf(files, min_L, max_L, L_bin_widths)
    elif format == "npy":
        return _load_R_from_npy(files, min_L, max_L, L_bin_widths)


def _load_R_from_fwf(
    files: list[Path], min_L: float = 0, max_L: float = 7, L_bin_widths: float = 0.005
) -> NDHistogram:
    """Loads a neutron response matrix from MCNP fixed width files.

    :param files: Data files to be used
    :type files: list[Path]
    :param min_L: minimum L value used in histogram, defaults to 0
    :type min_L: float, optional
    :param max_L: maximum L value used in histogram, defaults to 7
    :type max_L: float, optional
    :param L_bin_widths: width of L bins used in histogram, defaults to 0.005
    :type L_bin_widths: float, optional
    :raises ValueError: if no expected data files are given
    :raises RuntimeError: if data binning does not give expected data formats
    :return: Neutron response matrix
    :rtype: NDHistogram
    """
    widths = [15, 15, 20, 15, 15, 15, 15, 15]

    output_files = [
        file for file in files if "output" in file.stem and file.suffix == ".txt"
    ]
    if len(output_files) <= 0:
        raise ValueError("Provided folder contains no 'output' data files")

    response_sim_data: list[tuple[np.float64, np.ndarray]] = []
    response_Ls: np.ndarray | None = None

    for output_file in output_files:
        df = pd.read_fwf(output_file, widths=widths)

        source_e = df["source_e (MeV)"][0].astype(np.float64)
        assert all(df["source_e (MeV)"] == source_e)

        bins = np.arange(min_L, max_L + L_bin_widths, L_bin_widths)
        cut = pd.cut(df["det_pulse (MeVee)"], bins.tolist())
        cut_index = cut.cat.categories
        new_df = pd.DataFrame(
            df["NPS"].groupby(cut, observed=True).sum().reindex(cut_index, fill_value=0)
        )

        old_index = new_df.index
        if not isinstance(old_index, pd.IntervalIndex):
            raise RuntimeError(
                f"DataFrame created from cut/groupby for {output_file.name} "
                + "was not an IntervalIndex as expected"
            )
        mids = old_index.mid.to_series(index=cut_index)

        np_cps = new_df["NPS"].to_numpy()
        np_Ls = mids.to_numpy()

        response_sim_data.append((source_e, np_cps))
        if response_Ls is None:
            response_Ls = np_Ls

    response_sim_data = sorted(response_sim_data, key=lambda x: float(x[0]))
    response_Es, response_histos = zip(*response_sim_data)

    np_R = np.array(response_histos).T
    np_Es = np.array(response_Es)
    return NDHistogram(np_R, [np_Ls, np_Es])


def _load_R_from_npy(
    files: list[Path], min_L: float = 0, max_L: float = 7, L_bin_widths: float = 0.005
) -> NDHistogram:
    """Loads a neutron response matrix from GEANT4 .npy files.

    :param files: Data files to be used
    :type files: list[Path]
    :param min_L: minimum L value used in histogram, defaults to 0
    :type min_L: float, optional
    :param max_L: maximum L value used in histogram, defaults to 7
    :type max_L: float, optional
    :param L_bin_widths: width of L bins used in histogram, defaults to 0.005
    :type L_bin_widths: float, optional
    :raises ValueError: if no expected data files are given
    :return: Neutron response matrix
    :rtype: NDHistogram
    """
    neutron_files = [
        file
        for file in files
        if filestem_pattern.match(file.stem) is not None and file.suffix == ".npy"
    ]
    if len(neutron_files) <= 0:
        raise ValueError("Provided folder contains no 'neutron' data files")

    response_sim_data: list[tuple[float, np.ndarray]] = []
    response_Ls: np.ndarray | None = None

    for neutron_file in neutron_files:
        L_array = np.load(neutron_file)
        source_e_matches = filestem_pattern.match(neutron_file.stem)
        source_e = float(source_e_matches.group(1))  # type: ignore

        bins = np.arange(min_L, max_L + L_bin_widths, L_bin_widths)
        np_cps, *_ = np.histogram(L_array, bins=bins)
        np_Ls = (bins[1:] + bins[:-1]) / 2

        response_sim_data.append((source_e, np_cps))
        if response_Ls is None:
            response_Ls = np_Ls

    response_sim_data = sorted(response_sim_data, key=lambda x: float(x[0]))
    response_Es, response_histos = zip(*response_sim_data)

    np_R = np.array(response_histos).T
    np_Es = np.array(response_Es)
    return NDHistogram(np_R, [np_Ls, np_Es])


def _infer_format(files: list[Path]) -> Literal["fwf", "npy"]:
    """Infer response matrix data file type.

    Files can be in MCNP fixed width format or Numpy ndarray format.

    :raises ValueError: if format cannot be inferred
    :return: _description_
    :rtype: _type_
    """
    fwf_match = [
        file for file in files if "output" in file.stem and file.suffix == ".txt"
    ]
    if len(fwf_match) > 0:
        return "fwf"
    npy_match = [
        file
        for file in files
        if filestem_pattern.match(file.stem) is not None and file.suffix == ".npy"
    ]
    if len(npy_match) > 0:
        return "npy"
    raise ValueError("Cannot infer format")
