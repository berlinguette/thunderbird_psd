from pathlib import Path

import numpy as np
import pandas as pd
from data_processing.processing.spectrum_unfolding import NDHistogram


def load_neutron_response_matrix(
    dir: Path, min_L: float = 0, max_L: float = 7, L_bin_widths: float = 0.005
) -> NDHistogram:
    widths = [15, 15, 20, 15, 15, 15, 15, 15]
    output_files = [file for file in dir.iterdir() if "output" in file.stem]
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
