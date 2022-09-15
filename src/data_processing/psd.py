import pandas as pd
from data_processing.peak_finding import get_bases


def generate_psd(
    df: pd.DataFrame,
    left_bases: dict,
    right_bases: dict,
    amplitudes: dict,
    cutoff_voltage: float,
) -> pd.DataFrame:
    """Returns the Pulse Shape Discrimination report"""

    def process(series: pd.Series) -> dict:
        q_total = series[left_bases[series.name] :].sum()
        q_tail = series[right_bases[series.name] :].sum()

        res = {
            "total integral": q_total,
            "tail integral": q_tail,
            "tail / total": q_tail / q_total,
        }

        return res

    psd_report = pd.DataFrame(
        {signal_id: process(df[signal_id]) for signal_id in df.columns}
    )
    psd_report.loc["amplitude"] = amplitudes

    filt = psd_report.loc["tail / total"].between(-0.1, 0.8)
    psd_report = psd_report.T[filt].T

    amp_filt = psd_report.loc["amplitude"] >= cutoff_voltage

    return psd_report.T[amp_filt].T


def df_to_psd(
    df: pd.DataFrame,
    dc_offset: float,
    peak_offset: float,
    tail_onset: float,
    cutoff_voltage: float,
) -> pd.DataFrame:
    """
    This function applies the full workflow of this notebook to the input df.

    Args
    -----
    peak_offset: index shifted from the peak where its base may begin; used to calculate the peak's base value
    tail_onset: index shifted from the peak where the tail should start
    """
    df = df - dc_offset
    output = df.apply(lambda x: get_bases(x, x.idxmax(), peak_offset, tail_onset))
    left_bases, right_bases = output.iloc[0, :], output.iloc[1, :]
    return generate_psd(df, left_bases, right_bases, df.max(), cutoff_voltage)
