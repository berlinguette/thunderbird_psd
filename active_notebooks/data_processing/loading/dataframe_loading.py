import pandas as pd
import numpy as np
from data_processing.arc_paths import get_parq_root, get_signals_root
from data_processing.dataframe_validation import STARTING_COL_NAMES, STARTING_COL_TYPES, PsdDfColumn

def load_psd(experiment_name: str) -> pd.DataFrame:
    psd_folder = get_parq_root(experiment_name)
    
    # Load PSD data to "psd_report" DataFrame
    psd_df = pd.read_parquet(psd_folder, columns=STARTING_COL_NAMES)
    psd_df = psd_df.astype(STARTING_COL_TYPES)

    # Calculate PSD value as new column "tail / total"
    energy_col = psd_df[PsdDfColumn.ENERGY.value]
    short_col = psd_df[PsdDfColumn.ENERGYSHORT.value]
    psd_col = (energy_col - short_col) / energy_col
    psd_df[PsdDfColumn.PSD] = psd_col
    
    psd_df = psd_df.dropna()
    psd_df = psd_df[psd_col.between(0,0.5)]
    
    return psd_df

def load_signals(experiment_name: str) -> pd.DataFrame:
    signals_folder = get_signals_root(experiment_name)
    
    signals_data = pd.read_parquet(signals_folder)
    return signals_data

