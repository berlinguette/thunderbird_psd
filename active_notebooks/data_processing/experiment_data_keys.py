from enum import Enum
from typing import Any

class ExperimentDataKey(Enum):
    UNCLASSIFIED = "unclassified"
    BY_CHANNEL = "by_channel"
    CAEN_CALIBRATION = "caen_calibration"
    NEW_CALIBRATION = "new_calibration"
    PSD_HISTOGRAM = "psd_histogram"
    HISTOGRAM_X_EDGES = "histogram_x_edges"
    HISTOGRAM_Y_EDGES = "histogram_y_edges"
    END_SCAN_IDX = "end_scan_idx"
    VALID_SLICE_FITS = "valid_slice_fits"
    BAD_SLICE_INDEXES = "bad_slice_indexes"
    FOM_RESULTS = "fom_results"
    BORDERS = "borders"
    N_WINDOW_BORDERS = "n_window_borders"
    NASA_BORDERS = "nasa_borders"
    NASA_BORDERS_RECALC = "nasa_borders_recalc"
    BORDER_STRATEGY = "border_strategy"
    PSD_REPORT = "psd_report"
    START_TIME = "start_time"
    NEUTRONS_ONLY = "neutrons_only"
    GAMMA_ONLY = "gamma_only"
    TIME_BIN_EDGES = "time_bin_edges"
    BINNED_NEUTRONS = "binned_neutrons"
    BINNED_GAMMA = "binned_gamma"
    GAMMA_ENERGY_SPECTRUM = "gamma_energy_spectrum"
    GAMMA_ENERGY_BIN_EDGES = "gamma_energy_bin_edges"
    REACTOR_DATA = "reactor_data"
    BINNED_REACTOR_DATA = "binned_reactor_data"
    ALL_BINNED_DATA = "all_binned_data"
    PULSE_HEIGHT_DISTRIBUTION = "pulse_height_distribution"
    HISTOGRAM_ENERGY_BIN_EDGES = "histogram_energy_bin_edges"


ExperimentNeutronData = dict[str, dict[ExperimentDataKey, Any]]
