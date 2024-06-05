from enum import Enum

class ExperimentDataKey(Enum):
    UNCLASSIFIED = "unclassified"
    CAEN_CALIBRATION = "caen_calibration"
    NEW_CALIBRATION = "new_calibration"
    PSD_HISTOGRAM = "psd_histogram"
    HISTOGRAM_X_EDGES = "histogram_x_edges"
    HISTOGRAM_Y_EDGES = "histogram_y_edges"
    END_SCAN_IDX = "end_scan_idx"
    VALID_SLICE_FITS = "valid_slice_fits"
    BAD_SLICE_INDEXES = "bad_slice_indexes"
    FOM_RESULTS = "fom_results"
    N_WINDOW_BORDERS = "n_window_borders"
    NASA_BORDERS = "nasa_borders"
    NASA_BORDERS_RECALC = "nasa_borders_recalc"
    PSD_REPORT = "psd_report"