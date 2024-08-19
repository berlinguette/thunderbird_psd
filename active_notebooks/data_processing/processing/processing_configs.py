"""This module is responsible for storing processing configuration values."""
# [PROCESSING]
MAX_VOLTAGE = 2.5
FINE_DC_OFFSET = -0.00144

PEAK_OFFSET = 20
TAIL_ONSET = 8
CUTOFF_VOLTAGE = 0.38

OPTIMIZER_BOUNDS = (0.1, 0.35)
N_BINS = 100

CLASSIFIER_WINDOW_N = 5

# TODO get L0 from Californium source analysis
# for now, using value from "UBC Background 20221214-19 Update"
DEFAULT_LOWER_ENERGY_BOUND = 0.1966