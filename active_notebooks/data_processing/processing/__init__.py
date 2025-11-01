from .calibration import Detector, recalibrate
from .dataframe_manipulation import generate_full_neutron_df, generate_neutron_signals
from .figure_of_merit import FOM, FWHM, bimodal, gaussian, guess_bimodal_params
from .neutron_classification import classify
from .neutron_window_generation import (
    generate_n_distro_neutron_window,
    generate_nasa_neutron_window,
    generate_rectangle_neutron_window,
)
from .neutron_window_strategy import (
    AbstractNeutronStrategy,
    LoadingStrategy,
    MixedDistributionGenerationStrategy,
    NasaGenerationStrategy,
    NeutronDistributionGenerationStrategy,
    NeutronStrategyFactory,
    SquarishGenerationStrategy,
)
from .slice_fitting import (
    find_failed_slices,
    get_bimodal_fit,
    get_bimodal_fit_guess,
    get_psd_energy_histogram,
    scan_histogram_slices,
    split_params,
    unpack_slice_fit_pool_results,
)
from .spectrum_unfolding import (
    NDHistogram,
    UnfoldingProcessInfo,
    weight_factor,
    next_phi,
    stopping_criteria,
    unfold_spectrum,
    clean_data,
    cut_low_l,
    r_dot
)