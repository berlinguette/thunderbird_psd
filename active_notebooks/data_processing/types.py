from typing import Any, Callable, Literal, NamedTuple, TypeVar, Sequence

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pandas import Series


class BimodalParams(NamedTuple):
    mu1: float
    sigma1: float
    a1: float
    mu2: float
    sigma2: float
    a2: float


class GaussianParams(NamedTuple):
    mu: float
    sigma: float
    a: float


BimodalBounds = tuple[BimodalParams, BimodalParams]
BoundsSequence = Sequence[tuple[tuple[int, int], BimodalBounds]]


def unpack_bimodal_params(
    params: BimodalParams,
) -> tuple[float, float, float, float, float, float]:
    return (params.mu1, params.sigma1, params.a1, params.mu2, params.sigma2, params.a2)


def unpack_gaussian_params(params: GaussianParams) -> tuple[float, float, float]:
    return (params.mu, params.sigma, params.a)


class FitResult(NamedTuple):
    index: int
    gamma_params: GaussianParams | None
    neutron_params: GaussianParams | None
    slice_left_edge: float
    slice_right_edge: float
    fom: float | None


class FitErrorResult(NamedTuple):
    index: int
    gamma_params: GaussianParams | None
    neutron_params: GaussianParams | None
    slice_left_edge: float
    slice_right_edge: float


UnpackedFitResult = tuple[
    int,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float,
    float,
    float | None,
]
UnpackedFitErrorResult = tuple[
    int,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float,
    float,
]

VectorLike = TypeVar("VectorLike", float, Series, NDArray)
VectorLikeFunction = Callable[[VectorLike], VectorLike]


class WindowBorders(NamedTuple):
    left: float | None
    right: float | None
    bottom: VectorLikeFunction | None
    top: VectorLikeFunction | None


class NasaGenerationSettings(NamedTuple):
    window_offset: float = 0.2
    sigma: float = 5
    lower_energy_bound: float = 0.1966
    upper_energy_bound: float | None = None
    recalculate_lower_energy_bound: bool = False
    use_filter: bool = False
    filter_window: int = 21
    filter_order: int = 3


class NeutronDistributionGenerationSettings(NamedTuple):
    sigma: float = 3
    lower_energy_bound: float = 0.1966,
    upper_energy_bound: float = 0.688,
    recalculate_lower_energy_bound: bool = False,
    fom_energy_range: tuple[float, float] = (0.10, 0.35)


class SquarishGenerationSettings(NamedTuple):
    left: float
    bottom: float
    width: float = 0.1
    aspect_ratio: float = 2.5/0.5  # width/height


class MixedDistributionGenerationSettings(NamedTuple):
    gamma_sigma: float = 3,
    neutron_sigma: float = 3,
    lower_energy_bound: float = 0.1966,
    upper_energy_bound: float = 0.688


class BasicCutSettings(NamedTuple):
    bottom: float
    top: float | None = None


NeutronWindowSettings = (
    NasaGenerationSettings |
    NeutronDistributionGenerationSettings |
    SquarishGenerationSettings |
    MixedDistributionGenerationSettings |
    BasicCutSettings |
    str
)
SpecificNeutronWindowSettings = TypeVar(
    "SpecificNeutronWindowSettings",
    NasaGenerationSettings,
    NeutronDistributionGenerationSettings,
    SquarishGenerationSettings,
    MixedDistributionGenerationSettings,
    BasicCutSettings,
    str,
)


Kwargs = dict[str, Any]
GraphData = dict[Literal["x"] | Literal["y"], Series]
GraphingFunction = Callable[[Figure, Axes, GraphData, Kwargs], Axes]
AxesMatrix = list[list[Axes]]

DictKey = TypeVar("DictKey")
DictValue = TypeVar("DictValue")


class LinearCalibrationParams(NamedTuple):
    p1: float
    p2: float

class LogCurveCalibrationParams(NamedTuple):
    a: float
    b: float
    c: float

WindowType = Literal["nasa", "n_distro", "mixed_distro", "squarish", "basic_cut"]
SliceFitStyle = Literal["bounds", "peak_finder"]
NumberedSlice = tuple[int, Any]
SliceFitResult = tuple[FitResult, FitErrorResult]
