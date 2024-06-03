from typing import Callable, NamedTuple, TypeVar, Any, Literal

from numpy.typing import NDArray
from pandas import Series
from matplotlib.axes import Axes
from matplotlib.figure import Figure


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


def unpack_bimodal_params(
    params: BimodalParams
) -> tuple[float, float, float, float, float, float]:
    return (
        params.mu1,
        params.sigma1,
        params.a1,
        params.mu2,
        params.sigma2,
        params.a2
    )
    

def unpack_gaussian_params(
    params: GaussianParams
) -> tuple[float, float, float]:
    return (
        params.mu,
        params.sigma,
        params.a
    )


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


Kwargs = dict[str, Any]
GraphData = dict[Literal["x"] | Literal["y"], Series]
GraphingFunction = Callable[[Figure, Axes, GraphData, Kwargs], Axes]
AxesMatrix = list[list[Axes]]

DictKey = TypeVar("DictKey")
DictValue = TypeVar("DictValue")
