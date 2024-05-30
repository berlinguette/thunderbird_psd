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

VectorLike = TypeVar("VectorLike", float, Series, NDArray)
WindowBorderFunction = Callable[[VectorLike], VectorLike]


class WindowBorders(NamedTuple):
    left: float | None
    right: float | None
    bottom: WindowBorderFunction | None
    top: WindowBorderFunction | None

Kwargs = dict[str, Any]
GraphData = dict[Literal["x"] | Literal["y"], Series]
GraphingFunction = Callable[[Figure, Axes, GraphData, Kwargs], Axes]
AxesMatrix = list[list[Axes]]

DictKey = TypeVar("DictKey")
DictValue = TypeVar("DictValue")
