from pydantic import BaseModel, Field, model_validator, computed_field, field_validator
from numpy.typing import ArrayLike
from data_processing.reporting.plot_configs import *
from matplotlib import colormaps
from matplotlib.colors import Colormap
from data_processing.types import AxisLimits, AxesLimits
from typing import Any


class BasePlotConfig(BaseModel):
    width: float = FIG_DIM_X
    height: float = FIG_DIM_Y
    supertitle_fontsize: float = SUPTITLE_FONT_SIZE
    title_fontsize: float = TITLE_FONT_SIZE
    axis_label_fontsize: float = AXIS_FONT_SIZE
    axis_tick_fontsize: float = AXIS_TICK_FONT_SIZE
    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None

    @computed_field
    @property
    def figsize(self) -> tuple[float, float]:
        return self.width, self.height
    
    @computed_field
    @property
    def x_limits(self) -> AxisLimits:
        return self.x_min, self.x_max
    
    @computed_field
    @property
    def y_limits(self) -> AxisLimits:
        return self.y_min, self.y_max
    
    @computed_field
    @property
    def limits(self) -> AxesLimits:
        return self.x_limits, self.y_limits
    
def is_valid_cmap(value: str | Colormap) -> Colormap:
    if isinstance(value, str):
        cmap = colormaps.get(value)
        if cmap is None:
            raise ValueError(f"Colormap name given ({value}) is not a valid colormap")
        return cmap
    return value


class BaseHistogramConfig(BasePlotConfig):
    x_resolution: int = HISTOGRAM_RES
    y_resolution: int
    cmap: str | Colormap = "viridis"
    density: bool = False
    weights: ArrayLike | None = None
    cmin: float | None = None
    cmax: float | None = None
    vmin: float | None = None
    vmax: float | None = None

    @model_validator(mode="before")
    @classmethod
    def compute_y_resolution_if_missing(cls, values):
        y_res = values.get("y_resolution")
        x_res = int(values.get("x_resolution"))
        width = float(values.get("width"))
        height = float(values.get("height"))
        if y_res is None:
            values["y_resolution"] = x_res * width // height
        return values
    
    @computed_field
    @property
    def bins(self) -> tuple[int, int]:
        return self.x_resolution, self.y_resolution
    
    validate_cmap = field_validator("cmap")(is_valid_cmap)

class TailVsTotalConfig(BaseHistogramConfig):
    max_energy: float

    @model_validator(mode="before")
    @classmethod
    def set_limits(cls, values: dict[str, Any]):
        values_copy = {k: v for k, v in values.items()}
        max_energy = values.get("max_energy")
        if max_energy is not None:
            values_copy["x_min"] = 0
            values_copy["x_max"] = max_energy + 0.05
            values_copy["y_min"] = 0
            values_copy["y_max"] = 0.50
        return values_copy


class PsdHistogramConfig(BaseHistogramConfig):
    cmap: str | Colormap = "gnuplot"


class ClassificationConfig(BaseHistogramConfig):
    cmap: str | Colormap = "Greys"

