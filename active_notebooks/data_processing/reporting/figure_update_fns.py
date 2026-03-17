from data_processing.reporting.plot_configs import *
from data_processing.types import FigureUpdateFunction
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh

def make_supertitle_adder(supertitle: str, font_size: float = SUPTITLE_FONT_SIZE) -> FigureUpdateFunction:
    def add_supertitle_to_figure(fig: Figure) -> Figure:
        fig.suptitle(supertitle, fontsize=font_size)
        return fig
    
    return add_supertitle_to_figure


def make_colorbar_adder(image: QuadMesh, ax: Axes, **kwargs) -> FigureUpdateFunction:
    def add_colorbar_to_figure(fig: Figure) -> Figure:
        fig.colorbar(image, ax=ax, **kwargs)
        return fig
    
    return add_colorbar_to_figure