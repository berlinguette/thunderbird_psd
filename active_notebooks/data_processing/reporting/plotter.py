from __future__ import annotations
from math import ceil

from data_processing.reporting.plot_configs import *
from data_processing.types import (
    AxesMatrix,
    StrAnyDict,
    # GraphingFunction2,
    ConfigModel,
    GraphingFunction3,
    GraphData,
    AxesUpdateFunction,
    FigureUpdateFunction,
)
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Sequence
import matplotlib.pyplot as plt



class SinglePlot:
    def __init__(self, figsize: tuple[float, float] = (FIG_DIM_X, FIG_DIM_Y),
    **kwargs):
        subplots_return: tuple[Figure, Axes] = plt.subplots(figsize=figsize, **kwargs)  # type: ignore
        self._fig, self._ax = subplots_return
        self._plot_results: StrAnyDict | None = None

    @property
    def figure(self) -> Figure:
        return self._fig
    
    @property
    def axes(self) -> Axes:
        return self._ax

    @property
    def plot_objects(self) -> tuple[Figure, Axes]:
        return self._fig, self._ax
    
    @property
    def plot_results(self) -> StrAnyDict:
        if self._plot_results is None:
            raise ValueError("No plot results exist. Plot data first before calling this.")
        return self._plot_results
    
    def plot_data(self, plot_fn: GraphingFunction3, data: GraphData, config: ConfigModel) -> "SinglePlot":
        ax, return_data = plot_fn(self.axes, data, config)
        self._ax = ax
        self._plot_results = return_data
        return self
    
    def update_figure(self, update_fn: FigureUpdateFunction) -> "SinglePlot":
        fig = update_fn(self._fig)
        self._fig = fig
        return self
    
    def update_axes(self, update_fn: AxesUpdateFunction) -> "SinglePlot":
        ax = update_fn(self._ax)
        self._ax = ax
        return self
    

class MultiPlot:
    def __init__(self, axs_count: int, figsize: tuple[float, float] = (FIG_DIM_X, FIG_DIM_Y), max_cols: int = SUBPLOTS_MAX_COLS, **kwargs):
        n_cols = min(max_cols, axs_count)
        n_rows = ceil(axs_count / n_cols)
        subplots_result: tuple[Figure, AxesMatrix] = plt.subplots(
           figsize=figsize, ncols=n_cols, nrows=n_rows, **kwargs
        )

        self._n_cols = n_cols
        self._n_rows = n_rows
        self._figure, self._axs = subplots_result
        self._plot_results: list[StrAnyDict] | None = None
        
    @property
    def shape(self) -> tuple[int, int]:
        return self._n_rows, self._n_cols
    
    @property
    def figure(self) -> Figure:
        return self._figure
    
    @property
    def axes(self) -> AxesMatrix:
        return self._axs
    
    @property
    def plot_objects(self) -> tuple[Figure, AxesMatrix]:
        return self._fig, self._axs
    
    @property
    def plot_results(self) -> list[StrAnyDict]:
        if self._plot_results is None:
            raise ValueError("No plot results exist. Plot data first before calling this.")
        return self._plot_results
    
    def plot_data(self, plot_fn: GraphingFunction3, data: list[GraphData], plot_kwargs: Sequence[ConfigModel]) -> "MultiPlot":
        expected_count = self._n_cols * self._n_rows
        if len(data) != expected_count:
            raise ValueError("data must have the same count as plot axes")
        if len(plot_kwargs) != expected_count:
            raise ValueError("plot_kwargs must have the same count as plot axes")

        return_data_list: list[StrAnyDict] = []
        for i, ax_data in enumerate(data):
            row, col = self._get_coords(i)
            ax = self._get_ax(row, col)
            ax_kwargs = plot_kwargs[i]

            ax, return_data = plot_fn(ax, ax_data, ax_kwargs)
            self._set_ax(ax, row, col)
            return_data_list.append(return_data)

        self._plot_results = return_data_list
        return self

    def update_figure(self, update_fn: FigureUpdateFunction) -> "MultiPlot":
        self._figure = update_fn(self._figure)
        return self

    def update_axes(self, update_fns: Sequence[AxesUpdateFunction]) -> "MultiPlot":
        if len(update_fns) != self._n_cols * self._n_rows:
            raise ValueError("update_fns must have the same count as plot axes")
        
        for i, update_fn in enumerate(update_fns):
            row, col = self._get_coords(i)
            ax = self._get_ax(row, col)

            ax = update_fn(ax)
            self._set_ax(ax, row, col)
        
        return self

    def _get_coords(self, i: int) -> tuple[int, int]:
        row = i // self._n_cols
        col = i % self._n_cols
        return row, col
    
    def _get_ax(self, row: int, col: int) -> Axes:
        return self._axs[row][col]
    
    def _set_ax(self, ax: Axes, row: int, col: int):
        self._axs[row][col] = ax