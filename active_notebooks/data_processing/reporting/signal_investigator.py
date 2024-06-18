from abc import abstractmethod
from dataclasses import dataclass, field, fields
from datetime import datetime
from functools import lru_cache
from typing import Any, Generic, Protocol, TypeVar
from typing_extensions import Self

import matplotlib.pyplot as plt
import pandas as pd
from data_processing.reporting.plotting import plot_bounded_scatter
from data_processing.dataframe_validation import DetectorDataframeColumn, get_df_col
from data_processing.processing.dataframe_manipulation import \
    generate_neutron_signals
from data_processing.reporting.plot_configs import *
from data_processing.reporting.plotting import plot_psd_histogram
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
from numpy import arange

QueryFloatInput = tuple[float | None, float | None]
QueryDatetimeInput = tuple[datetime | None, datetime | None]

C = TypeVar("C", bound="QueryBoundary")
QueryEdge = C | None
QueryInput = tuple[QueryEdge, QueryEdge]


class QueryBoundary(Protocol):
    @abstractmethod
    def __hash__(self) -> int:
        ...

    @abstractmethod
    def __eq__(self, __value: Any) -> bool:
        ...

    @abstractmethod
    def __lt__(self: C, __value: C) -> bool:
        ...

    def __gt__(self: C, __value: C) -> bool:
        return (not self < __value) and self != __value

    def __le__(self: C, __value: C) -> bool:
        return self < __value or self == __value

    def __ge__(self: C, __value: C) -> bool:
        return (not self < __value)


class QueryRange(Generic[C]):
    def __init__(
        self,
        col: DetectorDataframeColumn,
        start: QueryEdge = None,
        end: QueryEdge = None
    ) -> None:
        self._start, self._end = self._reorder(start, end)
        self._col = col

    @property
    def start(self) -> QueryEdge:
        return self._start

    def set_start(self, value: QueryEdge) -> Self:
        return QueryRange[C](self._col, start=value, end=self._end)

    # @start.setter
    # def start(self, value: C | None):
    #     if (value is not None and
    #         self._start is not None and
    #         self._end is not None and
    #             value > self._end):
    #         self._start, self._end = self._end, value
    #     else:
    #         self._start = value

    @property
    def end(self) -> QueryEdge:
        return self._end

    def set_end(self, value: QueryEdge) -> Self:
        return QueryRange[C](self._col, start=self._start, end=value)

    # @end.setter
    # def end(self, value: C | None):
    #     if (value is not None and
    #         self._start is not None and
    #         self._end is not None and
    #             value < self._start):
    #         self._start, self._end = value, self._start
    #     else:
    #         self._end = value

    @property
    def range(self) -> QueryInput:
        return self._start, self._end

    def set_range(self, value: QueryInput) -> Self:
        start, end = value
        return QueryRange[C](self._col, start=start, end=end)

    # @range.setter
    # def range(self, value: tuple[C | None, C | None]):
    #     start, end = value
    #     if start is not None and end is not None and start > end:
    #         self._start = end
    #         self._end = start
    #     else:
    #         self._start = start
    #         self._end = end

    @property
    def column(self) -> str:
        return self._col.value

    def reset_range(self) -> Self:
        return QueryRange[C](self._col)

    def perform_range_query(self, df: pd.DataFrame) -> pd.DataFrame:
        query_col = get_df_col(df, self._col)

        if self._start is not None:
            if self._end is not None:
                if self._start == self._end:
                    return df[query_col == self._start]
                else:
                    return df[
                        (query_col >= self._start) &
                        (query_col < self._end)
                    ]
            else:
                return df[query_col >= self._start]
        else:
            if self._end is not None:
                return df[query_col < self._end]
            else:
                return df

    @staticmethod
    def _reorder(x: C | None, y: C | None) -> tuple[C | None, C | None]:
        if x is None or y is None:
            return x, y
        else:
            lo = x if x <= y else y
            hi = y if x <= y else x
            return lo, hi

    def __hash__(self) -> int:
        return hash((self._col, self._start, self._end))


class DatetimeQueryRange(QueryRange[datetime]):
    def perform_range_query(self, df: pd.DataFrame) -> pd.DataFrame:
        query_col = get_df_col(df, self._col)

        start = self._start if self._start is None else self._start.isoformat()
        end = self._end if self._end is None else self._end.isoformat()

        if start is not None:
            if end is not None:
                if start == end:
                    return df[query_col == start]
                else:
                    return df[(query_col >= start) & (query_col < end)]
            else:
                return df[query_col >= start]
        else:
            if end is not None:
                return df[query_col < end]
            else:
                return df


@dataclass(frozen=True)
class Query:
    psd: QueryRange = field(
        default_factory=lambda: QueryRange[float](
            DetectorDataframeColumn.PSD))
    energy: QueryRange = field(
        default_factory=lambda: QueryRange[float](
            DetectorDataframeColumn.CALIB_ENERGY))
    time: QueryRange = field(
        default_factory=lambda: DatetimeQueryRange(
            DetectorDataframeColumn.EVENT_TIME))

    def update_query(self,
                     psd: QueryFloatInput | None = None,
                     energy: QueryFloatInput | None = None,
                     time: QueryDatetimeInput | None = None):
        
        def update_query_range(query_range: QueryRange[C], 
                               update_value: QueryInput | None) -> QueryRange[C]:
            if update_value is None:
                return query_range
            else:
                return query_range.set_range(update_value)
            
        update_params = {
            'psd': update_query_range(self.psd, psd),
            'energy': update_query_range(self.energy, energy),
            'time': update_query_range(self.time, time)
        }
        return Query(**update_params)

    def reset_query(self):
        return Query(psd=self.psd.reset_range(),
                     energy=self.energy.reset_range(),
                     time=self.time.reset_range())

    def perform_query(self, df: pd.DataFrame) -> pd.DataFrame:
        query_result = df
        query_fields = fields(self)
        for query_field in query_fields:
            query_range: QueryRange = getattr(self, query_field.name)
            query_result = query_range.perform_range_query(query_result)
        return query_result


class SignalInvestigator:
    def __init__(self, neutron_events_df: pd.DataFrame):
        self._neutron_event_df = neutron_events_df
        self._neutron_signals_df = generate_neutron_signals(neutron_events_df)
        self.query = Query()
        
    def update_query(self, **kwargs):
        self.query = self.query.update_query(**kwargs)

    def perform_query(self) -> pd.DataFrame:
        return self._query_helper(self.query)

    def count_results(self) -> int:
        query_result = self.perform_query()
        return query_result.shape[0]

    def visualize_signals(
        self, max_legend_count: int = 10
    ) -> tuple[Figure, Axes]:
        query_result = self.perform_query()
        series_names = []

        fig, ax = plt.subplots(figsize=(FIG_DIM_X, FIG_DIM_Y))

        samples_count = 0
        for index, row in query_result.iterrows():
            n_psd = row[DetectorDataframeColumn.PSD.value]
            n_eng = row[DetectorDataframeColumn.CALIB_ENERGY.value]
            n_time = row[DetectorDataframeColumn.EVENT_TIME.value]
            n_ps_remain = row[DetectorDataframeColumn.EVENT_TIME_PS.value]

            series_name = (f"Time {n_time} +{n_ps_remain}ps, ",
                           f"PSD {n_psd:.4f}, E {n_eng:.4f} MeVee")
            series_names.append(series_name)
            
            amplitudes = self._neutron_signals_df[index]
            current_count = len(amplitudes)
            if current_count > samples_count:
                samples_count = current_count
            ax.plot(list(range(1,current_count+1)), 
                    (0-amplitudes)/1000)  # type: ignore

        ax.set_xticks(arange(0, samples_count + 1, 25))  # type: ignore
        ax.set_xlabel("Sample index", fontsize=AXIS_FONT_SIZE)
        ax.set_ylabel("Sample amplitude (ADC channels x 1000, inv.)",
                      fontsize=AXIS_FONT_SIZE)
        ax.tick_params(axis='both', which='major', 
                       labelsize=AXIS_TICK_FONT_SIZE)
        
        
        if len(series_names) <= max_legend_count:
            ax.legend(series_names)

        return fig, ax

    def visualize_psd(self, scatter_max: int = 1000) -> tuple[Figure, Axes]:
        if self.count_results() >= scatter_max:
            return self.visualize_histogram()
        else:
            return self.visualize_scatter()

    def visualize_histogram(self, log_scale: bool = True) -> tuple[Figure, Axes]:
        query_result = self.perform_query()
        if log_scale:
            options = {'norm': LogNorm()}
        else:
            options = {}
        return plot_psd_histogram(query_result, colorbar=True, **options)

    def visualize_scatter(self) -> tuple[Figure, Axes]:
        query_results = self.perform_query()

        energy_col = get_df_col(query_results, DetectorDataframeColumn.CALIB_ENERGY)
        psd_col = get_df_col(query_results, DetectorDataframeColumn.PSD)

        return plot_bounded_scatter(
            energy_col,
            psd_col,
            "Energy (MeVee)",
            "PSD",
            s=SCATTER_MARKER_SIZE_LARGE,
            # marker=?
        )

    @lru_cache
    def _query_helper(self, query: Query) -> pd.DataFrame:
        return query.perform_query(self._neutron_event_df)
