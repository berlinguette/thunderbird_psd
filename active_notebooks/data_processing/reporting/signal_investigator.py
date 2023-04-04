from abc import abstractmethod
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Any, Generic, Protocol, TypeVar

import matplotlib.pyplot as plt
import pandas as pd
from data_processing.dataframe_validation import DataframeColumn
from data_processing.processing.dataframe_manipulation import \
    generate_neutron_signals
from data_processing.reporting.plot_configs import *
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import arange

C = TypeVar("C", bound="Comparable")


class Comparable(Protocol):
    @abstractmethod
    def __eq__(self, __value: Any) -> bool:
        pass

    @abstractmethod
    def __lt__(self: C, __value: C) -> bool:
        pass

    def __gt__(self: C, __value: C) -> bool:
        return (not self < __value) and self != __value

    def __le__(self: C, __value: C) -> bool:
        return self < __value or self == __value

    def __ge__(self: C, __value: C) -> bool:
        return (not self < __value)


class QueryRange(Generic[C]):
    def __init__(self, col: DataframeColumn) -> None:
        self._start: C | None = None
        self._end: C | None = None
        self._col = col

    @property
    def start(self) -> C | None:
        return self._start

    @start.setter
    def start(self, value: C | None):
        if (value is not None and
            self._start is not None and
            self._end is not None and
                value > self._end):
            self._start, self._end = self._end, value
        else:
            self._start = value

    @property
    def end(self) -> C | None:
        return self._end

    @end.setter
    def end(self, value: C | None):
        if (value is not None and
            self._start is not None and
            self._end is not None and
                value < self._start):
            self._start, self._end = value, self._start
        else:
            self._end = value

    @property
    def range(self) -> tuple[C | None, C | None]:
        return self._start, self._end

    @range.setter
    def range(self, value: tuple[C | None, C | None]):
        start, end = value
        if start is not None and end is not None and end > start:
            self._start = end
            self._end = start
        else:
            self._start = start
            self._end = end

    @property
    def column(self) -> str:
        return self._col.value

    def perform_range_query(self, df: pd.DataFrame) -> pd.DataFrame:
        start = self._start
        end = self._end

        query_elements = []

        if self._start is not None:
            element = f"{self._col.value} >= @start"
            query_elements.append(element)
        if self._end is not None:
            element = f"{self._col.value} < @end"
            query_elements.append(element)

        if len(query_elements) == 0:
            return df
        elif len(query_elements) == 1:
            query = query_elements[0]
        else:
            query = ' & '.join(query_elements)
        return df.query(query)


@dataclass
class Query:
    psd: QueryRange = field(
        default_factory=lambda: QueryRange[float](DataframeColumn.PSD))
    energy: QueryRange = field(
        default_factory=lambda: QueryRange[float](DataframeColumn.ENERGY))
    time: QueryRange = field(
        default_factory=lambda: QueryRange[datetime](DataframeColumn.EVENT_TIME))

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

    def perform_query(self) -> pd.DataFrame:
        return self.query.perform_query(self._neutron_event_df)

    def visualize_query(self) -> tuple[Figure, Axes]:
        query_result = self.perform_query()
        series_names = []

        fig, ax = plt.subplots(figsize=(FIG_DIM_X, FIG_DIM_Y))

        for index, row in query_result.iterrows():
            n_psd = row[DataframeColumn.PSD.value]
            n_eng = row[DataframeColumn.ENERGY.value]
            n_time = row[DataframeColumn.EVENT_TIME.value]
            n_ps_remain = row[DataframeColumn.EVENT_TIME_PS.value]

            series_name = (f"Time {n_time} +{n_ps_remain}ps, ",
                           f"PSD {n_psd:.4f}, E {n_eng:.4f} MeVee")
            series_names.append(series_name)
            ax.plot(0-self._neutron_signals_df[index])  # type: ignore

        x_start, x_end = ax.get_xlim()
        ax.xaxis.set_ticks(arange(x_start, x_end, 10))  # type: ignore
        ax.legend(series_names)

        return fig, ax
