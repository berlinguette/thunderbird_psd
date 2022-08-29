from select import select
import time
from datetime import timedelta
from math import modf


class Timer:
    def __init__(self, start_now: bool = False) -> None:
        self._is_running = False
        if start_now:
            self._start_time: float = time.perf_counter()
        else:
            self._start_time = 0
        self._end_time: float = 0

    @property
    def is_running(self) -> bool:
        return self._is_running

    def get_elapsed_time(self) -> float:
        if self._is_running:
            return time.perf_counter() - self._start_time
        else:
            return self._end_time - self._start_time

    def format_elapsed_time(
        self, elapsed_time: float, decimals: int = 0
    ) -> str:
        hours = elapsed_time // 3600
        minutes = (elapsed_time // 60) - (hours * 60)
        seconds = elapsed_time - (hours*3600) - (minutes*60)
        
        hours_str = ''
        minutes_str = ''
        if hours > 0:
            hours_str = f'{hours} hr, '
        if hours > 0 or minutes > 0:
            minutes_str = f'{minutes} min, '
        seconds_str = f'{seconds:.{decimals}f} sec'
        return f'{hours_str}{minutes_str}{seconds_str}'
            

    def start_timer(self):
        self._start_time = time.perf_counter()
        self._is_running = True

    def stop_timer(self) -> float:
        self._end_time = time.perf_counter()
        self._is_running = False
        return self.get_elapsed_time()
