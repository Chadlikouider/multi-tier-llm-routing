from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
from pulp import LpVariable, value

DT_INDEX = pd.date_range("2023-01-01 00:00:00", "2023-12-31 23:00:00", freq="h")
def lt_indices(frequency: int) -> list[bool]:
    """Returns a list of boolean values indicating the indices for long-term optimization."""
    # Create a series of dates to match DT_INDEX and resample by the provided frequency
    dt_series = pd.Series(1, index=DT_INDEX)

    # Get the indices where the start of each period occurs according to the frequency
    # TODO what is happening here?
    resampled = dt_series.resample(timedelta(hours=frequency)).apply(lambda x: True).reindex(dt_series.index, fill_value=False)

    return resampled.tolist()


class Callback:

    def __init__(self,
                 gap: float = 0,
                 soft_limit: int = 0,
                 time_limit: Optional[int] = None,
                 time_limit_is_hard: bool = True):
        """Placeholder callback for compatibility with legacy interfaces.

        The open-source CBC solver used by PuLP does not expose the same callback
        API as Gurobi. The parameters are stored to keep backward compatibility,
        and the callable returned by ``__call__`` is a no-op.
        """
        self.gap = gap
        self.soft_limit = soft_limit
        self.time_limit = time_limit
        self.time_limit_is_hard = time_limit_is_hard
        self.metrics = []
        self.optimization_status = "OPTIMAL"

    def __str__(self):
        result = []
        if self.gap:
            result.append(f"gap={self.gap}")
        if self.soft_limit:
            result.append(f"soft={self.soft_limit}")
        if self.time_limit:
            result.append(f"time_limit={self.time_limit}")
        return ",".join(result)

    def __call__(self):
        def callback(*_args, **_kwargs):
            return None

        return callback


def extract_a(a: np.array) -> np.array:
    return _extract_solver_vars(a)


def extract_d(d: np.array) -> np.array:
    return np.rint(_extract_solver_vars(d))


def _extract_solver_vars(vec: np.array) -> np.array:
    """Extracts decision variable values from the underlying solver."""
    vfunc = np.vectorize(_extract_if_solver)
    x = vfunc(vec)
    return x


def _extract_if_solver(var):
    if isinstance(var, LpVariable):
        return value(var)
    if var is None:
        return 0
    return var


def get_validity_periods(window: list[int],
                         vp: int,
                         past: bool = True,
                         future: bool = True) -> list[list[int]]:
    periods = []
    for end in range(len(DT_INDEX)):
        start = end - vp + 1
        l = list(range(start, end + 1))
        if not future and end > window[-1]:
            break
        if 0 <= start <= window[-1] and end >= window[0]:
            if past or start >= window[0]:
                periods.append(l)
        if start > window[-1]:
            break
    return periods
