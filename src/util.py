from datetime import timedelta
from typing import Optional

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

DT_INDEX = pd.date_range("2023-01-01 00:00:00", "2023-12-31 23:00:00", freq="h")


class GurobiEnv:
    _value = None

    @classmethod
    def get_env(cls):
        if cls._value is None:
            cls._value = gp.Env(params={"OutputFlag": 0})
        return cls._value


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
        """Gurobi callback for stopping MIP solver

        Args:
            gap: MIP gap between the best solution and bound
            soft_limit: Minimum time to solve, even if already below gap. Ignored if no gap specified.
            time_limit: Maximum time to solve
            time_limit_is_hard: If False, only stops if a solution is found
        """
        self.gap = gap
        self.soft_limit = soft_limit
        self.time_limit = time_limit
        self.time_limit_is_hard = time_limit_is_hard
        self.metrics = []
        self.optimization_status = "OPTIMAL"
        self._lowest_objbst = np.inf
        self._highest_objbnd = 0

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
        """"""
        def callback(model, where):
            if (self.gap or self.time_limit is not None) and where == GRB.Callback.MIP:
                self.optimization_status = "OPTIMAL"
                runtime = model.cbGet(GRB.Callback.RUNTIME)
                work = model.cbGet(GRB.Callback.WORK)
                objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
                objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
                gap = _mip_gap(objbst, objbnd)

                if objbst < self._lowest_objbst or objbnd > self._highest_objbnd:
                    self.metrics.append({
                        "runtime": runtime,
                        "work": work,
                        "objbst": objbst if objbst < GRB.INFINITY else np.inf,
                        "objbnd": objbnd,
                        "gap": gap,
                    })
                    self._lowest_objbst = objbst
                    self._highest_objbnd = objbst

                # Stop if hard time limit is reached
                is_above_time_limit = self.time_limit is not None and runtime > self.time_limit
                is_valid_solution = objbst < GRB.INFINITY
                if is_above_time_limit and (self.time_limit_is_hard or is_valid_solution):
                    print(f"Callback: Time limit of {self.time_limit:.3f} seconds reached at gap {self.gap}.")
                    self.optimization_status = "TIME_LIMIT"
                    model.terminate()
                    return

                # Stop if there is a valid solution within gap after soft limit has passed
                is_above_soft_limit = runtime >= self.soft_limit
                is_below_mip_gap = gap < self.gap
                if is_valid_solution and is_above_soft_limit and is_below_mip_gap:
                    print(f"Callback: Gap of {self.gap} reached after {runtime:.3f} seconds.")
                    self.optimization_status = "GAP"
                    model.terminate()
                    return
        return callback


def _mip_gap(objbst: float, objbnd: float) -> float:
    try:
        return abs(objbst - objbnd) / abs(objbst)
    except ZeroDivisionError:
        return np.inf


def extract_a(a: np.array) -> np.array:
    return _extract_gurobi_vars(a)


def extract_d(d: np.array) -> np.array:
    return np.rint(_extract_gurobi_vars(d))


def _extract_gurobi_vars(vec: np.array) -> np.array:
    """Extracts Gurobi variables."""
    vfunc = np.vectorize(_extract_if_gurobi)
    x = vfunc(vec)
    return x


def _extract_if_gurobi(var):
    if isinstance(var, gp.Var):
        return var.X
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
