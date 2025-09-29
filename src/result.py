"""Utility helpers for working with solver results."""

import os
import pickle
from typing import Optional

import numpy as np

from src.qt_model import interval_emissions, interval_power_per_machine
from src.scenario import Scenario
from src.util import get_validity_periods

_BASE_DIR = os.path.dirname(os.path.dirname(__file__))


class Result:
    """Container storing allocations, deployments, and derived metrics."""

    def __init__(
        self,
        a: np.ndarray,
        d: np.ndarray,
        optimizer: Optional["src.optimizer.Optimizer"] = None,
        scenario: Optional[Scenario] = None,
        metrics: Optional[dict] = None,
        info: Optional[dict] = None,
    ):
        self.a = a.astype(np.float64)  # t, u, q
        self.d = d.astype(np.float64)  # t, q, m
        self.optimizer = optimizer
        self.scenario = scenario
        self.metrics = metrics if metrics is not None else {}
        self.info = info if info is not None else {}

        # Cache the QoR list to avoid recomputing expensive aggregations.
        self._qor_list = None

    @property
    def name(self) -> str:
        """Return a stable, human-readable identifier for persistence."""

        return get_result_name(self.scenario, self.optimizer, self.info)

    def emissions_over_time(self, C_hat: Optional[np.ndarray] = None, load_dependent: bool = False) -> np.ndarray:
        """Return hourly emissions using either actual or forecasted carbon data."""

        C = C_hat if C_hat is not None else self.scenario.C
        return np.array(
            [
                [
                    interval_emissions(i, q, m, self.d, self.scenario, C, load_dependent=load_dependent, a_=self.a)
                    for q in self.scenario.Q
                    for m in self.scenario.M
                ]
                for i in self.scenario.I
            ]
        ).sum(axis=1)

    def energy_over_time(self, load_dependent: bool = False) -> np.ndarray:
        """Return hourly energy consumption in kWh."""

        return (
            self.d.squeeze(axis=-1)
            * np.array(
                [
                    [
                        interval_power_per_machine(
                            i,
                            q,
                            m,
                            self.d,
                            self.scenario,
                            load_dependent=load_dependent,
                            a_=self.a,
                        )
                        for q in self.scenario.Q
                        for m in self.scenario.M
                    ]
                    for i in self.scenario.I
                ]
            )
        ).sum(axis=1)

    def print_stats(self) -> None:
        """Print a compact summary that mirrors the internal dashboards."""

        qor_list = self.qor_list()
        print(
            "Actual QoR: "
            f"min={np.min(qor_list):.3f}, "
            f"5th={self.qor_quantile(0.05):.3f}, "
            f"mean={np.mean(qor_list):.3f}±{np.std(qor_list):.3f} "
            f"at {self.emissions():.2f} tCO2e"
        )

    def emissions(
        self,
        i: Optional[int] = None,
        C_hat: Optional[np.ndarray] = None,
        load_dependent: bool = False,
    ) -> float:
        """Return total emissions or emissions for a specific hour in gCO2e."""

        z = self.emissions_over_time(C_hat, load_dependent=load_dependent)
        if i is not None:
            return float(z[i])
        return float(z.sum(axis=0))

    def qor_list(self) -> list[float]:
        """Return QoR for every validity period of the scenario."""

        if self._qor_list is None:
            self._qor_list = [
                self._qor(Ti, self.scenario.R)
                for Ti in get_validity_periods(self.scenario.I, self.scenario.vp)
            ]
        return self._qor_list

    def qor_quantile(self, quantile: float) -> float:
        """Return a QoR quantile for the cached QoR list."""

        return float(np.quantile(self.qor_list(), quantile))

    def qor_target(self) -> float:
        """Return the minimum QoR, i.e. the objective tracked by the optimizer."""

        return float(np.min(self.qor_list()))

    def qor_target_forecast(self, R_hat: np.ndarray) -> float:
        """Compute the QoR target using forecasted demand instead of actual demand."""

        return float(
            np.min([self._qor(Ti, R_hat) for Ti in get_validity_periods(self.scenario.I, self.scenario.vp)])
        )

    def _qor(self, validity_period, R_hat=None) -> float:
        """Return QoR for a specific validity period."""

        if R_hat is None:
            R_hat = self.scenario.R

        if R_hat[validity_period].sum() == 0:
            return 1.0

        service_levels = self.a[validity_period].sum(axis=0) / R_hat[validity_period].sum(axis=0)

        # Safely compute errors, treating zero denominators as np.inf
        nominator = np.abs(service_levels - self.scenario.slo_upper)
        denominator = np.abs(self.scenario.slo_lower - self.scenario.slo_upper)
        errors = nominator / np.where(denominator == 0, np.inf, denominator)
        highest_error = np.max(errors)

        if not 0 <= highest_error <= 1:
            raise RuntimeError(f"QoR error out of bounds: {highest_error}")
        return round(1 - highest_error, 3)

    def save(self, result_dir: str = "results") -> None:
        """Persist the result object as a pickle file."""

        os.makedirs(f"{_BASE_DIR}/{result_dir}", exist_ok=True)
        path = f"{_BASE_DIR}/{result_dir}/{self.name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.to_dict(), f)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation of the result."""

        return {
            "a": self.a,
            "d": self.d,
            "optimizer": self.optimizer,
            "metrics": self.metrics,
            "info": self.info,
            "scenario_name": self.scenario.name,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Result":
        """Rehydrate a :class:`Result` from :meth:`to_dict` output."""

        return cls(
            a=d["a"],
            d=d["d"],
            optimizer=d["optimizer"],
            metrics=d["metrics"],
            info=d["info"],
            scenario=Scenario.from_name(d["scenario_name"]),
        )


def load_result(name: str, result_dir: str = "results") -> Result:
    """Load a pickled result either by file name or logical name."""

    if ".pkl" not in name:
        name = f"{name}.pkl"
    with open(f"{_BASE_DIR}/{result_dir}/{name}", "rb") as f:
        result = Result.from_dict(pickle.load(f))
    return result


def load_results(result_dir: str = "results") -> list[Result]:
    """Load all results contained in ``result_dir``."""

    return [
        load_result(os.path.basename(filename), result_dir=result_dir)
        for filename in os.listdir(f"{_BASE_DIR}/{result_dir}")
        if "pkl" in filename
    ]


def get_result_name(scenario, optimizer, info) -> str:
    """Build a descriptive file name for a :class:`Result`."""

    n = f"{scenario.name}"
    for key, value in info.items():
        if isinstance(value, float):
            value = f"{value:.2f}"
        n += f",{key}={value}"
    n += f",{optimizer.name},seed={scenario.seed}"
    return n


def result_exists(result_name: str, result_dir: str = "results") -> bool:
    """Return ``True`` if a pickled result with ``result_name`` exists."""

    path = f"{_BASE_DIR}/{result_dir}/{result_name}.pkl"
    return os.path.exists(path)
