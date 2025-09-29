"""High level optimization strategies.

The module exposes two entry points: :class:`Qt` performs a single optimization
over the full horizon, while :class:`QtOnline` orchestrates rolling horizon
optimization based solely on short-term planning.  The goal of this file is to
document the control flow that wires together forecasting, optimization, and
fallback heuristics.
"""

from time import time
from typing import Optional

import numpy as np

from src.executor import execute_perfect_step
from src.qt_model import QtModel
from src.result import Result
from src.scenario import Scenario
from src.util import Callback, extract_a, extract_d


class Qt:
    """Convenience wrapper around :class:`src.qt_model.QtModel`.

    The class exists to expose a consistent API between the offline and the
    online optimizer.  It mainly handles default arguments and result
    serialization.
    """

    def __init__(self, callback: Optional[Callback] = None, silent: bool = False, relax: bool = False):
        self.callback = callback
        self.silent = silent
        self.relax = relax

    @property
    def name(self) -> str:
        result = []
        if self.callback:
            result.append(f"cb=({self.callback})")
        if self.relax:
            result.append("relax")
        return f"Qt({','.join(result)})"

    def minimize_emissions(
        self,
        scenario: Scenario,
        qor_target: float,
        R_hat: Optional[np.ndarray] = None,
        C_hat: Optional[np.ndarray] = None,
        info: Optional[dict] = None,
    ) -> Result:
        """Solve for the lowest emissions while maintaining ``qor_target``."""

        R_hat = R_hat if R_hat is not None else scenario.R
        C_hat = C_hat if C_hat is not None else scenario.C
        qt_model = QtModel(scenario, silent=self.silent, callback=self.callback)
        metrics = qt_model.minimize_emissions(
            qor_target,
            window=scenario.I,
            R_hat=R_hat,
            C_hat=C_hat,
            relax=self.relax,
        )
        info = info if info is not None else {}
        info["qor_target"] = qor_target
        return Result(
            extract_a(qt_model.a_),
            extract_d(qt_model.d_),
            optimizer=self,
            scenario=scenario,
            metrics=metrics,
            info=info,
        )

    def maximize_qor(self, scenario: Scenario, budget: float) -> Result:
        """Maximize the minimum QoR while respecting an emission budget."""

        qt_model = QtModel(scenario, silent=self.silent, callback=self.callback)
        metrics = qt_model.maximize_qor(
            budget,
            window=scenario.I,
            R_hat=scenario.R,
            C_hat=scenario.C,
            relax=self.relax,
        )
        return Result(
            extract_a(qt_model.a_),
            extract_d(qt_model.d_),
            optimizer=self,
            scenario=scenario,
            metrics=metrics,
            info={"budget": budget},
        )


class QtOnline:
    """Rolling horizon optimizer that repeatedly solves short-term problems."""

    def __init__(self,
                 horizon: int,
                 oracle: bool,
                 st_callback: Optional[Callback] = None,
                 st_silent: bool = False):
        """Short-term optimization model.

        Args:
            horizon: Short-term optimization window size, e.g. 24 for 24 hours
            oracle: If true, omits forecasting and only uses actual values
            st_callback: Short-term optimization callback
            st_silent: If true, suppress solver logs in short-term optimization
        """
        self.horizon = horizon
        self.oracle = oracle
        self.st_callback = st_callback
        self.st_silent = st_silent

    @property
    def name(self) -> str:
        result = [f"horizon={self.horizon}"]
        if self.st_callback:
            result.append(f"st_cb=({self.st_callback})")
        if self.oracle:
            result.append(f"oracle")
        return f"QtOnline({','.join(result)})"

    def minimize_emissions(self, scenario: Scenario, qor_target: float) -> Result:
        """Entry point for emission minimization with short-term corrections."""

        return self._optimize(scenario, qor_target=qor_target)

    def maximize_qor(self, scenario: Scenario, budget: float) -> Result:
        """Entry point for QoR maximization when a fixed budget is available."""

        return self._optimize(scenario, budget=budget)

    def _optimize(self, scenario: Scenario, qor_target: float = None, budget: float = None) -> Result:
        """Shared entry point for maximizing QoR or minimizing emissions."""

        horizon = min(self.horizon, scenario.vp)
        online_model = QtModel(scenario, silent=self.st_silent, callback=self.st_callback)
        R_hat, C_hat = self._initialize_forecasts(scenario)

        print("Starting online optimization...")

        t0 = time()

        for step in scenario.I:
            print(f"{step:<4} ------------------------")

            R_hat, C_hat = self._refresh_forecasts(scenario, step, R_hat, C_hat)

            # Replace the forecasts with the ground truth values we now observe.
            R_hat[step] = scenario.R[step]
            C_hat[step] = scenario.C[step]

            self._short_term_optimization(
                scenario,
                online_model,
                qor_target=qor_target,
                budget=budget,
                step=step,
                horizon=horizon,
                R_hat=R_hat,
                C_hat=C_hat,
            )

        metrics = {"runtime": time() - t0}
        info = {"budget": budget} if budget is not None else {"qor_target": qor_target}
        return Result(online_model.a_, online_model.d_, optimizer=self, scenario=scenario, metrics=metrics, info=info)

    def _initialize_forecasts(self, scenario: Scenario) -> tuple[np.ndarray, np.ndarray]:
        """Return mutable forecast arrays for demand and carbon data."""

        if self.oracle:
            return scenario.R.copy(), scenario.C.copy()

        return (
            scenario.generate_R_hat(0, kind="yhat"),
            scenario.generate_C_hat(0, kind="yhat"),
        )

    def _refresh_forecasts(
        self,
        scenario: Scenario,
        step: int,
        R_hat: np.ndarray,
        C_hat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update forecasts for future steps when operating without an oracle."""

        if self.oracle:
            return R_hat, C_hat

        return (
            scenario.generate_R_hat(step, kind="yhat"),
            scenario.generate_C_hat(step, kind="yhat"),
        )

    def _short_term_optimization(self,
                                 scenario: Scenario,
                                 online_model: QtModel,
                                 *,
                                 qor_target: Optional[float],
                                 budget: Optional[float],
                                 step: int,
                                 horizon,
                                 R_hat,
                                 C_hat):
        window = scenario.I[step:step + horizon]
        while True:
            try:
                if budget is not None:
                    metrics = online_model.maximize_qor(budget, window=window, R_hat=R_hat, C_hat=C_hat)
                else:
                    assert qor_target is not None, "qor_target must be provided for emission minimization"
                    metrics = online_model.minimize_emissions(qor_target, window=window, R_hat=R_hat, C_hat=C_hat)
                print(
                    f"{step:<4}: "
                    f"QoR: {metrics['qor_target']:.3f}, "
                    f"Runtime: {metrics['runtime']:.3f}s, "
                    f"Gap: {metrics['mip_gap']:.3f}, "
                    f"Status: {metrics['optimization_status']}"
                )

                a_act, d_act = perform_step(step, online_model.d_, online_model.a_, scenario)
                assert np.isclose(
                    a_act.sum(), scenario.R[step].sum()
                ), f"Sanity check: All requests should be served, but allocated {a_act.sum()} of {scenario.R[step].sum()}"
                break
            except RuntimeError:
                print("No solution found, allocating all to best model")
                a_act, d_act = execute_perfect_step(scenario.R[step], scenario)
                break

        # d_acts.append(d_act)

        online_model.a_[step] = a_act
        online_model.d_[step] = d_act

        # return metrics


def perform_step(i: int, d_: np.array, a_: np.array, scenario: Scenario) -> tuple[np.array, np.array]:
    """Extract the integer decisions for allocations and deployments at step ``i``."""

    d_act = np.array([[int(round(d_[i, q, m].X)) for m in scenario.M] for q in scenario.Q])
    a_act = np.array([[a_[i, u, q].X for q in scenario.Q] for u in scenario.U])
    return a_act, d_act
