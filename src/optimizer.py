"""High level optimization strategies.

The module exposes two entry points: :class:`Qt` performs a single optimization
over the full horizon, while :class:`QtOnline` orchestrates rolling horizon
optimization with optional long-term planning and outlook generation.  The goal
of this file is to document the control flow that wires together forecasting,
optimization, and fallback heuristics.
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
    """Rolling horizon optimizer that mixes short- and long-term planning."""

    def __init__(self,
                 horizon: int,
                 oracle: bool,
                 lt_frequency: int,
                 outlook_frequency: int,
                 st_callback: Optional[Callback] = None,
                 lt_callback: Optional[Callback] = None,
                 lt_relax: bool = False,
                 outlook_relax: bool = False,
                 st_silent: bool = False,
                 lt_silent: bool = False):
        """Multi-horizon optimization model.

        Args:
            horizon: Short-term optimization window size, e.g. 24 for 24 hours
            oracle: If true, omits forecasting and only uses actual values
            lt_frequency: Long-term optimization frequency, e.g. 24 for daily
            outlook_frequency: Frequency of long-term outlooks
            st_callback: Short-term optimization callback
            lt_callback: Long-term optimization callback
            lt_relax: If true, relax the d-matrix to continuous variables in long-term optimization
            outlook_relax: If true, relax the d-matrix to continuous variables in outlooks
            st_silent: If true, print Gurobi logs in short-term optimization
            lt_silent: If true, print Gurobi logs in long-term optimization
        """
        self.horizon = horizon
        self.oracle = oracle
        self.lt_frequency = lt_frequency
        self.outlook_frequency = outlook_frequency
        self.st_callback = st_callback
        self.lt_callback = lt_callback
        self.lt_relax = lt_relax
        self.outlook_relax = outlook_relax
        self.st_silent = st_silent
        self.lt_silent = lt_silent

    @property
    def name(self) -> str:
        result = [f"horizon={self.horizon}",
                  f"lt_freq=({self.lt_frequency})"]
        if self.lt_relax:
            result.append("lt_relax")
        if self.outlook_frequency:
            result.append(f"outlook={self.outlook_frequency}")
            if self.outlook_relax:
                result.append("outlook_relax")
        if self.st_callback:
            result.append(f"st_cb=({self.st_callback})")
        if self.lt_callback:
            result.append(f"lt_cb=({self.lt_callback})")
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
        """Shared entry point for maximizing QoR or minimizing emissions.

        The method orchestrates the interaction between long-term planning, optional
        outlook generation, and short-term corrective actions.  The original
        implementation grew organically and was difficult to follow because the
        responsibilities of each block were intertwined.  The refactored version
        delegates small, well defined steps to helper methods which makes the
        control flow clearer and easier to maintain.
        """

        horizon = min(self.horizon, scenario.vp)
        online_model = QtModel(scenario, silent=self.st_silent, callback=self.st_callback)
        long_term_params = self._prepare_long_term_params(budget, qor_target)
        R_hat, C_hat = self._initialize_forecasts(scenario)

        print("Starting online budget optimization...")

        t0 = time()
        retry_lt = False
        last_qor_target: Optional[float] = None
        qor_target_history: dict[int, float] = {}
        outlooks: dict[int, dict] = {}

        for i in scenario.I:
            print(f"{i:<4} ------------------------")
            if self._should_update_forecasts(i):
                R_hat, C_hat = self._generate_new_forecasts(scenario, i)

            qor_target, retry_lt, last_qor_target = self._maybe_run_long_term_optimization(
                scenario=scenario,
                online_model=online_model,
                window=scenario.I[i:],
                R_hat=R_hat,
                C_hat=C_hat,
                params=long_term_params,
                step=i,
                retry=retry_lt,
                budget=budget,
                qor_target=qor_target,
                last_qor_target=last_qor_target,
                qor_target_history=qor_target_history,
            )

            outlooks = self._maybe_generate_outlooks(
                scenario=scenario,
                step=i,
                R_hat=R_hat,
                C_hat=C_hat,
                outlooks=outlooks,
            )

            # Replace the forecasts with the ground truth values we now observe.
            R_hat[i] = scenario.R[i]
            C_hat[i] = scenario.C[i]

            self._short_term_optimization(
                scenario,
                online_model,
                qor_target,
                i,
                horizon,
                R_hat,
                C_hat,
                retry=(budget is not None),
            )

        metrics = self._build_metrics(t0, qor_target_history, outlooks)
        info = self._sanitize_long_term_params(long_term_params)
        return Result(online_model.a_, online_model.d_, optimizer=self, scenario=scenario, metrics=metrics, info=info)

    def _prepare_long_term_params(self, budget: Optional[float], qor_target: Optional[float]) -> dict:
        if budget is not None:
            return {"budget": budget, "past_vps": False}
        return {"qor_target": qor_target}

    def _initialize_forecasts(self, scenario: Scenario) -> tuple[np.ndarray, np.ndarray]:
        """Return mutable copies of the scenario demand and carbon data."""
        return scenario.R.copy(), scenario.C.copy()

    def _should_update_forecasts(self, step: int) -> bool:
        return step % self.lt_frequency == 0

    def _generate_new_forecasts(self, scenario: Scenario, step: int) -> tuple[np.ndarray, np.ndarray]:
        """Generate new forecast arrays for the current long-term step."""

        kind = "oracle" if self.oracle else "yhat"
        C_hat = scenario.generate_C_hat(step, kind=kind)
        R_hat = scenario.generate_R_hat(step, kind=kind)
        return R_hat, C_hat

    def _maybe_run_long_term_optimization(
        self,
        *,
        scenario: Scenario,
        online_model: QtModel,
        window: list[int],
        R_hat: np.ndarray,
        C_hat: np.ndarray,
        params: dict,
        step: int,
        retry: bool,
        budget: Optional[float],
        qor_target: Optional[float],
        last_qor_target: Optional[float],
        qor_target_history: dict[int, float],
    ) -> tuple[Optional[float], bool, Optional[float]]:
        if not (self._should_update_forecasts(step) or retry):
            return qor_target, False, last_qor_target

        try:
            print("Long-term optimization ...", flush=True)
            lt_result = self._long_term_optimization(
                R_hat,
                C_hat,
                scenario,
                online_model,
                window=window,
                relax=self.lt_relax,
                **params,
            )
            retry = False
        except RuntimeError:
            print("Long-term optimization failed, retrying next round")
            return qor_target, True, last_qor_target

        if budget is not None:
            qor_target = lt_result.metrics["qor_target"]
            if last_qor_target is not None:
                qor_target = self._smooth_qor_target(qor_target, last_qor_target)
            last_qor_target = qor_target
            qor_target_history[step] = qor_target
            print(f"Current min QoR: {qor_target:.3f} at {lt_result.metrics['emissions']} tCO2e")

        return qor_target, retry, last_qor_target

    def _smooth_qor_target(self, new_target: float, previous_target: float) -> float:
        lower_bound = previous_target * 0.95
        upper_bound = previous_target * 1.05
        return min(max(new_target, lower_bound), upper_bound)

    def _maybe_generate_outlooks(
        self,
        *,
        scenario: Scenario,
        step: int,
        R_hat: np.ndarray,
        C_hat: np.ndarray,
        outlooks: dict[int, dict],
    ) -> dict[int, dict]:
        if not self.outlook_frequency or step % self.outlook_frequency != 0:
            return outlooks

        R_hat_lower = scenario.generate_R_hat(step, kind="yhat_lower")
        R_hat_upper = scenario.generate_R_hat(step, kind="yhat_upper")

        outlook = {}
        for outlook_type, R_i in [("expected", R_hat), ("worst case", R_hat_lower), ("best case", R_hat_upper)]:
            outlook[outlook_type] = {}
            for qor in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
                print(f"Outlook {outlook_type} {qor} ...")
                outlook[outlook_type][qor] = self._emissions_outlook(step, R_i, C_hat, scenario, qor)

        outlooks[step] = outlook
        return outlooks

    def _build_metrics(
        self,
        start_time: float,
        qor_target_history: dict[int, float],
        outlooks: dict[int, dict],
    ) -> dict:
        metrics = {"runtime": time() - start_time}
        if qor_target_history:
            metrics["qor_target_history"] = qor_target_history
        if outlooks:
            metrics["outlooks"] = outlooks
        return metrics

    @staticmethod
    def _sanitize_long_term_params(params: dict) -> dict:
        info = dict(params)
        info.pop("past_vps", None)
        return info

    def _emissions_outlook(self, i: int, R_hat, C_hat, scenario, qor_target) -> float:
        outlook_model = QtModel(scenario, silent=self.lt_silent, callback=self.lt_callback)
        metrics = outlook_model.minimize_emissions(qor_target, window=scenario.I[i:], R_hat=R_hat, C_hat=C_hat,
                                                   relax=self.outlook_relax,
                                                   past_vps=False)  # we only optimize for the future
        return metrics["emissions"]

    def _long_term_optimization(self,
                                R_hat,
                                C_hat,
                                scenario: Scenario,
                                online_model: QtModel,
                                window: list[int],
                                relax: bool,
                                past_vps: bool = True,
                                budget: Optional[float] = None,
                                qor_target: Optional[float] = None) -> Result:
        """Currently updates a_, d_ inplace and returns R_hat, C_hat"""
        if budget is not None and qor_target is not None:
            raise ValueError("Only one of budget or qor_target must be specified")

        lt_model = QtModel(scenario, silent=self.lt_silent, callback=self.lt_callback)
        lt_model.a_[:window[0], :, :] = online_model.a_[:window[0], :, :]
        lt_model.d_[:window[0], :, :] = online_model.d_[:window[0], :, :]

        if qor_target is not None:
            metrics = lt_model.minimize_emissions(qor_target, window=window, R_hat=R_hat, C_hat=C_hat,
                                                  past_vps=past_vps, future_vps=False, relax=relax)
        elif budget is not None:
            metrics = lt_model.maximize_qor(budget, window=window, R_hat=R_hat, C_hat=C_hat,
                                            past_vps=past_vps, future_vps=False, relax=relax)
        else:
            raise ValueError("Either budget or qor_target must be specified")

        lt_result = Result(extract_a(lt_model.a_), extract_d(lt_model.d_),
                           optimizer=self, scenario=scenario, metrics=metrics)

        if window[-1] == 8760:
            lt_result.print_stats()
            print(f"Expected QoR: min={lt_result.qor_target_forecast(R_hat=R_hat):.3f}")

        # Update online_model
        online_model.a_[window, :, :] = lt_result.a[window, :, :]
        online_model.d_[window, :, :] = lt_result.d[window, :, :]

        return lt_result

    def _short_term_optimization(self,
                                 scenario: Scenario,
                                 online_model: QtModel,
                                 qor_target: float,
                                 i: int,
                                 horizon,
                                 R_hat,
                                 C_hat,
                                 retry=False):
        window = scenario.I[i:i + horizon]
        while True:
            try:
                metrics = online_model.minimize_emissions(qor_target, window=window, R_hat=R_hat, C_hat=C_hat)
                print(
                    f"{i:<4}: "
                    f"QoR: {metrics['qor_target']:.3f}, "
                    f"Runtime: {metrics['runtime']:.3f}s, "
                    f"Gap: {metrics['mip_gap']:.3f}, "
                    f"Status: {metrics['optimization_status']}"
                )

                a_act, d_act = perform_step(i, online_model.d_, online_model.a_, scenario)
                assert np.isclose(
                    a_act.sum(), scenario.R[i].sum()
                ), f"Sanity check: All requests should be served, but allocated {a_act.sum()} of {scenario.R[i].sum()}"
                break
            except RuntimeError:
                if retry:
                    # If we already spent the entire budget for the step we try
                    # again with a slightly relaxed QoR target.  This mirrors
                    # the original behaviour where the short-term solver would
                    # repeatedly degrade the QoR until feasible.
                    qor_target *= 0.95
                    continue
                print("No solution found, allocating all to best model")
                a_act, d_act = execute_perfect_step(scenario.R[i], scenario)
                break

        # d_acts.append(d_act)

        online_model.a_[i] = a_act
        online_model.d_[i] = d_act

        # return metrics


def perform_step(i: int, d_: np.array, a_: np.array, scenario: Scenario) -> tuple[np.array, np.array]:
    """Extract the integer decisions for allocations and deployments at step ``i``."""

    d_act = np.array([[int(round(d_[i, q, m].X)) for m in scenario.M] for q in scenario.Q])
    a_act = np.array([[a_[i, u, q].X for q in scenario.Q] for u in scenario.U])
    return a_act, d_act
