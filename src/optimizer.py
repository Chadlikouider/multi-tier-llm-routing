from time import time
from typing import Optional

import numpy as np

from src.executor import execute_perfect_step
from src.qt_model import QtModel
from src.result import Result
from src.scenario import Scenario
from src.util import Callback, extract_a, extract_d


class Qt:
    def __init__(self, callback: Optional[Callback] = None, silent: bool = False, relax: bool = False):
        """Global optimization model."""
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

    def minimize_emissions(self,
                           scenario: Scenario,
                           qor_target: float,
                           R_hat: Optional[np.array] = None,
                           C_hat: Optional[np.array] = None,
                           info: Optional[dict] = None) -> Result:
        R_hat = R_hat if R_hat is not None else scenario.R
        C_hat = C_hat if C_hat is not None else scenario.C
        qt_model = QtModel(scenario, silent=self.silent, callback=self.callback)
        metrics = qt_model.minimize_emissions(qor_target, window=scenario.I, R_hat=R_hat, C_hat=C_hat, relax=self.relax)
        info = info if info is not None else {}
        info["qor_target"] = qor_target
        return Result(extract_a(qt_model.a_), extract_d(qt_model.d_),
                      optimizer=self, scenario=scenario, metrics=metrics, info=info)

    def maximize_qor(self, scenario: Scenario, budget: float) -> Result:
        qt_model = QtModel(scenario, silent=self.silent, callback=self.callback)
        metrics = qt_model.maximize_qor(budget, window=scenario.I, R_hat=scenario.R, C_hat=scenario.C, relax=self.relax)
        return Result(extract_a(qt_model.a_), extract_d(qt_model.d_),
                      optimizer=self, scenario=scenario, metrics=metrics, info={"budget": budget})


class QtOnline:
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
        return self._optimize(scenario, qor_target=qor_target)

    def maximize_qor(self, scenario: Scenario, budget: float) -> Result:
        return self._optimize(scenario, budget=budget)

    def _optimize(self, scenario: Scenario, qor_target: float = None, budget: float = None) -> Result:
        horizon = min(self.horizon, scenario.vp)

        t0 = time()
        st_metrics = {}

        online_model = QtModel(scenario, silent=self.st_silent, callback=self.st_callback)

        d_sts = []
        d_acts = []
        qor_target_history = {}
        outlooks = {}

        # Init forecasts in case of oracle
        C_hat = scenario.C
        R_hat = scenario.R

        if budget:
            long_term_params = {"budget": budget, "past_vps": False}
        else:
            long_term_params = {"qor_target": qor_target}

        retry_lt = False
        print("Starting online budget optimization...")
        last_qor_target = None
        for i in scenario.I:
            print(f"{i:<4} ------------------------")
            if i % self.lt_frequency == 0:
                C_hat = scenario.generate_C_hat(i, kind="yhat" if not self.oracle else "oracle")
                R_hat = scenario.generate_R_hat(i, kind="yhat" if not self.oracle else "oracle")

            if i % self.lt_frequency == 0 or retry_lt:
                window = scenario.I[i:]

                try:
                    # TODO
                    # print(f"Long-term optimization {experiment_name}...", flush=True)
                    print(f"Long-term optimization ...", flush=True)
                    lt_result = self._long_term_optimization(R_hat, C_hat, scenario, online_model,
                                                             window=window,
                                                             relax=self.lt_relax,
                                                             **long_term_params)
                    retry_lt = False

                    if budget:  # TODO special case
                        qor_target = lt_result.metrics["qor_target"]
                        if last_qor_target is not None:
                            if qor_target < last_qor_target * 0.95:
                                qor_target = last_qor_target * 0.95
                            elif qor_target > last_qor_target * 1.05:
                                qor_target = last_qor_target * 1.05
                        last_qor_target = qor_target
                        qor_target_history[i] = qor_target
                        print(f"Current min QoR: {qor_target:.3f} at {lt_result.metrics['emissions']} tCO2e")
                except RuntimeError:
                    # The long-term optimization can fail if passed time steps make it impossible to reach the QoR
                    # In this case, we run a short-term optimization which will opt for an optimal solution
                    # and retry the long-term optimization in the next round
                    print("Long-term optimization failed, retrying next round")
                    retry_lt = True

            # Outlooks
            if self.outlook_frequency and i % self.outlook_frequency == 0:
                R_hat_lower = scenario.generate_R_hat(i, kind="yhat_lower")
                R_hat_upper = scenario.generate_R_hat(i, kind="yhat_upper")
                outlook = {}
                for outlook_type, R_i in [("expected", R_hat), ("worst case", R_hat_lower), ("best case", R_hat_upper)]:
                    outlook[outlook_type] = {}
                    for qor in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
                        print(f"Outlook {outlook_type} {qor} ...")
                        outlook[outlook_type][qor] = self._emissions_outlook(i, R_i, C_hat, scenario, qor)
                outlooks[i] = outlook

            R_hat[i] = scenario.R[i]
            C_hat[i] = scenario.C[i]

            self._short_term_optimization(scenario, online_model, qor_target, i, horizon, R_hat, C_hat, retry=(budget is not None))
            # st_metrics[i] = metrics

        # TODO bit dirty
        if "past_vps" in long_term_params:
            del long_term_params["past_vps"]

        metrics = {
            "runtime": time() - t0,
            # "lt_metrics": lt_metrics,
            # "st_metrics": st_metrics,
            # "d_lts": np.array(d_lts),
            # "d_sts": np.array(d_sts),
            # "d_acts": np.array(d_acts),
        }
        if qor_target_history:
            metrics["qor_target_history"] = qor_target_history
        if outlooks:
            metrics["outlooks"] = outlooks
        return Result(online_model.a_, online_model.d_, optimizer=self, scenario=scenario, metrics=metrics, info=long_term_params)

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
                print(f"{i:<4}: "
                      f"QoR: {metrics['qor_target']:.3f}, "
                      f"Runtime: {metrics['runtime']:.3f}s, "
                      f"Gap: {metrics['mip_gap']:.3f}, "
                      f"Status: {metrics['optimization_status']}")

                a_act, d_act = perform_step(i, online_model.d_, online_model.a_, scenario)
                assert np.isclose(a_act.sum(), scenario.R[i].sum()), f"Sanity check: All requests should be served, but allocated {a_act.sum()} of {scenario.R[i].sum()}"
                break
            except RuntimeError:
                if retry:
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
    d_act = np.array([[int(round(d_[i, q, m].X)) for m in scenario.M] for q in scenario.Q])
    a_act = np.array([[a_[i, u, q].X for q in scenario.Q] for u in scenario.U])
    return a_act, d_act
