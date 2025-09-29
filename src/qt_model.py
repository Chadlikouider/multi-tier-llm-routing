"""PuLP model describing the QualityTime optimization problem."""

from time import time
from typing import Optional

import numpy as np
from pulp import (
    LpContinuous,
    LpInteger,
    LpMaximize,
    LpMinimize,
    LpProblem,
    LpStatus,
    LpVariable,
    PULP_CBC_CMD,
    lpSum,
    value,
)

from src.scenario import Scenario
from src.util import Callback, get_validity_periods

# Keep a short alias for the sum helper to mirror the PuLP examples and avoid
# shadowing Python's built-in ``sum``.
sum_ = lpSum


class QtModel:

    def __init__(self, scenario: Scenario, silent: bool = False, callback: Optional[Callback] = None):
        self.scenario = scenario
        self.silent = silent
        self.callback = callback

        self.model: Optional[LpProblem] = None
        self.a_: Optional[np.ndarray] = None
        self.d_: Optional[np.ndarray] = None

    def minimize_emissions(self,
                           qor_target,
                           window,
                           R_hat,
                           C_hat,
                           past_vps: bool = True,  # for outlooks
                           future_vps: bool = True,
                           relax: bool = False):
        """Optimizes for the lowest budget that satisfies the QoR constraint."""
        t0 = time()
        self.model = LpProblem("qt_minimize_emissions", LpMinimize)
        self.a_, self.d_ = _init_variables(self.model, self.scenario, R_hat, window=window, relax=relax)

        validity_periods = get_validity_periods(window, self.scenario.vp, past=past_vps, future=future_vps)

        # Set QoR constraint
        qor = _qor(
            self.scenario.slo_lower,
            self.scenario.slo_upper,
            validity_periods,
            self.a_,
            self.scenario,
            self.model,
            R=R_hat,
        )
        self.model += qor >= qor_target, "qor_constr"

        # Configure model for budget optimization
        emissions = _emissions(self.scenario, self.d_, window=window, C_hat=C_hat)
        self.model += emissions

        # Optimize
        self._optimize()
        if LpStatus[self.model.status] != "Optimal":
            raise RuntimeError("Optimization did not converge to an optimal solution")

        metrics = self._collect_metrics()
        metrics["qor_target"] = value(qor)
        metrics["emissions"] = value(emissions)
        metrics["runtime"] = time() - t0
        return metrics

    def maximize_qor(self,
                     budget,
                     window,
                     R_hat,
                     C_hat,
                     past_vps: bool = True,
                     future_vps: bool = True,
                     relax: bool = False,
                     emissions_window: list[int] = None) -> dict:
        """Maximised the lowest QoR among the provided validity_periods under a budget."""
        if emissions_window is None:
            emissions_window = self.scenario.I

        t0 = time()
        self.model = LpProblem("qt_maximize_qor", LpMaximize)
        self.a_, self.d_ = _init_variables(self.model, self.scenario, R_hat, window=window, relax=relax)

        validity_periods = get_validity_periods(window, self.scenario.vp, past=past_vps, future=future_vps)

        # Set budget constraint
        emissions = _emissions(self.scenario, self.d_, window=emissions_window, C_hat=C_hat)
        self.model += emissions <= budget, "budget_constraint"

        # Configure model for QoR optimization
        qor = _qor(
            self.scenario.slo_lower,
            self.scenario.slo_upper,
            validity_periods,
            self.a_,
            self.scenario,
            self.model,
            R=R_hat,
        )
        self.model += qor

        # Optimize
        self._optimize()
        if LpStatus[self.model.status] != "Optimal":
            raise RuntimeError("Optimization did not converge to an optimal solution")

        metrics = self._collect_metrics()
        metrics["qor_target"] = value(qor)
        metrics["emissions"] = value(emissions)
        metrics["runtime"] = time() - t0
        return metrics

    def _optimize(self):
        solver = PULP_CBC_CMD(msg=0 if self.silent else 1)
        self.model.solve(solver)

    def _collect_metrics(self):
        metrics = {
            "work": None,
            "mip_gap": None,
            "status": LpStatus[self.model.status],
        }
        if self.callback:
            metrics["callback"] = self.callback.metrics
            metrics["optimization_status"] = self.callback.optimization_status
        else:
            # ``QtOnline`` expects the metrics dictionary to expose an
            # ``optimization_status`` field regardless of whether we used a
            # callback.  The Gurobi-backed implementation filled this field,
            # whereas the simplified PuLP code only reported ``status``.  The
            # mismatch caused a ``KeyError`` when printing diagnostics from the
            # short-term optimizer.  Fall back to the solver status so the
            # public API stays consistent with the historical behaviour.
            metrics["optimization_status"] = metrics["status"]
        return metrics


def _init_variables(
    model: LpProblem,
    scenario: Scenario,
    R_hat: np.ndarray,
    window: Optional[list[int]] = None,
    relax: bool = False,
    add_constraints: bool = True,
):
    """Create PuLP decision variables and add the structural constraints."""

    if window is None:
        window = scenario.I

    window_set = set(window)

    a_ = np.full((len(scenario.I), len(scenario.U), len(scenario.Q)), 0, dtype=object)
    d_ = np.full((len(scenario.I), len(scenario.Q), len(scenario.M)), 0, dtype=object)

    for i in scenario.I:
        for u in scenario.U:
            for q in scenario.Q:
                if i in window_set:
                    up_bound = float(R_hat[i, u]) if R_hat[i, u] is not None else None
                    a_[i, u, q] = LpVariable(f"a_{i}_{u}_{q}", lowBound=0, upBound=up_bound)
                else:
                    a_[i, u, q] = 0

        for q in scenario.Q:
            for m in scenario.M:
                if scenario.machines[m].performance[q] > 0:
                    if i in window_set:
                        category = LpContinuous if relax else LpInteger
                        d_[i, q, m] = LpVariable(f"d_{i}_{q}_{m}", lowBound=0, cat=category)
                    else:
                        d_[i, q, m] = 0
                else:
                    d_[i, q, m] = None

        if add_constraints and i in window_set:
            _constraint_request_allocation(scenario, model, a_, i, R_hat)
            _constraint_sufficient_resources(scenario, model, a_, d_, i)

    return a_, d_


def _emissions(scenario: Scenario, d_, window, C_hat):
    """Returns the emissions in gCO2e."""

    return sum_(
        interval_emissions(i, q, m, d_, scenario, C_hat)
        for i in window
        for q in scenario.Q
        for m in scenario.M
    )


def interval_emissions(i, q, m, d_, scenario: Scenario, C_hat, load_dependent=False, a_=None):
    if d_[i, q, m] is None:
        return 0
    p = interval_power_per_machine(i, q, m, d_, scenario, load_dependent=load_dependent, a_=a_)
    return d_[i, q, m] * (p * C_hat[i] + scenario.machines[m].embedded_carbon)


def interval_power_per_machine(i, q, m, d_, scenario: Scenario, load_dependent=False, a_=None) -> float:
    if load_dependent:
        raise NotImplementedError("Load-dependent power usage is not supported with the PuLP backend")
    else:
        return scenario.machines[m].load_independent_power_usage()


def _qor(slo_lower, slo_upper, validity_periods, a_, scenario, model, R):
    errors = []

    for vp_i, validity_period in enumerate(validity_periods):
        for u in scenario.U:
            request_sum = float(R[validity_period, u].sum())
            if request_sum == 0:
                continue
            for q in scenario.Q:
                # SLI
                lb = min(slo_upper[u][q], slo_lower[u][q])
                ub = max(slo_upper[u][q], slo_lower[u][q])
                sli = LpVariable(f"sli_{vp_i}_{u}_{q}", lowBound=lb, upBound=ub)
                allocated = sum_(a_[idx, u, q] for idx in validity_period)
                model += sli == allocated / request_sum, f"sli_constr_{vp_i}_{u}_{q}"

                # Errors
                denominator = abs(slo_lower[u][q] - slo_upper[u][q])
                if denominator == 0:
                    continue
                diff = LpVariable(f"diff_{vp_i}_{u}_{q}", lowBound=-1, upBound=1)
                model += diff == (sli - slo_upper[u][q]) / denominator, f"diff_constr_{vp_i}_{u}_{q}"

                error = LpVariable(f"error_{vp_i}_{u}_{q}", lowBound=0, upBound=1)
                errors.append(error)
                model += error >= diff, f"error_pos_{vp_i}_{u}_{q}"
                model += error >= -diff, f"error_neg_{vp_i}_{u}_{q}"

    if len(errors) > 0:
        highest_error = LpVariable("highest_error", lowBound=0, upBound=1)
        for idx, error in enumerate(errors):
            model += highest_error >= error, f"highest_error_ge_{idx}"
    else:
        highest_error = 0
    return 1 - highest_error


def _constraint_sufficient_resources(scenario: Scenario, model, a_, d_, i: int):
    """Ensure deployment can serve all requests for each level."""

    for q in scenario.Q:
        allocated_requests = sum_(a_[i, u, q] for u in scenario.U)
        servable_requests = sum_(
            d_[i, q, m] * scenario.machines[m].performance[q] for m in scenario.M if d_[i, q, m] is not None
        )
        model += allocated_requests <= servable_requests, f"constraint_sufficient_resources_{i}_{q}"


def _constraint_no_wasted_resources(scenario: Scenario, model, a_, d_, i: int):
    # TODO assumes only one machine
    constr = []
    for q in scenario.Q:
        allocated_requests = sum_(a_[i, u, q] for u in scenario.U)
        servable_requests = sum_(d_[i, q, m] * scenario.machines[m].performance[q]
                                 for m in scenario.M if d_[i, q, m] is not None)
        surplus = servable_requests - allocated_requests
        constr.append(surplus)
        model += surplus <= scenario.machines[0].performance[q], f"constraint_no_wasted_resources_{i}_{q}"
    return constr


def _constraint_request_allocation(scenario, model, a_, i: int, R):
    """Ensure all requests are allocated to a level."""

    for u in scenario.U:
        model += sum_(a_[i, u, q] for q in scenario.Q) == float(R[i, u]), f"constraint_request_allocation_{i}_{u}"
