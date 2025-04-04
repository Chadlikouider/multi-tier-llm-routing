from time import time
from typing import Optional

import gurobipy as gp
import numpy as np
from gurobipy import GRB, abs_

from src.scenario import Scenario
from src.util import GurobiEnv, Callback, get_validity_periods

sum_ = gp.quicksum


class QtModel:

    def __init__(self, scenario: Scenario, silent: bool = False, callback: Optional[Callback] = None):
        self.scenario = scenario
        if silent:
            self.model = gp.Model(env=GurobiEnv.get_env())
        else:
            self.model = gp.Model()
        self.callback = callback

        self.a_ = np.empty((len(scenario.I), len(scenario.U), len(scenario.Q)), dtype=object)
        self.d_ = np.empty((len(scenario.I), len(scenario.Q), len(scenario.M)), dtype=object)

        self._gc = []
        self._gc = []

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
        self._clean_gc()
        gc = _init_variables(self.model, self.a_, self.d_, self.scenario, R_hat, window=window, relax=relax)
        self._gc.extend(gc)

        validity_periods = get_validity_periods(window, self.scenario.vp, past=past_vps, future=future_vps)

        # Set QoR constraint
        qor, qor_gc = _qor(self.scenario.slo_lower, self.scenario.slo_upper, validity_periods, self.a_, self.scenario, self.model, R=R_hat)
        self._gc.extend(qor_gc)
        qor_constr = self.model.addConstr(qor >= qor_target, name="qor_constr")
        self._gc.append(qor_constr)

        # Configure model for budget optimization
        emissions = _emissions(self.scenario, self.d_, window=window, C_hat=C_hat)
        self.model.setObjective(emissions, GRB.MINIMIZE)

        # Optimize
        self._optimize(self.callback() if self.callback else None)
        if self.model.SolCount == 0:
            # We are always expecting a solution
            raise RuntimeError("This is bad")

        metrics = self._collect_metrics()
        metrics["qor_target"] = qor.getValue() if isinstance(qor, gp.LinExpr) else qor
        metrics["emissions"] = emissions.getValue()
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
        self._clean_gc()
        gc = _init_variables(self.model, self.a_, self.d_, self.scenario, R_hat, window=window, relax=relax)
        self._gc.extend(gc)

        validity_periods = get_validity_periods(window, self.scenario.vp, past=past_vps, future=future_vps)

        # Set budget constraint
        emissions = _emissions(self.scenario, self.d_, window=emissions_window, C_hat=C_hat)
        budget_constraint = self.model.addConstr(emissions <= budget, name="budget_constraint")
        self._gc.append(budget_constraint)

        # Configure model for QoR optimization
        qor, qor_gc = _qor(self.scenario.slo_lower, self.scenario.slo_upper, validity_periods, self.a_, self.scenario, self.model, R=R_hat)
        self._gc.extend(qor_gc)
        self.model.setObjective(qor, GRB.MAXIMIZE)

        # Optimize
        self._optimize(self.callback() if self.callback else None)
        if self.model.SolCount == 0:
            # We are always expecting a solution
            raise RuntimeError("This is bad")

        metrics = self._collect_metrics()
        metrics["qor_target"] = qor.getValue() if isinstance(qor, gp.LinExpr) else qor
        metrics["emissions"] = emissions.getValue()
        metrics["runtime"] = time() - t0
        return metrics

    def _clean_gc(self):
        for var_or_constr in self._gc:
            self.model.remove(var_or_constr)
        self._gc = []

    def _optimize(self, callback=None):
        self.model.reset()
        self.model.update()  # process any pending model modifications
        self.model.optimize(callback)

    def _collect_metrics(self):
        metrics = {
            "work": self.model.Work,
            "mip_gap": self.model.MIPGap if hasattr(self.model, 'MIPGap') else None,
        }
        if self.callback:
            metrics["callback"] = self.callback.metrics
            metrics["optimization_status"] = self.callback.optimization_status
        return metrics


def _init_variables(model, a_, d_, scenario, R_hat,
                    window: Optional[list[int]] = None,
                    relax: bool = False,
                    add_constraints=True):
    gc = []

    d_vtype = GRB.CONTINUOUS if relax else GRB.INTEGER
    if window is None:
        window = scenario.I

    for i in window:
        for u in scenario.U:
            for q in scenario.Q:
                var = model.addVar(name=f"a_{i}_{u}_{q}", lb=0, ub=R_hat[i, u])
                a_[i, u, q] = var
                gc.append(var)

        for q in scenario.Q:
            for m in scenario.M:
                if scenario.machines[m].performance[q] > 0:  # only introduce variable if level/machine combination is valid
                    var = model.addVar(name=f"d_{i}_{q}_{m}", lb=0, vtype=d_vtype)
                    d_[i, q, m] = var
                    gc.append(var)
                else:
                    d_[i, q, m] = None

        if add_constraints:
            gc.extend(_constraint_request_allocation(scenario, model, a_, i, R=R_hat))
            gc.extend(_constraint_sufficient_resources(scenario, model, a_, d_, i))

    return gc


def _emissions(scenario: Scenario, d_, window, C_hat):
    """Returns the emissions in gCO2e"""
    return sum_(interval_emissions(i, q, m, d_, scenario, C_hat) for i in window for q in scenario.Q for m in scenario.M)


def interval_emissions(i, q, m, d_, scenario: Scenario, C_hat, load_dependent=False, a_=None):
    p = interval_power_per_machine(i, q, m, d_, scenario, load_dependent=load_dependent, a_=a_)
    return d_[i, q, m] * (p * C_hat[i] + scenario.machines[m].embedded_carbon)


def interval_power_per_machine(i, q, m, d_, scenario: Scenario, load_dependent=False, a_=None) -> float:
    if load_dependent:
        util_ = sum(a_[i, u, q] for u in scenario.U) / sum(d_[i, q, m] * scenario.K[q, m] for m in scenario.M)
        return scenario.machines[m].load_dependent_power_usage(q, util_)
    else:
        return scenario.machines[m].load_independent_power_usage()


def _qor(slo_lower, slo_upper, validity_periods, a_, scenario, model, R):
    gc = []  # added variables and constraints
    errors = []

    for vp_i, validity_period in enumerate(validity_periods):
        for u in scenario.U:
            if R[validity_period, u].sum() == 0:
                continue
            for q in scenario.Q:
                # SLI
                lb = min(slo_upper[u][q], slo_lower[u][q])
                ub = max(slo_upper[u][q], slo_lower[u][q])
                sli = model.addVar(lb=lb, ub=ub, name=f"sli_{vp_i}_{u}_{q}")
                gc.append(sli)
                gc.append(model.addConstr(sli == sum_(a_[validity_period, u, q]) / R[validity_period, u].sum()))

                # Errors
                denominator = abs(slo_lower[u][q] - slo_upper[u][q])
                if denominator == 0:
                    continue
                diff = model.addVar(lb=-1, ub=1, name=f"diff_{vp_i}_{u}_{q}")
                gc.append(diff)
                gc.append(model.addConstr(diff == (sli - slo_upper[u][q]) / denominator, name=f"diff_constr_{vp_i}_{u}_{q}"))

                error = model.addVar(lb=0, ub=1, name=f"error_{vp_i}_{u}_{q}")
                errors.append(error)
                gc.append(error)
                gc.append(model.addConstr(error == abs_(diff), name=f"error_constr_{vp_i}_{u}_{q}"))

    if len(errors) > 0:
        highest_error = model.addVar(lb=0, ub=1, name="highest_error")
        gc.append(highest_error)
        gc.append(model.addConstr(highest_error == gp.max_(errors), name="highest_error_constraint"))
    else:
        highest_error = 0
    return 1 - highest_error, gc


def _constraint_sufficient_resources(scenario: Scenario, model, a_, d_, i: int):
    """Ensure deployment can serve all requests for each level"""
    constr = []
    for q in scenario.Q:
        allocated_requests = sum_(a_[i, u, q] for u in scenario.U)
        servable_requests = sum_(d_[i, q, m] * scenario.machines[m].performance[q] for m in scenario.M)
        constr.append(model.addConstr(allocated_requests <= servable_requests, name=f"constraint_sufficient_resources_{i}_{q}"))
    return constr


def _constraint_no_wasted_resources(scenario: Scenario, model, a_, d_, i: int):
    # TODO assumes only one machine
    constr = []
    for q in scenario.Q:
        allocated_requests = sum_(a_[i, u, q] for u in scenario.U)
        servable_requests = sum_(d_[i, q, m] * scenario.machines[m].performance[q] for m in scenario.M)
        constr.append(servable_requests - allocated_requests)
        model.addConstr(servable_requests - allocated_requests <= scenario.machines[0].performance[q],
                        name=f"constraint_no_wasted_resources_{i}_{q}")
    return constr


def _constraint_request_allocation(scenario, model, a_, i: int, R):
    """Ensure all requests are allocated to a level"""
    constr = []
    for u in scenario.U:
        constr.append(model.addConstr(sum_(a_[i, u, q] for q in scenario.Q) == R[i, u], name=f"constraint_request_allocation_{i}_{u}"))
    return constr
