from collections import deque
from time import time

import numpy as np

from src.executor import execute_minimal_step, execute_perfect_step
from src.optimizer import perform_step
from src.qt_model import QtModel
from src.result import Result
from src.scenario import Scenario


class Greedy:
    def __init__(self,
                 mode: str,
                 lt_frequency: str,
                 oracle: bool = False):
        self.mode = mode
        self.lt_frequency = lt_frequency
        self.oracle = oracle

    @property
    def name(self) -> str:
        result = [f"mode={self.mode},lt_frequency={self.lt_frequency}"]
        if self.oracle:
            result.append(f"oracle")
        return f"Greedy({','.join(result)})"

    def maximize_qor(self, scenario: Scenario, budget: float, experiment_name: str = None):
        t0 = time()

        a_ = np.empty((len(scenario.I), len(scenario.U), len(scenario.Q)), dtype=object)
        d_ = np.empty((len(scenario.I), len(scenario.Q), len(scenario.M)), dtype=object)

        C_hat = scenario.generate_C_hat(0, kind="yhat" if not self.oracle else "oracle")

        remaining_budget = budget
        for i in scenario.I:
            if i % self.lt_frequency == 0 and not self.oracle:
                print(f"{i:<4} -------- Budget: {remaining_budget:.2f} tCO2e")
                R_hat = scenario.generate_R_hat(i, kind="yhat")

            C_hat[i] = scenario.C[i]
            R_hat[i] = scenario.R[i]

            step_budgets = self._step_budgets(i, scenario, remaining_budget, R_hat, C_hat)
            step_budget = step_budgets.popleft()

            window = scenario.I[i:i + 1]
            qt_model = QtModel(scenario, silent=True, callback=None)
            qt_model.scenario.vp = 1
            try:
                metrics = qt_model.maximize_qor(step_budget, window=window, R_hat=R_hat, C_hat=C_hat, emissions_window=window)
                a_act, d_act = perform_step(i, qt_model.d_, qt_model.a_, scenario)
            except RuntimeError:
                a_act, d_act = execute_minimal_step(R_hat[i], scenario)
            if metrics["qor_target"] == 1:
                a_act, d_act = execute_perfect_step(R_hat[i], scenario)

            a_[i] = a_act
            d_[i] = d_act

            remaining_budget -= metrics["emissions"]

        metrics = {
            "runtime": time() - t0,
        }
        return Result(a_, d_, optimizer=self, scenario=scenario, metrics=metrics, info={"budget": budget})

    def _step_budgets(self, i, scenario: Scenario, budget: float, R_hat, C_hat):
        if self.mode == "constant":
            remaining_steps = len(scenario.I[i:])
            step_budget = budget / remaining_steps
            step_budgets = np.full(remaining_steps, fill_value=step_budget)
        elif self.mode == "weighted":
            weights = R_hat[i:].sum(axis=1) * C_hat[i:]
            weights = weights / weights.sum()
            step_budgets = weights * budget
        else:
            raise ValueError("Unknown mode")
        return deque(step_budgets)
