"""Simple baseline strategies for QualityTime experiments.

The :class:`Greedy` class implements a light-weight baseline that performs a
single step optimization for each time interval by distributing the remaining
budget across the rest of the horizon.  The implementation deliberately avoids
the orchestration machinery from :mod:`src.optimizer` which makes it easier to
reason about for quick experiments or ablation studies.
"""

from collections import deque
from time import time
from typing import Deque

import numpy as np

from src.executor import execute_minimal_step, execute_perfect_step
from src.optimizer import perform_step
from src.qt_model import QtModel
from src.result import Result
from src.scenario import Scenario


class Greedy:
    """Budget distributing baseline.

    Parameters
    ----------
    mode:
        Strategy used to split the remaining budget. ``"constant"`` allocates
        the same amount to each remaining interval, ``"weighted"`` biases
        towards intervals with a large product of forecasted demand and carbon
        intensity.
    lt_frequency:
        Number of hours between long-term forecast refreshes.  The parameter is
        expressed as an ``int`` in the configuration but stored as provided to
        match historical usage.
    oracle:
        If ``True`` the baseline skips the forecasting step and relies on
        ground-truth data for carbon intensity and demand.
    """

    def __init__(self, mode: str, lt_frequency: int, oracle: bool = False):
        self.mode = mode
        self.lt_frequency = lt_frequency
        self.oracle = oracle

    @property
    def name(self) -> str:
        """Return a human readable name for result artifacts."""

        result = [f"mode={self.mode},lt_frequency={self.lt_frequency}"]
        if self.oracle:
            result.append("oracle")
        return f"Greedy({','.join(result)})"

    def maximize_qor(
        self,
        scenario: Scenario,
        budget: float,
        experiment_name: str | None = None,
    ) -> Result:
        """Run the greedy optimizer for the provided :class:`Scenario`.

        The method iterates over each interval, allocates a share of the
        remaining budget, and solves a small PuLP model limited to that single
        step.  If the solver fails we fall back to simple execution helpers from
        :mod:`src.executor` to guarantee feasibility.
        """

        del experiment_name  # kept for backwards compatibility with callers

        t0 = time()

        # Allocate containers for the final integer decisions that the helper
        # methods will return.
        a_ = np.empty((len(scenario.I), len(scenario.U), len(scenario.Q)), dtype=object)
        d_ = np.empty((len(scenario.I), len(scenario.Q), len(scenario.M)), dtype=object)

        # Start from forecasts (or oracle data) and iteratively refine them with
        # ground truth as we move through the horizon.
        forecast_kind = "oracle" if self.oracle else "yhat"
        C_hat = scenario.generate_C_hat(0, kind=forecast_kind)
        R_hat = scenario.generate_R_hat(0, kind=forecast_kind)

        remaining_budget = budget
        for i in scenario.I:
            if i % self.lt_frequency == 0 and not self.oracle:
                print(f"{i:<4} -------- Budget: {remaining_budget:.2f} tCO2e")
                R_hat = scenario.generate_R_hat(i, kind="yhat")

            # Replace the forecast for the current step with the observed value.
            C_hat[i] = scenario.C[i]
            R_hat[i] = scenario.R[i]

            step_budget = self._step_budgets(i, scenario, remaining_budget, R_hat, C_hat).popleft()

            window = scenario.I[i : i + 1]
            qt_model = QtModel(scenario, silent=True, callback=None)
            qt_model.scenario.vp = 1
            try:
                metrics = qt_model.maximize_qor(
                    step_budget,
                    window=window,
                    R_hat=R_hat,
                    C_hat=C_hat,
                    emissions_window=window,
                )
                a_act, d_act = perform_step(i, qt_model.d_, qt_model.a_, scenario)
            except RuntimeError:
                # In rare cases the single step optimization fails.  Fall back to
                # minimal execution that simply provisions enough resources for
                # the worst quality level.
                a_act, d_act = execute_minimal_step(R_hat[i], scenario)
                metrics = {"emissions": 0, "qor_target": 0}
            if metrics["qor_target"] == 1:
                # If the solver decided to provide the best possible QoR anyway
                # we might as well provision the perfect solution to avoid
                # rounding artefacts.
                a_act, d_act = execute_perfect_step(R_hat[i], scenario)

            a_[i] = a_act
            d_[i] = d_act

            remaining_budget -= metrics["emissions"]

        metrics = {"runtime": time() - t0}
        return Result(a_, d_, optimizer=self, scenario=scenario, metrics=metrics, info={"budget": budget})

    def _step_budgets(
        self,
        start: int,
        scenario: Scenario,
        budget: float,
        R_hat: np.ndarray,
        C_hat: np.ndarray,
    ) -> Deque[float]:
        """Return a deque with per-interval budgets starting at ``start``."""

        remaining_steps = len(scenario.I[start:])

        if remaining_steps == 0:
            return deque()

        if self.mode == "constant":
            step_budget = budget / remaining_steps
            step_budgets = np.full(remaining_steps, fill_value=step_budget)
        elif self.mode == "weighted":
            weights = R_hat[start:].sum(axis=1) * C_hat[start:]
            weights_sum = weights.sum()
            if weights_sum == 0:
                step_budgets = np.full(remaining_steps, fill_value=budget / remaining_steps)
            else:
                weights = weights / weights_sum
                step_budgets = weights * budget
        else:
            raise ValueError(f"Unknown mode '{self.mode}'")
        return deque(step_budgets)
