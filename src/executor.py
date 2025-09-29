"""Execution helpers that convert solver decisions to integer actions.

The QualityTime optimizer internally stores PuLP decision variables.  When a
solver run fails, or when we intentionally bypass optimization, we fall back to
the utilities below to construct feasible allocations for a single time step.
The helpers intentionally stay minimal: they assume a single machine type and a
single user group which mirrors the experimental setup used throughout the
repository.
"""

import numpy as np

from src.scenario import Scenario


def execute_minimal_step(R_i: np.ndarray, scenario: Scenario) -> tuple[np.ndarray, np.ndarray]:
    """Allocate just enough low-quality capacity to satisfy the requests.

    This function is used as a last resort whenever the optimizer fails to
    produce a feasible solution.  All demand is mapped to the lowest quality
    tier which typically corresponds to the highest latency and lowest energy
    cost.
    """

    # TODO: assumes one machine type and one user group.
    machine = 0
    user_group = 0
    worst_quality = 0
    d_act = np.zeros((len(scenario.Q), len(scenario.M)))
    d_act[worst_quality, machine] = np.ceil(R_i[user_group] / scenario.K[worst_quality][machine])
    a_act = _find_optimal_allocation(d_act, R_i, scenario)
    return a_act, d_act


def execute_perfect_step(R_i: np.ndarray, scenario: Scenario) -> tuple[np.ndarray, np.ndarray]:
    """Allocate enough high-quality capacity to serve all requests."""

    # TODO: assumes one machine type and one user group.
    machine = 0
    user_group = 0
    best_quality = -1
    d_act = np.zeros((len(scenario.Q), len(scenario.M)))
    d_act[best_quality, machine] = np.ceil(R_i[user_group] / scenario.K[best_quality][machine])
    a_act = _find_optimal_allocation(d_act, R_i, scenario)
    return a_act, d_act


def _find_optimal_allocation(d: np.ndarray, R_i: np.ndarray, scenario: Scenario) -> np.ndarray:
    """Greedily allocate demand to the provisioned capacity.

    The helper iterates from highest quality to lowest (``[::-1]``) to match the
    intent of :func:`execute_perfect_step`.  It keeps allocating requests until
    either the capacity or the demand is exhausted.
    """

    capacity_per_quality = (d * scenario.K).sum(axis=1)

    # TODO: refine this for multiple user priority groups.
    assert len(R_i) == 1
    requests = R_i[0]  # user 0
    a = np.zeros((len(scenario.U), len(scenario.Q)))

    for backwards_index, capacity in enumerate(capacity_per_quality[::-1]):
        quality_index = len(capacity_per_quality) - 1 - backwards_index
        if capacity > requests:
            a[0, quality_index] = requests
            break
        a[0, quality_index] = capacity
        requests -= capacity

    assert np.isclose(a.sum(), R_i[0])
    return a
