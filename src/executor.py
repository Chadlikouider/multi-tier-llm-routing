import numpy as np

from src.scenario import Scenario


def execute_minimal_step(R_i: np.array, scenario: Scenario):
    """Executes a minimal step of the simulation in case no solution was found during optimization."""
    # TODO assumes one machine type and one user group
    machine = 0
    user_group = 0
    worst_quality = 0
    d_act = np.zeros((len(scenario.Q), len(scenario.M)))
    d_act[worst_quality, machine] = np.ceil(R_i[user_group] / scenario.K[worst_quality][machine])
    a_act = _find_optimal_allocation(d_act, R_i, scenario)
    return a_act, d_act


def execute_perfect_step(R_i: np.array, scenario: Scenario):
    """Executes a minimal step of the simulation in case no solution was found during optimization."""
    # TODO assumes one machine type and one user group
    machine = 0
    user_group = 0
    best_quality = -1
    d_act = np.zeros((len(scenario.Q), len(scenario.M)))
    d_act[best_quality, machine] = np.ceil(R_i[user_group] / scenario.K[best_quality][machine])
    a_act = _find_optimal_allocation(d_act, R_i, scenario)
    return a_act, d_act


def _find_optimal_allocation(d: np.array, R_i: np.array, scenario: Scenario):
    capacity_per_quality = (d * scenario.K).sum(axis=1)
    # TODO refine this for multiple user priority groups!
    assert len(R_i) == 1
    requests = R_i[0]  # user 0
    a = np.array(np.zeros((len(scenario.U), len(scenario.Q))))
    for bi, capacity in enumerate(capacity_per_quality[::-1]):  # iterate
        i = len(capacity_per_quality) - 1 - bi
        if capacity > requests:
            a[0, i] = requests
        else:
            a[0, i] = capacity
            requests -= capacity
    assert a.sum() == R_i[0]
    return a
