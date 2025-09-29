import pytest

pytest.importorskip("pulp")

from src.scenario import Machine, Scenario


def test_machine_performance_alignment():
    machine = Machine(
        name="hetero",
        performance={"tier_0": 10, "tier_2": 30},
        embedded_carbon=100,
        request_scaling_factor=1,
        power_usage=1,
        pue=1,
        quality_to_index={"tier_0": 0, "tier_1": 1, "tier_2": 2},
        quality_count=3,
    )
    assert machine.performance == [10, 0.0, 30]


def test_build_slo_matrix_validates_and_orders():
    scenario = Scenario.__new__(Scenario)
    scenario.model_qualities = ["tier_0", "tier_1", "tier_2"]
    scenario.quality_to_index = {quality: idx for idx, quality in enumerate(scenario.model_qualities)}

    user_groups = [
        {
            "name": "group_a",
            "slo_lower": {"tier_0": 1, "tier_1": 0.5, "tier_2": 0},
            "slo_upper": {"tier_0": 0.25, "tier_1": 0.75, "tier_2": 1},
        }
    ]

    matrix = Scenario._build_slo_matrix(scenario, user_groups, "slo_lower")
    assert matrix.tolist() == [[1, 0.5, 0]]


def test_build_slo_matrix_missing_quality():
    scenario = Scenario.__new__(Scenario)
    scenario.model_qualities = ["tier_0", "tier_1"]
    scenario.quality_to_index = {quality: idx for idx, quality in enumerate(scenario.model_qualities)}

    user_groups = [
        {
            "name": "group_a",
            "slo_lower": {"tier_0": 1},
            "slo_upper": {"tier_0": 0, "tier_1": 1},
        }
    ]

    with pytest.raises(ValueError):
        Scenario._build_slo_matrix(scenario, user_groups, "slo_lower")
