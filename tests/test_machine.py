"""Unit tests for the :class:`Machine` helper class."""

import pytest

from src.scenario import Machine


def test_machine_normalises_performance_with_quality_mapping():
    machine = Machine(
        name="accelerator",
        performance={"high": 20.0, "medium": 10.0},
        embedded_carbon=5_000.0,
        request_scaling_factor=0.1,
        quality_to_index={"medium": 0, "high": 1},
        quality_count=2,
        idle_power_usage=1.0,
        max_power_usage=[1.5, 2.5],
    )

    assert machine.performance == [pytest.approx(1.0), pytest.approx(2.0)]
    assert machine.embedded_carbon == pytest.approx(0.005)


def test_machine_load_dependent_power_usage():
    machine = Machine(
        name="gpu",
        performance={"standard": 5.0},
        embedded_carbon=1_000.0,
        request_scaling_factor=1.0,
        idle_power_usage=1.0,
        max_power_usage=[3.0],
        pue=2,
    )

    power = machine.load_dependent_power_usage(0, util=0.5)
    expected = 2 * (1.0 + (3.0 - 1.0) * 0.5)
    assert power == pytest.approx(expected)


def test_machine_unknown_quality_raises():
    with pytest.raises(ValueError):
        Machine(
            name="cpu",
            performance={"unknown": 1.0},
            embedded_carbon=1_000.0,
            request_scaling_factor=1.0,
            quality_to_index={"standard": 0},
            quality_count=1,
        )
