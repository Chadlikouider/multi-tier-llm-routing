"""Unit tests for :mod:`src.scenario`'s :class:`Scenario` class."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src import scenario as scenario_module
from src.scenario import Scenario
from src.util import DT_INDEX


@pytest.fixture
def patched_data(monkeypatch):
    """Provide deterministic inputs for scenario construction."""

    raw_request_length = len(DT_INDEX) + 8760
    request_raw = pd.DataFrame(
        {
            "ds": pd.date_range(datetime(2020, 1, 1), periods=raw_request_length, freq="h"),
            "y": np.linspace(1.0, 2.0, raw_request_length),
        }
    )

    def fake_load_requests(dataset, weights=None):
        scaling_factor = 0.5
        R = np.expand_dims(request_raw["y"].values[-len(DT_INDEX):], axis=1) * scaling_factor
        if weights:
            R = np.hstack([R * weight for weight in weights])
        return R, request_raw, scaling_factor

    carbon_raw = pd.DataFrame(
        {
            "ds": pd.date_range(datetime(2020, 1, 1), periods=len(DT_INDEX), freq="h"),
            "y": np.full(len(DT_INDEX), 200.0),
        }
    )
    carbon_series = carbon_raw["y"].values / 1_000_000

    def fake_load_carbon_intensity(region):
        return carbon_series, carbon_raw

    def fake_instantiate(config, **kwargs):
        params = {k: v for k, v in config.items() if k != "_target_"}
        params.update(kwargs)
        return scenario_module.Machine(**params)

    monkeypatch.setattr(scenario_module, "load_requests", fake_load_requests)
    monkeypatch.setattr(scenario_module, "load_carbon_intensity", fake_load_carbon_intensity)
    monkeypatch.setattr(scenario_module.hydra.utils, "instantiate", fake_instantiate)


def build_scenario(requests_dataset: str = "synthetic") -> Scenario:
    return Scenario(
        seed=123,
        requests_dataset=requests_dataset,
        region="test-region",
        vp=24,
        user_groups_scenario="default",
        user_groups=[
            {
                "name": "general",
                "weight": 1.0,
                "slo_lower": {"standard": 0.2},
                "slo_upper": {"standard": 0.8},
            }
        ],
        model_qualities=["standard"],
        machines=[
            {
                "_target_": "src.scenario.Machine",
                "name": "cpu",
                "performance": {"standard": 10.0},
                "embedded_carbon": 1000.0,
                "power_usage": 1.5,
                "idle_power_usage": 1.0,
                "max_power_usage": [2.0],
            }
        ],
    )


def test_scenario_initialisation_creates_structures(patched_data):
    scenario = build_scenario()

    assert scenario.name == "synthetic,test-region,default,vp=24"
    assert scenario.slo_lower.shape == (1, 1)
    assert scenario.slo_lower[0, 0] == pytest.approx(0.2)
    assert scenario.slo_upper[0, 0] == pytest.approx(0.8)

    machine = scenario.machines[0]
    assert machine.performance == [pytest.approx(5.0)]
    assert scenario.request_scaling_factor == pytest.approx(0.5)


def test_generate_r_hat_static_dataset(patched_data):
    scenario = build_scenario(requests_dataset="static_dataset")

    forecast = scenario.generate_R_hat(0, kind="yhat")
    assert np.allclose(forecast, np.ones_like(forecast))


def test_build_slo_matrix_missing_quality_raises(patched_data):
    scenario = build_scenario()
    with pytest.raises(ValueError):
        scenario._build_slo_matrix(
            [
                {
                    "name": "incomplete",
                    "weight": 1.0,
                    "slo_lower": {},
                }
            ],
            "slo_lower",
        )
