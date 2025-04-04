import os
from datetime import timedelta
from typing import Literal, Optional

import hydra
import numpy as np
import pandas as pd
from hydra import initialize, compose

from src.forecasting import load_prophet_forecast, generate_carbon_cast_96hrs
from src.util import DT_INDEX

_DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')


class Scenario:
    def __init__(self,
                 seed: int,
                 requests_dataset: str,
                 region: str,
                 vp: int,
                 user_groups_scenario: str,
                 user_groups: list[dict],
                 model_qualities: list[str],
                 machines: list[dict]):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.requests_dataset = requests_dataset
        self.region = region

        self.user_groups_scenario = user_groups_scenario
        self.user_group_weights = [u["weight"] for u in user_groups]
        # assert sum(self.user_group_weights) == 1, "Weights of user groups must sum to 1"
        self.slo_lower = np.array([list(u["slo_lower"].values()) for u in user_groups])  # TODO sort according to model_qualities
        self.slo_upper = np.array([list(u["slo_upper"].values()) for u in user_groups])  # TODO sort according to model_qualities

        self.vp = vp
        self.C, self.C_raw = load_carbon_intensity(region)
        self.R, self.R_raw, request_scaling_factor = load_requests(requests_dataset, weights=self.user_group_weights)
        self.request_scaling_factor = request_scaling_factor

        self.machines = {i: hydra.utils.instantiate(m, request_scaling_factor=request_scaling_factor) for i, m in enumerate(machines)}

        self.U = list(range(len(user_groups)))
        self.Q = list(range(len(model_qualities)))
        self.M = list(range(len(machines)))

        self.I = list(range(len(DT_INDEX)))  # set of intervals

    @property
    def name(self) -> str:
        return f"{self.requests_dataset},{self.region},{self.user_groups_scenario},vp={self.vp}"

    @classmethod
    def from_config(cls, cfg):
        return cls(seed=cfg.seed,
                   requests_dataset=cfg.requests_dataset,
                   region=cfg.region,
                   vp=cfg.vp,
                   user_groups_scenario=cfg.user_groups["name"],
                   user_groups=cfg.user_groups["groups"],
                   model_qualities=cfg.model_qualities,
                   machines=cfg.machines)

    @classmethod
    def from_name(cls, name: str):
        requests_dataset, region, user_group_scenario, vp_str = name.split(",")
        vp = vp_str.split("=")[1]
        with initialize(version_base=None, config_path="../config"):
            cfg = compose(config_name="config", overrides=[f"requests_dataset={requests_dataset}", f"region={region}", f"user_groups={user_group_scenario}", f"vp={vp}"])
        return cls.from_config(cfg)

    @property
    def K(self) -> np.array:
        return np.array([[self.machines[m].performance[q] for m in self.M] for q in self.Q])

    def generate_R_hat(self, i: int, kind: Literal["oracle", "yhat", "yhat_lower", "yhat_upper"]) -> np.array:
        # TODO implement multiple users
        R = np.copy(self.R)
        if kind == "oracle":
            return R
        if "static" in self.requests_dataset or "normal" in self.requests_dataset:
            R.fill(1)
            return R

        mean_of_previous_years = self.R_raw["y"][:-8760].mean()
        floor = mean_of_previous_years * 0.9
        cap = mean_of_previous_years * 1.1

        fc = load_prophet_forecast(self.R_raw, i, cache_key=f"R_hat_{self.requests_dataset}_{i}", forecast_params=dict(floor=floor, cap=cap))
        fc["yhat"] = fc["yhat"] * self.request_scaling_factor
        fc["yhat_lower"] = fc["yhat_lower"] * self.request_scaling_factor
        fc["yhat_upper"] = fc["yhat_upper"] * self.request_scaling_factor

        if self.requests_dataset == "wiki_de":
            floor_hard_cap = np.quantile(self.R_raw["y"][:-8760], 0.01) * self.request_scaling_factor
            fc.loc[fc["yhat"] < floor_hard_cap, "yhat"] = floor_hard_cap
            fc.loc[fc["yhat_lower"] < floor_hard_cap, "yhat_lower"] = floor_hard_cap
            fc.loc[fc["yhat_upper"] < floor_hard_cap, "yhat_upper"] = floor_hard_cap

        R[i:] = np.expand_dims(fc[kind].values, axis=1)

        # TODO implement multiple users
        if self.user_group_weights:
            R = np.hstack([R * weight for weight in self.user_group_weights])
        return R

    def generate_C_hat(self, i: int, kind: Literal["oracle", "yhat"]) -> np.array:
        C = np.copy(self.C)
        if kind == "oracle":
            return C

        # We fix LT forecasts to t=0
        fc = load_prophet_forecast(self.C_raw, i, cache_key=f"C_hat_{self.region}_{i}", forecast_params=dict(flat=True))
        # C[i:] = fc[kind][i:] / 1000000  # TODO make cleaner. THe [t:] is only necessary because we fix t=0
        C[i:] = fc[kind] / 1000000  # TODO make cleaner. THe [t:] is only necessary because we fix t=0
        max_index = min(i + 96, len(DT_INDEX))
        C[i:max_index] = generate_carbon_cast_96hrs(self.C[i:max_index], region=self.region, rng=self.rng)
        return C


class Machine:
    def __init__(self,
                 name: str,
                 performance: dict[str, float],
                 embedded_carbon: float,
                 request_scaling_factor: float,  # numerical stability
                 power_usage: Optional[float] = None,
                 idle_power_usage: Optional[float] = None,
                 max_power_usage: Optional[list[float]] = None,
                 pue: int = 1):
        self.name = name
        self.performance = {i: v * request_scaling_factor for i, v in enumerate(performance.values())}
        self.embedded_carbon = embedded_carbon / 1000000  # gCO₂eq to tCO₂eq

        # load-independent power model
        self._power_usage = power_usage

        # load-dependent power model
        self._idle_power_usage = idle_power_usage
        self._max_power_usage = max_power_usage

        self._pue = pue

    def load_independent_power_usage(self) -> float:
        return self._power_usage

    def load_dependent_power_usage(self, q, util: float, n: float = 1) -> float:
        return self._pue * (self._idle_power_usage + (self._max_power_usage[q] - self._idle_power_usage) * util ** n)


def load_requests(dataset: str, weights: Optional[list[float]] = None) -> tuple[np.array, pd.DataFrame, float]:
    """Load request dataset.

    # TODO implement multiple users
    # TODO explain weights
    # TODO remove power!!
    """
    if "static" in dataset:  # e.g. static_e6
        if "_e" in dataset:
            power = int(dataset.split("_e")[1])
        else:
            power = 6
        R_raw = pd.read_csv(f"{_DATA_DIR}/final/wiki_en.csv", parse_dates=True)
        R_raw["y"] = 10**power
    elif "normal" in dataset:  # e.g. normal_e6
        if "_e" in dataset:
            power = int(dataset.split("_e")[1])
        else:
            power = 6
        R_raw = pd.read_csv(f"{_DATA_DIR}/final/wiki_en.csv", parse_dates=True)
        rng = np.random.default_rng(42)
        y = rng.normal(10**power, 10**power / 3, len(R_raw))
        y[y < 0] = 0
        R_raw["y"] = y
    else:
        R_raw = pd.read_csv(f"{_DATA_DIR}/final/{dataset}.csv", parse_dates=True)

    request_scaling_factor = 1 / R_raw["y"].mean()
    R = np.expand_dims(R_raw["y"].values, axis=1)[-len(DT_INDEX):] * request_scaling_factor

    if weights:
        R = np.hstack([R * weight for weight in weights])
    return R, R_raw, request_scaling_factor


def load_carbon_intensity(region: str):
    # adding artificial 2020 bc we do not yet have the data
    _2020 = pd.read_csv(f'{_DATA_DIR}/electricitymaps/{region}_2021_hourly.csv', index_col=0, parse_dates=True)
    _2020.index = _2020.index - timedelta(days=365)
    dfs = [_2020]

    for year in [2021, 2022, 2023]:
        year_df = pd.read_csv(f'{_DATA_DIR}/electricitymaps/{region}_{year}_hourly.csv', index_col=0, parse_dates=True)
        dfs.append(year_df)
    ci = pd.concat(dfs)

    raw = ci["Carbon Intensity gCO₂eq/kWh (LCA)"].reset_index()
    raw = raw.rename(columns={"Datetime (UTC)": "ds", "Carbon Intensity gCO₂eq/kWh (LCA)": "y"})
    raw["y"] = raw["y"].ffill()  # some datasets have missing values

    C = raw["y"].values[-len(DT_INDEX):] / 1000000  # gCO₂eq to tCO₂eq
    return C, raw
