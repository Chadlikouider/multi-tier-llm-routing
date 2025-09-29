"""Scenario definitions and data loading helpers.

This module contains lightweight container classes that bundle together the
various data inputs used by the optimisation layer of the project.  The goal of
the refactor is to make the construction of scenarios easier to follow and to
document the data manipulation that happens along the way so future changes can
be made with confidence.
"""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Dict, Iterable, Literal, Optional

import hydra
import numpy as np
import pandas as pd
from hydra import compose, initialize

from src.forecasting import generate_carbon_cast_96hrs, load_prophet_forecast
from src.util import DT_INDEX

# ``_DATA_DIR`` is resolved relative to the repository so the module can be
# imported from both application code and tests without having to rely on the
# current working directory.
_DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")


class Scenario:
    """Container describing a full optimisation scenario.

    Parameters are passed directly from the Hydra configuration files used in
    the experiments.  During initialisation we load the request and carbon
    intensity data, normalise them for numerical stability, and materialise the
    machine definitions.
    """

    def __init__(
        self,
        seed: int,
        requests_dataset: str,
        region: str,
        vp: int,
        user_groups_scenario: str,
        user_groups: list[dict],
        model_qualities: list[str],
        machines: list[dict],
    ) -> None:
        # Random numbers are used for synthetic carbon intensity data.  We keep
        # a dedicated generator so repeated runs with the same configuration are
        # reproducible.
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Configuration identifiers used when serialising results.
        self.requests_dataset = requests_dataset
        self.region = region
        self.user_groups_scenario = user_groups_scenario
        self.vp = vp

        # User group metadata is re-shaped into numpy friendly structures.  The
        # weights are used to scale demand data per group, while the QoR targets
        # are mapped to matrices for fast lookups during optimisation.
        self.user_group_weights = [u["weight"] for u in user_groups]
        self.model_qualities = model_qualities
        self.quality_to_index = {quality: idx for idx, quality in enumerate(model_qualities)}

        self.slo_lower = self._build_slo_matrix(user_groups, "slo_lower")
        self.slo_upper = self._build_slo_matrix(user_groups, "slo_upper")

        # Load the time series backing the scenario.  ``request_scaling_factor``
        # keeps numerical values in a narrow range to improve solver stability
        # and is also passed to the machine definitions so their throughput is
        # scaled consistently.
        self.C, self.C_raw = load_carbon_intensity(region)
        self.R, self.R_raw, request_scaling_factor = load_requests(
            requests_dataset, weights=self.user_group_weights
        )
        self.request_scaling_factor = request_scaling_factor

        self.machines = self._instantiate_machines(machines, request_scaling_factor)

        # Convenience sets used by the optimisation models to iterate over the
        # scenario entities.
        self.U = list(range(len(user_groups)))
        self.Q = list(range(len(model_qualities)))
        self.M = list(range(len(machines)))
        self.I = list(range(len(DT_INDEX)))  # set of intervals

    @property
    def name(self) -> str:
        """Return a descriptive identifier used in file names and logs."""

        return f"{self.requests_dataset},{self.region},{self.user_groups_scenario},vp={self.vp}"

    def _instantiate_machines(
        self, machines: Iterable[Dict], request_scaling_factor: float
    ) -> dict[int, "Machine"]:
        """Instantiate machines defined in the configuration.

        Hydra allows us to specify different machine classes in configuration
        files; we pass the bookkeeping parameters required by :class:`Machine`
        so that custom classes can consume them as well.
        """

        return {
            i: hydra.utils.instantiate(
                machine_cfg,
                request_scaling_factor=request_scaling_factor,
                quality_to_index=self.quality_to_index,
                quality_count=len(self.model_qualities),
            )
            for i, machine_cfg in enumerate(machines)
        }

    @classmethod
    def from_config(cls, cfg):
        """Create a scenario directly from a Hydra configuration object."""

        return cls(
            seed=cfg.seed,
            requests_dataset=cfg.requests_dataset,
            region=cfg.region,
            vp=cfg.vp,
            user_groups_scenario=cfg.user_groups["name"],
            user_groups=cfg.user_groups["groups"],
            model_qualities=cfg.model_qualities,
            machines=cfg.machines,
        )

    @classmethod
    def from_name(cls, name: str):
        """Load a scenario by parsing the string returned from :attr:`name`."""

        requests_dataset, region, user_group_scenario, vp_str = name.split(",")
        vp = vp_str.split("=")[1]
        with initialize(version_base=None, config_path="../config"):
            cfg = compose(
                config_name="config",
                overrides=[
                    f"requests_dataset={requests_dataset}",
                    f"region={region}",
                    f"user_groups={user_group_scenario}",
                    f"vp={vp}",
                ],
            )
        return cls.from_config(cfg)

    @property
    def K(self) -> np.array:
        """Return a matrix of machine throughput indexed by quality and machine."""

        return np.array([[self.machines[m].performance[q] for m in self.M] for q in self.Q])

    def _build_slo_matrix(self, user_groups: list[dict], key: str) -> np.ndarray:
        """Convert QoR constraints per user group into a matrix.

        Parameters
        ----------
        user_groups:
            The raw configuration objects containing the QoR definitions.
        key:
            Either ``"slo_lower"`` or ``"slo_upper"`` specifying which bounds to
            extract.
        """

        matrix = np.zeros((len(user_groups), len(self.model_qualities)))
        expected_keys = set(self.model_qualities)
        for user_idx, user_group in enumerate(user_groups):
            slo_values = user_group.get(key, {})
            missing = expected_keys - set(slo_values)
            if missing:
                missing_str = ", ".join(sorted(missing))
                raise ValueError(
                    f"User group '{user_group.get('name', user_idx)}' is missing QoR targets for: {missing_str}"
                )
            for quality, value in slo_values.items():
                if quality not in self.quality_to_index:
                    raise ValueError(
                        f"Unknown model quality '{quality}' referenced in {key} for user group "
                        f"'{user_group.get('name', user_idx)}'"
                    )
                matrix[user_idx, self.quality_to_index[quality]] = value
        return matrix

    def generate_R_hat(self, i: int, kind: Literal["oracle", "yhat", "yhat_lower", "yhat_upper"]) -> np.array:
        """Generate a demand forecast starting from interval ``i``."""

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
        """Return a carbon intensity outlook starting at step ``i``."""

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
    """Description of a deployable hardware configuration."""

    def __init__(
        self,
        name: str,
        performance: dict[str, float],
        embedded_carbon: float,
        request_scaling_factor: float,  # numerical stability
        power_usage: Optional[float] = None,
        idle_power_usage: Optional[float] = None,
        max_power_usage: Optional[list[float]] = None,
        pue: int = 1,
        quality_to_index: Optional[dict[str, int]] = None,
        quality_count: Optional[int] = None,
    ) -> None:
        # Throughput is normalised so all optimisation inputs are scaled the
        # same way.  When ``quality_to_index`` is provided we produce a list that
        # can be indexed directly by a quality integer.
        self.name = name
        self.performance = self._normalize_performance(
            name,
            performance,
            request_scaling_factor,
            quality_to_index,
            quality_count,
        )
        # Convert to tonnes of CO₂ equivalent to align with other parts of the
        # model which operate on that unit.
        self.embedded_carbon = embedded_carbon / 1000000  # gCO₂eq to tCO₂eq

        # Power models can either be load-independent or load-dependent; both
        # representations are supported simultaneously and consumers can choose
        # which accessors to use.
        self._power_usage = power_usage
        self._idle_power_usage = idle_power_usage
        self._max_power_usage = max_power_usage
        self._pue = pue

    def load_independent_power_usage(self) -> float:
        """Return the constant power draw when the simple model is used."""

        return self._power_usage

    def load_dependent_power_usage(self, q, util: float, n: float = 1) -> float:
        """Return the dynamic power draw for a given utilisation.

        Parameters
        ----------
        q:
            Index of the model quality to look up in ``max_power_usage``.
        util:
            Current utilisation ratio, typically between 0 and 1.
        n:
            Exponent applied to the utilisation value.  A value of ``1`` keeps
            the relationship linear while larger values bias the curve towards
            high utilisation.
        """

        return self._pue * (self._idle_power_usage + (self._max_power_usage[q] - self._idle_power_usage) * util ** n)

    @staticmethod
    def _normalize_performance(machine_name: str,
                                performance: dict[str, float],
                                request_scaling_factor: float,
                                quality_to_index: Optional[dict[str, int]],
                                quality_count: Optional[int]) -> list[float]:
        """Map performance dictionaries to consistent list representations."""

        if quality_to_index is None or quality_count is None:
            return [v * request_scaling_factor for v in performance.values()]

        normalized = [0.0] * quality_count
        for quality, throughput in performance.items():
            if quality not in quality_to_index:
                raise ValueError(
                    f"Performance specified for unknown quality '{quality}' on machine '{machine_name}'"
                )
            normalized[quality_to_index[quality]] = throughput * request_scaling_factor
        return normalized


def load_requests(dataset: str, weights: Optional[list[float]] = None) -> tuple[np.ndarray, pd.DataFrame, float]:
    """Load the request dataset and normalise it for numerical stability."""

    # TODO implement multiple users
    # TODO explain weights
    # TODO remove power!!
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


def load_carbon_intensity(region: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Load carbon intensity data for ``region`` and return scaled values."""

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
