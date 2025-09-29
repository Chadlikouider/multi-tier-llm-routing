# Quality Time

Quality Time provides the reference implementation that accompanies the paper
["Carbon-Aware Quality Adaptation for Energy-Intensive Services"](https://arxiv.org/pdf/2411.19058).
The repository contains optimization models, simulators, and experiment harnesses for
studying how large-scale AI services can dynamically adapt quality targets to reduce
carbon emissions while respecting service-level objectives.

## Table of contents
- [Project structure](#project-structure)
- [Installation](#installation)
- [Running experiments](#running-experiments)
- [Configuration reference](#configuration-reference)
- [Data requirements](#data-requirements)
- [Forecast caching](#forecast-caching)
- [Development and testing](#development-and-testing)
- [Citing this work](#citing-this-work)

## Project structure

```
.
├── config/                # Hydra configuration tree (base config, optimizers, user groups)
├── src/                   # Core implementation (scenario, optimizers, forecasting, utilities)
├── main.py                # Entry point for carbon-aware optimization experiments
├── automatic_qor.py       # Experimental script for automatic QoR target selection
├── tests/                 # Pytest suite covering scenario and validity-period logic
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation (this file)
```

Key components:
- `src/scenario.py` defines the `Scenario` class, loading demand traces, carbon-intensity data,
  and machine configurations from Hydra configs and CSV datasets.
- `src/optimizer.py` implements both the global (`Qt`) and online (`QtOnline`) optimization
  models, which interact with the scenario to minimize emissions or maximize quality of result.
- `config/config.yaml` captures the default experiment setup, including regions, datasets,
  quality tiers, and machine definitions. Additional optimizer variants live in
  `config/optimizer/*.yaml`, and user population mixes are configured in `config/user_groups/`.

## Installation

1. Ensure you have Python 3.10+ installed. Prophet depends on a working C++ toolchain
   (for pystan); on Linux this typically means having `gcc`, `g++`, and `make` available.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Upgrade `pip` and install the dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Running experiments

Quality Time uses [Hydra](https://hydra.cc/) for configuration management. The main entry point
executes a carbon-aware optimization for the scenario described in the active config tree:

```bash
python main.py
```

Hydra makes it easy to override configuration values from the command line. For example, the
following command evaluates the online optimizer across several validity periods and datasets:

```bash
python main.py optimizer=qt_online \
               region=CISO \
               vp=1,24,168 \
               requests_dataset=wiki_en,wiki_de \
               qor_target=0.5 \
               result_dir=final_results
```

The script will execute one job per Cartesian product of the overrides, saving serialized
`Result` objects (including aggregated metrics) under the configured `result_dir`.
If a result already exists, it is skipped automatically using `src.result.result_exists()`.

Each optimizer exposes both emission minimization and quality maximization interfaces:
- `Qt.minimize_emissions` solves a global optimization over the full horizon.
- `QtOnline.minimize_emissions` performs rolling short-term optimizations using the
  latest demand and carbon intensity data, optionally relaxing integer constraints for
  the active horizon.

To explore additional scenarios you can override the seed, region, machines, QoR targets,
validity periods, or user group mixes directly from the command line or by editing the
YAML files in `config/`.

## Configuration reference

The top-level configuration (`config/config.yaml`) specifies:
- `seed`: random seed used for reproducible sampling and forecast perturbations.
- `requests_dataset`: name of the demand trace CSV to load (see [Data requirements](#data-requirements)).
- `region`: ElectricityMaps region code driving carbon-intensity data.
- `vp`: validity-period duration in hours.
- `model_qualities`: ordered list of quality tiers; these map to throughput values in each
  machine definition.
- `machines`: Hydra-instantiated `Machine` objects with power usage, PUE, and performance curves.
- `user_groups`: nested configuration defining user segments, weights, and service-level
  objectives (`slo_lower` / `slo_upper`) for each quality tier.
- `optimizer`: the optimization strategy to instantiate (e.g., `qt`, `qt_online`, or `greedy`).
- `result_dir`: output directory for serialized experiment results.

Optimizer-specific knobs (horizon, callbacks, relaxation flags, etc.) live under
`config/optimizer/`. User population scenarios are defined in `config/user_groups/`, allowing
experiments with different service mixes (e.g., percentile-based guarantees).

## Data requirements

The repository expects demand and carbon-intensity datasets to be stored under `data/` relative to
this project:
- Request traces live in `data/final/<dataset>.csv` and must contain hourly demand values in a
  column named `y`. Scenario loading scales the trace to per-request units based on the mean
  (`Scenario.load_requests`).
- Carbon-intensity histories are read from `data/electricitymaps/{REGION}_{YEAR}_hourly.csv`, using
  ElectricityMaps naming conventions (`Scenario.load_carbon_intensity`). The loader automatically
  synthesizes a 2020 trace by shifting 2021 data to provide a full multi-year history.

These datasets are not included in the public repository due to licensing constraints. Please
contact the authors if you require access, or replace the CSVs with your own data following the
same schema. When data files are absent the loaders will raise `FileNotFoundError`.

## Forecast caching

Short-term forecasts rely on Facebook Prophet. Generated forecasts are cached under
`cache/` to speed up repeated runs (`src.forecasting.load_prophet_forecast`). You can delete
this directory to force regeneration, or pre-populate it to share cached forecasts across runs.

## Development and testing

The project includes a small pytest suite that validates scenario construction and validity-period
helpers:

```bash
pytest
```

This is the recommended way to confirm that the environment and dependencies are installed
correctly before running larger experiments. When adding new functionality, consider extending the
tests in `tests/` to cover additional edge cases.

## Citing this work

If you use this codebase in academic or industrial research, please cite the accompanying paper:

> Gitansh Khirbat, Hritik Bansal, Sanket Patil, Marcel Pfeiffer, Siddharth Garg, Nati Srebro, and
> David Culler. "Carbon-Aware Quality Adaptation for Energy-Intensive Services." arXiv:2411.19058
> (2024).

For questions about the project or requests for datasets, please reach out to the authors listed in
the paper.
