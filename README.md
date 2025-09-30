# Multi-Tier LLM Routing

Multi-Tier LLM Routing provides an experimentation harness for studying how multi-quality
large language model (LLM) services can dynamically steer traffic across quality tiers while
respecting carbon, energy, or budget constraints. The project combines demand and carbon
intensity time series, machine performance profiles, and service-level objectives (SLOs) to
solve optimization problems that trade off emissions against quality of result (QoR).

The repository contains:

* **Scenario building blocks** that load datasets, normalize demand, and materialize fleets of
  accelerators with tier-dependent throughput.
* **Optimization models** that find the lowest-emission plan satisfying QoR targets or the
  highest QoR achievable within a fixed carbon budget.
* **Rolling-horizon controllers** and **baseline heuristics** for online execution studies.
* **Forecasting utilities** for demand and carbon intensity using Prophet-based models.

The codebase is designed for reproducible experiments, configuration via
[Hydra](https://hydra.cc/), and integration with custom datasets.

## Table of contents

- [Repository layout](#repository-layout)
- [Core concepts](#core-concepts)
- [Installation](#installation)
- [Running optimizations](#running-optimizations)
- [Configuration](#configuration)
- [Datasets](#datasets)
- [Forecast caching](#forecast-caching)
- [Baselines](#baselines)
- [Development](#development)
- [Citation](#citation)

## Repository layout

```
.
├── config/                # Hydra configuration tree (defaults, optimizers, user groups)
├── src/                   # Scenario, optimization, forecasting, and utility modules
├── main.py                # Entry point for batch experiments
├── automatic_qor.py       # Script for exploring QoR target selection
├── tests/                 # Pytest suite covering machines, scenarios, and utilities
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation (this file)
```

Key modules:

- `src/scenario.py` constructs `Scenario` objects by loading demand traces, carbon intensity
  data, and machine definitions. It also exposes helpers for generating Prophet forecasts and
  parsing serialized scenario names.
- `src/qt_model.py` defines the core PuLP optimization model used by both offline and online
  planners.
- `src/optimizer.py` wraps the model with high-level strategies: `Qt` for full-horizon solves and
  `QtOnline` for rolling-horizon execution with forecast refreshes.
- `src/baselines.py` hosts heuristic planners such as the `Greedy` allocator for comparison
  against the optimization-based approaches.
- `src/result.py` and `src/util.py` provide serialization, bookkeeping, and helper routines.

## Core concepts

The system models a multi-tier LLM service where each user group has QoR bounds it must respect
within a *validity period* (e.g., 24 hours). A scenario ties together:

- **Demand (`R`)**: per-tier request volumes derived from historical traces.
- **Carbon intensity (`C`)**: time series capturing regional grid intensity.
- **Machines**: accelerator types with tier-specific throughput, power usage, and embodied carbon.
- **User groups**: weights and QoR SLOs that define fairness across the demand mix.

Optimization tasks either minimize emissions subject to a QoR target or maximize QoR subject to a
budget. Results record the per-interval tier assignments (`a_`), machine activations (`d_`), and
aggregate metrics.

## Installation

1. Ensure Python 3.10+ is available. Prophet requires a working C++ toolchain (`gcc`, `g++`,
   `make`) on Linux/macOS.
2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Running optimizations

Hydra drives all experiment configuration. To run the default full-horizon optimization that
minimizes emissions for a QoR target of 0.5:

```bash
python main.py
```

Overriding configuration values is as simple as appending `key=value` pairs. For example, to run
the online optimizer with a 24 hour short-term horizon, multiple validity periods, and a custom
result directory:

```bash
python main.py optimizer=qt_online \
               optimizer.horizon=24 \
               optimizer.oracle=false \
               vp=1,24,168 \
               requests_dataset=wiki_en,wiki_de \
               qor_target=0.6 \
               result_dir=runs/online
```

Each combination of overrides spawns a job. Completed results are cached under `result_dir` and
skipped automatically if rerun.

`Result` objects provide convenience methods to inspect QoR, emissions, and solver statistics.
They can be reloaded via `Result.load(...)` or parsed by custom analysis scripts.

## Configuration

The base configuration lives in `config/config.yaml`. Important fields include:

- `seed`: random seed for reproducible sampling.
- `requests_dataset`: name of the demand CSV to load (under `data/final/`).
- `region`: ElectricityMaps region code for carbon intensity files
  (`data/electricitymaps/{REGION}_{YEAR}_hourly.csv`).
- `vp`: validity period length in hours.
- `model_qualities`: ordered list of QoR tiers; used to map throughput curves.
- `machines`: Hydra-instantiated accelerator definitions (`src.scenario.Machine` by default).
- `user_groups`: nested configuration defining group weights and QoR bounds.
- `optimizer`: choice of strategy (`qt`, `qt_online`, or a baseline from `config/optimizer/`).
- `result_dir`: destination for serialized optimization outputs.

Optimizer-specific options (e.g., rolling-horizon length, callback hooks, relaxation toggles) live
in `config/optimizer/*.yaml`. User mixes reside in `config/user_groups/` to make scenario
composition reusable.

## Datasets

The repository expects datasets to be placed under `data/` (not version-controlled):

- Demand traces: `data/final/<dataset>.csv` with a column `y` representing hourly aggregated
  requests.
- Carbon intensity: `data/electricitymaps/{REGION}_{YEAR}_hourly.csv` following ElectricityMaps
  naming conventions.

Dataset loaders normalize demand for numerical stability and align carbon intensity across multiple
years. If files are missing a `FileNotFoundError` is raised with the expected path.

## Forecast caching

Short-term planners use Prophet forecasts for both demand and carbon intensity. Generated
forecasts are cached in `cache/` so repeated experiments reuse serialized predictions. Delete the
folder to regenerate, or share it across environments for faster cold starts.

## Baselines

`src/baselines.py` contains heuristics for comparison studies. The `Greedy` allocator distributes a
budget across intervals using constant or weighted strategies and falls back to simple execution
helpers if the solver becomes infeasible. Instantiate it via Hydra
(`optimizer=greedy mode=weighted lt_frequency=24`) to include it in sweeps.

## Development

The project uses `pytest` for unit tests and scenario validation. Run the suite after installing
dependencies:

```bash
pytest
```

Pull requests should include tests that cover new behaviors, especially around scenario loading,
forecasting, and optimization invariants.

