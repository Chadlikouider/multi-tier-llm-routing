import logging

import hydra
from omegaconf import DictConfig

from src.result import result_exists, get_result_name
from src.scenario import Scenario

# hotfix for Prophet: https://github.com/facebook/prophet/issues/2595
import numpy as np

np.float_ = np.float64

logger = logging.getLogger("pulp")
logger.propagate = False


@hydra.main(version_base=None, config_path="./config", config_name="config")
def app(cfg: DictConfig):
    scenario = Scenario.from_config(cfg)
    optimizer = hydra.utils.instantiate(cfg.optimizer)

    result_name = get_result_name(scenario, optimizer, info={"qor_target": cfg.qor_target})
    if result_exists(result_name, result_dir=cfg.result_dir):
        print(f"Skipping existing result {result_name}")
    else:
        print(f"Running {result_name}...", flush=True)
        result = optimizer.minimize_emissions(scenario, qor_target=cfg.qor_target)
        result.print_stats()
        result.save(result_dir=cfg.result_dir)


if __name__ == "__main__":
    app()
