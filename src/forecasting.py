import os

import numpy as np
import pandas as pd

from src.util import DT_INDEX

_CACHE_DIR = os.path.join(os.path.dirname(__file__), '../cache')
_CARBON_CAST_MAPE_96HRS = {
    # US
    'CISO': [8.08, 11.19, 12.93, 13.62],
    'ERCOT': [9.78, 10.93, 11.61, 12.23],
    'PJM': [3.69, 4.93, 5.87, 6.67],
    'NYISO': [6.91, 9.06, 9.95, 10.42],
    # Europe
    'SE': [4.29, 5.64, 6.43, 6.74],
    'DE': [7.81, 10.69, 12.8, 15.55],
    'PL': [3.12, 4.14, 4.72, 5.50],
    'ES': [10.12, 16.00, 19.37, 21.12],
    'NL': [6.06, 7.87, 9.08, 9.99],
    # Australia
    'AU-QLD': [3.93, 3.98, 4.06, 5.87],
}


def load_prophet_forecast(data: pd.DataFrame, i: int, forecast_params: dict, cache_key: str) -> pd.DataFrame:
    """Loads prophet forecast from cache if available, otherwise generates it and saves it to cache."""
    path = f"{_CACHE_DIR}/{cache_key}.pkl"
    try:
        # Load forecast from cache if available
        fc = pd.read_pickle(path)
    except FileNotFoundError:
        print(f"Generating forecast for {cache_key}...")
        fc, _ = generate_prophet_forecast(data, i, forecast_params)
        fc = fc[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        # Save forecast to cache
        os.makedirs(_CACHE_DIR, exist_ok=True)
        fc.to_pickle(path)
    return fc


def generate_prophet_forecast(data: pd.DataFrame, i: int, forecast_params: dict, prophet_params: dict = {}) -> tuple[pd.DataFrame, "Prophet"]:
    from prophet import Prophet

    cutoff = -len(DT_INDEX) + i
    train = data[:cutoff].copy()

    # Fitting Prophet
    if "cap" in forecast_params:
        train['cap'] = forecast_params["cap"]
        train['floor'] = forecast_params["floor"]
        model = Prophet(growth='logistic', **prophet_params)
    elif "flat" in forecast_params:
        model = Prophet(growth='flat', **prophet_params)
    else:
        model = Prophet()
    model.fit(train)

    # Predicting
    future = model.make_future_dataframe(freq="h", periods=24 * 365)
    if "cap" in forecast_params:
        future['cap'] = forecast_params["cap"]
        future['floor'] = forecast_params["floor"]
    forecast = model.predict(future)

    return forecast[cutoff:], model


def generate_carbon_cast_96hrs(C: np.array, region: str, rng: np.random.Generator):
    mape_list = _CARBON_CAST_MAPE_96HRS[region]
    noise_daily = [rng.normal(loc=0, scale=_compute_sd_gauss(mape) / 100, size=24) for mape in mape_list]
    noise_96hrs = np.concatenate(np.array(noise_daily))
    return C * (1 + noise_96hrs[:len(C)])


def _compute_sd_gauss(mape: float):
    return mape * np.sqrt(np.pi / 2)
