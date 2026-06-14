"""Test NeuralProphet raw=True prediction extraction."""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "LD2011_2014.txt"

df = pd.read_csv(
    DATA_PATH, sep=";", decimal=",", parse_dates=[0], index_col=0, low_memory=False
)
df = df.loc["2012-01-01":"2012-12-31 23:59:59"]
hourly = df.resample("h").sum().sum(axis=1).asfreq("h").interpolate(method="linear").dropna()
ts_df = hourly.reset_index()
ts_df.columns = ["ds", "y"]

FORECAST_HORIZON = 168
train_df = ts_df.iloc[:-FORECAST_HORIZON].copy()
test_df = ts_df.iloc[-FORECAST_HORIZON:].copy()
y_test = test_df["y"].values

from neuralprophet import NeuralProphet

m = NeuralProphet(n_lags=24, n_forecasts=FORECAST_HORIZON, learning_rate=0.01, epochs=2, batch_size=64)
m.fit(train_df, freq="h")

# raw=True on train last row
fc_train = m.predict(train_df, raw=True)
last = fc_train.iloc[-1]
print("train raw cols:", [c for c in fc_train.columns if c.startswith("step")][:5])
steps = [f"step{i}" for i in range(1, FORECAST_HORIZON + 1)]
y_pred_train = np.array([last[c] for c in steps if c in fc_train.columns], dtype=float)
print("from train last raw: len", len(y_pred_train), "nan", np.isnan(y_pred_train).sum())
if not np.isnan(y_pred_train).all():
    print("rmse", np.sqrt(mean_squared_error(y_test[: len(y_pred_train)], y_pred_train)))

# raw=True on future dataframe
future = m.make_future_dataframe(train_df, periods=FORECAST_HORIZON)
fc_future = m.predict(future, raw=True)
last_train = train_df["ds"].iloc[-1]
origin = fc_future[fc_future["ds"] == last_train].iloc[0]
y_pred_future = np.array([origin.get(f"step{i}", np.nan) for i in range(1, FORECAST_HORIZON + 1)], dtype=float)
print("from future origin raw: nan", np.isnan(y_pred_future).sum())
if not np.isnan(y_pred_future).all():
    print("rmse", np.sqrt(mean_squared_error(y_test, y_pred_future)))

# n_forecasts=1 iterative
m1 = NeuralProphet(n_lags=24, n_forecasts=1, learning_rate=0.01, epochs=2, batch_size=64)
m1.fit(train_df, freq="h")
current = train_df.copy()
y_pred_iter = []
for _ in range(FORECAST_HORIZON):
    fc = m1.predict(current, raw=True)
    y_pred_iter.append(fc.iloc[-1]["step1"])
    next_ds = current["ds"].iloc[-1] + pd.Timedelta(hours=1)
    current = pd.concat([current, pd.DataFrame({"ds": [next_ds], "y": [y_pred_iter[-1]]})], ignore_index=True)
y_pred_iter = np.array(y_pred_iter, dtype=float)
print("iterative n_forecasts=1: nan", np.isnan(y_pred_iter).sum(), "rmse", np.sqrt(mean_squared_error(y_test, y_pred_iter)))
