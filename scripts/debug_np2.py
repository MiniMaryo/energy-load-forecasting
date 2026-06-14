"""NeuralProphet forecast column debug."""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

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

from neuralprophet import NeuralProphet

for n_forecasts in [1, 24, 168]:
    print(f"\n=== n_forecasts={n_forecasts} ===")
    m = NeuralProphet(n_lags=24, n_forecasts=n_forecasts, learning_rate=0.01, epochs=2, batch_size=64)
    m.fit(train_df, freq="h")
    future = m.make_future_dataframe(train_df, periods=FORECAST_HORIZON)
    print("future shape:", future.shape)
    forecast = m.predict(future)
    print("forecast shape:", forecast.shape)
    last_train_date = train_df["ds"].iloc[-1]
    origin = forecast[forecast["ds"] == last_train_date].iloc[0]
    cols = [f"yhat{i}" for i in range(1, min(n_forecasts, FORECAST_HORIZON) + 1)]
    vals = [origin[c] for c in cols if c in forecast.columns]
    print("origin non-nan yhat:", sum(not np.isnan(v) for v in vals), "/", len(vals))
    if vals:
        print("first vals:", vals[:3])

    test_fc = forecast[forecast["ds"].isin(test_df["ds"])]
    if len(test_fc) and "yhat1" in test_fc.columns:
        y1 = test_fc["yhat1"].values
        print("test yhat1 non-nan:", np.sum(~np.isnan(y1)), "/", len(y1))

    # try full train predict tail
    fc_train = m.predict(train_df)
    tail = fc_train.iloc[-1]
    if "yhat1" in fc_train.columns:
        non_nan = sum(not np.isnan(tail[f"yhat{i}"]) for i in range(1, min(n_forecasts, 10) + 1) if f"yhat{i}" in fc_train.columns)
        print("train tail yhat1..10 non-nan:", non_nan)
