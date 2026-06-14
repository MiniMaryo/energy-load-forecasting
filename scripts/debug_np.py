"""Quick NeuralProphet prediction debug (2 epochs, small sample)."""
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

m = NeuralProphet(
    n_lags=24,
    n_forecasts=FORECAST_HORIZON,
    learning_rate=0.01,
    epochs=2,
    batch_size=64,
)
m.fit(train_df, freq="h")

future = m.make_future_dataframe(train_df, periods=FORECAST_HORIZON)
forecast = m.predict(future)
last_train_date = train_df["ds"].iloc[-1]
print("last_train_date:", last_train_date)
print("forecast shape:", forecast.shape)
print("forecast tail ds:", forecast["ds"].tail(5).tolist())

matches = forecast[forecast["ds"] == last_train_date]
print("origin matches:", len(matches))
origin = matches.iloc[0]

yhat_cols = [f"yhat{i}" for i in range(1, FORECAST_HORIZON + 1)]
missing_cols = [c for c in yhat_cols if c not in forecast.columns]
print("missing yhat cols:", len(missing_cols), missing_cols[:5] if missing_cols else "none")

y_pred_origin = np.array([origin.get(f"yhat{i}", np.nan) for i in range(1, FORECAST_HORIZON + 1)])
print("origin y_pred nan count:", np.isnan(y_pred_origin).sum())
print("origin y_pred sample:", y_pred_origin[:5])

# Alternative: yhat1 on test rows
test_fc = forecast[forecast["ds"].isin(test_df["ds"])]
print("test rows in forecast:", len(test_fc))
if len(test_fc) and "yhat1" in test_fc.columns:
    y_pred_test = test_fc["yhat1"].values
    print("test yhat1 nan count:", np.isnan(y_pred_test).sum())

# Alternative: predict test_df directly
fc2 = m.predict(test_df)
print("predict(test_df) cols:", [c for c in fc2.columns if "yhat" in c][:5])
if "yhat1" in fc2.columns:
    print("predict(test_df) yhat1 nan:", fc2["yhat1"].isna().sum())
