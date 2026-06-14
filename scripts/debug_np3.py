"""Test NeuralProphet evaluation strategies."""
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

m = NeuralProphet(n_lags=24, n_forecasts=1, learning_rate=0.01, epochs=2, batch_size=64)
m.fit(train_df, freq="h")

# Strategy A: concat train + future NaN y
future = test_df[["ds"]].copy()
future["y"] = np.nan
extended = pd.concat([train_df, future], ignore_index=True)
forecast = m.predict(extended)
test_fc = forecast.iloc[-FORECAST_HORIZON:]
y_pred_a = test_fc["yhat1"].to_numpy(dtype=float)
print("Strategy A (concat NaN y): nan", np.isnan(y_pred_a).sum(), "rmse", np.sqrt(mean_squared_error(y_test, y_pred_a)))

# Strategy B: iterative 1-step
current = train_df.copy()
y_pred_b = []
for step in range(FORECAST_HORIZON):
    fc = m.predict(current)
    pred = fc.iloc[-1]["yhat1"]
    y_pred_b.append(pred)
    next_ds = current["ds"].iloc[-1] + pd.Timedelta(hours=1)
    current = pd.concat(
        [current, pd.DataFrame({"ds": [next_ds], "y": [pred]})], ignore_index=True
    )
y_pred_b = np.array(y_pred_b, dtype=float)
print("Strategy B (iterative): nan", np.isnan(y_pred_b).sum(), "rmse", np.sqrt(mean_squared_error(y_test, y_pred_b)))

# Strategy C: n_forecasts=168, extract from test rows yhat1
m2 = NeuralProphet(n_lags=24, n_forecasts=FORECAST_HORIZON, learning_rate=0.01, epochs=2, batch_size=64)
m2.fit(train_df, freq="h")
future2 = m2.make_future_dataframe(train_df, periods=FORECAST_HORIZON)
fc2 = m2.predict(future2)
test_fc2 = fc2[fc2["ds"].isin(test_df["ds"])]
print("Strategy C test yhat1 nan", test_fc2["yhat1"].isna().sum())
# try yhat step from origin on test rows
last_train = train_df["ds"].iloc[-1]
y_pred_c = []
for _, row in test_fc2.iterrows():
    step = int((row["ds"] - last_train) / pd.Timedelta(hours=1))
    col = f"yhat{step}"
    y_pred_c.append(row[col] if col in fc2.columns else np.nan)
y_pred_c = np.array(y_pred_c, dtype=float)
print("Strategy C (test row yhat step): nan", np.isnan(y_pred_c).sum())

# Strategy D: predict train, take last row yhat1..168
fc_train = m2.predict(train_df)
last = fc_train.iloc[-1]
y_pred_d = np.array([last[f"yhat{i}"] for i in range(1, FORECAST_HORIZON + 1)], dtype=float)
print("Strategy D (train last row yhat1..168): nan", np.isnan(y_pred_d).sum(), "sample", y_pred_d[:3])
