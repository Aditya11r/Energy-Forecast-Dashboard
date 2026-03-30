import streamlit as st
import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

from utils import load_and_preprocess, create_sequences

# ---------------------------
# FUTURE FORECAST FUNCTION
# ---------------------------
def forecast_future(model, last_window, steps, target_scaler):
    preds = []
    current_window = last_window.copy()

    for _ in range(steps):
        pred = model.predict(current_window, verbose=0)[0][0]
        preds.append(pred)

        # copy last row
        next_row = current_window[0, -1, :].copy()

        # update ONLY target (AEP_MW)
        next_row[0] = pred

        # shift window
        current_window = np.append(
            current_window[:, 1:, :],
            [[next_row]],
            axis=1
        )

    return target_scaler.inverse_transform(
        np.array(preds).reshape(-1, 1)
    )

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Energy Forecast", layout="wide")
st.title("⚡ Energy Forecast Dashboard")

# ---------------------------
# LOAD DATA
# ---------------------------
data = load_and_preprocess("data/AEP_hourly.csv")
values = data.values

st.write("### Dataset Preview", data.tail())

# ---------------------------
# LOAD BEST MODEL FROM MLFLOW
# ---------------------------
mlflow.set_tracking_uri("http://127.0.0.1:5000")

exp = mlflow.get_experiment_by_name("AEP_Energy_Forecasting")
runs = mlflow.search_runs([exp.experiment_id])

if runs.empty:
    st.error("❌ No MLflow runs found. Run train.py first.")
    st.stop()

best_run = runs.sort_values("metrics.test_mae").iloc[0]
model = mlflow.keras.load_model(f"runs:/{best_run.run_id}/model")

st.success(f"✅ Best Model: {best_run['tags.mlflow.runName']}")

# ---------------------------
# SCALING
# ---------------------------
feature_scaler = MinMaxScaler()
target_scaler  = MinMaxScaler()

X_scaled = feature_scaler.fit_transform(values)
y_scaled = target_scaler.fit_transform(values[:, 0].reshape(-1, 1))

WINDOW_SIZE = 30

X, y = create_sequences(X_scaled, y_scaled, WINDOW_SIZE)

# ---------------------------
# TEST PREDICTIONS (EVALUATION)
# ---------------------------
pred_scaled = model.predict(X, verbose=0)

pred = target_scaler.inverse_transform(pred_scaled)
actual = target_scaler.inverse_transform(y)

# ---------------------------
# EXTRA EVALUATION METRICS
# ---------------------------
y_true = actual.flatten()
y_pred = pred.flatten()

residuals = y_true - y_pred

# Core metrics
mae   = np.mean(np.abs(residuals))
rmse  = np.sqrt(np.mean(residuals**2))

mape = np.mean(
    np.abs((y_true - y_pred) / np.clip(y_true, 1, None))
) * 100

smape = 100 * np.mean(
    2 * np.abs(y_pred - y_true) /
    (np.abs(y_true) + np.abs(y_pred) + 1e-8)
)

# Directional accuracy
actual_dir = (y_true[1:] > y_true[:-1]).astype(int)
pred_dir   = (y_pred[1:] > y_pred[:-1]).astype(int)

dir_acc = np.mean(actual_dir == pred_dir)

# Bias
bias = np.mean(residuals)
# ---------------------------
# NEXT DAY FORECAST
# ---------------------------
last_window = X_scaled[-WINDOW_SIZE:]
last_window = last_window.reshape(1, WINDOW_SIZE, -1)

next_day_scaled = model.predict(last_window, verbose=0)
next_day = target_scaler.inverse_transform(next_day_scaled)

st.subheader("🔮 Next Day Prediction")

next_date = data.index[-1] + pd.Timedelta(days=1)

st.metric(
    label=f"Predicted Energy for {next_date.date()}",
    value=f"{next_day[0][0]:,.2f}"
)

# ---------------------------
# FUTURE FORECAST
# ---------------------------
st.subheader("📈 Future Forecast")

future_days = st.slider("Forecast future days", 1, 120, 6)

future_preds = forecast_future(
    model,
    last_window,
    future_days,
    target_scaler
)

# Create future dates
last_date = data.index[-1]
future_dates = pd.date_range(
    start=last_date,
    periods=future_days + 1,
    freq='D'
)[1:]

future_df = pd.DataFrame(
    future_preds,
    index=future_dates,
    columns=["Forecast"]
)

st.line_chart(future_df)

# ---------------------------
# ACTUAL vs PREDICTED GRAPH
# ---------------------------
st.subheader("📊 Model Performance")

days = st.slider("Select Days to View", 30, 200, 100)

dates = data.index[WINDOW_SIZE:]  # align with sequences

plot_dates = dates[-days:]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=plot_dates,
    y=actual[-days:].flatten(),
    mode='lines',
    name='Actual'
))

fig.add_trace(go.Scatter(
    x=plot_dates,
    y=pred[-days:].flatten(),
    mode='lines',
    name='Predicted'
))

fig.update_layout(
    template="plotly_dark",
    title="Actual vs Predicted",
    xaxis_title="Date",
    yaxis_title="Energy (MW)",
    xaxis=dict(
        tickformat="%Y-%m-%d",   # 👈 shows YEAR too
        dtick="M1"
    )
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# METRICS
# ---------------------------
mae = np.mean(np.abs(actual - pred))

st.metric("📉 MAE (Overall)", f"{mae:.2f} MW")

st.subheader("📊 Model Evaluation")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", f"{mae:.2f} MW")
col2.metric("RMSE", f"{rmse:.2f} MW")
col3.metric("MAPE", f"{mape:.2f} %")

col4, col5, col6 = st.columns(3)

col4.metric("SMAPE", f"{smape:.2f} %")
col5.metric("Direction Accuracy", f"{dir_acc:.2%}")
col6.metric("Bias", f"{bias:.2f} MW")

