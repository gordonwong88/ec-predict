# app.py — EC-Predict MVP (Streamlit)
# Consultant-grade layout + forecasting + alerts
# Requires: streamlit pandas numpy plotly scipy statsmodels openpyxl

from __future__ import annotations

import io
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# Optional imports
HAS_STATSMODELS = False
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

HAS_SARIMAX = False
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_SARIMAX = True
except Exception:
    HAS_SARIMAX = False


# ----------------------------
# Page + styling (Tableau-like)
# ----------------------------
st.set_page_config(
    page_title="EC-Predict (MVP)",
    page_icon="📈",
    layout="wide",
)

CUSTOM_CSS = """
<style>
/* Reduce top padding */
.block-container { padding-top: 1.25rem; padding-bottom: 2rem; }

/* Card style */
.ec-card {
  background: #0b1220;
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 16px;
  padding: 14px 16px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.20);
}
.ec-card .label { color: rgba(255,255,255,0.70); font-size: 12px; }
.ec-card .value { color: white; font-size: 24px; font-weight: 700; line-height: 1.2; }
.ec-card .sub { color: rgba(255,255,255,0.70); font-size: 12px; margin-top: 6px; }

/* Section title */
.ec-title {
  font-size: 22px;
  font-weight: 750;
  margin: 0 0 4px 0;
}
.ec-subtitle {
  color: rgba(255,255,255,0.70);
  margin: 0 0 14px 0;
  font-size: 13px;
}

/* Make plots blend nicely */
.js-plotly-plot .plotly .main-svg {
  border-radius: 14px;
}

/* Sidebar headings */
section[data-testid="stSidebar"] h2 {
  font-weight: 800;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_LAYOUT_BASE = dict(
    template="plotly_dark",
    margin=dict(l=20, r=20, t=45, b=20),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial"),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

# ----------------------------
# Helpers
# ----------------------------
def read_uploaded_file(uploaded) -> pd.DataFrame:
    if uploaded is None:
        return pd.DataFrame()

    name = uploaded.name.lower()
    data = uploaded.getvalue()

    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(data))
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(data))
    raise ValueError("Unsupported file format. Please upload CSV or Excel.")

def infer_datetime_col(df: pd.DataFrame) -> Optional[str]:
    # Prefer common names first
    candidates = [c for c in df.columns if str(c).lower() in ["date", "month", "time", "asof", "as_of_date", "period"]]
    for c in candidates:
        try:
            pd.to_datetime(df[c])
            return c
        except Exception:
            continue

    # Otherwise try any column that parses well
    best = None
    best_ok = -1
    for c in df.columns:
        try:
            x = pd.to_datetime(df[c], errors="coerce")
            ok = x.notna().sum()
            if ok > best_ok and ok >= max(10, int(0.6 * len(df))):
                best_ok = ok
                best = c
        except Exception:
            pass
    return best

def infer_numeric_cols(df: pd.DataFrame) -> List[str]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return numeric

def safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def resample_series(df: pd.DataFrame, dt_col: str, y_col: str, freq: str) -> pd.DataFrame:
    tmp = df[[dt_col, y_col]].copy()
    tmp[dt_col] = safe_to_datetime(tmp[dt_col])
    tmp = tmp.dropna(subset=[dt_col, y_col])
    tmp = tmp.sort_values(dt_col)
    tmp = tmp.set_index(dt_col)

    # Aggregate for duplicates
    agg = "sum" if freq in ["D", "W"] else "sum"
    # In banking, many series are monthly sums; you can change agg to mean if needed.
    out = tmp.resample(freq).agg({y_col: agg}).reset_index()
    out.columns = ["ds", "y"]
    return out

def train_test_split_ts(ts: pd.DataFrame, test_size: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if test_size <= 0:
        return ts, ts.iloc[0:0]
    if len(ts) <= test_size + 5:
        # Too short, fallback to no split
        return ts, ts.iloc[0:0]
    return ts.iloc[:-test_size].copy(), ts.iloc[-test_size:].copy()

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > 1e-9)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred))
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (denom > 1e-9)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100)

def coverage_pct(ts: pd.DataFrame) -> float:
    # coverage = non-null y / expected points
    if len(ts) == 0:
        return 0.0
    return float(ts["y"].notna().mean() * 100)

def detect_anomalies(ts: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    s = ts.copy()
    y = s["y"].astype(float).values
    if len(y) < 12:
        s["z"] = np.nan
        s["anomaly"] = False
        return s
    mu = np.nanmean(y)
    sd = np.nanstd(y)
    if sd < 1e-9:
        s["z"] = 0.0
        s["anomaly"] = False
        return s
    z = (y - mu) / sd
    s["z"] = z
    s["anomaly"] = np.abs(z) >= z_thresh
    return s

@dataclass
class ForecastResult:
    model_name: str
    forecast_df: pd.DataFrame  # columns: ds, yhat, yhat_lower, yhat_upper
    metrics: Dict[str, float]


def fit_forecast_ets(train: pd.DataFrame, horizon: int, seasonal: str) -> ForecastResult:
    # Holt-Winters ETS
    y = train["y"].astype(float).values

    season_map = {"None": None, "Additive": "add", "Multiplicative": "mul"}
    seasonal_type = season_map.get(seasonal, None)

    # Determine seasonal_periods for monthly/weekly
    # We infer by median diff of ds
    ds = train["ds"]
    diffs = ds.diff().dropna()
    if len(diffs) == 0:
        sp = None
    else:
        median_days = np.median(diffs.dt.days)
        # Monthly approx
        if median_days >= 25:
            sp = 12
        elif median_days >= 6:
            sp = 52
        else:
            sp = None

    if not HAS_STATSMODELS:
        # Fallback naive
        last = float(train["y"].iloc[-1])
        future_dates = pd.date_range(train["ds"].iloc[-1], periods=horizon + 1, freq=pd.infer_freq(train["ds"]) or "MS")[1:]
        f = pd.DataFrame({"ds": future_dates, "yhat": last})
        f["yhat_lower"] = f["yhat"] * 0.9
        f["yhat_upper"] = f["yhat"] * 1.1
        return ForecastResult("Naive (fallback)", f, {})

    model = ExponentialSmoothing(
        y,
        trend="add",
        seasonal=seasonal_type,
        seasonal_periods=sp if seasonal_type else None,
        initialization_method="estimated",
    )
    fit = model.fit(optimized=True)
    yhat = fit.forecast(horizon)

    # Simple interval: based on residual std
    resid = y - fit.fittedvalues
    sd = np.nanstd(resid) if np.isfinite(resid).any() else 0.0
    z = 1.96
    lower = yhat - z * sd
    upper = yhat + z * sd

    future_dates = pd.date_range(train["ds"].iloc[-1], periods=horizon + 1, freq=pd.infer_freq(train["ds"]) or "MS")[1:]
    f = pd.DataFrame({"ds": future_dates, "yhat": yhat, "yhat_lower": lower, "yhat_upper": upper})
    return ForecastResult("ETS (Holt-Winters)", f, {})


def fit_forecast_sarimax(train: pd.DataFrame, horizon: int) -> ForecastResult:
    if not HAS_SARIMAX:
        return fit_forecast_ets(train, horizon, seasonal="None")

    y = train["y"].astype(float).values
    # Simple robust default order
    order = (1, 1, 1)
    seasonal_order = (0, 0, 0, 0)

    model = SARIMAX(y, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)

    pred = fit.get_forecast(steps=horizon)
    yhat = pred.predicted_mean
    ci = pred.conf_int(alpha=0.05)  # 95%
    lower = ci.iloc[:, 0].values
    upper = ci.iloc[:, 1].values

    future_dates = pd.date_range(train["ds"].iloc[-1], periods=horizon + 1, freq=pd.infer_freq(train["ds"]) or "MS")[1:]
    f = pd.DataFrame({"ds": future_dates, "yhat": yhat, "yhat_lower": lower, "yhat_upper": upper})
    return ForecastResult("SARIMAX (1,1,1)", f, {})


def backtest_model(ts: pd.DataFrame, model_choice: str, horizon: int, seasonal: str, test_size: int) -> Tuple[ForecastResult, Dict[str, float]]:
    train, test = train_test_split_ts(ts, test_size=test_size)
    if len(test) == 0:
        # No backtest possible
        if model_choice == "Auto":
            res = fit_forecast_ets(train, horizon, seasonal)
        elif model_choice == "ETS":
            res = fit_forecast_ets(train, horizon, seasonal)
        else:
            res = fit_forecast_sarimax(train, horizon)
        return res, {"MAPE": np.nan, "sMAPE": np.nan}

    # Fit on train and predict exactly test horizon
    h = len(test)
    if model_choice == "ETS":
        res = fit_forecast_ets(train, h, seasonal)
    elif model_choice == "SARIMAX":
        res = fit_forecast_sarimax(train, h)
    else:
        # Auto: choose best between ETS and SARIMAX on test
        cands = []
        r1 = fit_forecast_ets(train, h, seasonal)
        y_pred1 = r1.forecast_df["yhat"].values
        cands.append(("ETS", smape(test["y"].values, y_pred1), r1))

        r2 = fit_forecast_sarimax(train, h)
        y_pred2 = r2.forecast_df["yhat"].values
        cands.append(("SARIMAX", smape(test["y"].values, y_pred2), r2))

        best = min(cands, key=lambda x: (np.nan_to_num(x[1], nan=1e9)))
        res = best[2]

    y_true = test["y"].values
    y_pred = res.forecast_df["yhat"].values
    metrics = {"MAPE": mape(y_true, y_pred), "sMAPE": smape(y_true, y_pred)}
    return res, metrics


def plot_forecast(ts: pd.DataFrame, fcst: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts["ds"], y=ts["y"],
        mode="lines+markers",
        name="Actual",
        line=dict(width=2),
    ))
    fig.add_trace(go.Scatter(
        x=fcst["ds"], y=fcst["yhat"],
        mode="lines+markers",
        name="Forecast",
        line=dict(width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([fcst["ds"], fcst["ds"][::-1]]),
        y=pd.concat([fcst["yhat_upper"], fcst["yhat_lower"][::-1]]),
        fill="toself",
        name="95% band",
        line=dict(width=0),
        opacity=0.25,
        hoverinfo="skip",
    ))
    fig.update_layout(**PLOTLY_LAYOUT_BASE, title=title)
    fig.update_yaxes(title_text="")
    fig.update_xaxes(title_text="")
    return fig


def plot_anomalies(ts_z: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts_z["ds"], y=ts_z["y"],
        mode="lines+markers",
        name="Actual",
        line=dict(width=2),
    ))
    anom = ts_z[ts_z["anomaly"] == True]
    if len(anom):
        fig.add_trace(go.Scatter(
            x=anom["ds"], y=anom["y"],
            mode="markers",
            name="Anomaly",
            marker=dict(size=10, symbol="diamond"),
        ))
    fig.update_layout(**PLOTLY_LAYOUT_BASE, title="Anomaly Scan (Z-score)")
    return fig


# ----------------------------
# UI: Header
# ----------------------------
st.markdown('<p class="ec-title">EC-Predict (MVP)</p>', unsafe_allow_html=True)
st.markdown('<p class="ec-subtitle">Upload a time series, forecast the next horizon, and generate early-warning signals — in a Tableau-like dashboard.</p>', unsafe_allow_html=True)

# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx", "xls"])
    st.caption("Tip: your dataset should contain a date column and at least one numeric metric column.")

    st.divider()

    freq_label = st.selectbox("Time aggregation", ["Monthly (MS)", "Weekly (W)", "Daily (D)"], index=0)
    freq_map = {"Monthly (MS)": "MS", "Weekly (W)": "W", "Daily (D)": "D"}
    freq = freq_map[freq_label]

    horizon = st.slider("Forecast horizon (periods)", 3, 24, 6)
    test_size = st.slider("Backtest window (periods)", 0, 24, 6)

    model_choice = st.selectbox("Model", ["Auto", "ETS", "SARIMAX"], index=0)
    seasonal = st.selectbox("Seasonality (ETS)", ["None", "Additive", "Multiplicative"], index=0)

    z_thresh = st.slider("Anomaly threshold (Z)", 2.0, 5.0, 3.0, 0.5)

    st.divider()
    st.caption("Status")
    st.write(f"statsmodels: {'✅' if HAS_STATSMODELS else '❌'}")
    st.write(f"sarimax: {'✅' if HAS_SARIMAX else '❌'}")


# ----------------------------
# Load data
# ----------------------------
df = pd.DataFrame()
if uploaded is not None:
    try:
        df = read_uploaded_file(uploaded)
    except Exception as e:
        st.error(f"Failed to read file: {e}")

if df.empty:
    st.info("Upload a dataset to begin. (CSV or Excel)")
    st.stop()

# Guess columns
dt_guess = infer_datetime_col(df)
num_cols = infer_numeric_cols(df)

colA, colB, colC = st.columns([1.2, 1.2, 2.0], gap="large")
with colA:
    dt_col = st.selectbox("Date column", options=df.columns.tolist(), index=df.columns.get_loc(dt_guess) if dt_guess in df.columns else 0)
with colB:
    if len(num_cols) == 0:
        st.error("No numeric columns detected. Please upload a dataset with numeric metrics.")
        st.stop()
    y_col = st.selectbox("Target metric (y)", options=num_cols, index=0)
with colC:
    st.caption("Preview")
    st.dataframe(df.head(8), use_container_width=True)

# Build time series
ts = resample_series(df, dt_col, y_col, freq=freq)

# Data quality metrics
cov = coverage_pct(ts)
n_points = len(ts)
start_date = ts["ds"].min()
end_date = ts["ds"].max()

# Anomalies
ts_z = detect_anomalies(ts, z_thresh=z_thresh)
anom_count = int(ts_z["anomaly"].sum())

# Backtest + Fit
res, bt_metrics = backtest_model(ts, model_choice=model_choice, horizon=horizon, seasonal=seasonal, test_size=test_size)

# Fit final on full data for horizon forecast (if backtest used train only)
if test_size > 0 and len(ts) > test_size + 5:
    final_train = ts.copy()
else:
    final_train = ts.copy()

if model_choice == "ETS":
    final_res = fit_forecast_ets(final_train, horizon, seasonal)
elif model_choice == "SARIMAX":
    final_res = fit_forecast_sarimax(final_train, horizon)
else:
    # Auto: pick best by backtest (if available)
    # If bt had no test, default ETS
    chosen = res.model_name
    if "SARIMAX" in chosen:
        final_res = fit_forecast_sarimax(final_train, horizon)
    else:
        final_res = fit_forecast_ets(final_train, horizon, seasonal)

model_used = final_res.model_name

# Create combined table
hist = ts.copy()
hist["type"] = "Actual"
hist = hist.rename(columns={"y": "value"})[["ds", "value", "type"]]

fc = final_res.forecast_df.copy()
fc["type"] = "Forecast"
fc = fc.rename(columns={"yhat": "value"})[["ds", "value", "type"]]

# KPI cards row
k1, k2, k3, k4, k5 = st.columns(5, gap="large")

def card(col, label, value, sub=""):
    col.markdown(
        f"""
        <div class="ec-card">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
          <div class="sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k1:
    card(k1, "Model", model_used, "Auto-selected if enabled")
with k2:
    card(k2, "Horizon", f"{horizon}", f"Periods ({freq})")
with k3:
    card(k3, "Coverage", f"{cov:0.1f}%", f"{n_points} points")
with k4:
    m = bt_metrics.get("sMAPE", np.nan)
    card(k4, "Backtest sMAPE", "—" if np.isnan(m) else f"{m:0.1f}%", f"Window: {test_size}")
with k5:
    card(k5, "Anomalies", f"{anom_count}", f"Z ≥ {z_thresh:0.1f}")

st.write("")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "🧩 Drivers (simple)", "🚨 Alerts", "🧼 Data Quality"])

with tab1:
    fig = plot_forecast(ts, final_res.forecast_df, title=f"{y_col} — Forecast")
    st.plotly_chart(fig, use_container_width=True)

    # Forecast table + download
    out = final_res.forecast_df.copy()
    out["yhat"] = out["yhat"].astype(float)
    out["yhat_lower"] = out["yhat_lower"].astype(float)
    out["yhat_upper"] = out["yhat_upper"].astype(float)

    c1, c2 = st.columns([1.2, 1.0], gap="large")
    with c1:
        st.subheader("Forecast table")
        st.dataframe(out, use_container_width=True)
    with c2:
        st.subheader("Download")
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download forecast CSV", data=csv_bytes, file_name="ec_predict_forecast.csv", mime="text/csv")

with tab2:
    # Simple drivers = decomposition proxy: mom change, rolling trend, volatility
    s = ts.copy()
    s["mom_change"] = s["y"].diff()
    s["mom_pct"] = (s["y"].pct_change() * 100).replace([np.inf, -np.inf], np.nan)
    s["roll_mean_3"] = s["y"].rolling(3).mean()
    s["roll_std_3"] = s["y"].rolling(3).std()

    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Scatter(x=s["ds"], y=s["y"], mode="lines+markers", name="Actual", line=dict(width=2)), secondary_y=False)
    fig2.add_trace(go.Scatter(x=s["ds"], y=s["roll_mean_3"], mode="lines", name="3-pt trend", line=dict(width=2, dash="dot")), secondary_y=False)
    fig2.add_trace(go.Bar(x=s["ds"], y=s["mom_change"], name="MoM Δ (abs)", opacity=0.65), secondary_y=True)
    fig2.update_layout(**PLOTLY_LAYOUT_BASE, title="Trend + Momentum (driver proxy)")
    fig2.update_yaxes(title_text="Level", secondary_y=False)
    fig2.update_yaxes(title_text="MoM Δ", secondary_y=True)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Recent driver stats")
    last = s.tail(8)[["ds", "y", "mom_change", "mom_pct", "roll_mean_3", "roll_std_3"]].copy()
    st.dataframe(last, use_container_width=True)

    st.caption("Next upgrade (Step 2/3): SHAP-style feature importance once we add external drivers (rates, FX, sector, pipeline).")

with tab3:
    st.subheader("Early-warning signals")

    # Rule-based alerts (bank-friendly defaults)
    alerts = []
    # 1) Big MoM drop
    if len(ts) >= 3:
        mom = float(ts["y"].iloc[-1] - ts["y"].iloc[-2])
        mom_pct = float((ts["y"].iloc[-1] / ts["y"].iloc[-2] - 1) * 100) if ts["y"].iloc[-2] != 0 else np.nan
        if np.isfinite(mom_pct) and mom_pct <= -15:
            alerts.append(("🔴 Sharp decline", f"Latest period dropped {mom_pct:0.1f}% vs prior. Investigate drivers."))

    # 2) Forecast downside risk (lower bound)
    if len(final_res.forecast_df):
        first = final_res.forecast_df.iloc[0]
        if first["yhat_lower"] < ts["y"].tail(6).mean() * 0.85:
            alerts.append(("🟠 Downside risk", "Forecast lower band implies meaningful downside vs recent average."))

    # 3) Anomalies
    if anom_count > 0:
        alerts.append(("🟡 Data anomaly detected", f"{anom_count} anomalous point(s) flagged. Verify data/one-offs."))

    if not alerts:
        st.success("No major alerts triggered by default rules.")
    else:
        for a, msg in alerts:
            st.warning(f"**{a}** — {msg}")

    st.plotly_chart(plot_anomalies(ts_z), use_container_width=True)

with tab4:
    st.subheader("Data quality summary")
    st.write(f"**Range:** {start_date.date()} → {end_date.date()}")
    st.write(f"**Points:** {n_points}")
    st.write(f"**Coverage:** {cov:0.1f}% (non-missing)")
    st.write("")

    # Missing timeline
    s = ts.copy()
    s["missing"] = s["y"].isna().astype(int)
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=s["ds"], y=s["missing"], name="Missing"))
    fig3.update_layout(**PLOTLY_LAYOUT_BASE, title="Missingness over time")
    fig3.update_yaxes(title_text="Missing (1=yes)")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Cleaned time series (after resampling)")
    st.dataframe(ts.head(24), use_container_width=True)

st.caption("EC-Predict MVP — next step: add bank KPIs & multi-metric portfolio forecasting.")
