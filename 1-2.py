#!/usr/bin/env python
# coding: utf-8

import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.stats import norm


st.set_page_config(
    page_title="Equity Options Risk Dashboard",
    layout="wide",
)


# ================================================================
# Constants
# ================================================================

DEFAULT_TICKER = "^GSPC"
DEFAULT_SPOT = 100.0
DEFAULT_VOLATILITY = 0.25
DEFAULT_RISK_FREE_RATE = 0.04
TRADING_DAYS = 252


# ================================================================
# Pricing functions
# ================================================================


def bs_call_metrics(S: float, K: float, T: float, r: float, vol: float) -> dict:
    """Calculate Black-Scholes price and Greeks for a European call option."""
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if vol <= 0:
        raise ValueError("Volatility must be positive.")

    d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * vol * np.sqrt(T))
    theta = (
        -(S * norm.pdf(d1) * vol) / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) * norm.cdf(d2)
    )
    vega = S * norm.pdf(d1) * np.sqrt(T)
    rho = K * T * np.exp(-r * T) * norm.cdf(d2)

    return {
        "price": float(call_price),
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta),
        "vega": float(vega),
        "rho": float(rho),
        "d1": float(d1),
        "d2": float(d2),
    }


# ================================================================
# Market-data helpers
# ================================================================


def _extract_close_series(data: pd.DataFrame, ticker: str) -> pd.Series:
    """Extract a close-price series from either standard or MultiIndex data."""
    if data is None or data.empty:
        raise ValueError(f"No historical data returned for {ticker}.")

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            raise ValueError("Close column was not returned by Yahoo Finance.")

        close_data = data["Close"]

        if isinstance(close_data, pd.Series):
            close_series = close_data
        elif ticker in close_data.columns:
            close_series = close_data[ticker]
        else:
            close_series = close_data.iloc[:, 0]
    else:
        if "Close" not in data.columns:
            raise ValueError("Close column was not returned by Yahoo Finance.")
        close_series = data["Close"]

    close_series = pd.to_numeric(close_series, errors="coerce").dropna()

    if close_series.empty:
        raise ValueError(f"No valid close prices returned for {ticker}.")

    return close_series


def _download_yahoo_data(
    ticker: str,
    period: str,
    interval: str = "1d",
    attempts: int = 3,
) -> pd.DataFrame:
    """Download Yahoo Finance data with limited retries."""
    ticker = ticker.strip().upper()

    if not ticker:
        raise ValueError("Ticker cannot be empty.")

    last_error: Exception | None = None

    for attempt in range(attempts):
        try:
            data = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,
                timeout=15,
            )

            if data is None or data.empty:
                raise ValueError(f"No historical data returned for {ticker}.")

            return data

        except Exception as error:
            last_error = error

            if attempt < attempts - 1:
                time.sleep(2**attempt)

    if last_error is None:
        raise RuntimeError(f"Unable to download market data for {ticker}.")

    raise last_error


@st.cache_data(ttl=3600, show_spinner=False)
def load_market_data(ticker: str):
    """Return recent history, close prices, spot, and annualized volatility."""
    ticker = ticker.strip().upper()
    history = _download_yahoo_data(ticker, period="3mo", interval="1d")
    close_series = _extract_close_series(history, ticker)

    if len(close_series) < 2:
        raise ValueError(f"Not enough price observations returned for {ticker}.")

    spot = float(close_series.iloc[-1])
    returns = np.log(close_series / close_series.shift(1)).dropna()
    volatility = float(returns.std() * np.sqrt(TRADING_DAYS))

    if not np.isfinite(spot) or spot <= 0:
        raise ValueError("The downloaded spot price is invalid.")

    if not np.isfinite(volatility) or volatility <= 0:
        raise ValueError("The calculated volatility is invalid.")

    return history, close_series, spot, volatility


@st.cache_data(ttl=3600, show_spinner=False)
def get_volatility(ticker: str, window: str = "1y") -> float | None:
    """Return annualized historical volatility, or None when unavailable."""
    try:
        history = _download_yahoo_data(ticker, period=window, interval="1d")
        close_series = _extract_close_series(history, ticker.strip().upper())

        if len(close_series) < 2:
            raise ValueError(f"Not enough price observations returned for {ticker}.")

        returns = np.log(close_series / close_series.shift(1)).dropna()
        volatility = float(returns.std() * np.sqrt(TRADING_DAYS))

        if not np.isfinite(volatility) or volatility <= 0:
            raise ValueError("Calculated volatility is invalid.")

        return volatility

    except Exception as error:
        print(
            f"Unable to retrieve volatility for {ticker}: "
            f"{type(error).__name__}: {error}"
        )
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_risk_free_rate() -> float | None:
    """Return the US 10-year Treasury yield as a decimal, or None."""
    try:
        history = _download_yahoo_data("^TNX", period="5d", interval="1d")
        close_series = _extract_close_series(history, "^TNX")
        rate = float(close_series.iloc[-1]) / 100

        if not np.isfinite(rate):
            raise ValueError("Downloaded interest rate is invalid.")

        return rate

    except Exception as error:
        print(
            "Unable to retrieve the risk-free rate: "
            f"{type(error).__name__}: {error}"
        )
        return None


# ================================================================
# Sidebar inputs and session-state initialization
# ================================================================

st.sidebar.header("Inputs")
ticker = st.sidebar.text_input("Ticker", value=DEFAULT_TICKER).strip().upper()

if not ticker:
    ticker = DEFAULT_TICKER
    st.sidebar.warning(f"Ticker was empty. Using {DEFAULT_TICKER}.")


# Initialize state once.
if "vol" not in st.session_state:
    initial_vol = get_volatility(ticker)
    st.session_state["vol"] = (
        initial_vol if initial_vol is not None else DEFAULT_VOLATILITY
    )

if "r" not in st.session_state:
    initial_rate = get_risk_free_rate()
    st.session_state["r"] = (
        initial_rate if initial_rate is not None else DEFAULT_RISK_FREE_RATE
    )

if "market_ticker" not in st.session_state:
    st.session_state["market_ticker"] = ticker


# Refresh button.
if st.sidebar.button("Get Market Data", key="refresh_btn"):
    st.cache_data.clear()

    refreshed_vol = get_volatility(ticker)
    refreshed_rate = get_risk_free_rate()

    if refreshed_vol is None:
        refreshed_vol = st.session_state.get("vol", DEFAULT_VOLATILITY)
        st.sidebar.warning(
            "Yahoo Finance volatility data is temporarily unavailable. "
            "Keeping the current volatility input."
        )

    if refreshed_rate is None:
        refreshed_rate = st.session_state.get("r", DEFAULT_RISK_FREE_RATE)
        st.sidebar.warning(
            "Yahoo Finance rate data is temporarily unavailable. "
            "Keeping the current rate input."
        )

    st.session_state["vol"] = float(refreshed_vol)
    st.session_state["r"] = float(refreshed_rate)
    st.session_state["market_ticker"] = ticker

    # Update widget state directly after the refresh.
    st.session_state["vol_percent"] = float(refreshed_vol * 100)
    st.session_state["r_percent"] = float(refreshed_rate * 100)


# Maturity.
T_days = st.sidebar.slider(
    label="Maturity (days)",
    min_value=1,
    max_value=252,
    value=30,
)


# Volatility.
if "vol_percent" not in st.session_state:
    st.session_state["vol_percent"] = float(
        np.clip(st.session_state["vol"] * 100, 0.1, 150.0)
    )

vol_percent = st.sidebar.slider(
    "Volatility (%)",
    min_value=0.1,
    max_value=150.0,
    step=0.1,
    format="%.2f",
    key="vol_percent",
)

vol = float(vol_percent / 100)
st.session_state["vol"] = vol


# Risk-free rate.
if "r_percent" not in st.session_state:
    st.session_state["r_percent"] = float(
        np.clip(st.session_state["r"] * 100, -5.0, 20.0)
    )

r_percent = st.sidebar.slider(
    label="Rate (%)",
    min_value=-5.0,
    max_value=20.0,
    step=0.01,
    format="%.2f",
    key="r_percent",
)

r = float(r_percent / 100)
st.session_state["r"] = r


# Strike selection.
strike_mode = st.sidebar.selectbox(
    "Strike setting",
    ["ATM", "ITM", "OTM", "Manual"],
)


# ================================================================
# Main market-data load with safe fallback
# ================================================================

market_data_available = True
market_data_error = None

try:
    hist, close_series, S, hist_vol = load_market_data(ticker)
except Exception as error:
    market_data_available = False
    market_data_error = error

    S = float(st.session_state.get("last_valid_spot", DEFAULT_SPOT))
    hist_vol = float(st.session_state.get("vol", DEFAULT_VOLATILITY))

    close_series = pd.Series([S], name="Close")
    hist = pd.DataFrame({"Close": close_series})
else:
    st.session_state["last_valid_spot"] = float(S)


if not market_data_available:
    st.warning(
        "Live Yahoo Finance price data is temporarily unavailable. "
        f"The dashboard is using a fallback spot price of {S:.2f}. "
        "You can still adjust volatility, rate, maturity, and strike manually."
    )
    print(
        f"Market-data fallback used for {ticker}: "
        f"{type(market_data_error).__name__}: {market_data_error}"
    )


if strike_mode == "ATM":
    K = S
elif strike_mode == "ITM":
    K = S * 0.8
elif strike_mode == "OTM":
    K = S * 1.2
else:
    K = st.sidebar.number_input(
        "Strike",
        min_value=0.01,
        value=float(S),
    )


T = T_days / TRADING_DAYS
metrics = bs_call_metrics(S, K, T, r, vol)


# ================================================================
# Dashboard title and top metrics
# ================================================================

st.title("Equity Options Risk Dashboard")
st.caption("Single-position Black-Scholes monitor for a long call")

st.subheader("Pricing")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Spot", f"{S:.2f}")
c2.metric("Strike", f"{K:.2f}")
c3.metric("Vol", f"{vol:.2%}")
c4.metric("Call Price", f"{metrics['price']:.4f}")
c5.metric("Rate", f"{r:.2%}")


# ================================================================
# Payoff diagram
# ================================================================

payoff_spot_range = np.linspace(S * 0.6, S * 1.4, 250)
payoff = np.maximum(payoff_spot_range - K, 0)
profit_loss = payoff - metrics["price"]
breakeven = K + metrics["price"]

payoff_fig = go.Figure()

payoff_fig.add_trace(
    go.Scatter(
        x=payoff_spot_range,
        y=profit_loss,
        mode="lines",
        name="Profit / Loss",
        line=dict(color="#4CC9F0", width=3),
    )
)

payoff_fig.add_hline(y=0, line_dash="dot", line_color="gray")
payoff_fig.add_vline(x=S, line_dash="dash", line_color="#FFD166")
payoff_fig.add_vline(x=K, line_dash="dash", line_color="#FFFFFF")
payoff_fig.add_vline(x=breakeven, line_dash="dot", line_color="#06D6A0")

annotation_y = float(max(profit_loss))

payoff_fig.add_annotation(
    x=S,
    y=annotation_y * 0.95,
    text="Spot",
    showarrow=False,
    yanchor="top",
    font=dict(color="#FFD166"),
)

payoff_fig.add_annotation(
    x=K,
    y=annotation_y * 0.82,
    text="Strike",
    showarrow=False,
    yanchor="top",
    font=dict(color="#FFFFFF"),
)

payoff_fig.add_annotation(
    x=breakeven,
    y=annotation_y * 0.69,
    text="Breakeven",
    showarrow=False,
    yanchor="top",
    font=dict(color="#06D6A0"),
)

payoff_fig.update_layout(
    title="Long Call Profit / Loss at Expiry",
    template="plotly_dark",
    xaxis_title="Underlying Price at Expiry",
    yaxis_title="Profit / Loss",
    height=420,
    legend=dict(
        x=0.02,
        y=0.98,
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(0,0,0,0)",
    ),
)

st.plotly_chart(payoff_fig, width="stretch")


# ================================================================
# Greeks
# ================================================================

st.subheader("Greeks")

c6, c7, c8, c9, c10 = st.columns(5)
c6.metric("Delta", f"{metrics['delta']:.4f}")
c7.metric("Gamma", f"{metrics['gamma']:.6f}")
c8.metric("Theta", f"{metrics['theta'] / 365:.6f}")
c9.metric("Vega", f"{metrics['vega'] / 100:.4f}")
c10.metric("Rho", f"{metrics['rho'] / 100:.4f}")


spot_range = np.linspace(S * 0.7, S * 1.3, 200)
delta_list, gamma_list, vega_list, theta_list, rho_list = [], [], [], [], []

for spot_value in spot_range:
    greek_metrics = bs_call_metrics(spot_value, K, T, r, vol)
    delta_list.append(greek_metrics["delta"])
    gamma_list.append(greek_metrics["gamma"])
    vega_list.append(greek_metrics["vega"] / 100)
    theta_list.append(greek_metrics["theta"] / 365)
    rho_list.append(greek_metrics["rho"] / 100)


def plot_greek(col, x, y, name, current_value, current_spot, strike):
    with col:
        greek_fig = go.Figure()

        greek_fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=name,
                showlegend=False,
            )
        )

        greek_fig.add_vline(x=strike, line_dash="dash", line_color="white")

        greek_fig.add_trace(
            go.Scatter(
                x=[current_spot],
                y=[current_value],
                mode="markers",
                marker=dict(size=6, color="red"),
                name="Current",
                showlegend=True,
            )
        )

        greek_fig.update_layout(
            template="plotly_dark",
            xaxis_title="Spot",
            yaxis_title=name,
            legend=dict(
                x=0.98,
                y=0.98,
                xanchor="right",
                yanchor="top",
                bgcolor="rgba(0,0,0,0)",
            ),
        )

        st.plotly_chart(greek_fig, width="stretch")


col1, col2, col3, col4, col5 = st.columns(5)

plot_greek(col1, spot_range, delta_list, "Delta", metrics["delta"], S, K)
plot_greek(col2, spot_range, gamma_list, "Gamma", metrics["gamma"], S, K)
plot_greek(
    col3,
    spot_range,
    theta_list,
    "Theta",
    metrics["theta"] / 365,
    S,
    K,
)
plot_greek(
    col4,
    spot_range,
    vega_list,
    "Vega",
    metrics["vega"] / 100,
    S,
    K,
)
plot_greek(
    col5,
    spot_range,
    rho_list,
    "Rho",
    metrics["rho"] / 100,
    S,
    K,
)


# ================================================================
# AI commentary
# ================================================================

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def get_openai_api_key():
    try:
        return st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    except st.errors.StreamlitSecretNotFoundError:
        return os.getenv("OPENAI_API_KEY")


def build_ai_commentary_prompt(ticker, S, K, T_days, r, vol, metrics):
    return f"""
You are an equity derivatives risk manager.
Analyze this single long call option position using the provided market inputs and Greeks.

Position data:
- Ticker: {ticker}
- Position: Long call
- Spot: {S:.4f}
- Strike: {K:.4f}
- Days to maturity: {T_days}
- Risk-free rate: {r:.4%}
- Implied/historical volatility input: {vol:.4%}
- Call price: {metrics['price']:.4f}
- Delta: {metrics['delta']:.4f}
- Gamma: {metrics['gamma']:.6f}
- Theta per day: {metrics['theta'] / 365:.6f}
- Vega per 1 vol point: {metrics['vega'] / 100:.4f}

Requirements:
- Write exactly 3 sentences.
- Focus on overall position characteristics, not individual Greek definitions.
- Explain what the position is exposed to.
- Mention the primary risk.
- Mention the market environment that benefits the position.
- Mention the market environment that hurts the position.
- Use professional risk-management language.
- Maximum 80 words.
- Do not use bullet points.

Return exactly:

Position Type:
<one sentence>

Key Risk:
<one sentence>

Market View:
<one sentence>
""".strip()


@st.cache_data(show_spinner=False)
def generate_ai_commentary(prompt, api_key):
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt,
    )
    return response.output_text.strip()


st.subheader("AI Commentary")

commentary_prompt = build_ai_commentary_prompt(
    ticker,
    S,
    K,
    T_days,
    r,
    vol,
    metrics,
)

with st.expander("Prompt", expanded=False):
    st.code(commentary_prompt, language="text")

api_key = get_openai_api_key()

if OpenAI is None:
    st.info("Install the OpenAI Python SDK to generate commentary: pip install openai")
elif not api_key:
    st.info(
        "Set OPENAI_API_KEY in Streamlit secrets or as an environment variable "
        "to generate commentary."
    )
elif st.button("Generate AI Commentary", type="primary"):
    with st.spinner("Generating risk commentary..."):
        try:
            commentary = generate_ai_commentary(commentary_prompt, api_key)
            st.markdown(commentary)
        except Exception as exc:
            st.error(f"Unable to generate AI commentary: {exc}")


# ================================================================
# Scenario analysis
# ================================================================

base_price = metrics["price"]


# Spot shocks.
spot_shocks_for_table = [-0.10, -0.07, -0.05, -0.03, -0.01, 0, 0.01, 0.03, 0.05, 0.07, 0.10]
spot_results = []

for shock in spot_shocks_for_table:
    scenario_spot = S * (1 + shock)
    scenario_metrics = bs_call_metrics(scenario_spot, K, T, r, vol)
    pnl = scenario_metrics["price"] - base_price

    dS = scenario_spot - S
    delta_pnl = metrics["delta"] * dS
    gamma_pnl = 0.5 * metrics["gamma"] * dS**2
    vega_pnl = 0.0

    approx_pnl = delta_pnl + gamma_pnl + vega_pnl
    residual = pnl - approx_pnl

    spot_results.append(
        {
            "Scenario": f"Spot {shock:+.0%}",
            "Spot": scenario_spot,
            "Vol": vol,
            "Price": scenario_metrics["price"],
            "PnL (%)": pnl / base_price if base_price != 0 else np.nan,
            "PnL": pnl,
            "Delta PnL": delta_pnl,
            "Gamma PnL": gamma_pnl,
            "Vega PnL": vega_pnl,
            "Residual": residual,
        }
    )

spot_scenario_df = pd.DataFrame(spot_results)


# Volatility shocks in percentage points.
vol_point_shocks = [-0.10, -0.07, -0.05, -0.03, -0.01, 0, 0.01, 0.03, 0.05, 0.07, 0.10]
vol_results = []

for vol_point_shock in vol_point_shocks:
    scenario_vol = max(0.0001, vol + vol_point_shock)
    scenario_metrics = bs_call_metrics(S, K, T, r, scenario_vol)
    pnl = scenario_metrics["price"] - base_price

    d_vol = scenario_vol - vol
    delta_pnl = 0.0
    gamma_pnl = 0.0
    vega_pnl = metrics["vega"] * d_vol

    approx_pnl = delta_pnl + gamma_pnl + vega_pnl
    residual = pnl - approx_pnl

    vol_results.append(
        {
            "Scenario": f"Vol {vol_point_shock * 100:+.0f} pts",
            "Spot": S,
            "Vol": scenario_vol,
            "Price": scenario_metrics["price"],
            "PnL (%)": pnl / base_price if base_price != 0 else np.nan,
            "PnL": pnl,
            "Delta PnL": delta_pnl,
            "Gamma PnL": gamma_pnl,
            "Vega PnL": vega_pnl,
            "Residual": residual,
        }
    )

vol_scenario_df = pd.DataFrame(vol_results)


# Spot x volatility heatmap.
spot_shocks = np.array([-10, -5, 0, 5, 10]) / 100
vol_shocks = np.array([-0.10, -0.05, 0, 0.05, 0.10])

pnl_matrix = []

for vol_shock in vol_shocks:
    row = []

    for spot_shock in spot_shocks:
        new_spot = S * (1 + spot_shock)
        new_vol = max(0.0001, vol + vol_shock)

        scenario_metrics = bs_call_metrics(new_spot, K, T, r, new_vol)
        pnl = scenario_metrics["price"] - base_price
        row.append(pnl)

    pnl_matrix.append(row)

heatmap_fig = go.Figure(
    data=go.Heatmap(
        z=pnl_matrix,
        x=spot_shocks * 100,
        y=vol_shocks * 100,
        colorscale="RdYlGn",
        reversescale=False,
        text=np.round(pnl_matrix, 2),
        texttemplate="%{text}",
    )
)

heatmap_fig.update_layout(
    title="P&L",
    xaxis_title="Spot Variation (%)",
    yaxis_title="Vol Variation (percentage points)",
    template="plotly_dark",
    height=600,
)


st.subheader("Scenario Analysis (Stress Test)")

tab1, tab2, tab3 = st.tabs(["Spot", "Volatility", "Spot x Vol"])


def highlight_spot_base(row):
    if row["Scenario"] == "Spot +0%":
        return ["background-color: #2E8B57; color: white"] * len(row)
    return [""] * len(row)


def highlight_vol_base(row):
    if row["Scenario"] == "Vol +0 pts":
        return ["background-color: #2E8B57; color: white"] * len(row)
    return [""] * len(row)


styled_spot_df = (
    spot_scenario_df.style.apply(highlight_spot_base, axis=1)
    .format(
        {
            "Spot": "{:.2f}",
            "Vol": "{:.2%}",
            "Price": "{:.4f}",
            "PnL (%)": "{:.2%}",
            "PnL": "{:.4f}",
            "Delta PnL": "{:.4f}",
            "Gamma PnL": "{:.4f}",
            "Vega PnL": "{:.4f}",
            "Residual": "{:.4f}",
        }
    )
)

styled_vol_df = (
    vol_scenario_df.style.apply(highlight_vol_base, axis=1)
    .format(
        {
            "Spot": "{:.2f}",
            "Vol": "{:.2%}",
            "Price": "{:.4f}",
            "PnL (%)": "{:.2%}",
            "PnL": "{:.4f}",
            "Delta PnL": "{:.4f}",
            "Gamma PnL": "{:.4f}",
            "Vega PnL": "{:.4f}",
            "Residual": "{:.4f}",
        }
    )
)

with tab1:
    st.dataframe(styled_spot_df, width="stretch")

with tab2:
    st.dataframe(styled_vol_df, width="stretch")

with tab3:
    st.plotly_chart(heatmap_fig, width="stretch")
