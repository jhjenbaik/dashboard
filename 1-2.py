#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import norm



st.set_page_config(page_title="Equity Options Risk Dashboard", layout="wide")

# ---------- pricing function ----------

def bs_call_metrics(S, K, T, r, vol):
    if S <=0 or K <= 0:
        raise ValueError("S and K must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if vol <= 0:
        raise ValueError("vol must be positive.")

    d1 = (np.log(S/K) + (r + 0.5*vol**2)*T) / (vol * np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)

    call_price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * vol * np.sqrt(T))
    theta = (
        -(S*norm.pdf(d1) * vol) / (2*np.sqrt(T))
        - r*K*np.exp(-r*T) * norm.cdf(d2)
    )
    vega = S * norm.pdf(d1) * np.sqrt(T)
    rho = K * T * np.exp(-r*T) * norm.cdf(d2)

    return {
        "price": call_price,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho,
        "d1": d1,
        "d2": d2
    }

    
# ---------- market data ----------

@st.cache_data
def load_market_data(ticker):
    tk = yf.Ticker(ticker)
    hist = tk.history(period="3mo", interval="1d")
    close_series = hist["Close"].dropna()

    if close_series.empty:
        raise ValueError(f"No price data found for {ticker}")

    S = float(close_series.iloc[-1])
    returns = np.log(close_series / close_series.shift(1)).dropna()
    vol = float(returns.std() * np.sqrt(252))

    return hist, close_series, S, vol

def get_risk_free_rate():
    tnx = yf.Ticker("^TNX")
    data = tnx.history(period="1d")
    rate = data["Close"].iloc[-1]
    r = rate/100
    
    return r


@st.cache_data(ttl=3600, show_spinner=False)
def get_volatility(ticker: str, window: str = "1y") -> float | None:
    """
    Download historical prices and calculate annualized volatility.

    Returns:
        Annualized volatility as a decimal, e.g. 0.25 = 25%.
        None when market data cannot be retrieved.
    """
    ticker = ticker.strip().upper()

    if not ticker:
        return None

    for attempt in range(3):
        try:
            hist = yf.download(
                ticker,
                period=window,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
                timeout=15,
            )

            if hist is None or hist.empty:
                raise ValueError(f"No historical data returned for {ticker}")

            # yfinance may return either a normal or MultiIndex DataFrame
            if isinstance(hist.columns, __import__("pandas").MultiIndex):
                close_series = hist["Close"][ticker]
            else:
                close_series = hist["Close"]

            close_series = close_series.dropna()

            if len(close_series) < 2:
                raise ValueError(
                    f"Not enough price observations returned for {ticker}"
                )

            returns = np.log(
                close_series / close_series.shift(1)
            ).dropna()

            volatility = float(returns.std() * np.sqrt(252))

            if not np.isfinite(volatility):
                raise ValueError("Calculated volatility is not finite")

            return vol

        except Exception as error:
            # Retry only after a short delay
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(
                    f"Unable to retrieve volatility for {ticker}: "
                    f"{type(error).__name__}: {error}"
                )

    return None

# ------------------------------------------------
# sidebar
# ------------------------------------------------

st.sidebar.header("Inputs")
ticker = st.sidebar.text_input("Ticker", value="^GSPC")

# -------------- refresh --------------
if st.sidebar.button("Get Market Data", key="refresh_btn"):
    st.cache_data.clear()

    mkt_vol = get_volatility(ticker)
    mkt_r = get_risk_free_rate()

    st.session_state["vol"] = mkt_vol
    st.session_state["r"] = mkt_r

    st.session_state["vol_percent"] = mkt_vol * 100
    st.session_state["r_percent"] = mkt_r * 100
    
# -------------- maturity --------------

T_days = st.sidebar.slider(
    label="Maturity (days)",
    min_value=1,
    max_value=252,
    value=30
)

# -------------- volatility --------------

if "vol" not in st.session_state:
    st.session_state["vol"] = get_volatility(ticker)

vol_percent = st.sidebar.slider(
    "Volatility (%)",
    min_value=0.0,
    max_value=100.0,
    value=st.session_state["vol"] * 100,
    step=0.1,
    format="%.2f",
    key="vol_percent"
)

vol = vol_percent/100
st.session_state["vol"] = vol

# -------------- risk-free rate --------------

if "r" not in st.session_state:
    st.session_state["r"] = get_risk_free_rate()

r_percent = st.sidebar.slider(
    label="Rate (%)",
    min_value=0.0,
    max_value=10.0,
    value=st.session_state["r"]*100,
    step=0.01,
    format="%.2f",
    key="r_percent"
)

r = r_percent / 100
st.session_state["r"] = r

# -------------- strike --------------

strike_mode = st.sidebar.selectbox("Strike setting", ["ATM", "ITM", "OTM", "Manual"])




# ---------- data load ----------

hist, close_series, S, hist_vol = load_market_data(ticker)

if strike_mode == "ATM":
    K = S * 1.0
elif strike_mode == "ITM":
    K = S * 0.8
elif strike_mode == "OTM":
    K = S * 1.2
else:
    K = st.sidebar.number_input("Strike", min_value=0.01, value=float(S))

T = T_days / 252
metrics = bs_call_metrics(S, K, T, r, vol)


# ---------- title ----------

st.title("Equity Options Risk Dashboard")
st.caption("Single-position Black-Scholes monitor for a long call")


# ------------------------------------------------
# top metrics
# ------------------------------------------------

st.subheader("Pricing")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Spot", f"{S:.2f}")
c2.metric("Strike", f"{K:.2f}")
c3.metric("Vol", f"{vol:.2%}")
c4.metric("Call Price", f"{metrics['price']:.4f}")
c5.metric("Rate", f"{r:.2%}")


# ---------- Payoff diagram ----------

payoff_spot_range = np.linspace(S*0.6, S*1.4, 250)
payoff = np.maximum(payoff_spot_range - K, 0)
profit_loss = payoff - metrics["price"]
breakeven = K + metrics["price"] # ????

payoff_fig = go.Figure()

payoff_fig.add_trace(go.Scatter(
    x=payoff_spot_range,
    y=payoff,
    mode="lines",
    name="Payoff",
    line=dict(color="#4CC9F0", width=3),
))

payoff_fig.add_hline(y=0, line_dash="dot", line_color="gray")
payoff_fig.add_vline(x=5, line_dash="dash", line_color="#FFD166")
payoff_fig.add_vline(x=5, line_dash="dash", line_color="#FFD166")
payoff_fig.add_vline(x=K, line_dash="dash", line_color="#FFFFFF")
payoff_fig.add_vline(x=breakeven, line_dash="dot", line_color="#06D6A0")

payoff_fig.add_annotation( # 'Spot' 글씨
    x=S,
    y=max(profit_loss) * 0.95, # ???왜 곱해
    text="Spot",
    showarrow=False,
    yanchor="top",
    font=dict(color="#FFD166"),
)

payoff_fig.add_annotation(
    x=K,
    y=max(profit_loss) * 0.82, # ???왜 곱해
    text="Strike",
    showarrow=False,
    yanchor="top",
    font=dict(color="#FFFFFF"),
)

payoff_fig.add_annotation(
    x=breakeven,
    y=max(profit_loss) * 0.69,
    text="Breakeven",
    showarrow=False,
    yanchor="top",
    font=dict(color="#06D6A0"),
)

payoff_fig.update_layout(
    title="Long Call Payoff at Expiry",
    template="plotly_dark",
    xaxis_title="Underlying Price at Expiry",
    yaxis_title="Value / P&L",
    height=420,
    legend=dict(
        x=0.02,
        y=0.98,
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(0,0,0,0)",
    ),
)

st.plotly_chart(payoff_fig, use_container_width=True)




st.subheader("Greeks")

c6, c7, c8, c9, c10 = st.columns(5)
c6.metric("Delta", f"{metrics['delta']:.4f}")
c7.metric("Gamma", f"{metrics['gamma']:.6f}")
c8.metric("Theta", f"{metrics['theta']/365:.6f}")
c9.metric("Vega", f"{metrics['vega']/100:.4f}")
c10.metric("Rho", f"{metrics['rho']/100:.4f}")

# ---------- Greeks vs Spot Visualization ----------

spot_range = np.linspace(S*0.7, S*1.3,  200)
delta_list, gamma_list, vega_list, theta_list, rho_list = [], [], [], [], []

for s in spot_range:
    m = bs_call_metrics(s, K, T, r, vol)
    delta_list.append(m["delta"])
    gamma_list.append(m["gamma"])
    vega_list.append(m["vega"]/100)
    theta_list.append(m["theta"]/365)
    rho_list.append(m["rho"]/100)

# ---------- plot ----------

col1, col2, col3, col4, col5 = st.columns(5)

def plot_greek(col, x, y, name, current_value, S, K):
    with col:
        fig = go.Figure()

        # line
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name=name,
            showlegend=False
        ))

        # ATM line
        fig.add_vline(x=K, line_dash="dash", line_color="white")

        # current point
        fig.add_trace(go.Scatter(
            x=[S],
            y=[current_value],
            mode='markers',
            marker=dict(size=6, color='red'),
            name='Current',
            showlegend=True
        ))

        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Spot",
            yaxis_title=name,
            legend=dict(
                x=0.98,
                y=0.98,
                xanchor="right",
                yanchor="top",
                bgcolor="rgba(0,0,0,0)"
            )
        )
        st.plotly_chart(fig, use_container_width=True)


col1, col2, col3, col4, col5 = st.columns(5)

plot_greek(col1, spot_range, delta_list, "Delta", metrics["delta"], S, K)
plot_greek(col2, spot_range, gamma_list, "Gamma", metrics["gamma"], S, K)
plot_greek(col3, spot_range, theta_list, "Theta", metrics["theta"]/365, S, K)
plot_greek(col4, spot_range, vega_list, "Vega", metrics["vega"]/100, S, K)
plot_greek(col5, spot_range, rho_list, "Rho", metrics["rho"]/100, S, K)


# ---------- AI commentary ----------

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
    -Ticker: {ticker}
    -Position: Long call
    -Spot: {S:.4f}
    -Strike: {K:.4f}
    -Days to maturity: {T_days}
    -Risk-free rate: {r:.4%}
    -Implied/historical volatility input: {vol:.4%}
    -Call price: {metrics["price"]:.4f}
    -Delta: {metrics["delta"]:.4f}
    -Gamma: {metrics["gamma"]:.6f}
    -Theta per day: {metrics["theta"] / 365:.6f}
    -Vega per 1 vol point: {metrics["vega"] / 100:.4f}

    Requirements:
    -Write exactly 3 sentences.
    -Focus on overall positions characteristics, not individual Greek definitions.
    -Explain what the position is exposed to.
    -Mention the primary risk.
    -Mention the market environment that benefits the position.
    -Mention the market environment that hurts the positions.
    -Use professional risk management language.
    -Maximum 80 words.
    -Do not use bullet points.

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

commentary_prompt = build_ai_commentary_prompt(ticker, S, K, T_days, r, vol, metrics)

with st.expander("Prompt", expanded=False):
    st.code(commentary_prompt, language="text")

api_key = get_openai_api_key()

if OpenAI is None:
    st.info("Install the OpenAI Python SDK to generate commentary: pip install openai")
elif not api_key:
    st.info("Enter an OpenAI API key in the sidebar or set OPEN_AI_KEY to generate commentary.")
elif st.button("Generate AI Commentary", type="primary"):
    with st.spinner("Generating risk commentary..."):
        try:
            commentary = generate_ai_commentary(commentary_prompt, api_key)
            st.markdown(commentary)
        except Exception as exc:
            st.error(f"(Unable to generate AI commentary: {exc}")



# ------------------------------------------------
# scenario analysis
# ------------------------------------------------

# ---------- spot shock ----------

base_price = metrics["price"]

spot_scenarios = [
    {"name": "Spot -10%", "S": S*0.90, "vol": vol},
    {"name": "Spot -7%", "S": S*0.93, "vol": vol},
    {"name": "Spot -5%", "S": S*0.95, "vol": vol},
    {"name": "Spot -3%", "S": S*0.97, "vol": vol},
    {"name": "Spot -1%", "S": S*0.99, "vol": vol},
    {"name": "Spot +0%", "S": S*1.00, "vol": vol},
    {"name": "Spot +1%", "S": S*1.01, "vol": vol},
    {"name": "Spot +3%", "S": S*1.03, "vol": vol},
    {"name": "Spot +5%", "S": S*1.05, "vol": vol},
    {"name": "Spot +7%", "S": S*1.07, "vol": vol},
    {"name": "Spot +10%", "S": S*1.10, "vol": vol},
]

spot_results = []
for sc in spot_scenarios:
    m = bs_call_metrics(sc["S"], K, T, r, sc["vol"])
    pnl = m["price"] - base_price

    dS = sc["S"] - S
    dVol = sc["vol"] - vol

    delta_pnl = metrics["delta"] * dS
    gamma_pnl = 0.5 * metrics["gamma"] * dS**2
    vega_pnl = metrics["vega"] * dVol

    approx_pnl = delta_pnl+gamma_pnl+vega_pnl
    residual = pnl - approx_pnl

    spot_results.append({
        "Scenario": sc["name"],
        "Spot": sc["S"],
        "Vol": sc["vol"],
        "Price": m["price"],
        "PnL (%)": pnl / base_price,
        "PnL": pnl,
        
        "Delta PnL": delta_pnl,
        "Gamma PnL": gamma_pnl,
        "Vega PnL": vega_pnl,
        "Residual": residual
    })

spot_scenario_df = pd.DataFrame(spot_results)


# ---------- vol shock ----------

vol_scenarios = [
    {"name": "Vol -10%", "vol": vol*0.90, "S": S},
    {"name": "Vol -7%", "vol": vol*0.93, "S": S},
    {"name": "Vol -5%", "vol": vol*0.95, "S": S},
    {"name": "Vol -3%", "vol": vol*0.97, "S": S},
    {"name": "Vol -1%", "vol": vol*0.99, "S": S},
    {"name": "Vol +0%", "vol": vol*1.0, "S": S},
    {"name": "Vol +1%", "vol": vol*1.01, "S": S},
    {"name": "Vol +3%", "vol": vol*1.03, "S": S},
    {"name": "Vol +5%", "vol": vol*1.05, "S": S},
    {"name": "Vol +7%", "vol": vol*1.07, "S": S},
    {"name": "Vol +10%", "vol": vol*1.10, "S": S},
]

vol_results = []
for sc in vol_scenarios:
    m = bs_call_metrics(S, K, T, r, sc["vol"])
    pnl = m["price"] - base_price

    dS = sc["S"]- S
    dVol = sc["vol"] - vol

    delta_pnl = metrics["delta"] * dS
    gamma_pnl = 0.5 * metrics["gamma"] * dS**2
    vega_pnl = metrics["vega"]/100 * dVol

    approx_pnl = delta_pnl + gamma_pnl + vega_pnl
    residual = pnl - approx_pnl

    vol_results.append({
        "Scenario": sc["name"],
        "Spot": sc["S"],
        "Vol": sc["vol"],
        "Price": m["price"],
        "PnL (%)": pnl / base_price,
        "PnL": pnl,
        
        "Delta PnL": delta_pnl,
        "Gamma PnL": gamma_pnl,
        "Vega PnL": vega_pnl,
        "Residual": residual
    })
        
vol_scenario_df = pd.DataFrame(vol_results)


# ---------- spot x vol shock ----------

spot_shocks = np.array([-10, -5, 0, 5, 10]) / 100 # %
vol_shocks = np.array([-0.10, -0.05, 0, 0.05, 0.10]) # pts

pnl_matrix = []

for v_shock in vol_shocks:
    row = []
    for s_shock in spot_shocks:
        new_S = S * (1 + s_shock)
        new_vol = vol + v_shock
        new_vol = max(0.0001, vol + v_shock)

        m = bs_call_metrics(new_S, K, T, r, new_vol)
        pnl = m["price"] - base_price

        row.append(pnl)
    pnl_matrix.append(row)

fig = go.Figure(data=go.Heatmap(
    z=pnl_matrix,
    x=spot_shocks * 100,
    y=vol_shocks * 100,
    colorscale="RdYlGn",
    reversescale=False,
    text=np.round(pnl_matrix, 2),
    texttemplate="%{text}",
))

fig.update_layout(
    title="P&L",
    xaxis_title="Spot Variation (%)",
    yaxis_title="Vol Variation (pts)",
    template="plotly_dark",
    height=600
)


# -------------------

st.subheader("Scenario Analysis (Stress Test)")

tab1, tab2, tab3 = st.tabs([
    "Spot",
    "Volatility",
    "Spot x Vol"
])

# ----------------------------


def highlight_base(row):
    if row["Scenario"] == "Spot +0%":
        return ["background-color: #2E8B57; color:white"] * len(row)
    else:
        return [""] * len(row)

styled_df = spot_scenario_df.style \
    .apply(highlight_base, axis=1) \
    .format({
        "Spot": "{:.2f}",
        "Price": "{:.4f}",
        "PnL": "{:.4f}",
    })


with tab1:
    st.dataframe(styled_df, use_container_width=True)

with tab2:
    st.dataframe(vol_scenario_df, use_container_width=True)

with tab3:
    st.plotly_chart(fig, use_container_width=True)




# In[ ]:




