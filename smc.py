import streamlit as st
import pandas as pd
import yfinance as yf
import time
import os

st.set_page_config(page_title="SMC + ChoCH Screener", layout="wide")
st.title("📊 SMC + ChoCH Screener Dashboard (S&P 500)")

# ==============================
# Load ticker list
# ==============================
@st.cache_data
def load_tickers():
    if not os.path.exists("sp500_tickers.csv"):
        st.error("sp500_tickers.csv not found! Generate it first.")
        st.stop()
    df = pd.read_csv("sp500_tickers.csv")
    return df["Ticker"].dropna().unique().tolist()

tickers = load_tickers()
st.sidebar.success(f"✅ Loaded {len(tickers)} tickers")

# ==============================
# Include your SMC & CHoCH functions
# ==============================
def is_local_pivot_high(df, idx, lookback=3):
    if idx < lookback or idx + lookback >= len(df): return False
    cand_high = float(df["High"].iloc[idx])
    left_max = float(df["High"].iloc[idx - lookback:idx].max())
    right_max = float(df["High"].iloc[idx + 1: idx + 1 + lookback].max())
    return (cand_high > left_max) and (cand_high >= right_max)

def is_local_pivot_low(df, idx, lookback=3):
    if idx < lookback or idx + lookback >= len(df): return False
    cand_low = float(df["Low"].iloc[idx])
    left_min = float(df["Low"].iloc[idx - lookback:idx].min())
    right_min = float(df["Low"].iloc[idx + 1: idx + 1 + lookback].min())
    return (cand_low < left_min) and (cand_low <= right_min)

def detect_market_structure(df, lookback=3):
    highs, lows = [], []
    for i in range(len(df)):
        if is_local_pivot_high(df, i, lookback):
            highs.append((i, float(df["High"].iloc[i])))
        if is_local_pivot_low(df, i, lookback):
            lows.append((i, float(df["Low"].iloc[i])))
    return highs, lows

def detect_order_blocks(df, lookback=10):
    bullish_obs, bearish_obs = [], []
    for i in range(lookback, len(df)):
        close_now = float(df["Close"].iloc[i])
        prev_high = float(df["High"].iloc[i - 1])
        prev_low = float(df["Low"].iloc[i - 1])
        if close_now > prev_high:
            bullish_obs.append((i, prev_high))
        elif close_now < prev_low:
            bearish_obs.append((i, prev_low))
    return bullish_obs, bearish_obs

def get_zone_and_bos(df, lookback=5):
    highs, lows = detect_market_structure(df, lookback)
    if not highs or not lows:
        return "Unknown", "No Structure"
    last_high = highs[-1][1]
    last_low = lows[-1][1]
    close = float(df["Close"].iloc[-1])
    equilibrium = (last_high + last_low) / 2
    if close > last_high:
        zone = "Premium"
    elif close < last_low:
        zone = "Discount"
    elif close > equilibrium:
        zone = "Premium"
    else:
        zone = "Discount"
    if close > last_high:
        bos = "Bullish BOS"
    elif close < last_low:
        bos = "Bearish BOS"
    else:
        bos = "None"
    return zone, bos

def generate_smc_signal(df):
    if df.empty or len(df) < 10:
        return "HOLD", "insufficient data"
    bullish_obs, bearish_obs = detect_order_blocks(df)
    last_close = float(df["Close"].iloc[-1])
    if bullish_obs and last_close > bullish_obs[-1][1]:
        return "BUY", f"Price broke above bullish OB @ {bullish_obs[-1][1]:.2f}"
    elif bearish_obs and last_close < bearish_obs[-1][1]:
        return "SELL", f"Price broke below bearish OB @ {bearish_obs[-1][1]:.2f}"
    else:
        return "HOLD", "No clear SMC signal"

def detect_swings(df, lookback=5):
    highs, lows = [], []
    for i in range(lookback, len(df) - lookback):
        high = float(df["High"].iloc[i])
        low = float(df["Low"].iloc[i])
        left_high = float(df["High"].iloc[i - lookback:i].max())
        right_high = float(df["High"].iloc[i + 1:i + 1 + lookback].max())
        left_low = float(df["Low"].iloc[i - lookback:i].min())
        right_low = float(df["Low"].iloc[i + 1:i + 1 + lookback].min())
        if high >= left_high and high >= right_high:
            highs.append((i, high))
        if low <= left_low and low <= right_low:
            lows.append((i, low))
    return highs, lows

def detect_choc(df, lookback=5):
    highs, lows = detect_swings(df, lookback)
    if len(highs) < 2 or len(lows) < 2:
        return "HOLD", "Not enough swing points"
    last_high, prev_high = float(highs[-1][1]), float(highs[-2][1])
    last_low, prev_low = float(lows[-1][1]), float(lows[-2][1])
    close = float(df["Close"].iloc[-1])
    if last_low > prev_low and close > prev_high:
        return "BUY", f"Bullish ChoCH (HL {last_low:.2f}>{prev_low:.2f}, break {prev_high:.2f})"
    if last_high < prev_high and close < prev_low:
        return "SELL", f"Bearish ChoCH (LH {last_high:.2f}<{prev_high:.2f}, break {prev_low:.2f})"
    return "HOLD", "No ChoCH"

# ==============================
# Analyze a single ticker
# ==============================
def analyze_ticker(ticker):
    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False)
        if df.empty:
            return None
        smc_sig, smc_reason = generate_smc_signal(df)
        choc_sig, choc_reason = detect_choc(df)
        zone, bos = get_zone_and_bos(df)
        last = float(df["Close"].iloc[-1])
        return pd.DataFrame([{
            "Ticker": ticker,
            "Close": last,
            "SMC_Signal": smc_sig,
            "SMC_Reason": smc_reason,
            "ChoCH_Signal": choc_sig,
            "ChoCH_Reason": choc_reason,
            "SMC_Zone": zone,
            "BOS": bos
        }])
    except Exception as e:
        return pd.DataFrame([{
            "Ticker": ticker,
            "Close": None,
            "SMC_Signal": "HOLD",
            "SMC_Reason": str(e),
            "ChoCH_Signal": "HOLD",
            "ChoCH_Reason": str(e),
            "SMC_Zone": "Error",
            "BOS": "Error"
        }])

# ==============================
# Run S&P 500 screener
# ==============================
run = st.button("▶️ Run Screener")

if run:
    results = []
    progress = st.progress(0)
    status = st.empty()
    for i, t in enumerate(tickers):
        status.text(f"Fetching {t} ({i+1}/{len(tickers)})")
        res = analyze_ticker(t)
        results.append(res)
        progress.progress((i+1)/len(tickers))
    df_result = pd.concat(results, ignore_index=True)
    df_result.to_csv("smc_signals_sp500.csv", index=False)
    st.success(f"✅ Completed analysis for {len(df_result)} tickers")

# ==============================
# Load previous CSV if exists
# ==============================
if os.path.exists("smc_signals_sp500.csv"):
    df_result = pd.read_csv("smc_signals_sp500.csv")
else:
    st.warning("Run the screener to generate data.")
    st.stop()

# ==============================
# Sidebar Filters
# ==============================
st.sidebar.header("🔍 Filters")
def safe_multiselect(label, colname):
    if colname in df_result.columns:
        return st.sidebar.multiselect(label, sorted(df_result[colname].dropna().unique()))
    return []

smc_filter = safe_multiselect("SMC Signal", "SMC_Signal")
choc_filter = safe_multiselect("ChoCH Signal", "ChoCH_Signal")
bos_filter = safe_multiselect("BOS Signal", "BOS")
zone_filter = safe_multiselect("Zone Signal", "SMC_Zone")

df_filtered = df_result.copy()
if smc_filter:
    df_filtered = df_filtered[df_filtered["SMC_Signal"].isin(smc_filter)]
if choc_filter:
    df_filtered = df_filtered[df_filtered["ChoCH_Signal"].isin(choc_filter)]
if bos_filter:
    df_filtered = df_filtered[df_filtered["BOS"].isin(bos_filter)]
if zone_filter:
    df_filtered = df_filtered[df_filtered["SMC_Zone"].isin(zone_filter)]

st.write(f"### Showing {len(df_filtered)} of {len(df_result)} results")
st.dataframe(df_filtered, use_container_width=True)

# Download filtered CSV
st.download_button(
    label="📥 Download Filtered CSV",
    data=df_filtered.to_csv(index=False).encode('utf-8'),
    file_name="filtered_smc_screener.csv",
    mime="text/csv"
)
