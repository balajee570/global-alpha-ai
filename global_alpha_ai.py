from __future__ import annotations
import time
import random
import re
import streamlit as st
import requests
import io
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────────────────────────────────────────────────────────
# CONFIG (RATE SAFE)
# ─────────────────────────────────────────────────────────────

ET = timezone(timedelta(hours=-4))

MAX_WORKERS = 4          # safe limit
CHUNK_SIZE = 60          # optimal batch size
MAX_RETRIES = 3
BASE_DELAY = 1.2

MA_PERIOD = 50
BREAKOUT_LOOKBACK = 20
VOL_LOOKBACK = 20
VOL_SURGE_THRESH = 1.8
BREAKOUT_TOLERANCE = 0.99
MIN_BARS = 60
RSI_PERIOD = 14
RSI_OVERBOUGHT = 72

NYSE_LISTINGS_URL = "https://datahub.io/core/nyse-other-listings/_r/-/data/nyse-listed.csv"

# ─────────────────────────────────────────────────────────────
# PAGE
# ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="Global Alpha AI", layout="wide")
st.title("Global Alpha AI — Rate Safe Scanner")

# ─────────────────────────────────────────────────────────────
# REGISTRY
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400)
def get_nyse_registry():
    try:
        r = requests.get(NYSE_LISTINGS_URL, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")

        if "SYMBOL" not in df.columns:
            return []

        df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip().str.upper()
        df = df[df["SYMBOL"].str.match(r"^[A-Z]{1,5}$", na=False)]

        return df["SYMBOL"].tolist()

    except Exception:
        return []

# ─────────────────────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────────────────────

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else np.nan

# ─────────────────────────────────────────────────────────────
# SAFE DOWNLOAD
# ─────────────────────────────────────────────────────────────

def safe_download(tickers):
    for attempt in range(MAX_RETRIES):
        try:
            data = yf.download(
                tickers,
                period="1y",
                interval="1d",
                group_by="ticker",
                threads=False,  # IMPORTANT
                progress=False,
                auto_adjust=True,
            )

            if data is not None and not data.empty:
                return data

        except Exception:
            pass

        sleep_time = BASE_DELAY * (2 ** attempt) + random.uniform(0.5, 1.5)
        time.sleep(sleep_time)

    return None

# ─────────────────────────────────────────────────────────────
# PROCESS CHUNK
# ─────────────────────────────────────────────────────────────

def process_chunk(tickers):
    results = []

    # jitter to avoid burst
    time.sleep(random.uniform(0.5, 1.5))

    data = safe_download(tickers)
    if data is None:
        return results

    for ticker in tickers:
        try:
            df = data[ticker] if len(tickers) > 1 else data
            df = df.dropna(subset=["Close", "Volume"])

            if len(df) < MIN_BARS:
                continue

            close = df["Close"]
            high = df["High"]
            volume = df["Volume"]

            price = float(close.iloc[-1])
            ma50 = float(close.rolling(MA_PERIOD).mean().iloc[-1])
            high20 = float(high.iloc[-(BREAKOUT_LOOKBACK + 1):-1].max())

            vol_avg = float(volume.iloc[-(VOL_LOOKBACK + 1):-1].mean())
            vol_now = float(volume.iloc[-1])
            vol_ratio = vol_now / vol_avg if vol_avg > 0 else 0

            rsi = compute_rsi(close)

            # filters
            if price <= 1 or price > 100000:
                continue

            if np.isnan(rsi) or rsi < 10 or rsi > 90:
                continue

            is_breakout = price >= high20 * BREAKOUT_TOLERANCE
            is_vol = vol_ratio > VOL_SURGE_THRESH

            if price > ma50 and (is_breakout or is_vol):
                results.append({
                    "Ticker": ticker,
                    "Price": round(price, 2),
                    "RSI": round(rsi, 1),
                    "Vol Ratio": round(vol_ratio, 2),
                    "Signal": "Breakout" if is_breakout else "Building",
                })

        except Exception:
            continue

    return results

# ─────────────────────────────────────────────────────────────
# MAIN SCAN
# ─────────────────────────────────────────────────────────────

def scan_market(symbols):
    if not symbols:
        return pd.DataFrame()

    chunks = [symbols[i:i + CHUNK_SIZE] for i in range(0, len(symbols), CHUNK_SIZE)]

    results = []

    pbar = st.progress(0)
    counter = st.empty()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_chunk, chunk): chunk for chunk in chunks}

        done = 0
        total = len(chunks)

        for future in as_completed(futures):
            try:
                results.extend(future.result())
            except Exception:
                pass

            done += 1
            pct = int(done / total * 100)

            pbar.progress(pct)
            counter.markdown(f"Processed {done}/{total} chunks")

    pbar.empty()
    counter.empty()

    if not results:
        return pd.DataFrame()

    return (
        pd.DataFrame(results)
        .sort_values(["Signal", "Vol Ratio"], ascending=[True, False])
        .reset_index(drop=True)
    )

# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────

if st.button("Run Market Scan"):

    symbols = get_nyse_registry()

    if not symbols:
        st.error("Failed to load NYSE symbols")
        st.stop()

    st.info(f"Scanning {len(symbols)} stocks safely...")

    df = scan_market(symbols)

    if df.empty:
        st.warning("No momentum signals found")
    else:
        st.success(f"Found {len(df)} signals")
        st.dataframe(df, use_container_width=True)