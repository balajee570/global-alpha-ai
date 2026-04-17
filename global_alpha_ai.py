from __future__ import annotations
import time
import random
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

MAX_WORKERS = 4
CHUNK_SIZE = 60

MA_PERIOD = 50
BREAKOUT_LOOKBACK = 20
VOL_LOOKBACK = 20
VOL_SURGE_THRESH = 1.8
BREAKOUT_TOLERANCE = 0.99
MIN_BARS = 60
RSI_PERIOD = 14

# ─────────────────────────────────────────────────────────────
# PAGE
# ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="Global Alpha AI", layout="wide")
st.title("Global Alpha AI")

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
# DOWNLOAD
# ─────────────────────────────────────────────────────────────

def download_data(tickers):
    try:
        data = yf.download(
            tickers,
            period="1y",
            interval="1d",
            group_by="ticker",
            threads=False,
            progress=False,
            auto_adjust=True,
        )
        return data
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────
# PROCESS CHUNK
# ─────────────────────────────────────────────────────────────

def process_chunk(tickers):
    results = []

    time.sleep(random.uniform(0.5, 1.2))  # small spread

    data = download_data(tickers)
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
# FILE UPLOAD
# ─────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload Nasdaq Screener CSV",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        df_raw = pd.read_csv(uploaded_file)
        df_raw.columns = df_raw.columns.str.strip()

        # rename columns
        df = df_raw.rename(columns={
            "Symbol": "Ticker",
            "Last Sale": "Price",
            "Market Cap": "MarketCap"
        })

        # clean numeric columns
        def clean_numeric(series):
            return (
                series.astype(str)
                .str.replace(r"[$,%]", "", regex=True)
                .replace("", np.nan)
                .astype(float)
            )

        if "Price" in df.columns:
            df["Price"] = clean_numeric(df["Price"])

        if "MarketCap" in df.columns:
            df["MarketCap"] = clean_numeric(df["MarketCap"])

        if "Volume" in df.columns:
            df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

        # filter valid tickers
        df = df[df["Ticker"].str.match(r"^[A-Z]{1,5}$", na=False)]

        symbols = df["Ticker"].dropna().unique().tolist()

        st.success(f"Loaded {len(symbols)} stocks")

        # run scan
        if st.button("Run Market Scan"):

            result = scan_market(symbols)

            if result.empty:
                st.warning("No momentum signals found")
            else:
                st.success(f"Found {len(result)} signals")
                st.dataframe(result, use_container_width=True)

    except Exception as e:
        st.error(f"File processing error: {e}")

else:
    st.info("Upload your Nasdaq screener CSV to begin")