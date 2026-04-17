from __future__ import annotations
import re
import streamlit as st
import requests
import io
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone, timedelta
from tavily import TavilyClient

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

ET = timezone(timedelta(hours=-4))  # Eastern Time

SARVAM_API_KEY = st.secrets["SARVAM_API_KEY"]
TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
SARVAM_URL = "https://api.sarvam.ai/v1/chat/completions"

tavily = TavilyClient(api_key=TAVILY_API_KEY)

MA_PERIOD          = 50
BREAKOUT_LOOKBACK  = 20
VOL_LOOKBACK       = 20
VOL_SURGE_THRESH   = 1.8
BREAKOUT_TOLERANCE = 0.99
MIN_BARS           = 60
RSI_PERIOD         = 14
RSI_OVERBOUGHT     = 72

# ─────────────────────────────────────────────────────────────
# PAGE SETUP & STYLING (Same beautiful NSE style)
# ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="Global Alpha AI", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700;900&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@400;600;700&display=swap');
:root {
    --bg-base:     #080c18;
    --gold:        #d4a843;
    --gold-light:  #f0c96a;
    --gold-dim:    rgba(212,168,67,0.18);
    --gold-border: rgba(212,168,67,0.28);
    --blue:        #4f8ef7;
    --emerald:     #00c896;
    --amber:       #f59e0b;
    --red:         #f87171;
    --text-1:      #eef2ff;
    --text-2:      #8896b3;
    --text-3:      #4a5578;
    --border:      rgba(255,255,255,0.07);
}
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; background:var(--bg-base); color:var(--text-1); }
.stApp {
    background: radial-gradient(ellipse 90% 55% at 15% -5%, rgba(79,142,247,0.08) 0%, transparent 55%),
                radial-gradient(ellipse 70% 45% at 85% 105%, rgba(212,168,67,0.07) 0%, transparent 55%),
                #080c18;
}
.nse-header { padding:36px 0 24px; }
.nse-wordmark {
    font-family:'Cinzel',serif; font-weight:900; font-size:2.9rem; letter-spacing:0.07em;
    background:linear-gradient(110deg,#b8873a 0%,#d4a843 25%,#f0c96a 50%,#d4a843 75%,#b8873a 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    animation:goldShimmer 5s linear infinite;
}
@keyframes goldShimmer{0%{background-position:0% center}100%{background-position:250% center}}
.pick-card { background:rgba(15,20,40,0.7); border:1px solid var(--border); border-radius:14px; padding:22px 24px; backdrop-filter:blur(16px); }
.ticker-name { font-family:'JetBrains Mono',monospace; font-size:1.3rem; font-weight:700; }
.upside-pill { background:linear-gradient(135deg,var(--gold),var(--gold-light)); color:#1a1000; font-weight:700; padding:4px 13px; border-radius:99px; }
.strategy-wrap { background:rgba(4,14,10,0.75); border:1px solid rgba(0,200,150,0.1); border-left:2px solid var(--emerald); border-radius:0 14px 14px 0; padding:26px 30px; }
.intel-wrap { background:rgba(8,14,30,0.75); border:1px solid var(--border); border-left:2px solid var(--blue); border-radius:0 12px 12px 0; padding:22px 24px; }
.stButton > button { background:linear-gradient(110deg,#b8873a 0%,#d4a843 35%,#f0c96a 65%,#d4a843 100%); color:#1a1000; font-family:'Cinzel',serif; font-weight:700; }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="nse-header">
    <div style="display:flex; align-items:baseline; gap:14px;">
        <span class="nse-wordmark">GLOBAL ALPHA</span>
        <span style="font-family:'JetBrains Mono',monospace; font-size:0.58rem; font-weight:700; background:linear-gradient(135deg,#d4a843,#f0c96a); color:#080c18; padding:3px 10px; border-radius:3px;">AI</span>
    </div>
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#8896b3; letter-spacing:0.18em;">
        US / Global Momentum Scanner &nbsp;&middot;&nbsp; 
        <em>{datetime.now(ET).strftime("%d %b %Y")}</em> &nbsp;&middot;&nbsp; 
        <em>{datetime.now(ET).strftime("%H:%M ET")}</em>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader("Upload Nasdaq / US Stock Screener CSV", type=["csv"])

if uploaded_file is None:
    st.info("Please upload your stock screener CSV file to begin.")
    st.stop()

# Read and clean uploaded file
df_raw = pd.read_csv(uploaded_file)
df_raw.columns = df_raw.columns.str.strip()

stocks_df = df_raw.rename(columns={
    "Symbol": "SYMBOL",
    "Sector": "Sector",
    "Industry": "Industry"
}).copy()

stocks_df["SYMBOL"] = stocks_df["SYMBOL"].astype(str).str.strip().str.upper()
stocks_df = stocks_df[stocks_df["SYMBOL"].str.match(r"^[A-Z]{1,5}$", na=False)]

symbols = stocks_df["SYMBOL"].tolist()

st.success(f"✅ Loaded **{len(symbols)}** US/Global stocks from uploaded file")

# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def call_ai(prompt: str, system: str = "", max_tokens: int = 6000) -> str:
    headers = {"Authorization": f"Bearer {SARVAM_API_KEY}", "Content-Type": "application/json"}
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "sarvam-105b",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.5,
        "top_p": 1,
    }
    try:
        r = requests.post(SARVAM_URL, headers=headers, json=payload, timeout=180)
        if r.status_code == 200:
            content = r.json()["choices"][0]["message"]["content"]
            return _strip_think(content)
    except Exception:
        pass
    return "AI analysis unavailable at the moment."

def compute_rsi(series: pd.Series) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
    loss = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 1) if not rsi.empty else np.nan

def scan_market(symbols: list) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    
    results = []
    data = yf.download(
        symbols, period="1y", interval="1d",
        group_by="ticker", threads=True, progress=False, auto_adjust=True
    )

    for ticker in symbols:
        try:
            df = data[ticker] if len(symbols) > 1 else data
            df = df.dropna(subset=["Close", "Volume", "High"])

            if len(df) < MIN_BARS:
                continue

            price = float(df["Close"].iloc[-1])
            ma50 = float(df["Close"].rolling(MA_PERIOD).mean().iloc[-1])
            high20 = float(df["High"].iloc[-(BREAKOUT_LOOKBACK + 1):-1].max())

            vol_avg = float(df["Volume"].iloc[-(VOL_LOOKBACK + 1):-1].mean())
            vol_now = float(df["Volume"].iloc[-1])
            vol_ratio = vol_now / vol_avg if vol_avg > 0 else 0

            rsi = compute_rsi(df["Close"])

            is_breakout = price >= high20 * BREAKOUT_TOLERANCE
            is_vol_surge = vol_ratio > VOL_SURGE_THRESH

            if price > ma50 and (is_breakout or is_vol_surge):
                results.append({
                    "Ticker": ticker,
                    "Price": round(price, 2),
                    "RSI": rsi,
                    "Vol Ratio": round(vol_ratio, 2),
                    "Signal": "Breakout" if is_breakout else "Building",
                })
        except:
            continue

    return pd.DataFrame(results)

def get_news() -> str:
    try:
        res = tavily.search(query="US stock market today Nasdaq S&P 500", max_results=6)
        return "\n".join([f"{i+1}. {r.get('title','')}" for i, r in enumerate(res.get("results", []))])
    except:
        return "Latest market news unavailable."

# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────

if st.button("🚀 Run Global Alpha AI", type="primary"):
    with st.spinner("Scanning market for momentum signals..."):
        momentum_df = scan_market(symbols)

    if momentum_df.empty:
        st.error("No momentum signals found in the uploaded list.")
        st.stop()

    # Merge sector information
    momentum_df = momentum_df.merge(
        stocks_df[["SYMBOL", "Sector"]],
        left_on="Ticker",
        right_on="SYMBOL",
        how="left"
    ).drop(columns=["SYMBOL"])

    st.success(f"Found **{len(momentum_df)}** momentum stocks")

    # Display top signals
    st.subheader("📊 Momentum Signals")
    st.dataframe(
        momentum_df.sort_values(["Signal", "Vol Ratio"], ascending=[False, False]).head(30),
        use_container_width=True
    )

    # AI Analysis
    st.subheader("🧠 AI Strategy Brief")

    context = "\n".join([
        f"{r['Ticker']} | {r['Signal']} | RSI {r['RSI']} | Vol {r['Vol Ratio']}x | {r.get('Sector', 'N/A')}"
        for _, r in momentum_df.head(20).iterrows()
    ])

    news = get_news()

    prompt = f"""
Today: {datetime.now(ET).strftime("%d %b %Y, %H:%M ET")}

Latest News:
{news}

Momentum Stocks Detected:
{context}

You are a senior US equity momentum analyst.
Write a professional market brief with:

**MARKET PULSE** (2-3 sentences)

**TOP 3 BUY PICKS**
Use only the stocks above. Make a clean markdown table with columns: Ticker | Signal | RSI | Vol Ratio | Sector | Why (short)

**SECTOR FOCUS**
Which sector looks strongest and why.

**RISKS TO WATCH**
2-3 key risks.

Be concise, professional, and data-driven.
"""

    with st.spinner("Generating AI strategy..."):
        strategy = call_ai(prompt)

    st.markdown("### Strategy Brief")
    st.markdown(strategy if strategy else "AI response unavailable.")

    # Debug option
    if st.checkbox("Show raw momentum data"):
        st.dataframe(momentum_df)

else:
    st.markdown("""
    <div style="text-align:center; padding:80px 20px;">
        <h2 style="color:#d4a843;">Global Alpha AI</h2>
        <p>Upload your Nasdaq / US stock screener CSV and click <strong>Run Global Alpha AI</strong></p>
    </div>
    """, unsafe_allow_html=True)