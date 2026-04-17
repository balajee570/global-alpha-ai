from __future__ import annotations
import re
import streamlit as st
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone, timedelta
from tavily import TavilyClient

ET = timezone(timedelta(hours=-4))

SARVAM_API_KEY = st.secrets["SARVAM_API_KEY"]
TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
SARVAM_URL = "https://api.sarvam.ai/v1/chat/completions"

tavily = TavilyClient(api_key=TAVILY_API_KEY)

# Parameters (same as NSE)
MA_PERIOD = 50
BREAKOUT_LOOKBACK = 20
VOL_LOOKBACK = 20
VOL_SURGE_THRESH = 1.8
BREAKOUT_TOLERANCE = 0.99
MIN_BARS = 60
RSI_PERIOD = 14
RSI_OVERBOUGHT = 72

# ==================== STYLING (same as NSE) ====================
st.set_page_config(page_title="Global Alpha AI", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700;900&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@400;600;700&display=swap');
:root {
    --bg-base: #080c18; --gold: #d4a843; --gold-light: #f0c96a; --emerald: #00c896; --amber: #f59e0b;
}
html, body { font-family:'DM Sans',sans-serif; background:var(--bg-base); color:#eef2ff; }
.stApp { background: var(--bg-base); }
.pick-card { background:rgba(15,20,40,0.8); border:1px solid rgba(212,168,67,0.3); border-radius:14px; padding:20px; margin-bottom:10px; }
.ticker-name { font-family:'JetBrains Mono',monospace; font-size:1.35rem; font-weight:700; }
.upside-pill { background:linear-gradient(135deg,var(--gold),var(--gold-light)); color:#1a1000; padding:5px 14px; border-radius:99px; font-weight:700; }
.strategy-wrap { background:rgba(4,14,10,0.8); border-left:3px solid var(--emerald); padding:25px; border-radius:12px; }
.intel-wrap { background:rgba(8,14,30,0.8); border-left:3px solid #4f8ef7; padding:22px; border-radius:12px; }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="padding:30px 0 20px;">
    <span style="font-family:'Cinzel',serif; font-size:2.8rem; font-weight:900; letter-spacing:0.07em; background:linear-gradient(110deg,#d4a843,#f0c96a); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        GLOBAL ALPHA
    </span>
    <span style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; background:#d4a843; color:#080c18; padding:4px 9px; border-radius:4px; margin-left:12px; vertical-align:middle;">AI</span>
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#8896b3; margin-top:8px;">
        US / Global Momentum Scanner • {datetime.now(ET).strftime("%d %b %Y %H:%M ET")}
    </div>
</div>
""", unsafe_allow_html=True)

# ==================== UPLOAD ====================
uploaded_file = st.file_uploader("Upload Nasdaq / US Stock Screener CSV", type=["csv"])

if uploaded_file is None:
    st.info("Please upload your screener CSV file (must contain Symbol and preferably Sector).")
    st.stop()

df_raw = pd.read_csv(uploaded_file)
df_raw.columns = df_raw.columns.str.strip()
stocks_df = df_raw.rename(columns={"Symbol": "SYMBOL", "Sector": "Sector"}).copy()
stocks_df["SYMBOL"] = stocks_df["SYMBOL"].astype(str).str.strip().str.upper()
stocks_df = stocks_df[stocks_df["SYMBOL"].str.match(r"^[A-Z]{1,5}$", na=False)]

symbols = stocks_df["SYMBOL"].tolist()
st.success(f"✅ Loaded **{len(symbols)}** US/Global stocks")

# ==================== HELPERS (same as NSE) ====================
def call_ai(prompt: str, system: str = "") -> str:
    headers = {"Authorization": f"Bearer {SARVAM_API_KEY}", "Content-Type": "application/json"}
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    try:
        r = requests.post(SARVAM_URL, headers=headers, json={
            "model": "sarvam-105b", "messages": messages, "temperature": 0.5, "max_tokens": 4000
        }, timeout=180)
        if r.status_code == 200:
            content = r.json()["choices"][0]["message"]["content"]
            return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    except:
        return "AI service temporarily unavailable."

def compute_rsi(series):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
    loss = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 1) if not rsi.empty else np.nan

def scan_market(symbols):
    results = []
    data = yf.download(symbols, period="1y", interval="1d", group_by="ticker", threads=True, progress=False, auto_adjust=True)
    for ticker in symbols:
        try:
            df = data[ticker] if len(symbols) > 1 else data
            df = df.dropna(subset=["Close", "High", "Volume"])
            if len(df) < MIN_BARS: continue
            price = float(df["Close"].iloc[-1])
            ma50 = float(df["Close"].rolling(MA_PERIOD).mean().iloc[-1])
            high20 = float(df["High"].iloc[-(BREAKOUT_LOOKBACK+1):-1].max())
            vol_avg = float(df["Volume"].iloc[-(VOL_LOOKBACK+1):-1].mean())
            vol_now = float(df["Volume"].iloc[-1])
            vol_ratio = vol_now / vol_avg if vol_avg > 0 else 0
            rsi = compute_rsi(df["Close"])
            is_breakout = price >= high20 * BREAKOUT_TOLERANCE
            if price > ma50 and (is_breakout or vol_ratio > VOL_SURGE_THRESH):
                results.append({
                    "Ticker": ticker, "Price": round(price,2), "RSI": rsi,
                    "Vol Ratio": round(vol_ratio,2), "Signal": "Breakout" if is_breakout else "Building"
                })
        except:
            continue
    return pd.DataFrame(results)

def get_live_snapshot():
    indices = {"^GSPC": "S&P 500", "^IXIC": "Nasdaq", "^VIX": "VIX"}
    lines = []
    for sym, name in indices.items():
        try:
            t = yf.Ticker(sym)
            hist = t.history(period="2d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
                chg = round((price / float(hist["Close"].iloc[-2]) - 1)*100, 2) if len(hist)>=2 else 0
                lines.append(f"{name}: {price:,.2f} ({chg:+.2f}%)")
        except:
            pass
    return "\n".join(lines) or "Live data unavailable"

# ==================== RUN BUTTON ====================
if st.button("🚀 Run Global Alpha AI", type="primary"):
    st.session_state["last_run"] = datetime.now(ET)

    # Phase 1: Scan
    with st.spinner("Scanning for momentum stocks..."):
        momentum_df = scan_market(symbols)

    if momentum_df.empty:
        st.error("No momentum signals found.")
        st.stop()

    # Merge sector
    momentum_df = momentum_df.merge(stocks_df[["SYMBOL", "Sector"]], left_on="Ticker", right_on="SYMBOL", how="left").drop(columns=["SYMBOL"])

    st.success(f"Found **{len(momentum_df)}** momentum stocks")

    # Show table
    st.subheader("📊 Momentum Signals")
    st.dataframe(momentum_df.sort_values(["Signal", "Vol Ratio"], ascending=[False, False]), use_container_width=True)

    # Phase 2: AI
    st.subheader("🧠 AI Market Intelligence")
    with st.spinner("Fetching news + generating strategy..."):
        live = get_live_snapshot()
        news = "\n".join([r.get("title","") for r in tavily.search(query="US stock market today Nasdaq", max_results=5).get("results", [])])

        intel_prompt = f"Today {datetime.now(ET).strftime('%d %b %Y')}\nLive: {live}\nNews: {news}\nWrite a short US market intelligence brief with ## headers."
        intel = call_ai(intel_prompt)

        strategy_prompt = f"""
Market Context:
{intel}

Momentum Stocks:
{"\n".join([f"{r['Ticker']} {r['Signal']} RSI{r['RSI']} Vol{r['Vol Ratio']}x Sector: {r.get('Sector','N/A')}" for _, r in momentum_df.head(15).iterrows()])}

Write professional strategy:
**MARKET PULSE** (2 sentences)
**TOP 3 BUY PICKS** (markdown table: Ticker | Price | RSI | Signal | Why)
**SECTOR FOCUS**
**RISKS**
"""
        strategy = call_ai(strategy_prompt)

    # Display
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<div class="strategy-wrap">### Strategy Brief\n' + strategy + '</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="intel-wrap">### Market Intelligence\n' + intel + '</div>', unsafe_allow_html=True)

    # Top 3 cards (like NSE)
    if not momentum_df.empty:
        st.markdown("### 🔥 Top Momentum Picks")
        top3 = momentum_df.nlargest(3, "Vol Ratio")
        cols = st.columns(3)
        for i, (_, row) in enumerate(top3.iterrows()):
            with cols[i]:
                sig_color = "color:#00c896" if row["Signal"] == "Breakout" else "color:#f59e0b"
                st.markdown(f"""
                <div class="pick-card">
                    <div class="ticker-name">{row['Ticker']}</div>
                    <div>Rs. {row['Price']} • RSI {row['RSI']}</div>
                    <div style="{sig_color}; font-weight:700;">{row['Signal'].upper()}</div>
                    <div>Vol {row['Vol Ratio']}x</div>
                </div>
                """, unsafe_allow_html=True)

else:
    st.info("Upload CSV → Click **Run Global Alpha AI** to start full analysis.")

st.caption("Global Alpha AI • Powered by Sarvam + Tavily + yfinance")