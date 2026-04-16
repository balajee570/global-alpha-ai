from __future__ import annotations
import re
import streamlit as st
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone, timedelta
from tavily import TavilyClient

# Timezone setup
UTC = timezone.utc

# Initialize API clients with error handling
@st.cache_resource
def get_tavily_client():
    try:
        api_key = st.secrets["TAVILY_API_KEY"]
        return TavilyClient(api_key=api_key)
    except Exception:
        return None

# API Configuration
SARVAM_URL = "https://api.sarvam.ai/v1/chat/completions"

# Technical Analysis Constants
MA_PERIOD = 50
BREAKOUT_LOOKBACK = 20
VOL_LOOKBACK = 20
VOL_SURGE_THRESH = 1.8
BREAKOUT_TOLERANCE = 0.99
MIN_BARS = 60
CHUNK_SIZE = 80
RSI_PERIOD = 14
RSI_OVERBOUGHT = 72

# Major US Stocks - Static list (works reliably on Streamlit Cloud)
MAJOR_US_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL",
    "ADBE", "CRM", "AMD", "INTC", "QCOM", "TXN", "IBM", "NOW", "PANW", "SNOW",
    "UBER", "PYPL", "SHOP", "CRWD", "DDOG", "PLTR", "NET", "ACN", "HPQ", "DELL",
    "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "BLK", "AXP", "C", 
    "USB", "PNC", "TFC", "COF", "SCHW", "SPGI", "CME", "ICE", "MCO", "KKR", "BX",
    "LLY", "JNJ", "UNH", "ABBV", "MRK", "TMO", "ABT", "DHR", "PFE", "AMGN", "BMY",
    "GILD", "REGN", "VRTX", "ISRG", "ZTS", "CVS", "CI", "HUM", "ELV", "SYK", "BDX",
    "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "MAR", "HLT", "LULU", 
    "WMT", "COST", "PG", "KO", "PEP", "DG", "DLTR", "MDLZ",
    "XOM", "CVX", "COP", "EOG", "SLB", "OXY", "MPC", "VLO", "PSX", "KMI",
    "GE", "CAT", "HON", "BA", "UPS", "RTX", "LMT", "DE", "ADP", "WM", "CSX", 
    "UNP", "FDX", "NSC", "ITW", "MMM", "EMR", "ETN", "CMI",
    "NFLX", "DIS", "VZ", "T", "CMCSA", "TMUS", "CHTR", "SPOT", "SNAP",
    "LIN", "APD", "SHW", "FCX", "NEM", "ECL", "DOW", "DD", "PPG", "NUE",
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL",
    "AMT", "PLD", "CCI", "EQIX", "PSA", "O", "DLR", "AVB", "SPG", "WELL",
    "ZTS", "CCI", "APD", "ECL", "NOC", "ITW", "BAX", "AIG",
    "MET", "PRU", "COF", "USB", "TFC", "BK", "STT", "ALL", "TRV",
    "RF", "CFG", "KEY", "HBAN", "FITB", "ZION",
    "ROKU", "SQ", "TWLO", "OKTA", "DOCU", "ZM", "DKNG", "RBLX", "COIN",
    "HOOD", "PLTR", "SOFI", "LCID", "RIVN", "TSM", "ASML", "AMAT", "LRCX", "KLAC",
]

@st.cache_data(ttl=3600)
def get_global_stocks() -> pd.DataFrame:
    stocks = []
    for symbol in list(set(MAJOR_US_STOCKS)):
        stocks.append({
            "SYMBOL": symbol,
            "NAME": symbol,
            "EXCHANGE": "US",
            "YF_SYMBOL": symbol,
        })
    return pd.DataFrame(stocks).drop_duplicates("SYMBOL").reset_index(drop=True)

# Global Sector Mapping
SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology", "GOOG": "Technology",
    "AMZN": "Technology", "NVDA": "Technology", "META": "Technology", "TSLA": "Technology",
    "AVGO": "Technology", "ORCL": "Technology", "ADBE": "Technology", "CRM": "Technology",
    "AMD": "Technology", "INTC": "Technology", "QCOM": "Technology", "TXN": "Technology",
    "IBM": "Technology", "NOW": "Technology", "PANW": "Technology", "SNOW": "Technology",
    "UBER": "Technology", "PYPL": "Technology", "SHOP": "Technology", "NET": "Technology",
    "CRWD": "Technology", "DDOG": "Technology", "PLTR": "Technology",
    "CDNS": "Technology", "ANSS": "Technology", "FTNT": "Technology", "ADSK": "Technology",
    "ROP": "Technology", "FICO": "Technology", "EPAM": "Technology", "IT": "Technology",
    "ACN": "Technology", "HPQ": "Technology", "DELL": "Technology",
    "BRK-B": "Financial Services", "JPM": "Financial Services", "V": "Financial Services",
    "MA": "Financial Services", "BAC": "Financial Services", "WFC": "Financial Services",
    "GS": "Financial Services", "MS": "Financial Services", "BLK": "Financial Services",
    "AXP": "Financial Services", "C": "Financial Services", "USB": "Financial Services",
    "PNC": "Financial Services", "TFC": "Financial Services", "COF": "Financial Services",
    "SCHW": "Financial Services", "SPGI": "Financial Services", "CME": "Financial Services",
    "ICE": "Financial Services", "MCO": "Financial Services", "KKR": "Financial Services",
    "BX": "Financial Services", "APO": "Financial Services", "CG": "Financial Services",
    "LLY": "Healthcare", "JNJ": "Healthcare", "UNH": "Healthcare", "ABBV": "Healthcare",
    "MRK": "Healthcare", "TMO": "Healthcare", "ABT": "Healthcare", "DHR": "Healthcare",
    "PFE": "Healthcare", "AMGN": "Healthcare", "BMY": "Healthcare", "GILD": "Healthcare",
    "REGN": "Healthcare", "VRTX": "Healthcare", "ISRG": "Healthcare", "ZTS": "Healthcare",
    "HD": "Consumer Cyclical", "MCD": "Consumer Cyclical", "NKE": "Consumer Cyclical",
    "LOW": "Consumer Cyclical", "SBUX": "Consumer Cyclical", "TJX": "Consumer Cyclical",
    "BKNG": "Consumer Cyclical", "MAR": "Consumer Cyclical", "HLT": "Consumer Cyclical",
    "NFLX": "Communication Services", "DIS": "Communication Services", 
    "VZ": "Communication Services", "T": "Communication Services",
    "CMCSA": "Communication Services", "TMUS": "Communication Services",
    "GE": "Industrials", "CAT": "Industrials", "HON": "Industrials", "BA": "Industrials",
    "UPS": "Industrials", "RTX": "Industrials", "LMT": "Industrials",
    "WMT": "Consumer Defensive", "COST": "Consumer Defensive", "PG": "Consumer Defensive",
    "KO": "Consumer Defensive", "PEP": "Consumer Defensive",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    "AMT": "Real Estate", "PLD": "Real Estate",
    "LIN": "Materials", "APD": "Materials", "SHW": "Materials",
}

st.set_page_config(page_title="Global Alpha AI", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.sig-breakout { background: linear-gradient(135deg, #00c89620, #00c89610); border-left: 3px solid #00c896; }
.sig-building { background: linear-gradient(135deg, #f59e0b20, #f59e0b10); border-left: 3px solid #f59e0b; }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="text-align: center; padding: 1.5rem 0; border-bottom: 1px solid #2d2d44; margin-bottom: 1rem;">
    <h1 style="margin: 0; font-size: 2.2rem; background: linear-gradient(90deg, #00c896, #00a8e8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        Global Alpha AI
    </h1>
    <p style="margin: 0.5rem 0 0 0; color: #888; font-size: 0.9rem;">
        Global Market Intelligence  ·
        <em>{datetime.now(UTC).strftime("%d %b %Y")}</em>  ·
        <em>{datetime.now(UTC).strftime("%H:%M UTC")}</em>
    </p>
</div>
""", unsafe_allow_html=True)

def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def call_ai(prompt: str, system: str = "", max_tokens: int = 6000) -> str:
    try:
        api_key = st.secrets["SARVAM_API_KEY"]
    except Exception:
        return "Error: SARVAM_API_KEY not configured in Streamlit secrets."
    
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    debug_log = st.session_state.get("ai_debug", [])
    
    payload = {
        "model": "sarvam-105b",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.5,
        "top_p": 1,
    }
    
    try:
        r = requests.post(SARVAM_URL, headers=headers, json=payload, timeout=240)
        entry = {"model": "sarvam-105b", "status": r.status_code, "body": r.text[:1200]}
        debug_log.append(entry)
        st.session_state["ai_debug"] = debug_log
        
        if r.status_code == 200:
            choices = r.json().get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                stripped = _strip_think(str(content))
                if stripped:
                    return stripped
        else:
            return f"Error: Sarvam API returned status {r.status_code}"
    except requests.exceptions.Timeout:
        debug_log.append({"model": "sarvam-105b", "error": "Timed out after 240s"})
        st.session_state["ai_debug"] = debug_log
        return "Error: Sarvam API timeout"
    except Exception as e:
        debug_log.append({"model": "sarvam-105b", "error": str(e)})
        st.session_state["ai_debug"] = debug_log
        return f"Error: {str(e)}"
    return ""

def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 1) if not rsi.empty else np.nan

def estimate_upside(df: pd.DataFrame, price: float, high20: float) -> dict:
    high52 = df["High"].iloc[-252:].max() if len(df) >= 252 else df["High"].max()
    low52 = df["Low"].iloc[-252:].min() if len(df) >= 252 else df["Low"].min()
    
    high, low, cp = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([(high - low), (high - cp).abs(), (low - cp).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    atr_target = price + 2 * (atr / price * price)
    
    fib_target = low52 + (high52 - low52) * 1.618
    targets = [t for t in [high52, atr_target, fib_target] if t > price * 1.02]
    target = round(min(targets), 2) if targets else round(price * 1.12, 2)
    
    return {
        "target": target,
        "upside_pct": round((target - price) / price * 100, 1),
        "high52": round(high52, 2),
        "high52_gap_pct": round((high52 - price) / price * 100, 1),
    }

def scan_global_market(symbols: list, _counter=None, _pbar=None) -> pd.DataFrame:
    results = []
    failed = []
    total = len(symbols)
    
    if _pbar is None:
        _pbar = st.progress(0)
    if _counter is None:
        _counter = st.empty()
    
    for i in range(0, total, CHUNK_SIZE):
        tickers = symbols[i:i + CHUNK_SIZE]
        current = min(i + CHUNK_SIZE, total)
        
        _pbar.progress(int(current / total * 100))
        _counter.markdown(
            f"<div style='text-align: center; color: #00c896;'>"
            f"Scanning Global Equities  <strong>{current}</strong> / {total}</div>",
            unsafe_allow_html=True,
        )
        
        try:
            data = yf.download(
                tickers, period="1y", interval="1d",
                group_by="ticker", threads=True, progress=False, auto_adjust=True,
            )
        except Exception:
            failed.extend([(t, "download_failed") for t in tickers])
            continue
        
        for ticker in tickers:
            try:
                df = data[ticker] if len(tickers) > 1 else data
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                df = df.dropna(subset=["Close", "Volume"])
                if len(df) < MIN_BARS:
                    continue
                
                price = float(df["Close"].iloc[-1])
                if price <= 0.5:
                    continue
                    
                ma50 = float(df["Close"].rolling(MA_PERIOD).mean().iloc[-1])
                high20 = float(df["High"].iloc[-(BREAKOUT_LOOKBACK + 1):-1].max())
                
                vol_avg = float(df["Volume"].iloc[-(VOL_LOOKBACK + 1):-1].mean())
                vol_now = float(df["Volume"].iloc[-1])
                vol_ratio = vol_now / vol_avg if vol_avg > 0 else 0
                
                rsi = compute_rsi(df["Close"])
                upside = estimate_upside(df, price, high20)
                
                is_breakout = price >= high20 * BREAKOUT_TOLERANCE
                is_vol_surge = vol_ratio > VOL_SURGE_THRESH
                
                if price > ma50 and (is_breakout or is_vol_surge):
                    results.append({
                        "Ticker": ticker,
                        "Price": round(price, 2),
                        "MA50": round(ma50, 2),
                        "RSI": rsi,
                        "Vol Ratio": round(vol_ratio, 2),
                        "Signal": "Breakout" if is_breakout else "Building",
                        "Target": upside["target"],
                        "Upside %": upside["upside_pct"],
                        "52W High": upside["high52"],
                        "Gap to 52W %": upside["high52_gap_pct"],
                    })
            except Exception:
                failed.append((ticker, "processing_error"))
    
    if _pbar:
        _pbar.empty()
    if _counter:
        _counter.empty()
    
    if failed:
        st.caption(f"Skipped {len(failed)} tickers")
    
    if not results:
        return pd.DataFrame()
    
    return (
        pd.DataFrame(results)
        .sort_values(["Signal", "Vol Ratio"], ascending=[True, False])
        .reset_index(drop=True)
    )

def get_live_global_snapshot() -> str:
    symbols = {
        "^GSPC": "S&P 500",
        "^IXIC": "NASDAQ Composite",
        "^DJI": "Dow Jones",
        "^VIX": "VIX",
        "EURUSD=X": "EUR/USD",
        "GC=F": "Gold (USD/oz)",
        "CL=F": "WTI Crude (USD/bbl)",
        "^TNX": "US 10Y Treasury",
    }
    
    lines = []
    for sym, label in symbols.items():
        try:
            t = yf.Ticker(sym)
            hist = t.history(period="2d", interval="1d")
            if hist.empty:
                continue
            price = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else price
            chg = round((price / prev - 1) * 100, 2)
            lines.append(f"- {label}: {price:,.2f} ({chg:+.2f}%)")
        except Exception:
            continue
    
    return "\n".join(lines) if lines else "Live data unavailable."

def get_raw_global_news() -> str:
    tavily_client = get_tavily_client()
    if tavily_client is None:
        return "Tavily API not configured."
    
    today = datetime.now(UTC).strftime("%d %B %Y")
    query = (
        f"Global stock market S&P 500 NASDAQ Dow Jones {today} "
        f"Fed interest rate FOMC inflation earnings "
        f"AI semiconductor oil gold dollar"
    )
    
    lines = []
    try:
        result = tavily_client.search(query=query, max_results=8)
        for i, r in enumerate(result.get("results", []), 1):
            title = r.get("title", "").strip()
            content = r.get("content", "").strip()
            if not title:
                continue
            lines.append(f"{i}. {title}\n   {content[:300]}")
    except Exception as e:
        lines.append(f"News fetch error: {e}")
    
    return "\n\n".join(lines) if lines else "No news retrieved."

def generate_global_intel_summary(raw_news: str, live_snapshot: str) -> str:
    system = (
        "You are a senior global market analyst. "
        "Be specific with numbers. Do not reproduce headlines."
    )
    
    prompt = f"""Today: {datetime.now(UTC).strftime("%d %B %Y, %H:%M UTC")}

LIVE MARKET DATA:
{live_snapshot}

NEWS CONTEXT:
{raw_news}

Write a Global Market Intelligence Brief:

## US Markets and Fed Policy
Current S&P 500, NASDAQ, Dow levels. Fed stance, rate expectations.

## Global Macro
Dollar strength, oil prices, gold, central bank moves.

## Sector Rotation
Which sectors are leading/lagging.

## Key Market Triggers This Week
3-5 bullets. Most important events.

## Risks to Watch
2-3 bullets. Biggest near-term risks.

Keep each section to 3-4 sentences.
"""
    
    result = call_ai(prompt, system=system, max_tokens=6000)
    return result if result.strip() else "Market intelligence could not be generated."

def build_global_sector_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    
    rows = []
    for _, r in df.iterrows():
        sector = SECTOR_MAP.get(r["Ticker"])
        if sector is None:
            continue
        rows.append({
            "Sector": sector,
            "Signal": r["Signal"],
            "Ticker": r["Ticker"],
            "Upside %": r["Upside %"]
        })
    
    if not rows:
        return pd.DataFrame()
    
    agg = pd.DataFrame(rows)
    summary = (
        agg.groupby("Sector")
        .agg(
            Breakouts=("Signal", lambda x: (x == "Breakout").sum()),
            Building=("Signal", lambda x: (x == "Building").sum()),
            Avg_Upside=("Upside %", "mean"),
            Top_Tickers=("Ticker", lambda x: ", ".join(x.tolist()[:4])),
        )
        .reset_index()
    )
    
    summary["Total Signals"] = summary["Breakouts"] + summary["Building"]
    summary["Avg Upside %"] = summary["Avg_Upside"].round(1)
    
    return (
        summary[["Sector", "Total Signals", "Breakouts", "Building", "Avg Upside %", "Top_Tickers"]]
        .sort_values("Total Signals", ascending=False)
        .reset_index(drop=True)
    )

def build_global_sector_context(sector_df: pd.DataFrame) -> str:
    if sector_df.empty:
        return "No sector data from scan."
    return "\n".join(
        f"- {r['Sector']}: {r['Total Signals']} signals "
        f"({r['Breakouts']} breakouts, {r['Building']} building) | "
        f"avg upside {r['Avg Upside %']}%"
        for _, r in sector_df.head(6).iterrows()
    )

def build_momentum_context(df: pd.DataFrame) -> str:
    if df.empty:
        return "No momentum candidates identified."
    return "\n".join(
        f"{r['Ticker']} ${r['Price']:.2f} {r['Signal']} RSI{r['RSI']:.1f} "
        f"Vol{r['Vol Ratio']:.1f}x Target ${r['Target']:.2f}(+{r['Upside %']:.1f}%)"
        for _, r in df.head(20).iterrows()
    )

def condense_intel(intel_summary: str) -> str:
    key_lines = []
    for line in intel_summary.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if any(c in line for c in ["%", "*", "-"]) or \
           any(k in line.lower() for k in ["s&p", "nasdaq", "dow", "fed", "rate", "oil", "gold", "dollar", "inflation"]):
            key_lines.append(line.lstrip("*- "))
    return "\n".join(key_lines[:12]) if key_lines else intel_summary[:400]

def generate_global_strategy(intel_summary: str, momentum_context: str, sector_context: str) -> str:
    macro_facts = condense_intel(intel_summary)
    stocks_short = "\n".join([l for l in momentum_context.splitlines() if l.strip()][:12])
    sector_short = "\n".join([l for l in sector_context.splitlines() if l.strip()][:4])
    
    system = "Global equity analyst. Cite both technical AND macro reason per pick. Only use tickers from STOCKS. TOP 3 are not AVOID."
    
    prompt = f"""{datetime.now(UTC).strftime("%d %b %Y")}
MACRO:{macro_facts}
SECTORS:{sector_short}
STOCK SCAN (use ONLY these tickers for TOP 3 BUY PICKS):{stocks_short}

Write markdown brief:
**MARKET PULSE** 2 sentences.

**TOP 3 BUY PICKS** - Pick ONLY from STOCK SCAN above.
RSI<={RSI_OVERBOUGHT}, breakout signal, macro tailwind. Why column: max 10 words.
|Ticker|Entry|Target|Upside|Why|Stop|Conviction|Horizon|
|------|-----|------|------|---|----|----------|-------|

**SECTOR TO OWN** 1 sector, 2 sentences.

**WATCHLIST** 3 stocks from STOCK SCAN.

**AVOID** 2 bullets.

**TAIL RISKS** 2 bullets."""
    
    result = call_ai(prompt, system=system, max_tokens=6000)
    if result.strip():
        return result
    
    return "Strategy unavailable - API timeout. Scan data above is valid."

def best_upside_picks(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    
    d = df.copy()
    d = d[d["RSI"] <= RSI_OVERBOUGHT] if "RSI" in d.columns else d
    if d.empty:
        d = df.copy()
    
    d["_sig_w"] = d["Signal"].map({"Breakout": 1.4, "Building": 1.0}).fillna(1.0)
    
    def rsi_score(r):
        if 40 <= r <= 65:
            return 1.3
        if 30 <= r < 40:
            return 1.0
        if 65 < r <= 72:
            return 0.8
        return 0.6
    
    d["_rsi_w"] = d["RSI"].apply(rsi_score) if "RSI" in d.columns else 1.0
    d["_vol_w"] = d["Vol Ratio"].clip(upper=3.0) if "Vol Ratio" in d.columns else 1.0
    d["_score"] = d["Upside %"] * d["_sig_w"] * d["_rsi_w"] * d["_vol_w"]
    
    return (
        d.nlargest(n, "_score")[["Ticker", "Price", "Target", "Upside %", "Signal", "RSI"]]
        .reset_index(drop=True)
    )

# Check secrets configuration
try:
    _ = st.secrets["SARVAM_API_KEY"]
    _ = st.secrets["TAVILY_API_KEY"]
    secrets_ok = True
except Exception:
    secrets_ok = False
    st.error("""
    ⚠️ **API Keys Not Configured!**
    
    Please add these secrets in your Streamlit Cloud dashboard:
    
    ```toml
    SARVAM_API_KEY = "your_sarvam_key_here"
    TAVILY_API_KEY = "your_tavily_key_here"
    ```
    """)

# UI Layout
col_btn, col_note = st.columns([1, 3])
with col_btn:
    run = st.button("Run Global Market Intelligence", disabled=not secrets_ok)
with col_note:
    st.markdown(
        "<span style='color: #888; font-size: 0.85rem;'>"
        "~200 Major US Stocks  · AI news + strategy  · ~3-5 min</span>",
        unsafe_allow_html=True,
    )

if run and secrets_ok:
    if "last_run" in st.session_state:
        elapsed = (datetime.now(UTC) - st.session_state["last_run"]).seconds
        if elapsed < 120:
            st.warning(f"Please wait {120 - elapsed}s before re-running.")
            st.stop()
    
    st.session_state["last_run"] = datetime.now(UTC)
    
    st.markdown("<h3>Phase 1 - Market Data</h3>", unsafe_allow_html=True)
    
    _eq_status = st.empty()
    _eq_status.markdown("Loading stock registry ...", unsafe_allow_html=True)
    
    registry = get_global_stocks()
    
    if registry.empty:
        st.error("Could not load stock registry.")
        st.stop()
    
    _eq_status.empty()
    
    eq_symbols = registry["YF_SYMBOL"].tolist()
    n_eq_total = len(eq_symbols)
    
    _eq_counter = st.empty()
    _eq_pbar = st.progress(0)
    momentum_df = scan_global_market(eq_symbols, _eq_counter, _eq_pbar)
    _eq_counter.empty()
    _eq_pbar.empty()
    
    st.markdown(
        f"<div style='display: flex; gap: 2rem; padding: 1rem; background: #1a1a2e; border-radius: 8px; margin: 1rem 0;'>"
        f"<div>Total Stocks: <strong>{n_eq_total:,}</strong></div>"
        f"<div>Momentum Signals: <strong>{len(momentum_df)}</strong></div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    
    st.markdown("<h3>Phase 2 - AI Analysis</h3>", unsafe_allow_html=True)
    _ai_status = st.empty()
    _ai_pbar = st.progress(0)
    
    def _ai(msg: str, pct: int):
        _ai_status.markdown(f"<div style='color: #00c896;'>{msg}</div>", unsafe_allow_html=True)
        _ai_pbar.progress(pct)
    
    _ai("Fetching live market prices ...", 10)
    live_snapshot = get_live_global_snapshot()
    
    _ai("Pulling latest news ...", 25)
    raw_news = get_raw_global_news()
    
    _ai("Analysing market intelligence ...", 40)
    intel_summary = generate_global_intel_summary(raw_news, live_snapshot)
    
    _ai("Building strategy brief ...", 70)
    momentum_context = build_momentum_context(momentum_df)
    sector_df = build_global_sector_heatmap(momentum_df)
    sector_context = build_global_sector_context(sector_df)
    strategy = generate_global_strategy(intel_summary, momentum_context, sector_context)
    
    _ai_status.empty()
    _ai_pbar.empty()
    
    st.session_state.update({
        "momentum_df": momentum_df,
        "sector_df": sector_df,
        "intel_summary": intel_summary,
        "strategy": strategy,
        "registry": registry,
    })

if "strategy" in st.session_state:
    momentum_df = st.session_state["momentum_df"]
    intel_summary = st.session_state.get("intel_summary", "")
    strategy = st.session_state["strategy"]
    
    if not momentum_df.empty:
        k1, k2, k3, k4 = st.columns(4)
        breakouts = int((momentum_df["Signal"] == "Breakout").sum())
        building = int((momentum_df["Signal"] == "Building").sum())
        
        k1.metric("Momentum Stocks", f"{len(momentum_df)}")
        k2.metric("Breakouts", f"{breakouts}", f"+{building} building")
        k3.metric("Avg Upside Target", f"{momentum_df['Upside %'].mean():.1f}%")
        k4.metric("Best Upside Found", f"{momentum_df['Upside %'].max():.1f}%")
    
    if not momentum_df.empty:
        st.markdown("<h3>Scanner - Highest Upside Potential</h3>", unsafe_allow_html=True)
        st.caption("Top 3 by composite score with RSI not overbought.")
        top3 = best_upside_picks(momentum_df, n=3)
        cols = st.columns(3)
        
        for idx, (_, row) in enumerate(top3.iterrows()):
            sig_cls = "sig-breakout" if row["Signal"] == "Breakout" else "sig-building"
            with cols[idx]:
                st.markdown(f"""
                <div class="{sig_cls}" style="padding: 1rem; border-radius: 8px;">
                    <div style="font-size: 1.5rem; font-weight: bold; color: #fff;">{row['Ticker']}</div>
                    <div style="color: #888;">${row['Price']:.2f}  RSI {row['RSI']:.0f}</div>
                    <div style="font-size: 1.3rem; color: #00c896; margin: 0.5rem 0;">+{row['Upside %']:.1f}%</div>
                    <div style="color: #888; font-size: 0.9rem;">Target  ${row['Target']:.2f}</div>
                    <div style="margin-top: 0.5rem; padding: 0.25rem 0.5rem; background: {'#00c89630' if row['Signal'] == 'Breakout' else '#f59e0b30'}; 
                         border-radius: 4px; display: inline-block; font-size: 0.8rem;">
                        {row['Signal'].upper()}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 2rem 0; border-color: #2d2d44;'>", unsafe_allow_html=True)
    
    col_l, col_r = st.columns([3, 2])
    
    with col_l:
        st.markdown("<h3>AI Strategy Brief</h3>", unsafe_allow_html=True)
        if strategy and not strategy.startswith("Error"):
            st.markdown(f"<div style='background: #1a1a2e; padding: 1.5rem; border-radius: 8px; border: 1px solid #2d2d44;'>\n\n{strategy}\n\n</div>", unsafe_allow_html=True)
        else:
            st.error(strategy if strategy else "Strategy generation failed.")
    
    with col_r:
        st.markdown("<h3>Global Market Intelligence</h3>", unsafe_allow_html=True)
        scan_time = st.session_state["last_run"].strftime("%d %b %Y - %H:%M UTC")
        st.markdown(
            f"<div style='background: #1a1a2e; padding: 1rem; border-radius: 8px; border: 1px solid #2d2d44;'>"
            f"<div style='color: #00c896; font-size: 0.85rem; margin-bottom: 0.5rem;'>AI Analysis - {scan_time}</div>"
            f"\n\n{intel_summary if intel_summary else ''}\n\n"
            f"</div>",
            unsafe_allow_html=True,
        )
    
    st.markdown("<h3>Full Momentum Candidates</h3>", unsafe_allow_html=True)
    if not momentum_df.empty:
        def color_signal(val):
            return "color:#00c896;font-weight:700" if val == "Breakout" else "color:#f59e0b;font-weight:600"
        
        def color_upside(val):
            if val >= 20:
                return "color:#00c896;font-weight:700"
            if val >= 10:
                return "color:#d4a843"
            return ""
        
        def color_rsi_stock(val):
            return "color:#f87171;font-weight:700" if val > RSI_OVERBOUGHT else ""
        
        st.dataframe(
            momentum_df.style
            .map(color_signal, subset=["Signal"])
            .map(color_upside, subset=["Upside %"])
            .map(color_rsi_stock, subset=["RSI"])
            .format({
                "Price": "${:.2f}",
                "MA50": "${:.2f}",
                "Target": "${:.2f}",
                "52W High": "${:.2f}",
                "RSI": "{:.1f}",
                "Vol Ratio": "{:.2f}x",
                "Upside %": "+{:.1f}%",
                "Gap to 52W %": "{:.1f}%",
            }),
            use_container_width=True,
            height=420,
        )
        st.caption(f"RSI > {RSI_OVERBOUGHT} highlighted red. All {len(momentum_df)} stocks above MA50 with breakout or volume signal.")
    else:
        st.info("No momentum candidates found in this scan.")
    
    if st.session_state.get("ai_debug"):
        with st.expander("AI Debug Log"):
            st.json(st.session_state["ai_debug"])
    
    st.markdown(
        f"<div style='text-align: center; color: #666; font-size: 0.8rem; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #2d2d44;'>"
        f"Last scan - {st.session_state['last_run'].strftime('%H:%M:%S UTC')} - Global Alpha AI"
        f"</div>",
        unsafe_allow_html=True,
    )

elif not secrets_ok:
    st.info("👆 Please configure your API keys in Streamlit Cloud secrets to use this app.")
else:
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; color: #666;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">🌍</div>
        <h2 style="color: #888;">Global Alpha AI</h2>
        <p>Hit <strong>Run Global Market Intelligence</strong> to begin</p>
        <p style="font-size: 0.85rem; color: #555;">
            ~200 Major US Stocks  · Live prices  · AI strategy
        </p>
    </div>
    """, unsafe_allow_html=True)
