"""
NASDAQ AI Integration - Sarvam + Tavily
Generates market intelligence, strategy briefs, and per-stock theses
"""

import streamlit as st
import requests
import re
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone, timedelta
from tavily import TavilyClient

ET = timezone(timedelta(hours=-4))

SARVAM_API_KEY = st.secrets.get("SARVAM_API_KEY", "")
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY", "")
SARVAM_URL = "https://api.sarvam.ai/v1/chat/completions"

tavily = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None


def _safe_sarvam_call(prompt: str, system: str = "") -> str:
    """Safely call Sarvam AI with fallback."""
    if not SARVAM_API_KEY:
        return "AI service not configured (missing SARVAM_API_KEY)."

    try:
        headers = {
            "Authorization": f"Bearer {SARVAM_API_KEY}",
            "Content-Type": "application/json"
        }
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = requests.post(
            SARVAM_URL,
            headers=headers,
            json={
                "model": "sarvam-105b",
                "messages": messages,
                "temperature": 0.5,
                "max_tokens": 2000
            },
            timeout=120
        )

        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        else:
            return f"Sarvam API error ({response.status_code})"
    except Exception as e:
        return f"AI service error: {str(e)}"


@st.cache_data(ttl=1800)  # 30 min cache
def get_live_market_snapshot() -> dict:
    """Fetch live market snapshot: SPY, QQQ, VIX, DXY."""
    snapshot = {}

    for ticker, name in [("SPY", "S&P 500"), ("QQQ", "Nasdaq"), ("^VIX", "VIX"), ("DXY=F", "DXY")]:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="2d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
                chg = (price / float(hist["Close"].iloc[-2]) - 1) * 100 if len(hist) >= 2 else 0
                snapshot[name] = {"price": price, "change": round(chg, 2)}
        except Exception:
            pass

    return snapshot


@st.cache_data(ttl=1800)
def fetch_market_news() -> list:
    """Fetch top market news headlines."""
    if not tavily:
        return []

    try:
        results = tavily.search(query="US stock market today NASDAQ", max_results=5)
        headlines = []
        for r in results.get("results", []):
            title = r.get("title", "")
            if title:
                headlines.append(title)
        return headlines
    except Exception:
        return []


@st.cache_data(ttl=1800)
def fetch_stock_news(ticker: str) -> list:
    """Fetch news for a specific stock."""
    if not tavily:
        return []

    try:
        results = tavily.search(query=f"{ticker} stock news", max_results=3)
        headlines = []
        for r in results.get("results", []):
            title = r.get("title", "")
            if title:
                headlines.append(title)
        return headlines
    except Exception:
        return []


def generate_market_intelligence(regime: dict, top_picks: pd.DataFrame, news: list) -> str:
    """Generate AI market intelligence brief."""
    regime_label = regime.get("label", "SELECTIVE")
    regime_desc = {
        "AGGRESSIVE": "Bullish market conditions with strong momentum",
        "SELECTIVE": "Mixed signals requiring careful stock selection",
        "DEFENSIVE": "Risk-off environment favoring quality names"
    }

    snapshot = get_live_market_snapshot()
    spy_price = snapshot.get("S&P 500", {}).get("price", "N/A")
    spy_chg = snapshot.get("S&P 500", {}).get("change", 0)
    vix = snapshot.get("VIX", {}).get("price", "N/A")

    context = f"""
Today: {datetime.now(ET).strftime("%d %b %Y %H:%M ET")}
Market Regime: {regime_label} — {regime_desc.get(regime_label, "")}
SPY: ${spy_price} ({spy_chg:+.2f}%)
VIX: {vix}

Top Stocks Found: {len(top_picks)}
News: {chr(10).join(news[:3])}

Generate a brief 3-line market intelligence report for US traders. Include outlook and key risks.
"""

    return _safe_sarvam_call(context)


def generate_strategy_brief(top_picks: pd.DataFrame, fund_map: dict, regime: dict) -> str:
    """Generate AI-powered strategy brief with Top 3 picks."""
    if top_picks.empty:
        return "No stocks meet criteria for strategy generation."

    top3 = top_picks.head(3)
    picks_text = ""
    for _, row in top3.iterrows():
        ticker = row["Ticker"]
        score = row.get("Score", 0)
        rec = fund_map.get(ticker, {}).get("recommendation", {})
        action = rec.get("action", "HOLD")
        conviction = rec.get("conviction", 0)
        picks_text += f"\n{ticker}: {action} (Score {score:.0f}, Conviction {conviction:.0f}%)"

    regime_label = regime.get("label", "SELECTIVE")

    prompt = f"""
Market Regime: {regime_label}
Top Picks: {picks_text}

Generate a professional strategy brief (under 200 words) with:
## Market Pulse
(1-2 sentences on current environment)

## Top 3 Buy Picks
(brief thesis for each, 1-2 lines)

## Sector Focus
(Which sectors to favor)

## Key Risks
(1-2 risk factors)
"""

    return _safe_sarvam_call(prompt)


def generate_stock_thesis(ticker: str, fundamentals: dict, technicals: dict, news: list) -> str:
    """Generate AI thesis for a single stock."""
    price = fundamentals.get("price", "N/A")
    piotroski = fundamentals.get("piotroski_score", 0)
    roe = fundamentals.get("roe", 0)
    fcf_yield = fundamentals.get("fcf_yield", 0)
    pe = fundamentals.get("pe_trailing", 0)
    stage = technicals.get("Stage", "N/A")
    rsi = technicals.get("RSI", "N/A")

    news_context = "\n".join([f"• {h}" for h in news[:3]])

    prompt = f"""
Stock: {ticker}
Price: ${price}
P/E: {pe}
ROE: {roe:.1%}
Piotroski: {piotroski}/9
FCF Yield: {fcf_yield:.2f}%
Stage: {stage}
RSI: {rsi}

News:
{news_context}

Write a concise bull/bear thesis (under 150 words) covering:
**Bull Case** (3 reasons to buy)
**Bear Case** (2 risks)
**Catalyst** (What could move it)
"""

    return _safe_sarvam_call(prompt)


@st.cache_data(ttl=1800)
def get_snapshot_summary() -> str:
    """Get cached live market snapshot as text."""
    snapshot = get_live_market_snapshot()
    lines = []
    for name in ["S&P 500", "Nasdaq", "VIX"]:
        if name in snapshot:
            data = snapshot[name]
            price = data["price"]
            chg = data["change"]
            lines.append(f"{name}: {price:.2f} ({chg:+.2f}%)")
    return "\n".join(lines) or "Market data unavailable"
