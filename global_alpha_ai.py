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
    # Mega Cap Tech
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL",
    "ADBE", "CRM", "AMD", "INTC", "QCOM", "TXN", "IBM", "NOW", "PANW", "SNOW",
    "UBER", "PYPL", "SHOP", "CRWD", "DDOG", "PLTR", "NET", "ACN", "HPQ", "DELL",
    # Financials
    "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "BLK", "AXP", "C", 
    "USB", "PNC", "TFC", "COF", "SCHW", "SPGI", "CME", "ICE", "MCO", "KKR", "BX",
    # Healthcare
    "LLY", "JNJ", "UNH", "ABBV", "MRK", "TMO", "ABT", "DHR", "PFE", "AMGN", "BMY",
    "GILD", "REGN", "VRTX", "ISRG", "ZTS", "CVS", "CI", "HUM", "ELV", "SYK", "BDX",
    # Consumer
    "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "MAR", "HLT", "LULU", 
    "WMT", "COST", "PG", "KO", "PEP", "DG", "DLTR", "MDLZ",
    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB", "OXY", "MPC", "VLO", "PSX", "KMI",
    # Industrial
    "GE", "CAT", "HON", "BA", "UPS", "RTX", "LMT", "DE", "ADP", "WM", "CSX", 
    "UNP", "FDX", "NSC", "ITW", "MMM", "EMR", "ETN", "CMI",
    # Communication
    "NFLX", "DIS", "VZ", "T", "CMCSA", "TMUS", "CHTR", "SPOT", "SNAP",
    # Materials
    "LIN", "APD", "SHW", "FCX", "NEM", "ECL", "DOW", "DD", "PPG", "NUE",
    # Utilities
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL",
    # Real Estate
    "AMT", "PLD", "CCI", "EQIX", "PSA", "O", "DLR", "AVB", "SPG", "WELL",
    # Additional Large Caps
    "ZTS", "CCI", "APD", "ECL", "NOC", "ITW", "BAX", "AIG",
    "MET", "PRU", "COF", "USB", "TFC", "BK", "STT", "ALL", "TRV",
    "RF", "CFG", "KEY", "HBAN", "FITB", "ZION",
    # Growth/Momentum
    "ROKU", "SQ", "TWLO", "OKTA", "DOCU", "ZM", "DKNG", "RBLX", "COIN",
    "HO
