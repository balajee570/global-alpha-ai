"""
NASDAQ Market Regime Analyzer
- Replaces NSE Nifty-based regime with SPY + VIX + breadth analysis
- Classifies market as AGGRESSIVE / SELECTIVE / DEFENSIVE
"""

import pandas as pd
import numpy as np
import streamlit as st
from nasdaq_data import _yf_ticker

logger = __import__('logging').getLogger(__name__)


@st.cache_data(ttl=3600, show_spinner=False)
def get_spy_technicals() -> dict:
    """Fetch SPY (S&P 500) technical indicators."""
    try:
        spy = _yf_ticker('^GSPC')  # S&P 500
        hist = spy.history(period='1y', interval='1d', auto_adjust=True)

        if hist.empty or len(hist) < 200:
            return {'above_50dma': False, 'above_200dma': False, 'ma50_slope': 0.0}

        close = hist['Close']
        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()
        last = float(close.iloc[-1])

        above_50 = bool(np.isfinite(ma50.iloc[-1]) and last > ma50.iloc[-1])
        above_200 = bool(np.isfinite(ma200.iloc[-1]) and last > ma200.iloc[-1])

        # Slope of MA50 over last 20 days
        if len(ma50.dropna()) >= 21:
            tail = ma50.dropna().iloc[-21:]
            slope = (tail.iloc[-1] / tail.iloc[0] - 1) * 100  # % change
        else:
            slope = 0.0

        return {
            'above_50dma': above_50,
            'above_200dma': above_200,
            'ma50_slope': round(slope, 2),
        }
    except Exception as e:
        logger.debug(f"SPY technicals error: {e}")
        return {'above_50dma': False, 'above_200dma': False, 'ma50_slope': 0.0}


@st.cache_data(ttl=3600, show_spinner=False)
def get_vix_level() -> float:
    """Fetch current VIX level."""
    try:
        vix = _yf_ticker('^VIX')
        hist = vix.history(period='5d', interval='1d')
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except Exception as e:
        logger.debug(f"VIX fetch error: {e}")
    return np.nan


def compute_nasdaq_regime(scan_df: pd.DataFrame | None = None) -> dict:
    """
    Score the US market regime 0-100 from SPY technicals + VIX + breadth.
    AGGRESSIVE ≥70, SELECTIVE 50-69, DEFENSIVE <50.
    """
    out = {
        'spy_above_50dma': False,
        'spy_above_200dma': False,
        'spy_ma50_slope_20d': 0.0,
        'vix_level': np.nan,
        'vix_band': '—',
        'breadth_above_ma50_pct': 0.0,
        'breadth_above_ma200_pct': 0.0,
        'score': 0.0,
        'label': 'DEFENSIVE',
        'multiplier': 0.70,
    }

    # SPY technicals
    spy_tech = get_spy_technicals()
    out.update({
        'spy_above_50dma': spy_tech.get('above_50dma', False),
        'spy_above_200dma': spy_tech.get('above_200dma', False),
        'spy_ma50_slope_20d': spy_tech.get('ma50_slope', 0.0),
    })

    # VIX level
    vix = get_vix_level()
    out['vix_level'] = vix
    if np.isfinite(vix):
        out['vix_band'] = 'Low' if vix < 12 else ('Normal' if vix < 20 else 'High')

    # Breadth (if scan data available)
    if scan_df is not None and not scan_df.empty:
        try:
            # Assume scan_df has 'Dist MA50 %' and 'Dist MA200 %' columns
            if 'Dist MA50 %' in scan_df.columns:
                out['breadth_above_ma50_pct'] = float((scan_df['Dist MA50 %'] > 0).mean() * 100)
            if 'Dist MA200 %' in scan_df.columns:
                out['breadth_above_ma200_pct'] = float((scan_df['Dist MA200 %'] > 0).mean() * 100)
        except Exception:
            pass

    # Composite score (0-100)
    vix_factor = 1.0 if (np.isfinite(vix) and vix < 14) else (
        0.6 if (np.isfinite(vix) and vix < 22) else 0.2
    )
    score = 100 * (
        0.25 * (1.0 if out['spy_above_50dma'] else 0.0) +
        0.25 * (1.0 if out['spy_above_200dma'] else 0.0) +
        0.15 * (1.0 if out['spy_ma50_slope_20d'] > 0 else 0.0) +
        0.15 * vix_factor +
        0.10 * min(out['breadth_above_ma50_pct'] / 100, 1.0) +
        0.10 * min(out['breadth_above_ma200_pct'] / 100, 1.0)
    )

    out['score'] = round(score, 1)
    if score >= 70:
        out['label'] = 'AGGRESSIVE'
        out['multiplier'] = 1.00
    elif score >= 50:
        out['label'] = 'SELECTIVE'
        out['multiplier'] = 0.85
    else:
        out['label'] = 'DEFENSIVE'
        out['multiplier'] = 0.70

    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_sector_etfs_raw(etf_symbols: tuple) -> dict:
    """
    Fetch NASDAQ sector ETF data (XLK, XLV, XLE, etc.).
    Returns dict by symbol with momentum metrics.
    """
    from nasdaq_data import _yf_download

    out = {}
    try:
        data = _yf_download(
            list(etf_symbols), period='6mo', interval='1d',
            group_by='ticker', threads=True, progress=False,
            auto_adjust=True,
        )
    except Exception as e:
        logger.debug(f"ETF fetch error: {e}")
        return out

    for sym in etf_symbols:
        try:
            df = data[sym] if len(etf_symbols) > 1 else data
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna(subset=['Close'])

            if len(df) < 22:
                continue

            close = df['Close']
            last = float(close.iloc[-1])

            # Returns
            ret_1m = float(close.iloc[-1] / close.iloc[-min(22, len(close) - 1) - 1] - 1) * 100 if len(close) > 22 else np.nan
            ret_3m = float(close.iloc[-1] / close.iloc[-min(63, len(close) - 1) - 1] - 1) * 100 if len(close) > 63 else np.nan
            ret_6m = float(close.iloc[-1] / close.iloc[0] - 1) * 100

            # MAs
            ma50 = close.rolling(50).mean()
            ma200 = close.rolling(200).mean()
            above_50 = bool(np.isfinite(ma50.iloc[-1]) and last > ma50.iloc[-1])
            above_200 = bool(ma200 is not None and np.isfinite(ma200.iloc[-1]) and last > ma200.iloc[-1])

            out[sym] = {
                'ret_1m': ret_1m,
                'ret_3m': ret_3m,
                'ret_6m': ret_6m,
                'above_50dma': above_50,
                'above_200dma': above_200,
                'last': last,
            }
        except Exception:
            continue

    return out


def compute_sector_strength(scan_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Compute sector strength scores for NASDAQ sectors (via SPDRs).
    Returns DataFrame with one row per sector + momentum metrics.
    """
    # NASDAQ sector ETFs
    SECTOR_ETFS = {
        'Technology': 'XLK',
        'Healthcare': 'XLV',
        'Financials': 'XLF',
        'Energy': 'XLE',
        'Industrials': 'XLI',
        'Consumer Disc': 'XLY',
        'Consumer Staples': 'XLP',
        'Utilities': 'XLU',
        'Real Estate': 'XLRE',
        'Materials': 'XLB',
        'Comm Services': 'XLC',
    }

    syms = tuple(sorted(set(SECTOR_ETFS.values())))
    raw = fetch_sector_etfs_raw(syms)
    rows = []

    for sector, sym in SECTOR_ETFS.items():
        m = raw.get(sym, {})
        if not m:
            continue

        rows.append({
            'Sector': sector,
            'ETF': sym,
            '1M %': round(m.get('ret_1m', 0), 2) if np.isfinite(m.get('ret_1m', np.nan)) else 0,
            '3M %': round(m.get('ret_3m', 0), 2) if np.isfinite(m.get('ret_3m', np.nan)) else 0,
            '6M %': round(m.get('ret_6m', 0), 2) if np.isfinite(m.get('ret_6m', np.nan)) else 0,
            'Above 50DMA': bool(m.get('above_50dma')),
            'Above 200DMA': bool(m.get('above_200dma')),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Score sectors
    def score_sector(r):
        r1 = r['1M %'] or 0
        r3 = r['3M %'] or 0
        dma = (1.0 if r['Above 50DMA'] else 0) + (1.0 if r['Above 200DMA'] else 0)
        return 100 * (0.4 * np.tanh(r3 / 20) + 0.3 * np.tanh(r1 / 10) + 0.3 * (dma / 2.0))

    df['Score'] = df.apply(score_sector, axis=1).round(1)
    df = df.sort_values('Score', ascending=False).reset_index(drop=True)
    df['Rank'] = df.index + 1
    df['Status'] = df['Rank'].apply(
        lambda r: 'In play' if r <= 5 else ('Watch' if r <= 10 else 'Laggard')
    )

    return df


def top_sectors_set(sector_df: pd.DataFrame, top_n: int = 5) -> set:
    """Return set of top N sectors (for grade boost)."""
    if sector_df is None or sector_df.empty:
        return set()
    return set(sector_df.head(top_n)['Sector'].tolist())
