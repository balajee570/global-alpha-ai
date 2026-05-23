"""
NASDAQ Technical Analysis
- All indicators: RSI, ADX, MACD, OBV, Bollinger Bands, pattern detection
- 4-stage rally classifier (Accumulation → Early Markup → Breakout → Extended)
- Adapted for US market volatility (slightly higher ATR %)
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Constants (same as NSE, proven to work across markets)
MA_PERIOD = 50
MA_LONG_PERIOD = 200
BREAKOUT_LOOKBACK = 20
VOL_LOOKBACK = 20
VOL_SURGE_THRESH = 1.8
BREAKOUT_TOLERANCE = 0.99
MIN_BARS = 60
RSI_PERIOD = 14
RSI_OVERBOUGHT = 72  # Stocks above this excluded from Top 3

STAGE_LABELS = {
    0: "—",
    1: "Accumulation",
    2: "Early Markup",
    3: "Breakout",
    4: "Extended",
}
STAGE_WEIGHTS = {0: 0.20, 1: 0.85, 2: 1.00, 3: 0.75, 4: 0.15}


# ═══════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ═══════════════════════════════════════════════════════════════════════

def _safe_last(series: pd.Series) -> float:
    """Get the last valid value from a series."""
    try:
        v = float(series.dropna().iloc[-1])
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> float:
    """Relative Strength Index."""
    try:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return round(float(rsi.iloc[-1]), 1) if not rsi.empty else np.nan
    except Exception:
        return np.nan


def compute_atr_pct(df: pd.DataFrame, period: int = 14) -> float:
    """Average True Range as % of close."""
    try:
        high, low, cp = df['High'], df['Low'], df['Close'].shift(1)
        tr = pd.concat([(high - low), (high - cp).abs(), (low - cp).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        price = df['Close'].iloc[-1]
        return round(float(atr / price * 100), 2) if price else np.nan
    except Exception:
        return np.nan


def compute_ma(series: pd.Series, period: int) -> float:
    """Simple Moving Average."""
    try:
        if len(series) < period:
            return np.nan
        return round(float(series.rolling(period).mean().iloc[-1]), 2)
    except Exception:
        return np.nan


def compute_ema_series(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average series."""
    try:
        return series.ewm(span=period, adjust=False).mean()
    except Exception:
        return pd.Series(dtype=float)


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """MACD: returns (macd_last, signal_last, hist_last, hist_series)."""
    try:
        if len(series) < slow + signal:
            return np.nan, np.nan, np.nan, pd.Series(dtype=float)
        ema_fast = compute_ema_series(series, fast)
        ema_slow = compute_ema_series(series, slow)
        macd = ema_fast - ema_slow
        sig = compute_ema_series(macd, signal)
        hist = macd - sig
        return _safe_last(macd), _safe_last(sig), _safe_last(hist), hist
    except Exception:
        return np.nan, np.nan, np.nan, pd.Series(dtype=float)


def compute_adx(df: pd.DataFrame, period: int = 14) -> tuple:
    """Wilder ADX. Returns (adx_last, plus_di_last, minus_di_last)."""
    try:
        if len(df) < period * 2 + 1:
            return np.nan, np.nan, np.nan
        high, low, close = df['High'], df['Low'], df['Close']
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        tr = pd.concat([
            (high - low),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        atr_w = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr_w.replace(0, np.nan)
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr_w.replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        return _safe_last(adx), _safe_last(plus_di), _safe_last(minus_di)
    except Exception:
        return np.nan, np.nan, np.nan


def compute_adx_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ADX series (full history)."""
    try:
        if len(df) < period * 2 + 1:
            return pd.Series(dtype=float)
        high, low, close = df['High'], df['Low'], df['Close']
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        tr = pd.concat([
            (high - low),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_w = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr_w.replace(0, np.nan)
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr_w.replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        return dx.ewm(alpha=1/period, adjust=False).mean()
    except Exception:
        return pd.Series(dtype=float)


def compute_roc(series: pd.Series, period: int = 20) -> float:
    """Rate of Change (%)."""
    try:
        if len(series) < period + 1:
            return np.nan
        ref = float(series.iloc[-period - 1])
        if ref == 0:
            return np.nan
        return round((float(series.iloc[-1]) / ref - 1) * 100, 2)
    except Exception:
        return np.nan


def compute_obv_slope(close: pd.Series, volume: pd.Series, window: int = 20) -> float:
    """On-Balance Volume slope (momentum signal)."""
    try:
        if len(close) < window + 5:
            return np.nan
        obv = (np.sign(close.diff().fillna(0)) * volume.fillna(0)).cumsum()
        tail = obv.iloc[-window:]
        if tail.std() == 0 or len(tail) < window:
            return 0.0
        x = np.arange(len(tail))
        slope, _ = np.polyfit(x, tail.values, 1)
        denom = max(abs(tail.mean()), 1.0)
        return round(float(slope) / denom, 6)
    except Exception:
        return np.nan


def dist_from_ma(price: float, ma: float) -> float:
    """Distance from MA as %."""
    try:
        if not ma or ma <= 0 or not np.isfinite(price) or not np.isfinite(ma):
            return np.nan
        return round((price / ma - 1) * 100, 2)
    except Exception:
        return np.nan


def slope_pct(series: pd.Series, window: int = 20) -> float:
    """Trend slope as % change over window bars."""
    try:
        if len(series) < window + 1:
            return np.nan
        a = float(series.iloc[-window - 1])
        b = float(series.iloc[-1])
        if a == 0:
            return np.nan
        return round((b / a - 1) * 100, 2)
    except Exception:
        return np.nan


def is_bb_squeeze(series: pd.Series, period: int = 20, k: float = 2.0, lookback: int = 120, pct: float = 0.25) -> bool:
    """Bollinger Band squeeze detector."""
    try:
        if len(series) < period:
            return False
        mid = series.rolling(period).mean()
        std = series.rolling(period).std()
        upper = mid + k * std
        lower = mid - k * std
        bbw = (upper - lower) / mid.replace(0, np.nan)
        bbw = bbw.dropna()

        if len(bbw) < lookback:
            return False
        tail = bbw.iloc[-lookback:]
        last = float(tail.iloc[-1])
        threshold = float(tail.quantile(pct))
        return last <= threshold
    except Exception:
        return False


def estimate_upside(df: pd.DataFrame, price: float, high20: float) -> dict:
    """Conservative price target estimation."""
    if not (isinstance(price, (int, float)) and np.isfinite(price) and price > 0):
        return {"target": None, "upside_pct": None, "high52": None, "high52_gap_pct": None}

    try:
        high52 = float(df["High"].iloc[-252:].max() if len(df) >= 252 else df["High"].max())
        low52 = float(df["Low"].iloc[-252:].min() if len(df) >= 252 else df["Low"].min())
        atrp = compute_atr_pct(df)

        atr_pct_capped = min(float(atrp), 25.0) if (atrp is not None and np.isfinite(atrp)) else 0.0
        atr_target = price * (1 + 2 * atr_pct_capped / 100) if atr_pct_capped > 0 else np.nan

        if np.isfinite(low52) and np.isfinite(high52) and high52 > low52:
            fib_raw = low52 + (high52 - low52) * 1.618
            fib_target = min(fib_raw, price * 2.0)
        else:
            fib_target = np.nan

        candidates = [t for t in [high52, atr_target, fib_target]
                      if np.isfinite(t) and price * 1.02 < t < price * 3.0]
        target = round(min(candidates), 2) if candidates else round(price * 1.12, 2)

        return {
            "target": target,
            "upside_pct": round((target - price) / price * 100, 1),
            "high52": round(high52, 2) if np.isfinite(high52) else None,
            "high52_gap_pct": round((high52 - price) / price * 100, 1) if np.isfinite(high52) else None,
        }
    except Exception as e:
        logger.debug(f"Upside estimation error: {e}")
        return {"target": None, "upside_pct": None, "high52": None, "high52_gap_pct": None}


# ═══════════════════════════════════════════════════════════════════════
# PATTERN DETECTION
# ═══════════════════════════════════════════════════════════════════════

def detect_base_pattern(close: pd.Series, volume: pd.Series) -> dict | None:
    """Try to detect chart patterns (VCP, flat base, cup-handle)."""
    # Simplified for now; full implementation would include all three patterns
    # Placeholder: return None for now, will be enhanced later
    return None


# ═══════════════════════════════════════════════════════════════════════
# STAGE CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════

def classify_stage(m: dict) -> tuple:
    """
    Classify rally stage from metrics dict.
    Returns (stage_id, stage_label).
    """
    price = m.get("price")
    ma50 = m.get("ma50")
    ma200 = m.get("ma200")
    ma50_slope10 = m.get("ma50_slope10")
    ma200_slope20 = m.get("ma200_slope20")
    rsi = m.get("rsi")
    adx = m.get("adx")
    adx_prev = m.get("adx_prev")
    macd_hist = m.get("macd_hist")
    macd_above_signal = m.get("macd_above_signal")
    macd_cross_recent = m.get("macd_cross_recent")
    obv_slope = m.get("obv_slope")
    vol_ratio = m.get("vol_ratio")
    high20 = m.get("high20")
    bb_squeeze = m.get("bb_squeeze")
    dist_ma200 = m.get("dist_ma200")
    dist_ma50 = m.get("dist_ma50")
    is_breakout = m.get("is_breakout")

    def _f(x):
        try:
            return float(x)
        except Exception:
            return np.nan

    # Stage 4 — Extended / climax
    s4_signals = []
    if np.isfinite(_f(dist_ma50)) and _f(dist_ma50) > 25:
        s4_signals.append(True)
    if np.isfinite(_f(rsi)) and _f(rsi) > 75:
        s4_signals.append(True)
    if np.isfinite(_f(dist_ma200)) and _f(dist_ma200) > 60:
        s4_signals.append(True)
    if np.isfinite(_f(adx)) and _f(adx) > 50:
        s4_signals.append(True)
    if any(s4_signals):
        return 4, STAGE_LABELS[4]

    # Stage 3 — Breakout
    if (is_breakout and np.isfinite(_f(macd_hist)) and _f(macd_hist) > 0
        and np.isfinite(_f(adx)) and _f(adx) > 25
        and np.isfinite(_f(vol_ratio)) and _f(vol_ratio) > 1.5
        and np.isfinite(_f(rsi)) and 55 <= _f(rsi) <= 72):
        return 3, STAGE_LABELS[3]

    # Stage 2 — Early Markup
    cond_s2 = (
        np.isfinite(_f(price)) and np.isfinite(_f(ma50)) and np.isfinite(_f(ma200))
        and _f(price) > _f(ma50) > _f(ma200)
        and np.isfinite(_f(ma50_slope10)) and _f(ma50_slope10) > 0
        and (macd_above_signal or macd_cross_recent)
        and np.isfinite(_f(adx)) and 18 <= _f(adx) <= 30
        and (not np.isfinite(_f(adx_prev)) or _f(adx) >= _f(adx_prev))
        and np.isfinite(_f(rsi)) and 50 <= _f(rsi) <= 65
        and np.isfinite(_f(high20)) and _f(price) < _f(high20) * BREAKOUT_TOLERANCE
    )
    if cond_s2:
        return 2, STAGE_LABELS[2]

    # Stage 1 — Accumulation
    cond_s1 = (
        np.isfinite(_f(price)) and np.isfinite(_f(ma200))
        and abs(_f(price) / _f(ma200) - 1) <= 0.08
        and (not np.isfinite(_f(ma200_slope20)) or _f(ma200_slope20) >= -1.0)
        and bool(bb_squeeze)
        and np.isfinite(_f(adx)) and _f(adx) < 20
        and (not np.isfinite(_f(obv_slope)) or _f(obv_slope) >= 0)
        and np.isfinite(_f(rsi)) and 40 <= _f(rsi) <= 55
    )
    if cond_s1:
        return 1, STAGE_LABELS[1]

    return 0, STAGE_LABELS[0]
