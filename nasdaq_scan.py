"""
NASDAQ Full Market Scanning Pipeline
- Integrates all modules: data, regime, tech, fundamentals, scoring
- Produces ranked stock recommendations with deep-dive data
"""

import pandas as pd
import numpy as np
import logging
from nasdaq_data import parallel_fetch_nasdaq, _yf_download
from nasdaq_regime import compute_nasdaq_regime, compute_sector_strength, top_sectors_set
from nasdaq_tech import (
    compute_ma, compute_rsi, compute_atr_pct, compute_macd, compute_adx, compute_adx_series,
    compute_roc, compute_obv_slope, is_bb_squeeze, dist_from_ma, slope_pct, estimate_upside,
    detect_base_pattern, classify_stage, MIN_BARS, BREAKOUT_LOOKBACK, VOL_LOOKBACK,
    VOL_SURGE_THRESH, BREAKOUT_TOLERANCE, RSI_OVERBOUGHT
)
from nasdaq_fundamentals import fetch_fundamentals_bulk_nasdaq
from nasdaq_scoring import (
    compute_composite_score_nasdaq, attach_grade_nasdaq, build_shortlist_nasdaq,
    recommend_action_nasdaq
)

logger = logging.getLogger(__name__)


def scan_nasdaq_equities(symbols: list, progress_cb=None) -> pd.DataFrame:
    """
    Full NASDAQ equity scan: technicals + stage classification + scoring.
    Returns DataFrame with all metrics, ranked by composite Score.
    """
    results = []
    failed = []
    total = len(symbols)

    # Download all data in parallel
    if progress_cb:
        try:
            progress_cb("scan", 0, total, f"Downloading {total} stocks...")
        except Exception:
            pass

    data = parallel_fetch_nasdaq(symbols, progress_cb=progress_cb)

    # Process each stock
    for i, ticker in enumerate(symbols):
        if progress_cb and i % 50 == 0:
            try:
                progress_cb("scan", i, total, f"Processing {i}/{total}...")
            except Exception:
                pass

        df = data.get(ticker)
        if df is None or df.empty or len(df) < MIN_BARS:
            failed.append((ticker, "Insufficient data"))
            continue

        try:
            # Clean data
            df = df.dropna(subset=["Close", "Volume"])
            if len(df) < MIN_BARS:
                continue

            close = df["Close"]
            price = float(close.iloc[-1])

            # Technical metrics
            ma50 = compute_ma(close, 50)
            ma200 = compute_ma(close, 200)
            ma50_series = close.rolling(50).mean()
            ma200_series = close.rolling(200).mean()
            ma50_slope10 = slope_pct(ma50_series.dropna(), window=10) if ma50_series.notna().sum() >= 11 else np.nan
            ma200_slope20 = slope_pct(ma200_series.dropna(), window=20) if ma200_series.notna().sum() >= 21 else np.nan

            high20 = float(df["High"].iloc[-(BREAKOUT_LOOKBACK + 1):-1].max()) if len(df) > BREAKOUT_LOOKBACK else price
            vol_avg = float(df["Volume"].iloc[-(VOL_LOOKBACK + 1):-1].mean()) if len(df) > VOL_LOOKBACK else 0
            vol_now = float(df["Volume"].iloc[-1])
            vol_ratio = vol_now / vol_avg if vol_avg > 0 else 0

            rsi = compute_rsi(close)
            atr_pct = compute_atr_pct(df)

            # MACD
            macd_last, sig_last, hist_last, hist_series = compute_macd(close)
            macd_above_signal = bool(np.isfinite(macd_last) and np.isfinite(sig_last) and macd_last > sig_last)
            macd_cross_recent = False
            if len(hist_series.dropna()) >= 11:
                tail = hist_series.dropna().iloc[-11:]
                macd_cross_recent = bool(((tail.shift(1) < 0) & (tail > 0)).any())

            # ADX
            adx_series = compute_adx_series(df)
            if len(adx_series.dropna()) >= 6:
                adx_now = float(adx_series.dropna().iloc[-1])
                adx_prev = float(adx_series.dropna().iloc[-6])
            else:
                adx_now, adx_prev = np.nan, np.nan
            _, plus_di, minus_di = compute_adx(df)

            # Other indicators
            roc20 = compute_roc(close, 20)
            obv_slope = compute_obv_slope(close, df["Volume"])
            bb_squeeze = is_bb_squeeze(close)
            d_ma50 = dist_from_ma(price, ma50)
            d_ma200 = dist_from_ma(price, ma200)

            # Breakout check
            is_breakout = price >= high20 * BREAKOUT_TOLERANCE
            is_vol_surge = vol_ratio > VOL_SURGE_THRESH

            # Upside estimate
            upside = estimate_upside(df, price, high20)

            # Stage classification
            metrics = {
                "price": price, "ma50": ma50, "ma200": ma200,
                "ma50_slope10": ma50_slope10, "ma200_slope20": ma200_slope20,
                "rsi": rsi, "adx": adx_now, "adx_prev": adx_prev,
                "macd_hist": hist_last, "macd_above_signal": macd_above_signal,
                "macd_cross_recent": macd_cross_recent,
                "obv_slope": obv_slope, "vol_ratio": vol_ratio, "high20": high20,
                "bb_squeeze": bb_squeeze,
                "dist_ma200": d_ma200, "dist_ma50": d_ma50,
                "is_breakout": is_breakout,
            }
            stage_id, stage_lbl = classify_stage(metrics)

            # Filter: keep meaningful stages or breakout signals
            stage_pass = stage_id in (1, 2, 3)
            legacy_pass = (np.isfinite(ma50) and price > ma50) and (is_breakout or is_vol_surge)
            if not (stage_pass or legacy_pass or stage_id == 4):
                continue

            # Pattern detection
            pat = detect_base_pattern(close, df["Volume"]) or {}

            signal = "Breakout" if is_breakout else ("Building" if is_vol_surge else stage_lbl)

            results.append({
                "Ticker": ticker,
                "Price $": round(price, 2),
                "MA50": round(ma50, 2) if np.isfinite(ma50) else np.nan,
                "MA200": round(ma200, 2) if np.isfinite(ma200) else np.nan,
                "RSI": rsi,
                "Vol Ratio": round(vol_ratio, 2),
                "Signal": signal,
                "Stage": stage_lbl,
                "StageId": stage_id,
                "ADX": round(adx_now, 1) if np.isfinite(adx_now) else np.nan,
                "+DI": round(plus_di, 1) if np.isfinite(plus_di) else np.nan,
                "-DI": round(minus_di, 1) if np.isfinite(minus_di) else np.nan,
                "MACD Hist": round(hist_last, 3) if np.isfinite(hist_last) else np.nan,
                "ROC20 %": roc20,
                "OBV Slope": obv_slope,
                "Dist MA50 %": d_ma50,
                "Dist MA200 %": d_ma200,
                "ATR %": atr_pct,
                "Target $": upside["target"],
                "Upside %": upside["upside_pct"],
                "52W High $": upside["high52"],
                "Gap to 52W %": upside["high52_gap_pct"],
                "Pattern": pat.get("name"),
                "Pattern Q": pat.get("quality"),
            })

        except Exception as e:
            failed.append((ticker, str(e)))
            logger.debug(f"Processing error for {ticker}: {e}")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Score and rank
    df = compute_composite_score_nasdaq(df)

    return df.sort_values("Score", ascending=False).reset_index(drop=True)


def build_scan_report(symbols: list, progress_cb=None) -> dict:
    """
    Full pipeline: scan → regime → sector strength → fundamentals → grades → shortlist.
    Returns comprehensive dict with all data needed for UI.
    """
    # Market regime
    if progress_cb:
        try:
            progress_cb("regime", 0, 100, "Computing market regime...")
        except Exception:
            pass

    # Scan equities
    if progress_cb:
        try:
            progress_cb("scan", 0, 100, "Scanning NASDAQ...")
        except Exception:
            pass

    momentum_df = scan_nasdaq_equities(symbols, progress_cb=progress_cb)

    if momentum_df.empty:
        return {
            "momentum_df": pd.DataFrame(),
            "regime": {},
            "sector_strength": pd.DataFrame(),
            "shortlist_df": pd.DataFrame(),
            "fund_map": {},
        }

    # Regime
    regime = compute_nasdaq_regime(momentum_df)

    # Sector strength
    sector_strength = compute_sector_strength(momentum_df)
    top5_sectors = top_sectors_set(sector_strength, top_n=5)

    # Add grades
    momentum_df = attach_grade_nasdaq(
        momentum_df,
        regime_label=regime.get("label", "SELECTIVE"),
        top_sectors=top5_sectors
    )

    # Build shortlist
    shortlist_df = build_shortlist_nasdaq(momentum_df)

    # Fetch fundamentals for shortlist
    if progress_cb:
        try:
            progress_cb("funds", 0, 100, "Fetching fundamentals...")
        except Exception:
            pass

    fund_map = {}
    if not shortlist_df.empty:
        sl_symbols = [f"{t}" for t in shortlist_df["Ticker"].tolist()]
        raw_funds = fetch_fundamentals_bulk_nasdaq(sl_symbols, progress_cb=progress_cb)

        for _, t_row in shortlist_df.iterrows():
            sym = t_row["Ticker"]
            m = raw_funds.get(sym, {})
            try:
                rec = recommend_action_nasdaq(m, t_row.to_dict())
            except Exception as e:
                rec = {
                    "action": "HOLD",
                    "conviction": 0,
                    "scores": {},
                    "bull_case": [],
                    "bear_case": [str(e)],
                    "entry_zone": (None, None),
                    "stop_loss": None,
                    "targets": {},
                }
            fund_map[sym] = {
                "metrics": m,
                "technicals": t_row.to_dict(),
                "recommendation": rec,
            }

    return {
        "momentum_df": momentum_df,
        "regime": regime,
        "sector_strength": sector_strength,
        "shortlist_df": shortlist_df,
        "fund_map": fund_map,
    }
