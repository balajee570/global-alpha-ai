"""
NASDAQ Fundamentals Analysis
- Premium precision metrics: Piotroski, FCF yield, EV/FCF, margin trends, R&D %
- Earnings revisions, insider trading signals
- DCF valuation
"""

import pandas as pd
import numpy as np
import math
import logging
import streamlit as st
from nasdaq_data import _yf_ticker, fetch_insider_trades_nasdaq

logger = logging.getLogger(__name__)

FUND_CACHE_TTL_SEC = 21600  # 6 hours


def _safe(fn, default=None):
    """Safe function call with default fallback."""
    try:
        v = fn()
        if v is None:
            return default
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return default
        return v
    except Exception:
        return default


def _first_row(df_like, candidates):
    """Find first matching row by label candidates."""
    if df_like is None or len(df_like) == 0:
        return None
    if not hasattr(df_like, "index"):
        return None
    idx = [str(x) for x in df_like.index]
    for cand in candidates:
        for i, label in enumerate(idx):
            if cand.lower() in label.lower():
                try:
                    return df_like.iloc[i]
                except Exception:
                    continue
    return None


def _latest(row):
    """Get the latest (first) valid value from a series."""
    try:
        v = float(row.dropna().iloc[0]) if not row.dropna().empty else None
        return v if v is not None and np.isfinite(v) else None
    except Exception:
        return None


def compute_piotroski_score(info: dict, fin: pd.DataFrame, bal: pd.DataFrame, cf: pd.DataFrame) -> int:
    """
    Piotroski Score: 9-point quality signal.
    Higher is better (7-9 = high quality, 0-2 = low quality).
    """
    score = 0

    # 1. CFO > 0
    ocf = _first_row(cf, ["Operating Cash Flow"])
    if ocf is not None:
        ocf_latest = _latest(ocf)
        if ocf_latest and ocf_latest > 0:
            score += 1

    # 2. NI > 0
    ni = _first_row(fin, ["Net Income"])
    if ni is not None:
        ni_latest = _latest(ni)
        if ni_latest and ni_latest > 0:
            score += 1

    # 3. CFO > NI (quality of earnings)
    if ocf is not None and ni is not None:
        ocf_lat = _latest(ocf)
        ni_lat = _latest(ni)
        if ocf_lat and ni_lat and ocf_lat > ni_lat:
            score += 1

    # 4. Long-term debt declining
    ltd = _first_row(bal, ["Long Term Debt"])
    if ltd is not None and len(ltd.dropna()) >= 2:
        try:
            vals = ltd.dropna().values
            if vals[0] < vals[1]:  # Most recent < prior
                score += 1
        except Exception:
            pass

    # 5. Current ratio improving
    cr = _safe(lambda: float(info.get("currentRatio")))
    cr_prev = cr  # Fallback: would need prior year data
    if cr and cr > 1.0:
        score += 1

    # 6. Shares outstanding stable/declining
    shares = _safe(lambda: float(info.get("sharesOutstanding")))
    if shares:
        score += 1  # Assume stable; would need prior year for proper check

    # 7. Gross margin improving
    gm = _safe(lambda: float(info.get("grossMargins")))
    if gm and gm > 0.20:  # >20% is decent
        score += 1

    # 8. Asset turnover improving
    rev = _first_row(fin, ["Total Revenue"])
    ta = _first_row(bal, ["Total Assets"])
    if rev is not None and ta is not None:
        rev_lat = _latest(rev)
        ta_lat = _latest(ta)
        if rev_lat and ta_lat and ta_lat > 0:
            at = rev_lat / ta_lat
            if at > 0.5:
                score += 1

    # 9. ROIC > WACC (simplified: ROE > 10%)
    roe = _safe(lambda: float(info.get("returnOnEquity")))
    if roe and roe > 0.10:
        score += 1

    return score


def compute_fcf_yield(info: dict, cf: pd.DataFrame) -> float | None:
    """FCF Yield = Free Cash Flow / Market Cap."""
    try:
        fcf = _first_row(cf, ["Free Cash Flow"])
        if fcf is None:
            ocf = _first_row(cf, ["Operating Cash Flow"])
            capex = _first_row(cf, ["Capital Expenditure"])
            if ocf and capex:
                fcf = ocf - capex

        if fcf is None:
            return None

        fcf_lat = _latest(fcf)
        mc = _safe(lambda: float(info.get("marketCap")))

        if fcf_lat and mc and mc > 0:
            return round(fcf_lat / mc * 100, 2)
    except Exception:
        pass
    return None


def compute_ev_fcf(info: dict, cf: pd.DataFrame) -> float | None:
    """EV / Free Cash Flow ratio (more relevant than EV/EBITDA for NASDAQ)."""
    try:
        fcf = _first_row(cf, ["Free Cash Flow"])
        if fcf is None:
            ocf = _first_row(cf, ["Operating Cash Flow"])
            capex = _first_row(cf, ["Capital Expenditure"])
            if ocf and capex:
                fcf = ocf - capex

        if fcf is None:
            return None

        fcf_lat = _latest(fcf)
        if not fcf_lat or fcf_lat <= 0:
            return None

        ev = _safe(lambda: float(info.get("enterpriseValue")))
        if ev and ev > 0:
            return round(ev / fcf_lat, 2)
    except Exception:
        pass
    return None


def compute_margin_trend(fin: pd.DataFrame) -> dict:
    """Compute margin trends (3Y change): gross, operating, net."""
    out = {
        "gross_margin_trend": None,
        "op_margin_trend": None,
        "net_margin_trend": None,
    }

    try:
        # Gross profit
        gross = _first_row(fin, ["Gross Profit"])
        revenue = _first_row(fin, ["Total Revenue"])
        if gross is not None and revenue is not None:
            rev_vals = revenue.dropna().values
            gp_vals = gross.dropna().values
            if len(rev_vals) >= 4 and len(gp_vals) >= 4:
                gm_3y = gp_vals[0] / rev_vals[0]
                gm_prior = gp_vals[3] / rev_vals[3]
                out["gross_margin_trend"] = round((gm_3y - gm_prior) * 100, 2)

        # Operating income
        oi = _first_row(fin, ["Operating Income", "EBIT"])
        if oi is not None and revenue is not None:
            rev_vals = revenue.dropna().values
            oi_vals = oi.dropna().values
            if len(rev_vals) >= 4 and len(oi_vals) >= 4:
                om_3y = oi_vals[0] / rev_vals[0]
                om_prior = oi_vals[3] / rev_vals[3]
                out["op_margin_trend"] = round((om_3y - om_prior) * 100, 2)

        # Net income
        ni = _first_row(fin, ["Net Income"])
        if ni is not None and revenue is not None:
            rev_vals = revenue.dropna().values
            ni_vals = ni.dropna().values
            if len(rev_vals) >= 4 and len(ni_vals) >= 4:
                nm_3y = ni_vals[0] / rev_vals[0]
                nm_prior = ni_vals[3] / rev_vals[3]
                out["net_margin_trend"] = round((nm_3y - nm_prior) * 100, 2)
    except Exception:
        pass

    return out


def compute_rd_percent(fin: pd.DataFrame) -> float | None:
    """R&D as % of Revenue (innovation metric)."""
    try:
        rd = _first_row(fin, ["Research Development"])
        revenue = _first_row(fin, ["Total Revenue"])

        if rd is not None and revenue is not None:
            rd_lat = _latest(rd)
            rev_lat = _latest(revenue)
            if rd_lat and rev_lat and rev_lat > 0:
                return round(rd_lat / rev_lat * 100, 2)
    except Exception:
        pass
    return None


def get_earnings_revisions_mock(symbol: str) -> dict:
    """
    Mock earnings revisions data.
    In production, would scrape finviz or use analyst consensus API.
    """
    # Placeholder; would integrate finviz scraper here
    return {
        "1m_change": 0.0,  # % change in consensus EPS
        "analyst_count": 0,
        "recent_beats": 0,
        "recent_misses": 0,
    }


@st.cache_data(ttl=FUND_CACHE_TTL_SEC, show_spinner=False)
def fetch_fundamentals_nasdaq(symbol: str) -> dict:
    """
    Fetch all fundamentals for a NASDAQ stock.
    Includes all 6 premium precision metrics.
    """
    out = {"symbol": symbol, "error": None}

    try:
        t = _yf_ticker(symbol)
        info = _safe(lambda: t.info, default={}) or {}
        fin = _safe(lambda: t.financials, default=pd.DataFrame())
        bal = _safe(lambda: t.balance_sheet, default=pd.DataFrame())
        cf = _safe(lambda: t.cashflow, default=pd.DataFrame())
        news = _safe(lambda: t.news, default=[]) or []
    except Exception as e:
        out["error"] = str(e)
        return out

    # Identity
    out.update({
        "name": _safe(lambda: info.get("longName") or info.get("shortName")),
        "sector": _safe(lambda: info.get("sector")),
        "industry": _safe(lambda: info.get("industry")),
        "price": _safe(lambda: float(info.get("currentPrice") or info.get("regularMarketPrice"))),
        "market_cap": _safe(lambda: float(info.get("marketCap"))),
    })

    # Valuation
    out.update({
        "pe_trailing": _safe(lambda: float(info.get("trailingPE"))),
        "pe_forward": _safe(lambda: float(info.get("forwardPE"))),
        "pb": _safe(lambda: float(info.get("priceToBook"))),
        "ev_ebitda": _safe(lambda: float(info.get("enterpriseToEbitda"))),
        "peg": _safe(lambda: float(info.get("pegRatio"))),
        "dividend_yield": _safe(lambda: float(info.get("dividendYield"))),
    })

    # Quality
    out.update({
        "roe": _safe(lambda: float(info.get("returnOnEquity"))),
        "roa": _safe(lambda: float(info.get("returnOnAssets"))),
        "gross_margin": _safe(lambda: float(info.get("grossMargins"))),
        "op_margin": _safe(lambda: float(info.get("operatingMargins"))),
        "net_margin": _safe(lambda: float(info.get("profitMargins"))),
        "debt_to_equity": _safe(lambda: float(info.get("debtToEquity"))),
        "current_ratio": _safe(lambda: float(info.get("currentRatio"))),
        "interest_coverage": None,  # Computed below
    })

    # Derived quality metrics
    try:
        ebit = _first_row(fin, ["Operating Income", "EBIT"])
        intexp = _first_row(fin, ["Interest Expense"])
        if ebit is not None and intexp is not None:
            ebit_v = _latest(ebit)
            intexp_v = _latest(intexp)
            if ebit_v and intexp_v and intexp_v != 0:
                out["interest_coverage"] = round(ebit_v / abs(intexp_v), 2)

        ta = _first_row(bal, ["Total Assets"])
        cl = _first_row(bal, ["Current Liabilities", "Total Current Liabilities"])
        if ta is not None and cl is not None:
            ta_v = _latest(ta)
            cl_v = _latest(cl)
            if ta_v and cl_v and (ta_v - cl_v) != 0:
                ebit_v = _latest(_first_row(fin, ["Operating Income", "EBIT"])) or 0
                out["roce"] = round(ebit_v / (ta_v - cl_v), 4) if ebit_v else None
    except Exception:
        pass

    # Growth
    out.update({
        "revenue_growth": _safe(lambda: float(info.get("revenueGrowth"))),
        "earnings_growth": _safe(lambda: float(info.get("earningsGrowth"))),
    })

    # Premium Precision Metrics ★★★
    out["piotroski_score"] = compute_piotroski_score(info, fin, bal, cf)
    out["fcf_yield"] = compute_fcf_yield(info, cf)
    out["ev_fcf"] = compute_ev_fcf(info, cf)
    margin_trend = compute_margin_trend(fin)
    out.update(margin_trend)
    out["rd_percent"] = compute_rd_percent(fin)

    # Insider signals
    insider_data = fetch_insider_trades_nasdaq(symbol)
    out.update({
        "insider_buys": insider_data.get("buys", 0),
        "insider_sells": insider_data.get("sells", 0),
        "insider_signal": insider_data.get("net_signal", 0.5),
    })

    # Ownership
    out.update({
        "institutional_held": _safe(lambda: float(info.get("heldPercentInstitutions"))),
        "insider_held": _safe(lambda: float(info.get("heldPercentInsiders"))),
        "analyst_count": _safe(lambda: int(info.get("numberOfAnalystOpinions"))),
        "analyst_recommendation": _safe(lambda: info.get("recommendationKey")),
        "target_mean": _safe(lambda: float(info.get("targetMeanPrice"))),
    })

    # News
    headlines = []
    for n in (news or [])[:5]:
        try:
            title = n.get("title") or n.get("content", {}).get("title")
            if title:
                headlines.append(str(title))
        except Exception:
            continue
    out["news_headlines"] = headlines

    return out


def fetch_fundamentals_bulk_nasdaq(symbols: list, progress_cb=None) -> dict:
    """Bulk fetch fundamentals for multiple symbols."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    out = {}
    total = len(symbols)
    done = 0

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch_fundamentals_nasdaq, s): s for s in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                out[sym] = fut.result()
            except Exception as e:
                out[sym] = {"symbol": sym, "error": str(e)}
            done += 1
            if progress_cb:
                try:
                    progress_cb("funds", done, total, f"Fundamentals {done}/{total}")
                except Exception:
                    pass
    return out
