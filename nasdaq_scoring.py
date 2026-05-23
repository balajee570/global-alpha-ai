"""
NASDAQ Scoring & Recommendation Engine
- New NASDAQ weights: 35% tech, 25% quality, 20% value, 15% growth, 5% macro
- Premium precision: Piotroski, FCF yield, EV/FCF, earnings revisions, insider signals
- Action triggers: STRONG_BUY / BUY / ACCUMULATE / HOLD / AVOID
"""

import pandas as pd
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)

NASDAQ_WEIGHTS = {
    "tech": 0.35,
    "quality": 0.25,
    "value": 0.20,
    "growth": 0.15,
    "macro": 0.05,
}


def _coalesce(v, default):
    """Return default if v is None or NaN."""
    try:
        if v is None:
            return default
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return default
        return v
    except Exception:
        return default


def _num(v, decimals=2, prefix="", suffix=""):
    """Format number."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "N/A"
    return f"{prefix}{v:,.{decimals}f}{suffix}"


def _pct(v, decimals=1):
    """Format percentage."""
    if v is None or not np.isfinite(v):
        return "N/A"
    return f"{v * 100:.{decimals}f}%"


def compute_composite_score_nasdaq(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute NASDAQ-optimized composite Score using new weights.
    Uses Technical + Quality + Value + Growth + Macro components.
    """
    if df.empty:
        return df

    d = df.copy()

    # ── Component 1: Technical (35%)
    # stage(30%) + vol(18%) + RSI(12%) + MACD(20%) + ADX(15%) + pattern(5%)
    stage_w = d.get("StageId", pd.Series([0]*len(d))).map(
        {0: 0.20, 1: 0.85, 2: 1.00, 3: 0.75, 4: 0.15}
    ).fillna(0.2)

    def _rsi_health(r):
        if not np.isfinite(r):
            return 0.4
        if 50 <= r <= 65:
            return 1.0
        if 40 <= r < 50 or 65 < r <= 72:
            return 0.8
        return 0.4

    vol_w = (d.get("Vol Ratio", pd.Series([0]*len(d))).fillna(0) / 3.0).clip(0, 1)
    rsi_w = d.get("RSI", pd.Series([np.nan]*len(d))).apply(_rsi_health)

    macd_hist = d.get("MACD Hist", pd.Series([0.0]*len(d))).fillna(0.0)
    raw_std = macd_hist.std()
    hist_std = raw_std if (pd.notna(raw_std) and raw_std > 0) else 1.0
    macd_w = (macd_hist / hist_std).clip(lower=0, upper=1)

    adx_w = ((d.get("ADX", pd.Series([0]*len(d))).fillna(0) - 15).clip(lower=0) / 25).clip(upper=1)
    pattern_w = (d.get("Pattern Q", pd.Series([0]*len(d))).fillna(0) / 100).clip(0, 1)

    tech_w = (
        0.30 * stage_w +
        0.18 * vol_w +
        0.12 * rsi_w +
        0.20 * macd_w +
        0.15 * adx_w +
        0.05 * pattern_w
    )

    # ── Component 2: Quality (25%)
    # ROE(30%) + ROCE(25%) + margin_trend(20%) + FCF_margin(15%) + Piotroski(10%)
    roe_score = ((d.get("ROE", pd.Series([0]*len(d))).fillna(0)) / 0.25).clip(0, 1)
    roce_score = ((d.get("ROCE", pd.Series([0]*len(d))).fillna(0)) / 0.20).clip(0, 1)

    # Margin trend: assume >0 is good, <-2 is bad
    margin_trend_score = pd.Series([0.5]*len(d), index=d.index)
    if "Op Margin Trend" in d.columns:
        margin_trend_score = ((d["Op Margin Trend"].fillna(0) + 5) / 10).clip(0, 1)

    # FCF margin: FCF / Revenue (not always available)
    fcf_margin_score = pd.Series([0.5]*len(d), index=d.index)
    if "FCF Yield" in d.columns:
        fcf_margin_score = ((d["FCF Yield"].fillna(0) + 2) / 8).clip(0, 1)

    # Piotroski: 0-9 scale
    piotroski_score = (d.get("Piotroski Score", pd.Series([4]*len(d))).fillna(4) / 9).clip(0, 1)

    quality_w = (
        0.30 * roe_score +
        0.25 * roce_score +
        0.20 * margin_trend_score +
        0.15 * fcf_margin_score +
        0.10 * piotroski_score
    )

    # ── Component 3: Value (20%)
    # PEG_score(35%) + EV_FCF(30%) + DCF_upside(20%) + P_S(15%)
    pe_fwd = d.get("PE Forward", d.get("PE Trailing", pd.Series([25]*len(d))))
    peg = d.get("PEG", pd.Series([1.5]*len(d))).fillna(1.5)
    peg_score = (1 - (peg / 2)).clip(0, 1)  # <1 is great, >2 is bad

    ev_fcf = d.get("EV/FCF", pd.Series([20]*len(d))).fillna(20)
    ev_fcf_score = (30 / ev_fcf).clip(0, 1)  # <20x is good, >30x is bad

    dcf_up = d.get("DCF Upside %", pd.Series([10]*len(d))).fillna(10)
    dcf_score = ((dcf_up + 20) / 60).clip(0, 1)  # >20% upside is great

    ps = d.get("P/S", pd.Series([2.0]*len(d))).fillna(2.0)
    ps_score = (4 / ps).clip(0, 1)  # <2 is great, >4 is bad

    value_w = (
        0.35 * peg_score +
        0.30 * ev_fcf_score +
        0.20 * dcf_score +
        0.15 * ps_score
    )

    # ── Component 4: Growth (15%)
    # rev_cagr(40%) + eps_cagr(30%) + R&D%(20%) + guidance_beat(10%)
    rev_cagr = d.get("Rev CAGR 3Y", pd.Series([0.10]*len(d))).fillna(0.10)
    rev_score = ((rev_cagr + 0.05) / 0.30).clip(0, 1)  # 15%+ is great

    eps_cagr = d.get("EPS CAGR 3Y", pd.Series([0.10]*len(d))).fillna(0.10)
    eps_score = ((eps_cagr + 0.05) / 0.30).clip(0, 1)

    rd_pct = d.get("R&D %", pd.Series([5.0]*len(d))).fillna(5.0)
    rd_score = (rd_pct / 30).clip(0, 1)  # >30% R&D is exceptional

    guidance_beat = d.get("Earnings Beats", pd.Series([0.5]*len(d))).fillna(0.5)
    guidance_score = guidance_beat.clip(0, 1)

    growth_w = (
        0.40 * rev_score +
        0.30 * eps_score +
        0.20 * rd_score +
        0.10 * guidance_score
    )

    # ── Component 5: Macro (5%)
    # regime_multiplier × sector_momentum
    macro_w = d.get("Regime Multiplier", pd.Series([0.85]*len(d))).fillna(0.85) * \
              d.get("Sector Momentum", pd.Series([0.5]*len(d))).fillna(0.5)
    macro_w = macro_w.clip(0, 1)

    # ── Final Score ──
    score = 100 * (
        NASDAQ_WEIGHTS["tech"] * tech_w +
        NASDAQ_WEIGHTS["quality"] * quality_w +
        NASDAQ_WEIGHTS["value"] * value_w +
        NASDAQ_WEIGHTS["growth"] * growth_w +
        NASDAQ_WEIGHTS["macro"] * macro_w
    )
    d["Score"] = score.round(1)

    return d


def attach_grade_nasdaq(df: pd.DataFrame, regime_label: str = "SELECTIVE",
                        top_sectors: set | None = None) -> pd.DataFrame:
    """
    Compute Conviction Grade A+/A/B+/B/C using NASDAQ formula.
    Uses: Score, RS Rank, sector position, pattern quality, regime, stage.
    """
    if df.empty or "Score" not in df.columns:
        return df

    top_sectors = top_sectors or set()
    regime_w = {"AGGRESSIVE": 1.0, "SELECTIVE": 0.7, "DEFENSIVE": 0.4}.get(regime_label, 0.7)
    stage_w_map = {1: 0.8, 2: 1.0, 3: 0.8, 4: 0.2, 0: 0.5}

    d = df.copy()
    score_n = (d.get("Score", pd.Series([0]*len(d))).fillna(0) / 100).clip(0, 1)
    rs_n = (d.get("RS Rank", pd.Series([50]*len(d))).fillna(50) / 100).clip(0, 1)

    # Sector bonus
    sec_n = d.get("Sector", pd.Series(["—"]*len(d))).apply(
        lambda t: 1.0 if t in top_sectors else 0.3
    )
    pat_n = (d.get("Pattern Q", pd.Series([0]*len(d))).fillna(0) / 100).clip(0, 1)
    reg_n = regime_w
    stg_n = d.get("StageId", pd.Series([0]*len(d))).map(stage_w_map).fillna(0.5)

    grade_score = 100 * (
        0.35 * score_n +
        0.20 * rs_n +
        0.15 * sec_n +
        0.10 * pat_n +
        0.10 * reg_n +
        0.10 * stg_n
    )
    d["GradeScore"] = grade_score.round(1)

    def _g(s):
        if s >= 85: return "A+"
        if s >= 75: return "A"
        if s >= 65: return "B+"
        if s >= 50: return "B"
        return "C"
    d["Grade"] = d["GradeScore"].apply(_g)

    return d


def build_shortlist_nasdaq(df: pd.DataFrame, target: int = 25, min_score: float = 45.0, floor: int = 5) -> pd.DataFrame:
    """Top N rows above score threshold, with floor. Lowered threshold for inclusivity."""
    if df.empty or "Score" not in df.columns:
        return df.head(0)

    above = df[df["Score"] >= min_score].sort_values("Score", ascending=False)
    if len(above) >= floor:
        return above.head(target).reset_index(drop=True)

    return df.sort_values("Score", ascending=False).head(floor).reset_index(drop=True)


def recommend_action_nasdaq(metrics: dict, tech: dict) -> dict:
    """
    Recommend BUY/HOLD/AVOID action.
    metrics = fundamentals dict, tech = single-stock technicals row.
    Returns {action, conviction, scores, bull_case, bear_case, entry_zone, stop_loss, targets}.
    """
    try:
        price = _coalesce(tech.get("Price $"), _coalesce(metrics.get("price"), 100.0))
        atrp = _coalesce(tech.get("ATR %"), 3.0)
        stage_id = int(_coalesce(tech.get("StageId"), 0))

        # Component scores
        tech_score = float(_coalesce(tech.get("Score"), 50.0))

        # Quality score from fundamentals
        piotroski = _coalesce(metrics.get("piotroski_score"), 4)
        roe = _coalesce(metrics.get("roe"), 0.10)
        roce = _coalesce(metrics.get("roce"), 0.12)
        op_margin = _coalesce(metrics.get("op_margin"), 0.10)
        fcf_yield = _coalesce(metrics.get("fcf_yield"), 2.0)

        quality_parts = [
            1.0 if roe > 0.18 else (0.6 if roe > 0.12 else 0.2),
            1.0 if roce > 0.15 else (0.6 if roce > 0.10 else 0.2),
            0.8 + (piotroski / 10),
            1.0 if op_margin > 0.15 else (0.6 if op_margin > 0.08 else 0.2),
            1.0 if (fcf_yield and fcf_yield > 3) else (0.6 if fcf_yield and fcf_yield > 1 else 0.2),
        ]
        quality_score = 100 * (sum(quality_parts) / len(quality_parts)) if quality_parts else 50.0

        # Value score
        pe = metrics.get("pe_forward") or metrics.get("pe_trailing")
        peg = metrics.get("peg")
        dcf_up = (metrics.get("dcf") or {}).get("upside_pct") or metrics.get("dcf_upside")
        ev_fcf = metrics.get("ev_fcf")

        val_parts = []
        if pe and pe > 0:
            val_parts.append(1.0 if pe < 20 else (0.7 if pe < 35 else 0.2))
        if peg and peg > 0:
            val_parts.append(1.0 if peg < 1 else (0.6 if peg < 2 else 0.2))
        if dcf_up is not None:
            val_parts.append(1.0 if dcf_up > 20 else (0.6 if dcf_up > 0 else 0.2))
        if ev_fcf and ev_fcf > 0:
            val_parts.append(1.0 if ev_fcf < 20 else (0.6 if ev_fcf < 30 else 0.2))

        value_score = 100 * (sum(val_parts) / len(val_parts)) if val_parts else 50.0

        # Growth score
        growth_parts = []
        if metrics.get("revenue_growth"):
            growth_parts.append(1.0 if metrics["revenue_growth"] > 0.15 else 0.6)
        if metrics.get("earnings_growth"):
            growth_parts.append(1.0 if metrics["earnings_growth"] > 0.15 else 0.6)
        growth_score = 100 * (sum(growth_parts) / len(growth_parts)) if growth_parts else 50.0

        # Final conviction
        conviction = 0.4 * tech_score + 0.3 * quality_score + 0.2 * value_score + 0.1 * growth_score
        conviction = round(max(0, min(100, conviction)), 1)

        # Action determination
        de = _coalesce(metrics.get("debt_to_equity"), 1.0)
        critical_bear = (de > 2.0) or stage_id == 4

        if stage_id == 4:
            action = "AVOID"
        elif conviction >= 80 and not critical_bear:
            action = "STRONG_BUY"
        elif conviction >= 70:
            action = "BUY"
        elif conviction >= 50:
            action = "ACCUMULATE"
        elif conviction >= 35:
            action = "HOLD"
        else:
            action = "AVOID"

        # Bull & bear cases
        bull = []
        if roe and roe > 0.18:
            bull.append(f"ROE {roe*100:.1f}% > 18%")
        if op_margin and op_margin > 0.15:
            bull.append(f"Op margin {op_margin*100:.1f}% > 15%")
        if piotroski >= 7:
            bull.append(f"Piotroski {piotroski} (high quality)")
        if stage_id in (1, 2):
            bull.append(f"Stage {stage_id} setup (early rally)")
        if conviction >= 75:
            bull.append(f"Tech+Fundamental strength (conviction {conviction:.0f})")

        bear = []
        if de > 1.5:
            bear.append(f"D/E {de:.2f} > 1.5")
        if stage_id == 4:
            bear.append("Stage 4 extended (climax risk)")
        if pe and pe > 50:
            bear.append(f"Expensive: P/E {pe:.0f}")
        if fcf_yield and fcf_yield < 1.0:
            bear.append("Weak FCF yield (burning cash)")

        # Entry & stop
        atrp_use = atrp if (isinstance(atrp, (int, float)) and np.isfinite(atrp) and atrp > 0) else 3.0
        entry_low = round(price * (1 - 0.5 * atrp_use / 100), 2) if price else None
        entry_high = round(price, 2) if price else None
        stop = round(price * (1 - 2 * atrp_use / 100), 2) if price else None

        # Targets
        targets = {}
        if tech.get("Target $"):
            targets["technical"] = round(float(tech["Target $"]), 2)
        if metrics.get("target_mean"):
            targets["analyst"] = round(float(metrics["target_mean"]), 2)

        return {
            "action": action,
            "conviction": conviction,
            "scores": {
                "technical": round(tech_score, 1),
                "quality": round(quality_score, 1),
                "value": round(value_score, 1),
                "growth": round(growth_score, 1),
            },
            "bull_case": bull[:3],
            "bear_case": bear[:3],
            "entry_zone": (entry_low, entry_high),
            "stop_loss": stop,
            "targets": targets,
        }

    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        return {
            "action": "HOLD",
            "conviction": 0,
            "scores": {"technical": 0, "quality": 0, "value": 0, "growth": 0},
            "bull_case": [],
            "bear_case": [str(e)],
            "entry_zone": (None, None),
            "stop_loss": None,
            "targets": {},
        }
