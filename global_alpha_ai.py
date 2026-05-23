"""
Global Alpha AI - NASDAQ Recommendation Engine
Streamlit app for comprehensive stock analysis with premium precision metrics
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import json
import os
import tempfile

# Import NASDAQ engine modules
from nasdaq_scan import build_scan_report
from nasdaq_data import parse_nasdaq_screener_csv, get_sp500_list
from nasdaq_fundamentals import fetch_fundamentals_nasdaq
from nasdaq_scoring import recommend_action_nasdaq

# ==================== CONFIG ====================
ET = timezone(timedelta(hours=-4))
st.set_page_config(
    page_title="Global Alpha AI - NASDAQ Engine",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== STYLING ====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700;900&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@400;600;700&display=swap');
:root {
    --bg-base: #080c18;
    --gold: #d4a843;
    --gold-light: #f0c96a;
    --emerald: #00c896;
    --amber: #f59e0b;
    --red: #ef4444;
    --blue: #3b82f6;
}
html, body {
    font-family: 'DM Sans', sans-serif;
    background: var(--bg-base);
    color: #eef2ff;
}
.stApp { background: var(--bg-base); }
.card {
    background: rgba(15,20,40,0.8);
    border: 1px solid rgba(212,168,67,0.3);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 10px;
}
.pick-card {
    background: rgba(15,20,40,0.8);
    border: 2px solid var(--gold);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 10px;
}
.ticker-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--gold);
}
.action-strong-buy { color: var(--emerald); font-weight: 700; }
.action-buy { color: var(--emerald); font-weight: 600; }
.action-avoid { color: var(--red); font-weight: 700; }
.action-hold { color: var(--amber); font-weight: 600; }
.metric-row { display: flex; justify-content: space-between; margin: 8px 0; font-size: 0.9rem; }
.regime-aggressive { color: var(--emerald); font-weight: 700; }
.regime-selective { color: var(--amber); font-weight: 700; }
.regime-defensive { color: var(--red); font-weight: 700; }
.grade-a-plus { color: var(--emerald); font-weight: 700; }
.grade-a { color: var(--emerald); font-weight: 600; }
.grade-b { color: var(--amber); }
.grade-c { color: var(--red); }
</style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================
st.markdown(f"""
<div style="padding:30px 0 20px;">
    <span style="font-family:'Cinzel',serif; font-size:2.8rem; font-weight:900; letter-spacing:0.07em; background:linear-gradient(110deg,#d4a843,#f0c96a); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        NASDAQ ALPHA ENGINE
    </span>
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#8896b3; margin-top:8px;">
        Premium Precision Metrics • {datetime.now(ET).strftime("%d %b %Y %H:%M ET")}
    </div>
    <div style="font-family:'DM Sans',sans-serif; font-size:0.85rem; color:#a0b4d8; margin-top:6px;">
        6 Precision Metrics: Piotroski • FCF Yield • EV/FCF • Margin Trends • R&D % • Insider Signals
    </div>
</div>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if "scan_result" not in st.session_state:
    st.session_state.scan_result = None
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None

# ==================== INPUT SECTION ====================
st.divider()
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("#### 📋 Input: CSV Screener or Full NASDAQ")
    input_mode = st.radio("Choose input method:", ["Upload CSV", "Use S&P 500 Default"], horizontal=True)

with col2:
    st.markdown("#### 🎯")

if input_mode == "Upload CSV":
    uploaded_file = st.file_uploader("Upload NASDAQ Screener CSV (Symbol, Sector, etc.)", type=["csv"])
    if uploaded_file:
        try:
            symbols = parse_nasdaq_screener_csv(uploaded_file)
            st.success(f"✅ Loaded **{len(symbols)}** stocks from CSV")
        except Exception as e:
            st.error(f"CSV parsing error: {e}")
            symbols = None
    else:
        symbols = None
else:
    symbols = get_sp500_list()
    st.info(f"📊 Using S&P 500 default universe ({len(symbols)} stocks for speed)")

# ==================== MANUAL SEARCH ====================
with st.expander("🔍 Add Stocks Manually (Search Tab)"):
    search_ticker = st.text_input("Search ticker (e.g., MSFT, NVDA)", key="search_ticker").upper().strip()
    if search_ticker and len(search_ticker) <= 5:
        st.session_state.selected_ticker = search_ticker

# ==================== RUN SCAN ====================
if symbols and st.button("🚀 Run Full Scan", type="primary", use_container_width=True):
    with st.spinner("🔄 Running full NASDAQ analysis..."):
        try:
            st.session_state.scan_result = build_scan_report(symbols, progress_cb=None)
            st.success("✅ Scan complete!")
        except Exception as e:
            st.error(f"Scan error: {str(e)}")

# ==================== RESULTS SECTION ====================
if st.session_state.scan_result:
    result = st.session_state.scan_result
    momentum_df = result.get("momentum_df", pd.DataFrame())
    regime = result.get("regime", {})
    sector_strength = result.get("sector_strength", pd.DataFrame())
    shortlist_df = result.get("shortlist_df", pd.DataFrame())
    fund_map = result.get("fund_map", {})

    st.divider()

    # Market regime banner
    regime_label = regime.get("label", "SELECTIVE")
    regime_class = f"regime-{regime_label.lower()}"
    regime_desc = {
        "AGGRESSIVE": "✅ Bullish market conditions",
        "SELECTIVE": "⚖️ Mixed signals",
        "DEFENSIVE": "⚠️ Risk-off environment"
    }
    st.markdown(f"""
    <div style="background:rgba(20,30,50,0.9); border-left:4px solid #d4a843; padding:15px; border-radius:8px; margin-bottom:20px;">
        <div style="font-size:1.1rem; font-weight:700;">Market Regime</div>
        <div class="{regime_class}" style="font-size:1.4rem; margin-top:4px;">{regime_label}</div>
        <div style="color:#a0b4d8; font-size:0.85rem; margin-top:6px;">{regime_desc.get(regime_label, "")}</div>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Scan", len(momentum_df))
    with col2:
        stage_1_2 = len(momentum_df[momentum_df["StageId"].isin([1, 2])])
        st.metric("Stage 1-2", stage_1_2)
    with col3:
        breakouts = len(momentum_df[momentum_df["StageId"] == 3])
        st.metric("Breakouts", breakouts)
    with col4:
        avg_score = momentum_df["Score"].mean() if "Score" in momentum_df.columns else 0
        st.metric("Avg Score", f"{avg_score:.0f}")
    with col5:
        shortlist_count = len(shortlist_df)
        st.metric("Top Picks (A/A+)", shortlist_count)

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Full Scan",
        "🔥 Top Picks",
        "🔍 Search Stock",
        "📈 Sector Strength",
        "💾 Download"
    ])

    # ========== TAB 1: Full Scan ==========
    with tab1:
        st.subheader("All Scanned Stocks (Ranked by Score)")

        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            min_score = st.slider("Min Score", 0, 100, 0)
        with col2:
            stage_filter = st.multiselect("Stage Filter", [1, 2, 3, 4], default=[1, 2, 3])
        with col3:
            grade_filter = st.multiselect("Grade Filter", ["A+", "A", "B+", "B", "C"], default=["A+", "A", "B+"])

        # Apply filters
        display_df = momentum_df.copy()
        if min_score > 0:
            display_df = display_df[display_df["Score"] >= min_score]
        if stage_filter:
            display_df = display_df[display_df["StageId"].isin(stage_filter)]
        if "Grade" in display_df.columns:
            display_df = display_df[display_df["Grade"].isin(grade_filter)]

        display_df = display_df.sort_values("Score", ascending=False)

        # Show table
        cols_to_show = ["Ticker", "Price ₹", "Score", "Grade", "Stage", "RSI", "ADX", "Vol Ratio", "Target ₹", "Upside %"]
        cols_to_show = [c for c in cols_to_show if c in display_df.columns]
        st.dataframe(display_df[cols_to_show], use_container_width=True, height=400)

        st.caption(f"Showing {len(display_df)} of {len(momentum_df)} stocks")

    # ========== TAB 2: Top Picks ==========
    with tab2:
        st.subheader("🏆 Top Recommended Stocks")

        if shortlist_df.empty:
            st.info("No stocks meet top-pick criteria yet.")
        else:
            top_picks = shortlist_df.head(10)

            for idx, (_, row) in enumerate(top_picks.iterrows(), 1):
                ticker = row["Ticker"]
                rec = fund_map.get(ticker, {}).get("recommendation", {})
                action = rec.get("action", "HOLD")
                conviction = rec.get("conviction", 0)

                # Color code by action
                action_class = {
                    "STRONG_BUY": "action-strong-buy",
                    "BUY": "action-buy",
                    "AVOID": "action-avoid",
                    "HOLD": "action-hold"
                }.get(action, "action-hold")

                st.markdown(f"""
                <div class="pick-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                        <div class="ticker-name">{ticker}</div>
                        <div class="{action_class}" style="font-size:1.1rem;">{action}</div>
                    </div>
                    <div class="metric-row">
                        <span>Price:</span>
                        <strong>${row.get("Price ₹", "N/A")}</strong>
                    </div>
                    <div class="metric-row">
                        <span>Score:</span>
                        <strong>{row.get("Score", "N/A"):.0f}</strong>
                    </div>
                    <div class="metric-row">
                        <span>Conviction:</span>
                        <strong>{conviction:.0f}%</strong>
                    </div>
                    <div class="metric-row">
                        <span>Stage:</span>
                        <strong>{row.get("Stage", "N/A")}</strong>
                    </div>
                    <div class="metric-row">
                        <span>Target:</span>
                        <strong>${rec.get("targets", {}).get("target1", "N/A")}</strong>
                    </div>
                """, unsafe_allow_html=True)

                # Bull case
                bull_case = rec.get("bull_case", [])
                if bull_case:
                    st.markdown("**Bull Case:**")
                    for point in bull_case[:3]:
                        st.markdown(f"- {point}")

                st.markdown("</div>", unsafe_allow_html=True)

    # ========== TAB 3: Search Stock ==========
    with tab3:
        st.subheader("🔍 Deep Dive: Individual Stock Analysis")

        search_input = st.text_input("Enter ticker symbol (e.g., MSFT, NVDA, TSLA)", key="search_input").upper().strip()

        if search_input and len(search_input) <= 5:
            with st.spinner(f"Fetching data for {search_input}..."):
                try:
                    # Fetch fundamentals
                    fund = fetch_fundamentals_nasdaq(search_input)

                    if fund.get("error"):
                        st.error(f"Error: {fund['error']}")
                    else:
                        # Display basic info
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Price", f"${fund.get('price', 'N/A')}")
                        with col2:
                            st.metric("Market Cap", f"${fund.get('market_cap', 0) / 1e9:.1f}B")
                        with col3:
                            st.metric("P/E Ratio", f"{fund.get('pe_trailing', 'N/A')}")
                        with col4:
                            st.metric("Div Yield", f"{fund.get('dividend_yield', 0) * 100:.2f}%")

                        st.divider()

                        # Premium Metrics (6 precision metrics)
                        st.markdown("#### ⭐ Premium Precision Metrics")
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            piotroski = fund.get("piotroski_score", 0)
                            color = "🟢" if piotroski >= 7 else "🟡" if piotroski >= 5 else "🔴"
                            st.metric(
                                f"{color} Piotroski Score",
                                f"{piotroski}/9",
                                help="Quality signal: 7-9 = high quality, 0-2 = distressed"
                            )

                        with col2:
                            fcf_yield = fund.get("fcf_yield", 0)
                            st.metric(
                                "FCF Yield",
                                f"{fcf_yield:.2f}%",
                                help="FCF / Market Cap: >3% = attractive"
                            )

                        with col3:
                            ev_fcf = fund.get("ev_fcf", 0)
                            st.metric(
                                "EV/FCF Ratio",
                                f"{ev_fcf:.1f}x",
                                help="<20x = attractive, >30x = expensive"
                            )

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            margin_trend = fund.get("net_margin_trend", 0)
                            st.metric(
                                "Margin Trend (3Y)",
                                f"{margin_trend:+.1f}%",
                                help="Net margin change: improving >0"
                            )

                        with col2:
                            rd_pct = fund.get("rd_percent", 0)
                            st.metric(
                                "R&D % Revenue",
                                f"{rd_pct:.1f}%",
                                help="Innovation capacity"
                            )

                        with col3:
                            insider_signal = fund.get("insider_signal", 0.5)
                            signal_text = "Buying" if insider_signal > 0.6 else "Selling" if insider_signal < 0.4 else "Neutral"
                            st.metric(
                                "Insider Signal",
                                signal_text,
                                help=f"Score: {insider_signal:.2f}"
                            )

                        st.divider()

                        # Traditional fundamentals
                        st.markdown("#### 📊 Valuation & Quality")
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("P/B Ratio", f"{fund.get('pb', 'N/A')}")
                        with col2:
                            st.metric("PEG Ratio", f"{fund.get('peg', 'N/A')}")
                        with col3:
                            st.metric("ROE", f"{fund.get('roe', 0) * 100:.1f}%")
                        with col4:
                            st.metric("D/E Ratio", f"{fund.get('debt_to_equity', 0):.2f}")

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("ROCE", f"{fund.get('roce', 0) * 100:.1f}%")
                        with col2:
                            st.metric("Op Margin", f"{fund.get('op_margin', 0) * 100:.1f}%")
                        with col3:
                            st.metric("Rev Growth", f"{fund.get('revenue_growth', 0) * 100:.1f}%")
                        with col4:
                            st.metric("EPS Growth", f"{fund.get('earnings_growth', 0) * 100:.1f}%")

                        st.divider()

                        # Analyst & Ownership
                        st.markdown("#### 🎯 Analyst & Ownership")
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Target Price", f"${fund.get('target_mean', 'N/A')}")
                        with col2:
                            st.metric("Analyst Count", f"{fund.get('analyst_count', 0)}")
                        with col3:
                            st.metric("Institutional Held", f"{fund.get('institutional_held', 0) * 100:.1f}%")

                        # News headlines
                        if fund.get("news_headlines"):
                            st.markdown("#### 📰 Recent News")
                            for headline in fund.get("news_headlines", [])[:3]:
                                st.markdown(f"- {headline}")

                except Exception as e:
                    st.error(f"Error fetching stock data: {str(e)}")

    # ========== TAB 4: Sector Strength ==========
    with tab4:
        st.subheader("🌐 Sector Leadership")

        if sector_strength.empty:
            st.info("Sector analysis not yet available.")
        else:
            # Top sectors
            top_sectors = sector_strength.head(5)
            st.markdown("**Top 5 Sectors (by momentum)**")
            for idx, (_, row) in enumerate(top_sectors.iterrows(), 1):
                sector = row.get("Sector", "N/A")
                score = row.get("Score", 0)
                st.progress(min(score / 100, 1.0), text=f"{sector}: {score:.0f}")

            # Show full table
            with st.expander("Show all sectors"):
                cols_to_show = [c for c in ["Sector", "Score", "1M Return %", "3M Return %", "6M Return %"] if c in sector_strength.columns]
                st.dataframe(sector_strength[cols_to_show], use_container_width=True)

    # ========== TAB 5: Download ==========
    with tab5:
        st.subheader("💾 Export Results")

        # Full scan CSV
        if not momentum_df.empty:
            csv_data = momentum_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Scan (CSV)",
                data=csv_data,
                file_name=f"nasdaq_scan_{datetime.now(ET).strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

        # Shortlist CSV
        if not shortlist_df.empty:
            shortlist_with_recs = shortlist_df.copy()
            shortlist_with_recs["Action"] = [
                fund_map.get(t, {}).get("recommendation", {}).get("action", "HOLD")
                for t in shortlist_df["Ticker"]
            ]
            shortlist_with_recs["Conviction"] = [
                fund_map.get(t, {}).get("recommendation", {}).get("conviction", 0)
                for t in shortlist_df["Ticker"]
            ]

            csv_data = shortlist_with_recs.to_csv(index=False)
            st.download_button(
                label="📥 Download Top Picks (CSV)",
                data=csv_data,
                file_name=f"nasdaq_picks_{datetime.now(ET).strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

        st.divider()
        st.caption("📊 Data last updated: " + datetime.now(ET).strftime("%d %b %Y %H:%M ET"))

# ==================== FOOTER ====================
st.divider()
st.caption("""
**NASDAQ Alpha Engine** • Powered by Piotroski + FCF Yield + EV/FCF •
6 Premium Precision Metrics for World-Class Stock Analysis
""")
