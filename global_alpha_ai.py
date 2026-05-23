"""
Global Alpha AI - NASDAQ Recommendation Engine
Complete Streamlit UI with AI intelligence layer
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# Import NASDAQ engine modules
from nasdaq_scan import build_scan_report
from nasdaq_data import parse_nasdaq_screener_csv, get_sp500_list
from nasdaq_fundamentals import fetch_fundamentals_nasdaq
from nasdaq_ai import (
    get_live_market_snapshot, fetch_market_news, fetch_stock_news,
    generate_market_intelligence, generate_strategy_brief, generate_stock_thesis,
    get_snapshot_summary
)
from nasdaq_export import build_excel_workbook

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
.intel-brief { background: rgba(4,14,10,0.8); border-left: 4px solid var(--emerald); padding: 20px; border-radius: 12px; margin: 20px 0; }
.strategy-brief { background: rgba(20,30,50,0.8); border-left: 4px solid var(--blue); padding: 20px; border-radius: 12px; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================
st.markdown(f"""
<div style="padding:30px 0 20px;">
    <span style="font-family:'Cinzel',serif; font-size:2.8rem; font-weight:900; letter-spacing:0.07em; background:linear-gradient(110deg,#d4a843,#f0c96a); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        NASDAQ ALPHA ENGINE
    </span>
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#8896b3; margin-top:8px;">
        AI-Powered Precision Metrics • {datetime.now(ET).strftime("%d %b %Y %H:%M ET")}
    </div>
</div>
""", unsafe_allow_html=True)

# ==================== LIVE MARKET SNAPSHOT ====================
snapshot = get_live_market_snapshot()
cols = st.columns(4)
for idx, (name, color) in enumerate([("S&P 500", "#1f77b4"), ("Nasdaq", "#ff7f0e"), ("VIX", "#d62728"), ("DXY", "#2ca02c")]):
    with cols[idx]:
        if name in snapshot:
            data = snapshot[name]
            price = data.get("price", "N/A")
            chg = data.get("change", 0)
            st.metric(
                name,
                f"{price:.2f}" if isinstance(price, (int, float)) else price,
                f"{chg:+.2f}%"
            )

# ==================== SESSION STATE ====================
if "scan_result" not in st.session_state:
    st.session_state.scan_result = None

# ==================== INPUT SECTION ====================
st.divider()
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("#### 📋 Input: CSV Screener or NASDAQ Default")
    input_mode = st.radio("Choose input method:", ["Upload CSV", "Use NASDAQ Default"], horizontal=True)

with col2:
    st.markdown("#### 🎯")

symbols = None

if input_mode == "Upload CSV":
    uploaded_file = st.file_uploader("Upload NASDAQ Screener CSV (Symbol, Sector, etc.)", type=["csv"])
    if uploaded_file:
        try:
            symbols = parse_nasdaq_screener_csv(uploaded_file)
            if symbols:
                st.success(f"✅ Loaded **{len(symbols)}** stocks from CSV")
            else:
                st.error("CSV parsing failed or no valid symbols found.")
                symbols = None
        except Exception as e:
            st.error(f"CSV parsing error: {e}")
            symbols = None
else:
    symbols = get_sp500_list()
    st.info(f"📊 Using NASDAQ default universe ({len(symbols)} stocks - tech-heavy, growth-focused)")

# ==================== DEBUG / MANUAL SEARCH ====================
with st.expander("🔍 Quick Diagnostics"):
    if st.button("Test Data Download (5 mega-caps)"):
        with st.spinner("Testing yfinance downloads..."):
            try:
                test_symbols = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN']
                from nasdaq_data import parallel_fetch_nasdaq
                test_data = parallel_fetch_nasdaq(test_symbols)

                success_count = sum(1 for v in test_data.values() if v is not None and not v.empty)
                st.success(f"✅ Downloaded {success_count}/{len(test_symbols)} stocks successfully")

                for sym, df in test_data.items():
                    if df is not None and not df.empty:
                        st.caption(f"{sym}: {len(df)} candles, latest price ${df['Close'].iloc[-1]:.2f}")
                    else:
                        st.caption(f"{sym}: ⚠️ No data")
            except Exception as e:
                st.error(f"Download test failed: {str(e)}")

with st.expander("🔍 Add Stocks Manually (Search Tab)"):
    search_ticker = st.text_input("Search ticker (e.g., MSFT, NVDA)", key="search_ticker").upper().strip()

# ==================== RUN SCAN ====================
if symbols and st.button("🚀 Run Full Scan", type="primary", use_container_width=True):
    with st.spinner("🔄 Running full NASDAQ analysis (this may take 3-5 minutes)..."):
        try:
            st.session_state.scan_result = build_scan_report(symbols, progress_cb=None)

            # Check if results are empty
            momentum_df = st.session_state.scan_result.get("momentum_df", pd.DataFrame())
            shortlist_df = st.session_state.scan_result.get("shortlist_df", pd.DataFrame())

            if momentum_df.empty:
                st.error(f"❌ No scan results. Checked {len(symbols)} stocks but none returned valid data.")
                with st.expander("🔧 Troubleshooting"):
                    st.write("""
**Possible causes:**
1. **yfinance rate limit** - Try again in 30 seconds
2. **Network issue** - Check your internet connection
3. **Bad symbols** - Some stocks may not exist or be delisted
4. **Sparse data** - Stocks may not have enough price history

**Solutions:**
- Click "Test Data Download" in Quick Diagnostics to check if data works
- Use CSV upload with your own stock list instead of defaults
- Try uploading a smaller CSV (10-20 stocks) to test
- If test download works but scan fails, the issue is data availability
                    """)
            elif shortlist_df.empty:
                st.warning(f"📊 Scan found {len(momentum_df)} stocks with momentum, but none qualified as 'Top Picks' (Score < 45).")
                st.info("Check the Full Scan tab to see all candidates.")
            else:
                st.success(f"✅ Scan complete! Found {len(momentum_df)} candidates, {len(shortlist_df)} top picks.")

        except Exception as e:
            st.error(f"Scan error: {str(e)}")
            st.info("This may indicate a network issue or data availability problem. Try again in a moment.")

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
        stage_1_2 = len(momentum_df[momentum_df["StageId"].isin([1, 2])]) if "StageId" in momentum_df.columns else 0
        st.metric("Stage 1-2", stage_1_2)
    with col3:
        breakouts = len(momentum_df[momentum_df["StageId"] == 3]) if "StageId" in momentum_df.columns else 0
        st.metric("Breakouts", breakouts)
    with col4:
        avg_score = momentum_df["Score"].mean() if "Score" in momentum_df.columns else 0
        st.metric("Avg Score", f"{avg_score:.0f}")
    with col5:
        st.metric("Top Picks", len(shortlist_df))

    # AI Market Intelligence
    with st.spinner("🧠 Generating AI market intelligence..."):
        try:
            news = fetch_market_news()
            intelligence = generate_market_intelligence(regime, shortlist_df, news)
            st.markdown(f'<div class="intel-brief">{intelligence}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"AI brief unavailable: {str(e)}")

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Full Scan",
        "🔥 Top Picks",
        "🔍 Search Stock",
        "📈 Sector Strength",
        "🧠 AI Strategy",
        "💾 Download"
    ])

    # ========== TAB 1: Full Scan ==========
    with tab1:
        st.subheader("All Scanned Stocks (Ranked by Score)")

        col1, col2, col3 = st.columns(3)
        with col1:
            min_score = st.slider("Min Score", 0, 100, 0)
        with col2:
            stage_filter = st.multiselect("Stage Filter", [1, 2, 3, 4], default=[1, 2, 3])
        with col3:
            grade_filter = st.multiselect("Grade Filter", ["A+", "A", "B+", "B", "C"], default=["A+", "A", "B+"])

        display_df = momentum_df.copy()
        if min_score > 0 and "Score" in display_df.columns:
            display_df = display_df[display_df["Score"] >= min_score]
        if stage_filter and "StageId" in display_df.columns:
            display_df = display_df[display_df["StageId"].isin(stage_filter)]
        if "Grade" in display_df.columns:
            display_df = display_df[display_df["Grade"].isin(grade_filter)]

        if "Score" in display_df.columns:
            display_df = display_df.sort_values("Score", ascending=False)

        cols_to_show = ["Ticker", "Price $", "Score", "Grade", "Stage", "RSI", "ADX", "Vol Ratio", "Target $", "Upside %"]
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
                        <strong>${row.get("Price $", "N/A")}</strong>
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
                """, unsafe_allow_html=True)

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
                    fund = fetch_fundamentals_nasdaq(search_input)

                    if fund.get("error"):
                        st.error(f"Error: {fund['error']}")
                    else:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Price", f"${fund.get('price', 'N/A')}")
                        with col2:
                            mc = fund.get('market_cap', 0)
                            st.metric("Market Cap", f"${mc / 1e9:.1f}B" if mc else "N/A")
                        with col3:
                            st.metric("P/E Ratio", f"{fund.get('pe_trailing', 'N/A')}")
                        with col4:
                            st.metric("Div Yield", f"{fund.get('dividend_yield', 0) * 100:.2f}%")

                        st.divider()

                        # Premium Metrics
                        st.markdown("#### ⭐ Premium Precision Metrics")
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            piotroski = fund.get("piotroski_score", 0)
                            color = "🟢" if piotroski >= 7 else "🟡" if piotroski >= 5 else "🔴"
                            st.metric(f"{color} Piotroski", f"{piotroski}/9")

                        with col2:
                            fcf_yield = fund.get("fcf_yield", 0)
                            st.metric("FCF Yield", f"{fcf_yield:.2f}%")

                        with col3:
                            ev_fcf = fund.get("ev_fcf", 0)
                            st.metric("EV/FCF", f"{ev_fcf:.1f}x")

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            margin_trend = fund.get("net_margin_trend", 0)
                            st.metric("Margin Trend (3Y)", f"{margin_trend:+.1f}%")

                        with col2:
                            rd_pct = fund.get("rd_percent", 0)
                            st.metric("R&D % Revenue", f"{rd_pct:.1f}%")

                        with col3:
                            insider_signal = fund.get("insider_signal", 0.5)
                            signal_text = "Buying" if insider_signal > 0.6 else "Selling" if insider_signal < 0.4 else "Neutral"
                            st.metric("Insider Signal", signal_text)

                        st.divider()

                        # Fundamentals
                        st.markdown("#### 📊 Valuation & Quality")
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("P/B", f"{fund.get('pb', 'N/A')}")
                        with col2:
                            st.metric("PEG", f"{fund.get('peg', 'N/A')}")
                        with col3:
                            st.metric("ROE", f"{fund.get('roe', 0) * 100:.1f}%")
                        with col4:
                            st.metric("D/E", f"{fund.get('debt_to_equity', 0):.2f}")

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

                        # AI Thesis
                        st.markdown("#### 🧠 AI Analysis")
                        with st.spinner("Generating AI thesis..."):
                            try:
                                news = fetch_stock_news(search_input)
                                technicals = {}  # Would need to fetch from scan
                                thesis = generate_stock_thesis(search_input, fund, technicals, news)
                                st.markdown(f'<div class="intel-brief">{thesis}</div>', unsafe_allow_html=True)
                            except Exception as e:
                                st.warning(f"AI thesis unavailable: {str(e)}")

                        # News
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
            st.info("Sector analysis not available.")
        else:
            top_sectors = sector_strength.head(5)
            st.markdown("**Top 5 Sectors (by momentum)**")
            for idx, (_, row) in enumerate(top_sectors.iterrows(), 1):
                sector = row.get("Sector", "N/A")
                score = row.get("Score", 0)
                st.progress(min(score / 100, 1.0), text=f"{sector}: {score:.0f}")

            with st.expander("Show all sectors"):
                cols_to_show = [c for c in ["Sector", "Score", "1M Return %", "3M Return %", "6M Return %"] if c in sector_strength.columns]
                st.dataframe(sector_strength[cols_to_show], use_container_width=True)

    # ========== TAB 5: AI Strategy ==========
    with tab5:
        st.subheader("🧠 AI Strategy Brief")

        with st.spinner("Generating strategy..."):
            try:
                strategy = generate_strategy_brief(shortlist_df, fund_map, regime)
                st.markdown(f'<div class="strategy-brief">{strategy}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Strategy generation failed: {str(e)}")

    # ========== TAB 6: Download ==========
    with tab6:
        st.subheader("💾 Export Results")

        if not momentum_df.empty:
            csv_data = momentum_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Scan (CSV)",
                data=csv_data,
                file_name=f"nasdaq_scan_{datetime.now(ET).strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

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

        # Excel export
        with st.spinner("Building Excel workbook..."):
            try:
                excel_buffer = build_excel_workbook(result, fund_map)
                st.download_button(
                    label="📊 Download Full Report (Excel 6+ sheets)",
                    data=excel_buffer,
                    file_name=f"nasdaq_report_{datetime.now(ET).strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"Excel export failed: {str(e)}")

        st.divider()
        st.caption("📊 Data last updated: " + datetime.now(ET).strftime("%d %b %Y %H:%M ET"))

# ==================== FOOTER ====================
st.divider()
st.caption("""
**NASDAQ Alpha Engine** • AI-Powered Precision Metrics •
Piotroski + FCF Yield + EV/FCF + Margin Trends + R&D % + Insider Signals
""")
