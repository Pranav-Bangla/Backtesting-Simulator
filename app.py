import streamlit as st
import pandas as pd
from datetime import date, timedelta

from data import StockDataFetcher
from strategies import MovingAverageCrossover, BollingerBands
from backtester import Backtester
from monte_carlo import MonteCarloSimulator
from charts import ChartBuilder

# ─────────────────────────────────────────────────────────────────────────────
#  Page Config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Backtesting Simulator",
    page_icon="📊",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0D1117;
    color: #C9D1D9;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background-color: #161B22;
    border-right: 1px solid #21262D;
  }
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 {
    color: #8B949E !important;
    font-size: 1.7rem !important;
    font-weight: 600 !important;
  }
  [data-testid="stSidebar"] p strong {
    color: #8B949E !important;
  }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stTextInput label,
  [data-testid="stSidebar"] .stSlider label,
  [data-testid="stSidebar"] .stDateInput label,
  [data-testid="stSidebar"] .stNumberInput label {
    color: #8B949E;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  /* ── Metric cards ── */
  [data-testid="stMetric"] {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 8px;
    padding: 14px 18px;
  }
  [data-testid="stMetricLabel"] { color: #8B949E !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.05em; }
  [data-testid="stMetricValue"] { color: #C9D1D9 !important; font-family: 'JetBrains Mono', monospace !important; font-size: 1.15rem !important; }
  [data-testid="stMetricDelta"] { font-family: 'JetBrains Mono', monospace !important; font-size: 0.8rem !important; }

  /* ── Page header ── */
  .app-header {
    padding: 20px 0 12px 0;
    border-bottom: 1px solid #21262D;
    margin-bottom: 24px;
  }
  .app-header h1 {
    font-size: 2.3rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #000000;
    margin: 0;
  }
  .app-header p {
    color: #6E7681;
    font-size: 0.88rem;
    margin: 4px 0 0 0;
  }

  /* ── Section labels ── */
  .section-label {
    font-size: 1.0rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #8B949E;
    margin: 28px 0 10px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #21262D;
  }


  /* ── Run button ── */
  div.stButton > button {
    width: 100%;
    background: #00C896;
    color: #0D1117;
    font-weight: 700;
    font-size: 0.88rem;
    border: none;
    border-radius: 6px;
    padding: 12px;
    margin-top: 12px;
    letter-spacing: 0.02em;
    transition: opacity 0.15s;
  }
  div.stButton > button:hover { opacity: 0.82; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: #161B22;
    border-radius: 8px 8px 0 0;
    gap: 4px;
    padding: 4px;
    border-bottom: 1px solid #21262D;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8B949E;
    border-radius: 6px;
    font-size: 0.83rem;
    padding: 8px 16px;
  }
  .stTabs [aria-selected="true"] {
    background: #21262D !important;
    color: #C9D1D9 !important;
  }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Stock Selection & Strategy")

    ticker = st.text_input("Ticker Symbol", value="AAPL",
                           placeholder="e.g. AAPL, MSFT, TSLA").upper()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date",
                                   value=date.today() - timedelta(days=5 * 365))
    with col2:
        end_date = st.date_input("End Date", value=date.today())

    initial_cash = st.number_input("Initial Investment ($)", min_value=1_000,
                                   max_value=1_000_000, value=10_000, step=1_000)

    st.divider()
    strategy_name = st.selectbox(
        "Strategy",
        ["Moving Average Crossover", "Bollinger Bands"],
    )

    st.markdown(
        '<p style="color:#8B949E;font-weight:600;text-transform:uppercase;'
        'font-size:0.75rem;letter-spacing:0.08em;margin-top:12px;">'
        'Strategy Parameters</p>',
        unsafe_allow_html=True,
    )

    if strategy_name == "Moving Average Crossover":
        short_w = st.slider("Short Window (days)", 10, 100, 50)
        long_w  = st.slider("Long Window (days)",  50, 300, 200)
    else:
        bb_window  = st.slider("Band Window (days)", 5, 50, 20,
                               help="How many days to use for the rolling mean and standard deviation")
        bb_num_std = st.slider("Band Width (std devs)", 1.0, 3.0, 2.0, step=0.1,
                               help="How far the bands sit from the middle — wider = fewer but stronger signals")

    st.divider()
    st.markdown(
        '<p style="color:#8B949E;font-weight:600;text-transform:uppercase;'
        'font-size:0.75rem;letter-spacing:0.08em;">'
        'Monte Carlo Settings</p>',
        unsafe_allow_html=True,
    )
    n_simulations = st.slider("Number of Simulations", 100, 1000, 500, step=100)
    n_days        = st.slider("Days to Project Forward", 63, 504, 252, step=63,
                               help="63=3mo  126=6mo  252=1yr  504=2yr")

    st.divider()
    run_button = st.button("Backtest & Simulate")


# ─────────────────────────────────────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
  <h1>Backtesting Simulator</h1>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

if run_button:

    # ── 1. Fetch data ───────────────────────────────────────────────────
    with st.spinner(f"Fetching data for {ticker}…"):
        try:
            fetcher = StockDataFetcher(ticker, str(start_date), str(end_date))
            raw_df  = fetcher.fetch()
        except ValueError as e:
            st.error(str(e))
            st.stop()

    # ── 2. Generate signals ─────────────────────────────────────────────
    if strategy_name == "Moving Average Crossover":
        strategy = MovingAverageCrossover(raw_df, short_w, long_w)
    else:
        strategy = BollingerBands(raw_df, bb_window, bb_num_std)

    signal_df = strategy.generate_signals()

    # ── 3. Backtest ─────────────────────────────────────────────────────
    bt        = Backtester(signal_df, float(initial_cash))
    result_df = bt.run()
    bt_stats  = bt.metrics()

    if bt_stats["Number of Buys"] == 0:
        st.warning(
            "No buy signals were triggered. Try a longer date range or "
            "loosen the strategy parameters in the sidebar."
        )
        st.stop()

    # ── 4. Monte Carlo ──────────────────────────────────────────────────
    with st.spinner("Running Monte Carlo simulation…"):
        mc = MonteCarloSimulator(
            result_df["Portfolio"],
            initial_cash=float(initial_cash),
            n_simulations=n_simulations,
            n_days=n_days,
        )
        paths    = mc.run()
        mc_stats = mc.metrics()

    # ═══════════════════════════════════════════════════════════════════
    #  TAB LAYOUT
    # ═══════════════════════════════════════════════════════════════════
    tab1, tab2, tab3 = st.tabs([
        "📊  Backtest Results",
        "🎲  Monte Carlo Analysis",
        "📋  Trade Log",
    ])

    # ────────────────────────────────────────────────────────────────────
    #  TAB 1 — Backtest
    # ────────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-label">Performance Summary</div>',
                    unsafe_allow_html=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        total_ret = bt_stats["Total Return (%)"]
        bh_ret    = bt_stats["Buy & Hold Return (%)"]
        edge      = total_ret - bh_ret

        c1.metric("Final Portfolio",
                  f"${bt_stats['Final Portfolio ($)']:,.0f}",
                  f"{total_ret:+.2f}%")
        c2.metric("vs Buy & Hold",
                  f"{bh_ret:+.1f}%",
                  f"{edge:+.1f}% edge")
        c3.metric("Sharpe Ratio", bt_stats["Sharpe Ratio"])
        c4.metric("Max Drawdown", f"{bt_stats['Max Drawdown (%)']:.1f}%")
        c5.metric("Win Rate",
                  f"{bt_stats['Win Rate (%)']}%",
                  f"{bt_stats['Number of Sells']} trades")

        st.markdown('<div class="section-label">Price & Signals</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(ChartBuilder.price_chart(result_df, strategy_name),
                        use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.markdown('<div class="section-label">Portfolio Growth</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(ChartBuilder.portfolio_chart(result_df, float(initial_cash)),
                            use_container_width=True)
        with right:
            st.markdown('<div class="section-label">Drawdown</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(ChartBuilder.drawdown_chart(result_df),
                            use_container_width=True)

    # ────────────────────────────────────────────────────────────────────
    #  TAB 2 — Monte Carlo
    # ────────────────────────────────────────────────────────────────────
    with tab2:

        # ── Decision metrics grid ───────────────────────────────────────
        st.markdown('<div class="section-label">Probability Metrics</div>',
                    unsafe_allow_html=True)

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Probability of Profit",
                  f"{mc_stats['Probability of Any Profit (%)']:.1f}%",
                  help="% of simulations where you end up with more than you started")
        p2.metric("Beats the Market",
                  f"{mc_stats['Probability of Beating Market (%)']:.1f}%",
                  help="% of simulations where this strategy outperforms buy-and-hold")
        p3.metric("Risk of Losing >10%",
                  f"{mc_stats['Probability of Losing >10% (%)']:.1f}%",
                  help="% of simulations where you lose more than 10% of your investment")
        p4.metric("Risk of Losing >20%",
                  f"{mc_stats['Probability of Losing >20% (%)']:.1f}%",
                  help="% of simulations where you lose more than 20% of your investment")

        st.markdown('<div class="section-label">Return Scenarios</div>',
                    unsafe_allow_html=True)

        r1, r2, r3, r4, r5, r6 = st.columns(6)
        r1.metric("Best Case", f"{mc_stats['Best Case Return — top 10% (%)']:+.1f}%",
                  help="Return in the top 10% of simulations")
        r2.metric("Good Case", f"{mc_stats['Good Case Return — top 25% (%)']:+.1f}%",
                  help="Return in the top 25% of simulations")
        r3.metric("Median",    f"{mc_stats['Median Projected Return (%)']:+.1f}%",
                  help="The middle outcome — 50% of simulations do better, 50% worse")
        r4.metric("Bad Case",  f"{mc_stats['Bad Case Return — bottom 25% (%)']:+.1f}%",
                  help="Return in the bottom 25% of simulations")
        r5.metric("Worst Case",f"{mc_stats['Worst Case Return — bottom 10% (%)']:+.1f}%",
                  help="Return in the bottom 10% of simulations")
        r6.metric("Consistency",f"{mc_stats['Consistency Score (%)']:.0f}%",
                  help="% of simulations within ±15% of the median — higher means more predictable")

        st.markdown('<div class="section-label">Risk Metrics</div>',
                    unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Value at Risk (95%)",
                  f"{mc_stats['Value at Risk 95% (%)']:.1f}%",
                  help="In the worst 5% of scenarios, you lose at least this much")
        k2.metric("Expected Loss (Worst 5%)",
                  f"{mc_stats['Expected Loss in Worst 5% (%)']:.1f}%",
                  help="The average loss across all worst-case scenarios")
        k3.metric("Median Max Drawdown",
                  f"{mc_stats['Median Max Drawdown (%)']:.1f}%",
                  help="The typical worst dip you'd experience before recovering")
        k4.metric("Worst Drawdown (95th %ile)",
                  f"{mc_stats['Worst Drawdown — top 95% (%)']:.1f}%",
                  help="In the worst 5% of simulations, drawdown gets at least this bad")

        # ── Charts ──────────────────────────────────────────────────────
        st.markdown('<div class="section-label">Projected Outcome Paths</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            ChartBuilder.monte_carlo_fan(
                mc_stats["_paths"], float(initial_cash), n_days
            ),
            use_container_width=True,
        )

        st.markdown('<div class="section-label">Return Distribution</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            ChartBuilder.return_distribution(mc_stats["_returns_pct"]),
            use_container_width=True,
        )

    # ────────────────────────────────────────────────────────────────────
    #  TAB 3 — Trade Log
    # ────────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-label">All Trades</div>',
                    unsafe_allow_html=True)
        trades = result_df[result_df["Signal"] != 0][
            ["Close", "Signal", "Portfolio"]
        ].copy()
        trades["Action"] = trades["Signal"].map({1: "🟢 BUY", -1: "🔴 SELL"})
        trades = trades.rename(columns={
            "Close":     "Price ($)",
            "Portfolio": "Portfolio Value ($)",
        })
        st.dataframe(
            trades[["Action", "Price ($)", "Portfolio Value ($)"]],
            use_container_width=True,
        )

else:
    st.info("👈  Configure a strategy in the sidebar, then click **Run Backtest + Simulation**.")
    st.markdown("""
    ### How it works
    1. **Fetch** — Real historical price data is downloaded for your chosen ticker
    2. **Signal** — The strategy scans every trading day and triggers buy/sell signals
    3. **Backtest** — Simulates trading with your initial investment, tracking portfolio value day by day
    4. **Monte Carlo** — Runs simulations by resampling historical daily returns to project a range of realistic future outcomes

    ### Strategies
    | Strategy | Type | Logic |
    |---|---|---|
    | Moving Average Crossover | Trend following | Buy when the 50-day average crosses above the 200-day average; sell on the reverse |
    | Bollinger Bands | Mean reversion | Buy when price touches the lower band (oversold); sell when it touches the upper band (overbought) |

    ### What the Monte Carlo tells you
    | Metric | What it answers |
    |---|---|
    | Probability of Profit | "Will I likely make money?" |
    | Probability of Beating Market | "Is this better than doing nothing?" |
    | Risk of Losing >10/20% | "What's the realistic downside?" |
    | Value at Risk | "In a bad scenario, how much could I lose?" |
    | Consistency Score | "Are the outcomes predictable or all over the place?" |
    | Return Scenarios | "What does best/worst/median look like?" |
    """)
