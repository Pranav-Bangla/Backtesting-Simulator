# Backtesting Simulator

A Python web app for simulating quantitative trading strategies on real historical stock data, with Monte Carlo analysis to project future performance.

---

## What it does

The app has two core parts:

**Backtester** — Pick a stock, a date range, an initial investment, and a strategy. The app fetches real historical price data and simulates what would have happened if you had followed that strategy day by day. It tracks your portfolio value, compares it against a buy-and-hold benchmark, and computes performance metrics like return, Sharpe ratio, max drawdown, and win rate.

**Monte Carlo Simulation** — Rather than showing a single historical outcome, the app runs multiple forward projections by resampling the strategy's historical daily returns. This produces a realistic range of best, median, and worst case outcomes — along with decision-relevant metrics that tell you whether the strategy is actually worth using.

---

## Features

- 2 quantitative strategies — Moving Average Crossover and Bollinger Bands
- Interactive price chart with buy/sell markers and indicator lines plotted directly on the price history
- Portfolio growth chart vs buy-and-hold benchmark
- Drawdown chart showing the worst dips the strategy experienced
- Monte Carlo fan chart showing the spread of projected paths across percentile bands
- Return distribution histogram with Value at Risk markers
- Automatic strategy verdict — reads the Monte Carlo results and tells you if the strategy looks viable, risky, or mixed
- Fully configurable — all strategy parameters and Monte Carlo settings are adjustable from the sidebar
- Trade log showing every buy and sell with price and portfolio value at the time

---

## Strategies

| Strategy | Type | Logic |
|---|---|---|
| Moving Average Crossover | Trend following | Buy when the short-term average crosses above the long-term average (Golden Cross); sell on the reverse |
| Bollinger Bands | Mean reversion | Buy when price touches the lower band (oversold); sell when it touches the upper band (overbought) |

These two strategies represent opposite philosophies — Moving Average Crossover bets on trends continuing, while Bollinger Bands bets on prices snapping back to their average. Running the same stock through both strategies and comparing the results is one of the most useful things you can do with this app.

---

## Monte Carlo Metrics

The simulation returns metrics across four categories designed to help decide whether to trust and use the strategy:

**Return Scenarios** — Projects a range of outcomes from best case to worst case, giving you a realistic spread rather than a single number to anchor on.

**Probability Metrics** — Answers the core questions: how likely is it that you profit, how likely is it that you beat simply holding the stock, and what is the realistic chance of a significant loss.

**Risk Metrics** — Value at Risk tells you the minimum loss in the worst scenarios. Expected Shortfall tells you the average loss across those worst scenarios. Drawdown metrics show how bad the dips could get before recovery.

**Reliability** — The consistency score measures how predictable the outcomes are across simulations. A low score means the strategy's results are highly sensitive to market conditions, which is itself an important signal.

---

## Project Structure

```
backtesting-simulator/
├── app.py              # Streamlit UI — sidebar, tabs, charts, metrics dashboard
├── data.py             # StockDataFetcher — downloads and cleans OHLCV data via yfinance
├── strategies.py       # BaseStrategy (ABC) + MovingAverageCrossover + BollingerBands
├── backtester.py       # Backtester — core simulation loop and performance metrics
├── monte_carlo.py      # MonteCarloSimulator — bootstrapped projections and risk metrics
├── charts.py           # ChartBuilder — all Plotly figures
├── requirements.txt
└── README.md
```

### Data flow

​```
StockDataFetcher → BaseStrategy → Backtester → MonteCarloSimulator → ChartBuilder
  (fetch data)    (add signals)   (simulate)    (project forward)    (visualise)
​```

Adding a new strategy is straightforward — subclass `BaseStrategy` and implement `generate_signals()`.

---

## Getting Started

### Prerequisites
- Python 3.10 or higher

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-username/backtesting-simulator.git
cd backtesting-simulator

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python3 -m streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

To stop the app, press `Ctrl + C` in the terminal.

---

## How to use it

1. Enter a **ticker symbol** in the sidebar — e.g. `AAPL`, `MSFT`, `TSLA`, `NVDA`
2. Set a **date range** 
3. Set your **initial investment**
4. Choose a **strategy** and adjust its parameters using the sliders
5. Configure **Monte Carlo settings** — number of simulations and how far forward to project
6. Click **Backtest & Simulate**

Results are split across three tabs: Backtest Results, Monte Carlo Analysis, and Trade Log.

---

## Tech Stack

| Library | Purpose |
|---|---|
| `yfinance` | Free historical OHLCV price data — no API key required |
| `pandas` | Data manipulation and indicator calculations |
| `numpy` | Numerical computations — Sharpe ratio, drawdown, Monte Carlo sampling |
| `plotly` | Interactive charts |
| `streamlit` | Web UI |

---

## License

MIT
