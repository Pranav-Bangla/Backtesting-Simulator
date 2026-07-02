import pandas as pd
import numpy as np


class Backtester:
    """
    Core loop: walks through a signal DataFrame day by day, simulates trades,
    and tracks portfolio value.

    Parameters
    ----------
    df            : pd.DataFrame  Output from a strategy's generate_signals()
    initial_cash  : float         Starting capital in USD (default $10,000)
    """

    def __init__(self, df: pd.DataFrame, initial_cash: float = 10_000.0):
        self.df = df.copy()
        self.initial_cash = initial_cash

    # ------------------------------------------------------------------ #
    #  Core Loop                                                           #
    # ------------------------------------------------------------------ #

    def run(self) -> pd.DataFrame:
        """
        Simulate trades day by day.
        Returns enriched DataFrame with portfolio value, cash, holdings columns.
        """
        cash     = self.initial_cash
        shares   = 0.0
        cash_log    = []
        shares_log  = []
        portfolio_log = []

        for _, row in self.df.iterrows():
            price  = float(row["Close"])
            signal = int(row["Signal"])

            if signal == 1 and cash > 0:          # BUY — spend all cash
                shares = cash / price
                cash   = 0.0

            elif signal == -1 and shares > 0:     # SELL — liquidate all shares
                cash   = shares * price
                shares = 0.0

            portfolio_value = cash + shares * price
            cash_log.append(cash)
            shares_log.append(shares)
            portfolio_log.append(portfolio_value)

        self.df["Cash"]      = cash_log
        self.df["Shares"]    = shares_log
        self.df["Portfolio"] = portfolio_log

        # Buy-and-hold benchmark
        first_price = float(self.df["Close"].iloc[0])
        bh_shares   = self.initial_cash / first_price
        self.df["BuyAndHold"] = bh_shares * self.df["Close"]

        return self.df

    # ------------------------------------------------------------------ #
    #  Performance Metrics                                                 #
    # ------------------------------------------------------------------ #

    def metrics(self) -> dict:
        """Compute summary statistics after run() has been called."""
        if "Portfolio" not in self.df.columns:
            raise RuntimeError("Call run() before metrics().")

        final   = self.df["Portfolio"].iloc[-1]
        bh_final = self.df["BuyAndHold"].iloc[-1]

        total_return = (final - self.initial_cash) / self.initial_cash * 100
        bh_return    = (bh_final - self.initial_cash) / self.initial_cash * 100

        # Daily returns for Sharpe / max-drawdown
        daily_ret = self.df["Portfolio"].pct_change().dropna()
        sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
                  if daily_ret.std() > 0 else 0.0)

        roll_max  = self.df["Portfolio"].cummax()
        drawdown  = (self.df["Portfolio"] - roll_max) / roll_max
        max_dd    = drawdown.min() * 100

        # Trade count
        buys  = (self.df["Signal"] ==  1).sum()
        sells = (self.df["Signal"] == -1).sum()

        # Win rate — compare return of each completed trade
        win_rate = self._win_rate()

        return {
            "Final Portfolio ($)":      round(final, 2),
            "Total Return (%)":         round(total_return, 2),
            "Buy & Hold Return (%)":    round(bh_return, 2),
            "Sharpe Ratio":             round(sharpe, 2),
            "Max Drawdown (%)":         round(max_dd, 2),
            "Number of Buys":           int(buys),
            "Number of Sells":          int(sells),
            "Win Rate (%)":             win_rate,
        }

    def _win_rate(self) -> float:
        """Percentage of sell trades that closed at a profit."""
        buy_price  = None
        wins, total = 0, 0

        for _, row in self.df.iterrows():
            if row["Signal"] == 1:
                buy_price = float(row["Close"])
            elif row["Signal"] == -1 and buy_price is not None:
                total += 1
                if float(row["Close"]) > buy_price:
                    wins += 1
                buy_price = None

        return round(wins / total * 100, 1) if total > 0 else 0.0
