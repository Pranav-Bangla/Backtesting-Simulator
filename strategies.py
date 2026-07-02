import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


# ─────────────────────────────────────────────────────────────────────────────
#  Base Strategy
# ─────────────────────────────────────────────────────────────────────────────

class BaseStrategy(ABC):
    """
    Every strategy receives a price DataFrame, adds indicator columns to a
    working copy, and produces a 'Signal' column:
        +1  →  buy
        -1  →  sell
         0  →  hold / no position
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    @abstractmethod
    def generate_signals(self) -> pd.DataFrame:
        """Compute indicators, set self.df['Signal'], return self.df."""

    def _init_signal_col(self):
        self.df["Signal"] = 0


# ─────────────────────────────────────────────────────────────────────────────
#  Strategy 1 – Moving Average Crossover
# ─────────────────────────────────────────────────────────────────────────────

class MovingAverageCrossover(BaseStrategy):
    """
    Buy when the short-term rolling mean crosses above the long-term rolling
    mean (Golden Cross); sell when it crosses back below (Death Cross).
    """

    def __init__(self, df: pd.DataFrame, short_window: int = 50, long_window: int = 200):
        super().__init__(df)
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self) -> pd.DataFrame:
        self._init_signal_col()
        self.df["MA_Short"] = self.df["Close"].rolling(self.short_window).mean()
        self.df["MA_Long"]  = self.df["Close"].rolling(self.long_window).mean()
        position = (self.df["MA_Short"] > self.df["MA_Long"]).astype(int)
        self.df["Signal"] = position.diff().fillna(0).astype(int)
        return self.df


# ─────────────────────────────────────────────────────────────────────────────
#  Strategy 2 – Bollinger Bands
# ─────────────────────────────────────────────────────────────────────────────

class BollingerBands(BaseStrategy):
    """
    Mean reversion strategy using Bollinger Bands.

    Three lines are drawn around the price:
        Middle Band  = N-day rolling mean
        Upper Band   = Middle + (std_dev × multiplier)
        Lower Band   = Middle − (std_dev × multiplier)

    Logic
    -----
    - Price touches / drops below the Lower Band → unusually oversold → BUY
    - Price touches / rises above the Upper Band → unusually overbought → SELL

    Parameters
    ----------
    window     : int    Rolling window for mean and std (default 20 days)
    num_std    : float  Number of standard deviations for the bands (default 2.0)
    """

    def __init__(self, df: pd.DataFrame, window: int = 20, num_std: float = 2.0):
        super().__init__(df)
        self.window  = window
        self.num_std = num_std

    def generate_signals(self) -> pd.DataFrame:
        self._init_signal_col()

        close = self.df["Close"]
        self.df["BB_Middle"] = close.rolling(self.window).mean()
        self.df["BB_Std"]    = close.rolling(self.window).std()
        self.df["BB_Upper"]  = self.df["BB_Middle"] + self.num_std * self.df["BB_Std"]
        self.df["BB_Lower"]  = self.df["BB_Middle"] - self.num_std * self.df["BB_Std"]

        close_arr  = close.to_numpy()
        upper_arr  = self.df["BB_Upper"].to_numpy()
        lower_arr  = self.df["BB_Lower"].to_numpy()
        sigs       = [0] * len(close_arr)
        in_pos     = False

        for i in range(len(close_arr)):
            if np.isnan(lower_arr[i]):
                continue
            if not in_pos and close_arr[i] <= lower_arr[i]:
                sigs[i] = 1        # price hit lower band → buy
                in_pos  = True
            elif in_pos and close_arr[i] >= upper_arr[i]:
                sigs[i] = -1       # price hit upper band → sell
                in_pos  = False

        self.df["Signal"] = sigs
        return self.df
