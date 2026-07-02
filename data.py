import yfinance as yf
import pandas as pd


class StockDataFetcher:
    """
    Responsible for downloading and preparing historical stock price data.
    All other classes depend on the DataFrame this class produces.
    """

    def __init__(self, ticker: str, start: str, end: str):
        self.ticker = ticker.upper()
        self.start = start
        self.end = end
        self._raw: pd.DataFrame | None = None

    def fetch(self) -> pd.DataFrame:
        """Download OHLCV data and return a clean DataFrame."""
        raw = yf.download(self.ticker, start=self.start, end=self.end, progress=False)

        if raw.empty:
            raise ValueError(f"No data found for ticker '{self.ticker}'. "
                             "Check the symbol and date range.")

        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.dropna(inplace=True)
        self._raw = df
        return df

    @property
    def data(self) -> pd.DataFrame:
        if self._raw is None:
            raise RuntimeError("Call fetch() before accessing data.")
        return self._raw
