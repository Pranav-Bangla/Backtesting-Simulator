import numpy as np
import pandas as pd


class MonteCarloSimulator:
    """
    Runs N simulations by resampling the strategy's historical daily returns
    (bootstrapping) to project a range of possible future outcomes.

    Why bootstrapping instead of pure random walks?
    ------------------------------------------------
    A pure random walk assumes returns are independent and normally distributed.
    Bootstrapping resamples the *actual* daily returns the strategy produced —
    preserving the real distribution, including fat tails and volatility
    clusters — giving more realistic projections.

    Parameters
    ----------
    portfolio_series : pd.Series   Daily portfolio values from Backtester.run()
    initial_cash     : float       Starting capital
    n_simulations    : int         Number of paths to simulate (default 500)
    n_days           : int         How many trading days to project (default 252 = 1 year)
    """

    def __init__(
        self,
        portfolio_series: pd.Series,
        initial_cash: float,
        n_simulations: int = 500,
        n_days: int = 252,
    ):
        self.portfolio   = portfolio_series
        self.initial_cash = initial_cash
        self.n_sim       = n_simulations
        self.n_days      = n_days
        self._paths: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    #  Run                                                                 #
    # ------------------------------------------------------------------ #

    def run(self) -> np.ndarray:
        """
        Returns shape (n_simulations, n_days+1) array of projected portfolio
        values, starting from the last known portfolio value.
        """
        daily_returns = self.portfolio.pct_change().dropna().to_numpy()
        start_value   = float(self.portfolio.iloc[-1])

        rng   = np.random.default_rng(seed=42)
        paths = np.empty((self.n_sim, self.n_days + 1))
        paths[:, 0] = start_value

        for day in range(1, self.n_days + 1):
            sampled = rng.choice(daily_returns, size=self.n_sim, replace=True)
            paths[:, day] = paths[:, day - 1] * (1 + sampled)

        self._paths = paths
        return paths

    # ------------------------------------------------------------------ #
    #  Decision-relevant metrics                                           #
    # ------------------------------------------------------------------ #

    def metrics(self) -> dict:
        """
        Returns metrics a real investor would use to decide whether to trust
        and deploy this strategy.
        """
        if self._paths is None:
            raise RuntimeError("Call run() before metrics().")

        final_values = self._paths[:, -1]

        # ── Return distribution ─────────────────────────────────────────
        returns_pct = (final_values - self.initial_cash) / self.initial_cash * 100

        median_return  = float(np.median(returns_pct))
        mean_return    = float(np.mean(returns_pct))
        p10            = float(np.percentile(returns_pct, 10))   # bad scenario
        p25            = float(np.percentile(returns_pct, 25))   # below average
        p75            = float(np.percentile(returns_pct, 75))   # above average
        p90            = float(np.percentile(returns_pct, 90))   # great scenario

        # ── Probability of profit ───────────────────────────────────────
        prob_profit    = float(np.mean(final_values > self.initial_cash) * 100)

        # ── Probability of beating buy-and-hold ─────────────────────────
        # We approximate buy-and-hold using the median of all simulated paths
        # (since we only have strategy paths here, this is relative comparison)
        bh_daily_ret   = self.portfolio.pct_change().dropna()
        bh_mean        = float(bh_daily_ret.mean())
        bh_projected   = self.initial_cash * ((1 + bh_mean) ** self.n_days)
        prob_beat_bh   = float(np.mean(final_values > bh_projected) * 100)

        # ── Risk of significant loss ────────────────────────────────────
        prob_loss_10   = float(np.mean(returns_pct < -10) * 100)   # lose >10%
        prob_loss_20   = float(np.mean(returns_pct < -20) * 100)   # lose >20%

        # ── Value at Risk (VaR) — 95% confidence ───────────────────────
        # "In the worst 5% of scenarios, we lose at least this much"
        var_95         = float(np.percentile(returns_pct, 5))

        # ── Conditional VaR / Expected Shortfall ───────────────────────
        # Average loss in those worst 5% of scenarios
        cvar_95        = float(np.mean(returns_pct[returns_pct <= var_95]))

        # ── Max drawdown across all simulated paths ─────────────────────
        drawdowns = []
        for path in self._paths:
            roll_max = np.maximum.accumulate(path)
            dd = (path - roll_max) / roll_max
            drawdowns.append(dd.min() * 100)

        median_max_dd  = float(np.median(drawdowns))
        worst_max_dd   = float(np.percentile(drawdowns, 95))

        # ── Consistency score ───────────────────────────────────────────
        # % of simulations that end within ±15% of the median outcome
        consistency = float(
            np.mean(np.abs(returns_pct - median_return) < 15) * 100
        )

        return {
            # Return projections
            "Median Projected Return (%)":    round(median_return, 1),
            "Mean Projected Return (%)":      round(mean_return, 1),
            "Best Case Return — top 10% (%)": round(p90, 1),
            "Good Case Return — top 25% (%)": round(p75, 1),
            "Bad Case Return — bottom 25% (%)": round(p25, 1),
            "Worst Case Return — bottom 10% (%)": round(p10, 1),

            # Probability metrics
            "Probability of Any Profit (%)":      round(prob_profit, 1),
            "Probability of Beating Market (%)":  round(prob_beat_bh, 1),
            "Probability of Losing >10% (%)":     round(prob_loss_10, 1),
            "Probability of Losing >20% (%)":     round(prob_loss_20, 1),

            # Risk metrics
            "Value at Risk 95% (%)":          round(var_95, 1),
            "Expected Loss in Worst 5% (%)":  round(cvar_95, 1),
            "Median Max Drawdown (%)":         round(median_max_dd, 1),
            "Worst Drawdown — top 95% (%)":   round(worst_max_dd, 1),

            # Reliability
            "Consistency Score (%)":          round(consistency, 1),

            # Raw paths for charting
            "_paths":        self._paths,
            "_returns_pct":  returns_pct,
        }
