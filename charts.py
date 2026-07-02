import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ACCENT  = "#00C896"
DANGER  = "#FF4D6D"
BG      = "#0D1117"
SURFACE = "#161B22"
GRID    = "#21262D"
TEXT    = "#C9D1D9"
MUTED   = "#8B949E"
GOLD    = "#F0B429"
PURPLE  = "#A78BFA"

LAYOUT_BASE = dict(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(family="'Inter', 'Segoe UI', sans-serif", color=TEXT, size=12),
    margin=dict(l=16, r=16, t=48, b=16),
    legend=dict(bgcolor=SURFACE, bordercolor=GRID, borderwidth=1,
                font=dict(color=MUTED)),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, showspikes=True,
               spikecolor=MUTED, spikethickness=1),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
)


class ChartBuilder:

    # ── Price + signals ───────────────────────────────────────────────────
    @staticmethod
    def price_chart(df: pd.DataFrame, strategy_name: str) -> go.Figure:
        has_bb = "BB_Upper" in df.columns

        fig = go.Figure()

        # Bollinger Band shaded region (draw first so it sits behind price)
        if has_bb:
            fig.add_trace(go.Scatter(
                x=list(df.index) + list(df.index[::-1]),
                y=list(df["BB_Upper"]) + list(df["BB_Lower"][::-1]),
                fill="toself",
                fillcolor="rgba(167,139,250,0.08)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Band region",
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=df.index, y=df["BB_Upper"], name="Upper Band",
                line=dict(color=PURPLE, width=1.2, dash="dot"),
                hovertemplate="Upper: $%{y:.2f}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=df.index, y=df["BB_Middle"], name="Middle Band",
                line=dict(color=MUTED, width=1.0, dash="dot"),
                hovertemplate="Middle: $%{y:.2f}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=df.index, y=df["BB_Lower"], name="Lower Band",
                line=dict(color=PURPLE, width=1.2, dash="dot"),
                hovertemplate="Lower: $%{y:.2f}<extra></extra>",
            ))

        # Moving average lines
        for col, color, label in [
            ("MA_Short", "#4F91FF", "MA Short"),
            ("MA_Long",  GOLD,      "MA Long"),
        ]:
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col], name=label,
                    line=dict(color=color, width=1.5, dash="dot"),
                    hovertemplate=f"{label}: $%{{y:.2f}}<extra></extra>",
                ))

        # Close price (on top of bands)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"], name="Close",
            line=dict(color=TEXT, width=1.6),
            hovertemplate="%{x|%b %d %Y}<br>$%{y:.2f}<extra></extra>",
        ))

        # Buy markers
        buys = df[df["Signal"] == 1]
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys.index, y=buys["Close"], mode="markers", name="Buy",
                marker=dict(symbol="triangle-up", size=12, color=ACCENT,
                            line=dict(color="white", width=1)),
                hovertemplate="BUY $%{y:.2f}<extra></extra>",
            ))

        # Sell markers
        sells = df[df["Signal"] == -1]
        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells.index, y=sells["Close"], mode="markers", name="Sell",
                marker=dict(symbol="triangle-down", size=12, color=DANGER,
                            line=dict(color="white", width=1)),
                hovertemplate="SELL $%{y:.2f}<extra></extra>",
            ))

        fig.update_layout(
            **LAYOUT_BASE,
            title=dict(text=f"<b>{strategy_name}</b> — Price & Signals",
                       font=dict(size=16, color=TEXT)),
            hovermode="x unified",
        )
        return fig

    # ── Portfolio vs buy-and-hold ─────────────────────────────────────────
    @staticmethod
    def portfolio_chart(df: pd.DataFrame, initial_cash: float) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Portfolio"], name="Strategy",
            fill="tozeroy", fillcolor="rgba(0,200,150,0.08)",
            line=dict(color=ACCENT, width=2),
            hovertemplate="Strategy: $%{y:,.2f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BuyAndHold"], name="Buy & Hold",
            line=dict(color=GOLD, width=1.8, dash="dot"),
            hovertemplate="Buy & Hold: $%{y:,.2f}<extra></extra>",
        ))
        fig.add_hline(y=initial_cash, line_dash="dash", line_color=MUTED,
                      opacity=0.4, annotation_text="Initial Investment",
                      annotation_font_color=MUTED)
        fig.update_layout(
            **LAYOUT_BASE,
            title=dict(text="<b>Portfolio Value</b> vs Buy & Hold",
                       font=dict(size=16, color=TEXT)),
            yaxis_title="Portfolio Value ($)",
            hovermode="x unified",
        )
        return fig

    # ── Drawdown ──────────────────────────────────────────────────────────
    @staticmethod
    def drawdown_chart(df: pd.DataFrame) -> go.Figure:
        roll_max = df["Portfolio"].cummax()
        drawdown = (df["Portfolio"] - roll_max) / roll_max * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=drawdown, name="Drawdown",
            fill="tozeroy", fillcolor="rgba(255,77,109,0.15)",
            line=dict(color=DANGER, width=1.5),
            hovertemplate="Drawdown: %{y:.2f}%<extra></extra>",
        ))
        fig.update_layout(
            **LAYOUT_BASE,
            title=dict(text="<b>Portfolio Drawdown</b>",
                       font=dict(size=16, color=TEXT)),
            yaxis_title="Drawdown (%)",
            hovermode="x unified",
        )
        return fig

    # ── Monte Carlo fan chart ─────────────────────────────────────────────
    @staticmethod
    def monte_carlo_fan(paths: np.ndarray, initial_cash: float,
                        n_days: int) -> go.Figure:
        """
        Fan chart: percentile bands around the median path.
        Shows the realistic spread of outcomes — not individual spaghetti lines.
        """
        x = list(range(n_days + 1))

        p10  = np.percentile(paths, 10,  axis=0)
        p25  = np.percentile(paths, 25,  axis=0)
        p50  = np.percentile(paths, 50,  axis=0)
        p75  = np.percentile(paths, 75,  axis=0)
        p90  = np.percentile(paths, 90,  axis=0)

        fig = go.Figure()

        # 10–90 outer band
        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=list(p90) + list(p10[::-1]),
            fill="toself",
            fillcolor="rgba(0,200,150,0.07)",
            line=dict(color="rgba(0,0,0,0)"),
            name="10th–90th percentile",
            hoverinfo="skip",
        ))

        # 25–75 inner band
        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=list(p75) + list(p25[::-1]),
            fill="toself",
            fillcolor="rgba(0,200,150,0.18)",
            line=dict(color="rgba(0,0,0,0)"),
            name="25th–75th percentile",
            hoverinfo="skip",
        ))

        # Median
        fig.add_trace(go.Scatter(
            x=x, y=p50, name="Median path",
            line=dict(color=ACCENT, width=2.5),
            hovertemplate="Day %{x}<br>Median: $%{y:,.0f}<extra></extra>",
        ))

        # Worst / best paths
        fig.add_trace(go.Scatter(
            x=x, y=p10, name="Bad scenario (10th %ile)",
            line=dict(color=DANGER, width=1.2, dash="dot"),
            hovertemplate="Bad: $%{y:,.0f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=x, y=p90, name="Great scenario (90th %ile)",
            line=dict(color=GOLD, width=1.2, dash="dot"),
            hovertemplate="Great: $%{y:,.0f}<extra></extra>",
        ))

        fig.add_hline(y=initial_cash, line_dash="dash", line_color=MUTED,
                      opacity=0.5, annotation_text="Break-even",
                      annotation_font_color=MUTED)

        fig.update_layout(
            **LAYOUT_BASE,
            title=dict(
                text="<b>Monte Carlo Simulation</b> — Projected Outcomes",
                font=dict(size=16, color=TEXT),
            ),
            xaxis_title=f"Trading Days (projecting {n_days} days forward)",
            yaxis_title="Portfolio Value ($)",
            hovermode="x unified",
        )
        return fig

    # ── Return distribution histogram ─────────────────────────────────────
    @staticmethod
    def return_distribution(returns_pct: np.ndarray) -> go.Figure:
        fig = go.Figure()

        colors = [ACCENT if r >= 0 else DANGER for r in returns_pct]

        fig.add_trace(go.Histogram(
            x=returns_pct,
            nbinsx=50,
            name="Simulated Returns",
            marker_color=ACCENT,
            opacity=0.75,
            hovertemplate="Return: %{x:.1f}%<br>Count: %{y}<extra></extra>",
        ))

        # Mark key percentiles
        for pct, label, color in [
            (5,  "VaR 95%",  DANGER),
            (50, "Median",   ACCENT),
            (95, "Top 5%",   GOLD),
        ]:
            val = float(np.percentile(returns_pct, pct))
            fig.add_vline(x=val, line_dash="dash", line_color=color,
                          annotation_text=f"{label}: {val:.1f}%",
                          annotation_font_color=color,
                          annotation_position="top")

        fig.add_vline(x=0, line_color=MUTED, opacity=0.4)

        fig.update_layout(
            **LAYOUT_BASE,
            title=dict(
                text="<b>Return Distribution</b> — All Simulated Outcomes",
                font=dict(size=16, color=TEXT),
            ),
            xaxis_title="Projected Return (%)",
            yaxis_title="Number of Simulations",
            showlegend=False,
        )
        return fig
