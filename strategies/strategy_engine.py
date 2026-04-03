"""
2. Strategy Engine
──────────────────
Calculates RSI and Linear Regression indicators with pandas-ta,
then generates buy (+1) / sell (-1) / hold (0) signals.

Signal modes
────────────
RSI_ONLY  – buy on RSI cross-below oversold; sell on cross-above overbought
LR_ONLY   – buy when LR slope turns positive; sell when it turns negative
COMBINED  – RSI entry signal filtered by LR trend direction
"""
from __future__ import annotations

import pandas as pd
import pandas_ta as ta

from models.dto import StrategyConfig


class StrategyEngine:

    def apply(self, df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
        df = df.copy()
        df = self._add_indicators(df, config)
        df = self._generate_signals(df, config)
        return df

    # ──────────────────────────────────────────
    # Indicator calculation
    # ──────────────────────────────────────────

    def _add_indicators(self, df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
        rsi_cfg = config.rsi
        lr_cfg = config.linear_regression

        # RSI
        df["rsi"] = ta.rsi(df["close"], length=rsi_cfg.period)

        # Linear Regression value (fitted price on the LR line)
        df["lr"] = ta.linreg(df["close"], length=lr_cfg.period)

        # LR slope (first derivative of the regression line)
        df["lr_slope"] = ta.linreg(df["close"], length=lr_cfg.period, slope=True)

        return df

    # ──────────────────────────────────────────
    # Signal generation
    # ──────────────────────────────────────────

    def _generate_signals(self, df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
        mode = config.signal_mode

        if mode == "RSI_ONLY":
            df["signal"] = self._rsi_signal(df, config)
        elif mode == "LR_ONLY":
            df["signal"] = self._lr_signal(df, config)
        elif mode == "COMBINED":
            df["signal"] = self._combined_signal(df, config)
        else:
            raise ValueError(f"Unknown signal_mode: {mode}")

        return df

    def _rsi_signal(self, df: pd.DataFrame, config: StrategyConfig) -> pd.Series:
        """Cross-below oversold → BUY ; cross-above overbought → SELL."""
        rsi = df["rsi"]
        oversold = config.rsi.oversold
        overbought = config.rsi.overbought

        buy = (rsi < oversold) & (rsi.shift(1) >= oversold)
        sell = (rsi > overbought) & (rsi.shift(1) <= overbought)

        return self._to_signal(buy, sell, df.index)

    def _lr_signal(self, df: pd.DataFrame, config: StrategyConfig) -> pd.Series:
        """Slope crosses threshold upward → BUY ; downward → SELL."""
        slope = df["lr_slope"]
        thr = config.linear_regression.slope_threshold

        buy = (slope > thr) & (slope.shift(1) <= thr)
        sell = (slope < -thr) & (slope.shift(1) >= -thr)

        return self._to_signal(buy, sell, df.index)

    def _combined_signal(self, df: pd.DataFrame, config: StrategyConfig) -> pd.Series:
        """RSI crossover entry, gated by LR slope trend direction."""
        rsi_sig = self._rsi_signal(df, config)
        lr_trend_up = df["lr_slope"] > config.linear_regression.slope_threshold

        signal = pd.Series(0, index=df.index, dtype=int)
        signal[rsi_sig == 1] = 0          # tentative buy (will be filtered)
        signal[(rsi_sig == 1) & lr_trend_up] = 1   # confirmed buy
        signal[rsi_sig == -1] = -1                  # sell always honoured

        return signal

    # ──────────────────────────────────────────
    # Helper
    # ──────────────────────────────────────────

    @staticmethod
    def _to_signal(
        buy: pd.Series, sell: pd.Series, index: pd.Index
    ) -> pd.Series:
        signal = pd.Series(0, index=index, dtype=int)
        signal[buy] = 1
        signal[sell] = -1
        return signal
