"""
2. Strategy Engine
──────────────────
Signal modes
────────────
RSI_ONLY  – RSI 과매도/과매수 돌파 신호
LR_ONLY   – 선형회귀 기울기 전환 신호
COMBINED  – RSI 진입 + LR 추세 필터
LR_V2     – 롤링 선형회귀 채널 (DonovanWall) 양방향 전략
            Long  : 가격이 lower_band 하향 돌파
            Short : 가격이 upper_band 상향 돌파 (+ RSI 필터 선택)
            익절  : 가격이 lr_center 로 회귀 시
"""
from __future__ import annotations

import numpy as np
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
    # 지표 계산
    # ──────────────────────────────────────────

    def _add_indicators(self, df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
        mode = config.signal_mode

        if mode in ("RSI_ONLY", "COMBINED", "LR_V2"):
            df["rsi"] = ta.rsi(df["close"], length=config.rsi.period)

        if mode in ("LR_ONLY", "COMBINED"):
            lr_cfg = config.linear_regression
            df["lr"]       = ta.linreg(df["close"], length=lr_cfg.period)
            df["lr_slope"] = ta.linreg(df["close"], length=lr_cfg.period, slope=True)

        if mode == "LR_V2":
            df = self._lr_v2_indicators(df, config)

        return df

    def _lr_v2_indicators(self, df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
        """
        매 봉마다 직전 window_size 봉에 numpy.polyfit 을 적용하여
        lr_center / upper_band / lower_band 를 롤링 계산한다.
        """
        w    = config.lr_v2.window_size
        mult = config.lr_v2.multiplier
        n    = len(df)

        lr_center  = np.full(n, np.nan)
        upper_band = np.full(n, np.nan)
        lower_band = np.full(n, np.nan)

        closes = df["close"].values
        x      = np.arange(w)          # 공통 x 축 (매 반복 재사용)

        for i in range(w - 1, n):
            y = closes[i - w + 1 : i + 1]          # 직전 w 봉 종가
            slope, intercept = np.polyfit(x, y, 1)  # 1차 선형 회귀
            reg_line = slope * x + intercept

            center  = reg_line[-1]                   # 현재 봉의 회귀선 값
            std_dev = np.std(y - reg_line)           # 가격 편차 표준편차

            lr_center[i]  = center
            upper_band[i] = center + mult * std_dev
            lower_band[i] = center - mult * std_dev

        df["lr_center"]  = lr_center
        df["upper_band"] = upper_band
        df["lower_band"] = lower_band
        return df

    # ──────────────────────────────────────────
    # 신호 생성
    # ──────────────────────────────────────────

    def _generate_signals(self, df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
        mode = config.signal_mode
        if   mode == "RSI_ONLY": df["signal"] = self._rsi_signal(df, config)
        elif mode == "LR_ONLY":  df["signal"] = self._lr_signal(df, config)
        elif mode == "COMBINED": df["signal"] = self._combined_signal(df, config)
        elif mode == "LR_V2":    df["signal"] = self._lr_v2_signal(df, config)
        else: raise ValueError(f"Unknown signal_mode: {mode}")
        return df

    def _rsi_signal(self, df: pd.DataFrame, config: StrategyConfig) -> pd.Series:
        rsi        = df["rsi"]
        oversold   = config.rsi.oversold
        overbought = config.rsi.overbought
        buy  = (rsi < oversold)   & (rsi.shift(1) >= oversold)
        sell = (rsi > overbought) & (rsi.shift(1) <= overbought)
        return self._to_signal(buy, sell, df.index)

    def _lr_signal(self, df: pd.DataFrame, config: StrategyConfig) -> pd.Series:
        slope = df["lr_slope"]
        thr   = config.linear_regression.slope_threshold
        buy  = (slope >  thr) & (slope.shift(1) <=  thr)
        sell = (slope < -thr) & (slope.shift(1) >= -thr)
        return self._to_signal(buy, sell, df.index)

    def _combined_signal(self, df: pd.DataFrame, config: StrategyConfig) -> pd.Series:
        rsi_sig     = self._rsi_signal(df, config)
        lr_trend_up = df["lr_slope"] > config.linear_regression.slope_threshold
        signal = pd.Series(0, index=df.index, dtype=int)
        signal[(rsi_sig == 1) & lr_trend_up] = 1
        signal[rsi_sig == -1]                = -1
        return signal

    def _lr_v2_signal(self, df: pd.DataFrame, config: StrategyConfig) -> pd.Series:
        """
        신호값 정의
        ───────────
         1  : Long  진입 (가격이 lower_band 하향 돌파)
        -1  : Short 진입 (가격이 upper_band 상향 돌파, RSI 필터 적용 가능)
         2  : Long  익절 (가격이 lr_center 상향 회귀)
        -2  : Short 익절 (가격이 lr_center 하향 회귀)
         0  : 관망
        """
        price  = df["close"]
        upper  = df["upper_band"]
        lower  = df["lower_band"]
        center = df["lr_center"]

        # ── 진입 신호 (크로스오버 방식) ──────────────────────────────────
        long_entry  = (price < lower) & (price.shift(1) >= lower.shift(1))
        short_entry = (price > upper) & (price.shift(1) <= upper.shift(1))

        # RSI 필터 (숏 진입 시 RSI 과매수 확인)
        if config.lr_v2.rsi_filter and "rsi" in df.columns:
            short_entry = short_entry & (df["rsi"] > config.lr_v2.rsi_threshold)

        # ── 익절 신호 (lr_center 회귀) ───────────────────────────────────
        long_exit  = (price >= center) & (price.shift(1) < center.shift(1))
        short_exit = (price <= center) & (price.shift(1) > center.shift(1))

        # ── 신호 합성 (우선순위: 진입 > 익절) ────────────────────────────
        signal = pd.Series(0, index=df.index, dtype=int)
        signal[long_exit]   = 2
        signal[short_exit]  = -2
        signal[long_entry]  = 1    # 진입이 익절보다 우선
        signal[short_entry] = -1

        return signal

    # ──────────────────────────────────────────
    # 헬퍼
    # ──────────────────────────────────────────

    @staticmethod
    def _to_signal(buy: pd.Series, sell: pd.Series, index: pd.Index) -> pd.Series:
        signal = pd.Series(0, index=index, dtype=int)
        signal[buy]  = 1
        signal[sell] = -1
        return signal
