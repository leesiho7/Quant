"""
4. Result Analyzer
──────────────────
equity_df (봉별 자산 가치) 와 trade_log (왕복 거래 기록) 을 받아
프론트엔드(Chart.js) 와 Spring Boot 에 전달할 BacktestResult 를 생성한다.

계산 항목
─────────
[시계열]
  labels         : 날짜 문자열 리스트  (Chart.js x축)
  equity_curve   : 봉별 포트폴리오 가치 리스트
  drawdown_curve : 봉별 낙폭(%) 리스트  ― 모두 0 이하

[성과 지표]
  total_return_pct : 총 수익률(%)
  mdd_pct          : 최대낙폭(%) + 시작·종료일
  sharpe_ratio     : 연환산 샤프지수 (252 거래일, 무위험수익률=0)
  win_rate_pct     : 승률(%)

[거래 통계]
  total_trades, winning_trades, losing_trades
  avg_profit, avg_loss, profit_factor
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from models.dto import (
    BacktestResult,
    IndicatorSnapshot,
    TradeLog,
)


class ResultAnalyzer:

    def analyze(
        self,
        df: pd.DataFrame,           # OHLCV + 지표 DataFrame (indicators 스냅샷용)
        equity_df: pd.DataFrame,    # index=날짜, column='equity'
        trade_log: List[TradeLog],  # BacktestCore 가 생성한 왕복 거래 목록
        initial_capital: float,
        final_capital: float,
        ticker: str,
        start_date: str,
        end_date: str,
        signal_mode: str,
    ) -> BacktestResult:

        equity: pd.Series = equity_df["equity"].astype(float)

        # ── 1. 시계열 데이터 생성 ─────────────────────────────────────────
        labels, equity_curve, drawdown_curve = self._build_chart_series(equity)

        # ── 2. MDD 계산 ───────────────────────────────────────────────────
        dd_series = pd.Series(drawdown_curve, index=equity.index)
        mdd_pct, mdd_start, mdd_end = self._calc_mdd(equity, dd_series)

        # ── 3. 총 수익률 ──────────────────────────────────────────────────
        total_return_pct = (final_capital - initial_capital) / initial_capital * 100.0

        # ── 4. 샤프지수 (연환산) ──────────────────────────────────────────
        sharpe_ratio = self._calc_sharpe(equity)

        # ── 5. 거래 통계 ──────────────────────────────────────────────────
        (
            total_trades,
            winning_trades,
            losing_trades,
            win_rate_pct,
            avg_profit,
            avg_loss,
            profit_factor,
        ) = self._calc_trade_stats(trade_log)

        # ── 6. 지표 스냅샷 ────────────────────────────────────────────────
        indicators = self._build_indicator_snapshot(df)

        return BacktestResult(
            # 메타
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            signal_mode=signal_mode,
            # 자본
            initial_capital=initial_capital,
            final_capital=round(final_capital, 2),
            total_return_pct=round(total_return_pct, 4),
            # 리스크
            mdd_pct=round(mdd_pct, 4),
            mdd_start=mdd_start,
            mdd_end=mdd_end,
            sharpe_ratio=round(sharpe_ratio, 4),
            # 거래 통계
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate_pct=round(win_rate_pct, 2),
            avg_profit=round(avg_profit, 2),
            avg_loss=round(avg_loss, 2),
            profit_factor=round(profit_factor, 4) if profit_factor != float("inf") else profit_factor,
            # Chart.js 시리즈
            labels=labels,
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve,
            # 거래 로그
            trade_log=trade_log,
            # 진단
            indicators=indicators,
        )

    # ══════════════════════════════════════════════════════════════════════
    # 내부 메서드
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _build_chart_series(
        equity: pd.Series,
    ) -> tuple[List[str], List[float], List[float]]:
        """
        Chart.js 에 바로 넘길 수 있는 1-D 리스트 세 개를 반환한다.

        drawdown 계산 방식
        ──────────────────
        rolling_peak = 해당 봉까지의 자산 최고점 (전고점)
        drawdown     = (equity - rolling_peak) / rolling_peak × 100  ← 항상 0 이하
        """
        rolling_peak: pd.Series = equity.cummax()
        drawdown_series: pd.Series = (equity - rolling_peak) / rolling_peak * 100.0

        labels: List[str] = [_fmt(d) for d in equity.index]
        equity_curve: List[float] = [round(float(v), 2) for v in equity]
        drawdown_curve: List[float] = [round(float(v), 4) for v in drawdown_series]

        return labels, equity_curve, drawdown_curve

    @staticmethod
    def _calc_mdd(
        equity: pd.Series,
        dd_series: pd.Series,
    ) -> tuple[float, str, str]:
        """
        최대낙폭(MDD) 수치와 시작·종료일을 반환한다.

        mdd_end   : 낙폭이 가장 깊었던 날 (최저점)
        mdd_start : 해당 최저점 직전의 전고점 날
        """
        mdd_pct: float = float(dd_series.min())
        mdd_end_idx = dd_series.idxmin()
        # 전고점 = mdd_end_idx 이전 구간에서 equity 가 가장 높았던 날
        mdd_start_idx = equity.loc[:mdd_end_idx].idxmax()

        return mdd_pct, _fmt(mdd_start_idx), _fmt(mdd_end_idx)

    @staticmethod
    def _calc_sharpe(equity: pd.Series) -> float:
        """
        일간 수익률로 연환산 샤프지수를 계산한다.
        연환산 계수 = √252 (252 거래일 가정)
        무위험수익률 = 0 (초과수익률 = 일간 수익률)
        """
        daily_returns: pd.Series = equity.pct_change().dropna()
        std = float(daily_returns.std())
        if std == 0:
            return 0.0
        return float(daily_returns.mean() / std * np.sqrt(252))

    @staticmethod
    def _calc_trade_stats(
        trade_log: List[TradeLog],
    ) -> tuple[int, int, int, float, float, float, float]:
        """
        거래 로그에서 승/패 통계를 추출한다.

        Returns (total, winning, losing, win_rate_pct, avg_profit, avg_loss, profit_factor)
        """
        winners = [t for t in trade_log if t.is_winner]
        losers  = [t for t in trade_log if not t.is_winner]

        total     = len(trade_log)
        winning   = len(winners)
        losing    = len(losers)
        win_rate  = winning / total * 100.0 if total > 0 else 0.0

        avg_profit = float(np.mean([t.net_pnl for t in winners])) if winners else 0.0
        avg_loss   = float(np.mean([t.net_pnl for t in losers]))  if losers  else 0.0

        gross_profit = sum(t.net_pnl for t in winners)
        gross_loss   = abs(sum(t.net_pnl for t in losers))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        return total, winning, losing, win_rate, avg_profit, avg_loss, profit_factor

    @staticmethod
    def _build_indicator_snapshot(df: pd.DataFrame) -> IndicatorSnapshot:
        last = df.iloc[-1]
        return IndicatorSnapshot(
            last_close=round(float(last["close"]), 4),
            last_rsi=_safe_float(last.get("rsi")),
            last_lr=_safe_float(last.get("lr")),
            last_lr_slope=_safe_float(last.get("lr_slope")),
        )


# ── 헬퍼 ────────────────────────────────────────────────────────────────────

def _fmt(date) -> str:
    try:
        return str(date.date())
    except AttributeError:
        return str(date)


def _safe_float(val) -> Optional[float]:
    try:
        f = float(val)
        return None if np.isnan(f) else round(f, 4)
    except (TypeError, ValueError):
        return None
