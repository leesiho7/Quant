"""
DTO definitions for the Quant Backtesting Engine.
Spring Boot ↔ FastAPI contract.

변경 이력
─────────
v2  TradeLog (왕복 거래 레코드), Chart.js 전용 labels / equity_curve /
    drawdown_curve 필드 추가. BacktestConfig 에 slippage_pct 추가.
"""
from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════
# REQUEST DTOs  (Spring Boot → FastAPI)
# ══════════════════════════════════════════════════════

class DataProviderConfig(BaseModel):
    """어떤 종목의 OHLCV 데이터를 가져올지 지정."""
    ticker: str = Field(..., example="AAPL", description="Yahoo Finance 티커")
    start_date: str = Field(..., example="2020-01-01", description="YYYY-MM-DD")
    end_date: str = Field(..., example="2024-12-31", description="YYYY-MM-DD")
    interval: Literal["5m", "15m", "60m", "1d"] = Field(default="1d", description="봉 단위 (5m|15m|60m|1d)")


class RSIConfig(BaseModel):
    period: int = Field(default=14, ge=2, le=200)
    oversold: float = Field(default=30.0, ge=0.0, le=100.0, description="매수 기준선")
    overbought: float = Field(default=70.0, ge=0.0, le=100.0, description="매도 기준선")


class LinearRegressionConfig(BaseModel):
    period: int = Field(default=20, ge=2, le=500)
    slope_threshold: float = Field(default=0.0, description="추세 판단 최소 기울기")


class StrategyConfig(BaseModel):
    signal_mode: Literal["RSI_ONLY", "LR_ONLY", "COMBINED"] = Field(
        default="RSI_ONLY",
        description="RSI_ONLY | LR_ONLY | COMBINED",
    )
    rsi: RSIConfig = Field(default_factory=RSIConfig)
    linear_regression: LinearRegressionConfig = Field(
        default_factory=LinearRegressionConfig
    )


class BacktestConfig(BaseModel):
    """거래 시뮬레이션 파라미터."""
    initial_capital: float = Field(default=10_000.0, gt=0, description="초기 자본(USD)")
    position_size_pct: float = Field(
        default=1.0, ge=0.01, le=1.0,
        description="진입 시 사용할 자본 비율 (1.0 = 전액, 복리 적용)",
    )
    commission_pct: float = Field(
        default=0.001, ge=0.0,
        description="편도 수수료율 (0.001 = 0.1 %)",
    )
    slippage_pct: float = Field(
        default=0.0005, ge=0.0,
        description="슬리피지율 – 매수 시 체결가를 높이고 매도 시 체결가를 낮춤 (0.0005 = 0.05 %)",
    )


class BacktestRequest(BaseModel):
    """Spring Boot 가 보내는 최상위 요청 바디."""
    data_provider: DataProviderConfig
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)

    class Config:
        json_schema_extra = {
            "example": {
                "data_provider": {
                    "ticker": "AAPL",
                    "start_date": "2020-01-01",
                    "end_date": "2024-12-31",
                    "interval": "1d",
                },
                "strategy": {
                    "signal_mode": "COMBINED",
                    "rsi": {"period": 14, "oversold": 30, "overbought": 70},
                    "linear_regression": {"period": 20, "slope_threshold": 0.0},
                },
                "backtest": {
                    "initial_capital": 10000.0,
                    "position_size_pct": 1.0,
                    "commission_pct": 0.001,
                    "slippage_pct": 0.0005,
                },
            }
        }


# ══════════════════════════════════════════════════════
# RESPONSE DTOs  (FastAPI → Spring Boot / 프론트엔드)
# ══════════════════════════════════════════════════════

class TradeLog(BaseModel):
    """
    매수→매도 한 사이클의 완전한 거래 기록.
    BacktestCore 가 생성하고 ResultAnalyzer 가 그대로 전달한다.
    """
    trade_no: int = Field(description="거래 순번 (1부터 시작)")

    # 진입 정보
    entry_date: str
    entry_price: float = Field(description="슬리피지 반영 실제 체결가")
    entry_commission: float = Field(description="진입 수수료")
    entry_slippage: float = Field(description="진입 슬리피지 비용")

    # 청산 정보
    exit_date: str
    exit_price: float = Field(description="슬리피지 반영 실제 체결가")
    exit_commission: float = Field(description="청산 수수료")
    exit_slippage: float = Field(description="청산 슬리피지 비용")
    exit_type: str = Field(description="SELL | SELL_FORCED")

    # 포지션
    shares: float
    holding_bars: int = Field(description="보유 봉 수")

    # 손익
    gross_pnl: float = Field(description="수수료·슬리피지 차감 전 손익")
    total_cost: float = Field(description="수수료 + 슬리피지 합계")
    net_pnl: float = Field(description="순 손익")
    return_pct: float = Field(description="거래 수익률 (%)")
    is_winner: bool


class IndicatorSnapshot(BaseModel):
    """마지막 봉의 지표 값 – 진단용."""
    last_close: float
    last_rsi: Optional[float]
    last_lr: Optional[float]
    last_lr_slope: Optional[float]


class BacktestResult(BaseModel):
    """
    최종 백테스트 결과.
    Chart.js 에서 바로 쓸 수 있도록 labels / equity_curve /
    drawdown_curve 를 1-D 리스트로 제공한다.
    """
    # ── 메타 ──────────────────────────────────────────
    ticker: str
    start_date: str
    end_date: str
    signal_mode: str

    # ── 자본 요약 ─────────────────────────────────────
    initial_capital: float
    final_capital: float
    total_return_pct: float

    # ── 리스크 지표 ───────────────────────────────────
    mdd_pct: float = Field(description="최대낙폭 MDD (%)")
    mdd_start: str = Field(description="MDD 시작일 (전고점)")
    mdd_end: str = Field(description="MDD 종료일 (최저점)")
    sharpe_ratio: float = Field(description="연환산 샤프지수 (무위험수익률=0)")

    # ── 거래 통계 ─────────────────────────────────────
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    avg_profit: float
    avg_loss: float
    profit_factor: float

    # ── Chart.js 전용 1-D 시리즈 ─────────────────────
    labels: List[str] = Field(
        description="x축 날짜 레이블 – equity_curve / drawdown_curve 와 인덱스 1:1 대응"
    )
    equity_curve: List[float] = Field(
        description="봉별 포트폴리오 가치 (Chart.js datasets[].data)"
    )
    drawdown_curve: List[float] = Field(
        description="봉별 낙폭 % – 모두 0 이하 (Chart.js 낙폭 차트용)"
    )

    # ── 거래 로그 ─────────────────────────────────────
    trade_log: List[TradeLog] = Field(
        description="왕복 거래 레코드 목록 (진입·청산 날짜/가격/수익률 포함)"
    )

    # ── 진단 ──────────────────────────────────────────
    indicators: IndicatorSnapshot
