"""
DTO definitions for the Quant Backtesting Engine.
Spring Boot ↔ FastAPI contract.

변경 이력
─────────
v3  LrV2Config 추가 (롤링 선형회귀 채널),
    signal_mode 에 LR_V2 추가,
    TradeLog 에 position_type 추가 (LONG / SHORT 구분).
"""
from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════
# REQUEST DTOs
# ══════════════════════════════════════════════════════

class DataProviderConfig(BaseModel):
    ticker: str = Field(..., example="BTCUSDT")
    start_date: str = Field(..., example="2024-01-01")
    end_date: str   = Field(..., example="2024-12-31")
    interval: Literal["5m", "15m", "60m", "1d"] = Field(default="1d")


class RSIConfig(BaseModel):
    period:     int   = Field(default=14,   ge=2,   le=200)
    oversold:   float = Field(default=30.0, ge=0.0, le=100.0)
    overbought: float = Field(default=70.0, ge=0.0, le=100.0)


class LinearRegressionConfig(BaseModel):
    period:          int   = Field(default=20,  ge=2,  le=500)
    slope_threshold: float = Field(default=0.0)


class LrV2Config(BaseModel):
    """롤링 선형회귀 채널 (DonovanWall 방식) + 숏 전략"""
    window_size:   int   = Field(default=100,  ge=10,  le=500,  description="회귀선 윈도우 크기")
    multiplier:    float = Field(default=2.0,  ge=0.1, le=10.0, description="채널 폭 표준편차 배수")
    rsi_filter:    bool  = Field(default=False, description="숏 진입 시 RSI 필터 적용 여부")
    rsi_threshold: float = Field(default=70.0, ge=50.0, le=90.0, description="숏 진입 RSI 최솟값")


class StrategyConfig(BaseModel):
    signal_mode: Literal["RSI_ONLY", "LR_ONLY", "COMBINED", "LR_V2"] = Field(default="RSI_ONLY")
    rsi:               RSIConfig              = Field(default_factory=RSIConfig)
    linear_regression: LinearRegressionConfig = Field(default_factory=LinearRegressionConfig)
    lr_v2:             LrV2Config             = Field(default_factory=LrV2Config)


class BacktestConfig(BaseModel):
    initial_capital:   float = Field(default=10_000.0, gt=0)
    position_size_pct: float = Field(default=1.0, ge=0.01, le=1.0)
    commission_pct:    float = Field(default=0.001, ge=0.0)
    slippage_pct:      float = Field(default=0.0005, ge=0.0)
    take_profit_pct:   float = Field(default=0.0, ge=0.0, description="익절 비율 (0 = 비활성, 0.1 = 10%)")
    stop_loss_pct:     float = Field(default=0.0, ge=0.0, description="손절 비율 (0 = 비활성, 0.05 = 5%)")


class BacktestRequest(BaseModel):
    data_provider: DataProviderConfig
    strategy:      StrategyConfig  = Field(default_factory=StrategyConfig)
    backtest:      BacktestConfig  = Field(default_factory=BacktestConfig)


# ══════════════════════════════════════════════════════
# RESPONSE DTOs
# ══════════════════════════════════════════════════════

class TradeLog(BaseModel):
    trade_no: int

    # 진입
    entry_date:       str
    entry_price:      float
    entry_commission: float
    entry_slippage:   float

    # 청산
    exit_date:        str
    exit_price:       float
    exit_commission:  float
    exit_slippage:    float
    exit_type:        str   # SELL | SELL_FORCED | SHORT_COVER | SHORT_COVER_FORCED

    # 포지션
    position_type: str  # LONG | SHORT
    shares:        float
    holding_bars:  int

    # 손익
    gross_pnl:  float
    total_cost: float
    net_pnl:    float
    return_pct: float
    is_winner:  bool


class IndicatorSnapshot(BaseModel):
    last_close:    float
    last_rsi:      Optional[float]
    last_lr:       Optional[float]
    last_lr_slope: Optional[float]


class BacktestResult(BaseModel):
    ticker:      str
    start_date:  str
    end_date:    str
    signal_mode: str

    initial_capital:   float
    final_capital:     float
    total_return_pct:  float

    mdd_pct:     float
    mdd_start:   str
    mdd_end:     str
    sharpe_ratio: float

    total_trades:   int
    winning_trades: int
    losing_trades:  int
    win_rate_pct:   float
    avg_profit:     float
    avg_loss:       float
    profit_factor:  float

    labels:         List[str]
    equity_curve:   List[float]
    drawdown_curve: List[float]

    # LR Channel V2 밴드 시계열 (프론트 차트용, non-LR_V2 전략일 때 빈 리스트)
    upper_band: List[Optional[float]] = Field(default_factory=list)
    lower_band: List[Optional[float]] = Field(default_factory=list)
    lr_center:  List[Optional[float]] = Field(default_factory=list)

    trade_log:  List[TradeLog]
    indicators: IndicatorSnapshot
