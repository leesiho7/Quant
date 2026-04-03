"""
Quant Backtesting Engine  –  FastAPI entry-point
─────────────────────────────────────────────────
Layers
  1. DataProvider    → fetches OHLCV from Yahoo Finance
  2. StrategyEngine  → computes RSI / Linear Regression indicators & signals
  3. BacktestCore    → simulates order execution bar-by-bar
  4. ResultAnalyzer  → derives equity curve, MDD, Sharpe, trade stats

Run:
  uvicorn main:app --reload --port 8000

Spring Boot sample call:
  POST http://localhost:8000/api/v1/backtest
  Content-Type: application/json
  Body: see BacktestRequest schema in /docs
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from models.dto import BacktestRequest, BacktestResult  # noqa: F401
from providers.data_provider import DataProvider
from strategies.strategy_engine import StrategyEngine
from backtest.backtest_core import BacktestCore
from analyzers.result_analyzer import ResultAnalyzer

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger("quant")

app = FastAPI(
    title="Quant Backtesting Engine",
    description=(
        "RSI + Linear Regression strategy backtester. "
        "Accepts BacktestRequest from Spring Boot and returns "
        "equity curve, MDD, Sharpe ratio, and trade records."
    ),
    version="1.0.0",
)

# ── Singleton services ──────────────────────────────────────────────────────
_data_provider = DataProvider()
_strategy_engine = StrategyEngine()
_backtest_core = BacktestCore()
_result_analyzer = ResultAnalyzer()


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
async def health_check():
    return {"status": "ok"}


@app.post(
    "/api/v1/backtest",
    response_model=BacktestResult,
    status_code=status.HTTP_200_OK,
    summary="Run a backtest",
    tags=["backtest"],
)
async def run_backtest(request: BacktestRequest) -> BacktestResult:
    """
    Execute a full backtest and return performance results.

    **Flow**
    1. `DataProvider`   – download OHLCV data from Yahoo Finance
    2. `StrategyEngine` – attach RSI & LR indicators, emit signals
    3. `BacktestCore`   – simulate trades with commission
    4. `ResultAnalyzer` – compute equity curve, MDD, Sharpe, win rate…
    """
    logger.info(
        "Backtest request  ticker=%s  %s → %s  mode=%s",
        request.data_provider.ticker,
        request.data_provider.start_date,
        request.data_provider.end_date,
        request.strategy.signal_mode,
    )

    # 1. Data
    try:
        df = _data_provider.fetch(request.data_provider)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    # 2. Strategy
    try:
        df = _strategy_engine.apply(df, request.strategy)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Strategy error: {exc}",
        )

    # 3. Backtest
    equity_df, trade_log, final_capital = _backtest_core.run(df, request.backtest)

    # 4. Analyse
    result = _result_analyzer.analyze(
        df=df,
        equity_df=equity_df,
        trade_log=trade_log,
        initial_capital=request.backtest.initial_capital,
        final_capital=final_capital,
        ticker=request.data_provider.ticker,
        start_date=request.data_provider.start_date,
        end_date=request.data_provider.end_date,
        signal_mode=request.strategy.signal_mode,
    )

    logger.info(
        "Backtest complete  ticker=%s  return=%.2f%%  MDD=%.2f%%  trades=%d",
        result.ticker,
        result.total_return_pct,
        result.mdd_pct,
        result.total_trades if result.total_trades else 0,
    )

    return result


# ── Global error handler ─────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error. Check server logs."},
    )
