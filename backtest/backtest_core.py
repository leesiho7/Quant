"""
3. Backtest Core
────────────────
봉 단위(Bar-by-bar) 거래 시뮬레이션.

핵심 규칙
─────────
* 동시에 1 포지션만 허용 (피라미딩 없음).
* signal == +1  → 매수  (포지션 없을 때만)
* signal == -1  → 매도  (포지션 보유 중일 때만)
* 마지막 봉까지 미청산이면 강제 청산 (SELL_FORCED).

복리 계산
─────────
매 거래는 잔여 현금 전체(또는 position_size_pct 비율)를 재투자한다.
→ 이전 거래 수익이 다음 거래 투자금에 그대로 포함되어 자연스럽게 복리 적용.

비용 처리
─────────
매수 체결가 = Close × (1 + slippage_pct)   ← 불리하게 체결
매도 체결가 = Close × (1 - slippage_pct)   ← 불리하게 체결
수수료     = 투자금액(또는 회수금액) × commission_pct  (편도)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

from models.dto import BacktestConfig, TradeLog


# ── 내부 포지션 상태 ────────────────────────────────────────────────────────

@dataclass
class _OpenPosition:
    """매수 진입 후 청산 전까지의 상태를 담는 내부 자료구조."""
    entry_date: str
    entry_price: float          # 슬리피지 반영 실제 체결가
    entry_commission: float     # 진입 수수료
    entry_slippage: float       # 진입 슬리피지 비용
    shares: float               # 보유 주수
    entry_bar: int              # 진입 봉 인덱스 (holding_bars 계산용)


# ── 메인 클래스 ─────────────────────────────────────────────────────────────

class BacktestCore:
    """
    df 의 'close' 와 'signal' 컬럼을 읽어 봉 단위 시뮬레이션을 실행한다.

    Returns
    ───────
    equity_df     : DatetimeIndex DataFrame, 컬럼 'equity' (봉별 포트폴리오 가치)
    trade_log     : List[TradeLog]  왕복 거래 기록
    final_capital : float           시뮬레이션 종료 시 현금 잔고
    """

    def run(
        self,
        df: pd.DataFrame,
        config: BacktestConfig,
    ) -> Tuple[pd.DataFrame, List[TradeLog], float]:

        capital: float = config.initial_capital
        position: _OpenPosition | None = None
        trade_log: List[TradeLog] = []
        equity_records: List[dict] = []
        trade_counter: int = 0

        rows = list(df.iterrows())
        n = len(rows)

        for bar_idx, (date, row) in enumerate(rows):
            signal: int = int(row["signal"])
            close: float = float(row["close"])
            is_last_bar: bool = bar_idx == n - 1

            # ── 매수 진입 ────────────────────────────────────────────────
            if signal == 1 and position is None:
                invest: float = capital * config.position_size_pct

                # 슬리피지: 매수는 불리하게(높은 가격으로) 체결
                slip_cost_entry: float = invest * config.slippage_pct
                commission_entry: float = invest * config.commission_pct
                total_entry_cost: float = slip_cost_entry + commission_entry

                net_invest: float = invest - total_entry_cost
                eff_buy_price: float = close * (1.0 + config.slippage_pct)
                shares: float = net_invest / eff_buy_price

                capital -= invest  # 복리: 현재 잔고에서 차감

                position = _OpenPosition(
                    entry_date=_fmt(date),
                    entry_price=round(eff_buy_price, 6),
                    entry_commission=round(commission_entry, 4),
                    entry_slippage=round(slip_cost_entry, 4),
                    shares=shares,
                    entry_bar=bar_idx,
                )

            # ── 청산 ─────────────────────────────────────────────────────
            elif position is not None and (signal == -1 or is_last_bar):
                exit_type: str = "SELL" if signal == -1 else "SELL_FORCED"

                # 슬리피지: 매도는 불리하게(낮은 가격으로) 체결
                eff_sell_price: float = close * (1.0 - config.slippage_pct)
                gross_proceeds: float = position.shares * eff_sell_price

                slip_cost_exit: float = position.shares * close * config.slippage_pct
                commission_exit: float = gross_proceeds * config.commission_pct
                net_proceeds: float = gross_proceeds - commission_exit

                # 손익 계산
                cost_basis: float = position.shares * position.entry_price
                gross_pnl: float = gross_proceeds - cost_basis
                total_cost: float = (
                    position.entry_commission
                    + position.entry_slippage
                    + commission_exit
                    + slip_cost_exit
                )
                net_pnl: float = net_proceeds - cost_basis - position.entry_commission - position.entry_slippage

                # 수익률(%) – 진입 시 실제 투자금 대비
                invested_amount: float = cost_basis + position.entry_commission + position.entry_slippage
                return_pct: float = (net_pnl / invested_amount * 100.0) if invested_amount > 0 else 0.0

                holding_bars: int = bar_idx - position.entry_bar

                trade_counter += 1
                trade_log.append(
                    TradeLog(
                        trade_no=trade_counter,
                        entry_date=position.entry_date,
                        entry_price=round(position.entry_price, 4),
                        entry_commission=round(position.entry_commission, 4),
                        entry_slippage=round(position.entry_slippage, 4),
                        exit_date=_fmt(date),
                        exit_price=round(eff_sell_price, 4),
                        exit_commission=round(commission_exit, 4),
                        exit_slippage=round(slip_cost_exit, 4),
                        exit_type=exit_type,
                        shares=round(position.shares, 6),
                        holding_bars=holding_bars,
                        gross_pnl=round(gross_pnl, 4),
                        total_cost=round(total_cost, 4),
                        net_pnl=round(net_pnl, 4),
                        return_pct=round(return_pct, 4),
                        is_winner=net_pnl > 0,
                    )
                )

                capital += net_proceeds  # 복리: 청산 대금을 현금에 합산
                position = None

            # ── 봉별 포트폴리오 가치 스냅샷 ─────────────────────────────
            # 현금 + 보유 주식의 현재가 평가액 (미실현 손익 포함)
            market_value: float = (position.shares * close) if position is not None else 0.0
            equity_records.append({"date": date, "equity": capital + market_value})

        # equity DataFrame 생성
        equity_df = (
            pd.DataFrame(equity_records)
            .set_index("date")
        )
        equity_df.index = pd.to_datetime(equity_df.index)

        return equity_df, trade_log, capital


# ── 헬퍼 ────────────────────────────────────────────────────────────────────

def _fmt(date) -> str:
    try:
        return str(date.date())
    except AttributeError:
        return str(date)
