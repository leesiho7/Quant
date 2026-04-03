"""
3. Backtest Core  (양방향 포지션 지원)
포지션 상태: None(관망) | LONG(매수) | SHORT(공매도)

신호 체계
---------
 1  : Long  진입
-1  : Short 진입
 2  : Long  익절 신호  (LR_V2)
-2  : Short 익절 신호  (LR_V2)

LR_V2 추가 청산 조건
---------------------
Long  보유 중 : close >= lr_center 일 때 자동 청산
Short 보유 중 : close <= lr_center 일 때 자동 청산
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

from models.dto import BacktestConfig, TradeLog


@dataclass
class _OpenPosition:
    position_type:    str    # "LONG" | "SHORT"
    entry_date:       str
    entry_price:      float
    entry_commission: float
    entry_slippage:   float
    shares:           float
    invest:           float
    entry_bar:        int


class BacktestCore:

    def run(
        self,
        df: pd.DataFrame,
        config: BacktestConfig,
    ) -> Tuple[pd.DataFrame, List[TradeLog], float]:

        capital        = config.initial_capital
        position       = None
        trade_log      = []
        equity_records = []
        trade_counter  = 0
        rows = list(df.iterrows())
        n    = len(rows)

        for bar_idx, (date, row) in enumerate(rows):
            signal  = int(row.get("signal", 0))
            close   = float(row["close"])
            is_last = bar_idx == n - 1

            try:
                lrc       = row["lr_center"] if "lr_center" in row.index else float("nan")
                lr_center = float(lrc) if not math.isnan(float(lrc)) else float("nan")
            except Exception:
                lr_center = float("nan")

            # STEP 1: 청산 판단
            if position is not None:
                should_exit, exit_type = self._check_exit(
                    position, signal, close, lr_center, is_last, config
                )
                if should_exit:
                    log_entry, net_proceeds = self._close_position(
                        position, trade_counter + 1, date, close, exit_type, config
                    )
                    trade_log.append(log_entry)
                    trade_counter += 1
                    capital  += net_proceeds
                    position  = None

            # STEP 2: 신규 진입
            if position is None and not is_last:
                if signal == 1:
                    position, capital = self._open_long(capital, close, bar_idx, date, config)
                elif signal == -1:
                    position, capital = self._open_short(capital, close, bar_idx, date, config)

            # STEP 3: 봉별 포트폴리오 평가
            market_value = self._market_value(position, close)
            equity_records.append({"date": date, "equity": capital + market_value})

        equity_df = pd.DataFrame(equity_records).set_index("date")
        equity_df.index = pd.to_datetime(equity_df.index)
        return equity_df, trade_log, capital

    def _open_long(self, capital, close, bar_idx, date, config):
        invest     = capital * config.position_size_pct
        slip_cost  = invest  * config.slippage_pct
        commission = invest  * config.commission_pct
        eff_buy    = close   * (1.0 + config.slippage_pct)
        net_invest = invest - slip_cost - commission
        shares     = net_invest / eff_buy if eff_buy > 0 else 0
        pos = _OpenPosition(
            position_type="LONG", entry_date=_fmt(date),
            entry_price=round(eff_buy, 6), entry_commission=round(commission, 4),
            entry_slippage=round(slip_cost, 4), shares=shares, invest=invest, entry_bar=bar_idx,
        )
        return pos, capital - invest

    def _open_short(self, capital, close, bar_idx, date, config):
        invest     = capital * config.position_size_pct
        eff_sell   = close   * (1.0 - config.slippage_pct)
        slip_cost  = invest  * config.slippage_pct
        commission = invest  * config.commission_pct
        shares     = invest  / eff_sell if eff_sell > 0 else 0
        pos = _OpenPosition(
            position_type="SHORT", entry_date=_fmt(date),
            entry_price=round(eff_sell, 6), entry_commission=round(commission, 4),
            entry_slippage=round(slip_cost, 4), shares=shares, invest=invest, entry_bar=bar_idx,
        )
        return pos, capital - invest

    @staticmethod
    def _check_exit(pos, signal, close, lr_center, is_last, config):
        if is_last:
            return True, ("SELL_FORCED" if pos.position_type == "LONG" else "SHORT_COVER_FORCED")

        tp = config.take_profit_pct
        sl = config.stop_loss_pct

        if pos.position_type == "LONG":
            # 익절 / 손절
            if tp > 0 and close >= pos.entry_price * (1.0 + tp):
                return True, "TAKE_PROFIT"
            if sl > 0 and close <= pos.entry_price * (1.0 - sl):
                return True, "STOP_LOSS"
            # 신호 청산
            if signal in (-1, 2):
                return True, "SELL"
            if not math.isnan(lr_center) and close >= lr_center:
                return True, "SELL"
        elif pos.position_type == "SHORT":
            # 익절 / 손절 (숏: 가격이 내려가면 수익)
            if tp > 0 and close <= pos.entry_price * (1.0 - tp):
                return True, "TAKE_PROFIT"
            if sl > 0 and close >= pos.entry_price * (1.0 + sl):
                return True, "STOP_LOSS"
            # 신호 청산
            if signal in (1, -2):
                return True, "SHORT_COVER"
            if not math.isnan(lr_center) and close <= lr_center:
                return True, "SHORT_COVER"
        return False, ""

    def _close_position(self, pos, trade_no, date, close, exit_type, config):
        if pos.position_type == "LONG":
            return self._close_long(pos, trade_no, date, close, exit_type, config)
        return self._close_short(pos, trade_no, date, close, exit_type, config)

    def _close_long(self, pos, trade_no, date, close, exit_type, config):
        eff_sell        = close * (1.0 - config.slippage_pct)
        gross_proceeds  = pos.shares * eff_sell
        slip_cost_exit  = pos.shares * close * config.slippage_pct
        commission_exit = gross_proceeds * config.commission_pct
        net_proceeds    = gross_proceeds - commission_exit
        cost_basis      = pos.shares * pos.entry_price
        gross_pnl       = gross_proceeds - cost_basis
        total_cost      = pos.entry_commission + pos.entry_slippage + commission_exit + slip_cost_exit
        net_pnl         = net_proceeds - cost_basis - pos.entry_commission - pos.entry_slippage
        return_pct      = (net_pnl / pos.invest * 100.0) if pos.invest > 0 else 0.0
        holding_bars    = trade_no  # simplified
        log = TradeLog(
            trade_no=trade_no, position_type="LONG",
            entry_date=pos.entry_date, entry_price=round(pos.entry_price, 4),
            entry_commission=round(pos.entry_commission, 4), entry_slippage=round(pos.entry_slippage, 4),
            exit_date=_fmt(date), exit_price=round(eff_sell, 4),
            exit_commission=round(commission_exit, 4), exit_slippage=round(slip_cost_exit, 4),
            exit_type=exit_type, shares=round(pos.shares, 6), holding_bars=0,
            gross_pnl=round(gross_pnl, 4), total_cost=round(total_cost, 4),
            net_pnl=round(net_pnl, 4), return_pct=round(return_pct, 4), is_winner=net_pnl > 0,
        )
        return log, net_proceeds

    def _close_short(self, pos, trade_no, date, close, exit_type, config):
        eff_cover       = close * (1.0 + config.slippage_pct)
        gross_cover     = pos.shares * eff_cover
        slip_cost_exit  = pos.shares * close * config.slippage_pct
        commission_exit = gross_cover * config.commission_pct
        short_received  = pos.shares * pos.entry_price
        gross_pnl       = short_received - gross_cover
        total_cost      = pos.entry_commission + pos.entry_slippage + commission_exit + slip_cost_exit
        net_pnl         = gross_pnl - commission_exit - pos.entry_commission - pos.entry_slippage
        return_pct      = (net_pnl / pos.invest * 100.0) if pos.invest > 0 else 0.0
        net_proceeds    = pos.invest + net_pnl
        log = TradeLog(
            trade_no=trade_no, position_type="SHORT",
            entry_date=pos.entry_date, entry_price=round(pos.entry_price, 4),
            entry_commission=round(pos.entry_commission, 4), entry_slippage=round(pos.entry_slippage, 4),
            exit_date=_fmt(date), exit_price=round(eff_cover, 4),
            exit_commission=round(commission_exit, 4), exit_slippage=round(slip_cost_exit, 4),
            exit_type=exit_type, shares=round(pos.shares, 6), holding_bars=0,
            gross_pnl=round(gross_pnl, 4), total_cost=round(total_cost, 4),
            net_pnl=round(net_pnl, 4), return_pct=round(return_pct, 4), is_winner=net_pnl > 0,
        )
        return log, net_proceeds

    @staticmethod
    def _market_value(pos, close):
        if pos is None:
            return 0.0
        if pos.position_type == "LONG":
            return pos.shares * close
        return pos.invest + (pos.entry_price - close) * pos.shares


def _fmt(date) -> str:
    try:
        if hasattr(date, "hour") and (date.hour != 0 or date.minute != 0):
            return date.strftime("%Y-%m-%d %H:%M")
        return str(date.date())
    except AttributeError:
        return str(date)
