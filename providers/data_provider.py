"""
1. Data Provider  (Bybit REST API 버전)
────────────────────────────────────────
Bybit V5 kline 엔드포인트로 OHLCV 데이터를 가져와 DataFrame 으로 반환한다.
날짜 범위가 길면 1000봉씩 페이지네이션하여 전체 데이터를 수집한다.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import requests

from models.dto import DataProviderConfig

BYBIT_URL = "https://api.bybit.com/v5/market/kline"

# 사용자 interval → Bybit interval 매핑
INTERVAL_MAP = {
    "5m":  "5",
    "15m": "15",
    "60m": "60",
    "1d":  "D",
}


def _normalize_symbol(ticker: str) -> str:
    """
    다양한 입력 형식을 Bybit 심볼(예: BTCUSDT)로 정규화한다.
    BTC-USD / BTC/USDT / BTC / BTCUSDT → BTCUSDT
    """
    s = ticker.upper().replace("-", "").replace("/", "").replace("_", "")
    # 이미 USDT로 끝나면 그대로, 아니면 USDT 붙이기
    if not s.endswith("USDT"):
        # USD 로 끝나는 경우 (예: BTCUSD) → T 추가
        if s.endswith("USD"):
            s = s + "T"
        else:
            s = s + "USDT"
    return s


def _to_ms(date_str: str) -> int:
    """'YYYY-MM-DD' 문자열 → UTC 밀리초 타임스탬프"""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


class DataProvider:
    """Bybit V5 kline API 로 OHLCV 데이터를 가져온다."""

    def fetch(self, config: DataProviderConfig) -> pd.DataFrame:
        symbol   = _normalize_symbol(config.ticker)
        interval = INTERVAL_MAP.get(config.interval, "D")
        start_ms = _to_ms(config.start_date)
        end_ms   = _to_ms(config.end_date)

        all_rows: list[list] = []
        cursor_end = end_ms

        while True:
            params = {
                "category": "linear",
                "symbol":   symbol,
                "interval": interval,
                "start":    start_ms,
                "end":      cursor_end,
                "limit":    1000,
            }
            resp = requests.get(BYBIT_URL, params=params, timeout=15)
            resp.raise_for_status()
            body = resp.json()

            if body.get("retCode") != 0:
                raise ValueError(
                    f"Bybit API 오류 ({body.get('retCode')}): {body.get('retMsg')}"
                )

            rows = body["result"]["list"]  # 최신 → 과거 순서
            if not rows:
                break

            all_rows.extend(rows)

            oldest_ts = int(rows[-1][0])
            if oldest_ts <= start_ms or len(rows) < 1000:
                break

            cursor_end = oldest_ts - 1  # 다음 페이지: 그 이전 데이터

        if not all_rows:
            raise ValueError(
                f"'{symbol}' 에 대한 데이터가 없습니다. "
                f"({config.start_date} ~ {config.end_date})"
            )

        # 오름차순 정렬 (과거 → 현재)
        all_rows.sort(key=lambda x: int(x[0]))

        df = pd.DataFrame(
            all_rows,
            columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
        df = df.set_index("timestamp")
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        df = df[df["close"] > 0]
        df.sort_index(inplace=True)

        return df
