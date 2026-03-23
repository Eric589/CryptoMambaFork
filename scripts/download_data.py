"""
Download daily OHLCV candles from Binance and save in CryptoMamba format.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --symbol BTCUSDT --start 2018-09-17 --end 2024-09-16
    python scripts/download_data.py --symbol ETHUSDT --start 2020-01-01 --end 2024-09-16
"""

import os
import time
import argparse
import requests
import pandas as pd
from datetime import datetime, timezone

BINANCE_URL = "https://api.binance.com/api/v3/klines"
INTERVAL = "1d"
INTERVAL_SECONDS = 86400
MAX_LIMIT = 1000


def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list:
    """Fetch all daily klines between start_ms and end_ms (milliseconds, UTC)."""
    all_klines = []
    current_ms = start_ms

    while current_ms < end_ms:
        params = {
            "symbol": symbol,
            "interval": INTERVAL,
            "startTime": current_ms,
            "endTime": end_ms - 1,
            "limit": MAX_LIMIT,
        }
        resp = requests.get(BINANCE_URL, params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json()

        if not batch:
            break

        all_klines.extend(batch)
        print(f"  Fetched {len(all_klines)} candles so far...", end="\r")

        # Next batch starts after the last candle's open time
        last_open_ms = batch[-1][0]
        current_ms = last_open_ms + INTERVAL_SECONDS * 1000

        if len(batch) < MAX_LIMIT:
            break

        time.sleep(0.2)  # stay within rate limits

    print(f"  Fetched {len(all_klines)} candles total.    ")
    return all_klines


def klines_to_dataframe(klines: list) -> pd.DataFrame:
    rows = []
    for k in klines:
        rows.append({
            "Open":      float(k[1]),
            "High":      float(k[2]),
            "Low":       float(k[3]),
            "Close":     float(k[4]),
            "Volume":    float(k[5]),
            "Timestamp": int(k[0]) // 1000,   # ms → seconds
        })
    df = pd.DataFrame(rows)
    df.index.name = None
    return df


def save_split(df: pd.DataFrame, out_dir: str, name: str, start: str, end: str):
    start_ts = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    end_ts   = int(datetime.strptime(end,   "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    split = df[(df["Timestamp"] >= start_ts) & (df["Timestamp"] < end_ts)].reset_index(drop=True)
    path = os.path.join(out_dir, f"{name}.csv")
    split.to_csv(path)
    print(f"  {name:5s}: {len(split)} rows → {path}")
    return split


def main():
    parser = argparse.ArgumentParser(description="Download Binance daily candles for CryptoMamba")
    parser.add_argument("--symbol", default="BTCUSDT", help="Binance trading pair (default: BTCUSDT)")
    parser.add_argument("--start",  default="2018-09-17", help="Start date YYYY-MM-DD (inclusive)")
    parser.add_argument("--end",    default="2026-03-17", help="End date YYYY-MM-DD (inclusive)")
    parser.add_argument("--train_end", default="2024-10-01", help="Train/val split date")
    parser.add_argument("--val_end",   default="2025-10-01", help="Val/test split date")
    parser.add_argument("--out_dir",   default="./data",     help="Root output directory")
    args = parser.parse_args()

    end_next_day = pd.Timestamp(args.end) + pd.Timedelta(days=1)
    end_next_day_str = end_next_day.strftime("%Y-%m-%d")

    start_ms = int(datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms   = int(end_next_day.timestamp() * 1000)

    folder_name = f"{args.start}_{args.end}_{INTERVAL_SECONDS}"
    out_dir = os.path.join(args.out_dir, folder_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Downloading {args.symbol} daily candles: {args.start} → {args.end}")
    klines = fetch_klines(args.symbol, start_ms, end_ms)

    if not klines:
        print("No data returned. Check symbol/dates.")
        return

    df = klines_to_dataframe(klines)

    print(f"\nSaving splits to {out_dir}/")
    save_split(df, out_dir, "train", args.start,     args.train_end)
    save_split(df, out_dir, "val",   args.train_end, args.val_end)
    save_split(df, out_dir, "test",  args.val_end,   end_next_day_str)

    print(f"\nDone. {len(df)} total candles from {args.start} to {args.end}.")


if __name__ == "__main__":
    main()
