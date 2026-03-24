from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


BASE_URL = "https://api.marketdata.app/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover SPX expiries near a target DTE using raw HTTP."
    )
    parser.add_argument("--symbol", type=str, default="SPX")
    parser.add_argument("--start-date", type=str, default="2026-01-05")
    parser.add_argument("--target-dte", type=int, default=30)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--save-path",
        type=str,
        default="data/hedging_data/raw/_expiry_candidates_VIX_120125.csv",
    )
    return parser.parse_args()


def get_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def http_get(url: str, headers: dict[str, str], params: dict) -> dict:
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if data.get("s") != "ok":
        raise RuntimeError(f"API returned non-ok status: {data}")
    return data


def expirations_to_df(data: dict, start_date: str) -> pd.DataFrame:
    expirations = data.get("expirations", [])
    if not expirations:
        raise RuntimeError("No expirations returned.")

    df = pd.DataFrame({"expiration": pd.to_datetime(expirations).normalize()})
    df["start_date"] = pd.Timestamp(start_date).normalize()
    df["dte_days"] = (df["expiration"] - df["start_date"]).dt.days
    df = df[df["dte_days"] >= 0].copy()
    return df.sort_values("expiration").reset_index(drop=True)


def main() -> None:
    load_dotenv()
    args = parse_args()

    token = os.getenv("MARKETDATA_TOKEN")
    if not token:
        raise RuntimeError("MARKETDATA_TOKEN not found in environment or .env")

    url = f"{BASE_URL}/options/expirations/{args.symbol}/"
    params = {"date": args.start_date}

    print("Requesting expirations with:")
    print("  URL   :", url)
    print("  params:", params)

    data = http_get(url, get_headers(token), params)
    df = expirations_to_df(data, args.start_date)

    df["abs_error_to_target"] = (df["dte_days"] - args.target_dte).abs()
    df = df.sort_values(["abs_error_to_target", "dte_days"]).reset_index(drop=True)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)

    print("\nNearest expiries:")
    print(df.head(args.top_n).to_string(index=False))

    best = df.iloc[0]
    print("\nSuggested expiry:")
    print(f"  start date : {args.start_date}")
    print(f"  target DTE : {args.target_dte}")
    print(f"  expiry     : {best['expiration'].date()}")
    print(f"  actual DTE : {int(best['dte_days'])}")
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()