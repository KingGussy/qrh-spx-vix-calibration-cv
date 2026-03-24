from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from dotenv import load_dotenv


BASE_URL = "https://api.marketdata.app/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch a small SPX hedging dataset using raw HTTP only."
    )
    parser.add_argument("--symbol", type=str, default="SPX")
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--expiration", type=str, required=True)
    parser.add_argument("--hedge-days", type=int, default=20)
    parser.add_argument("--side", type=str, default="call", choices=["call", "put", "both"])
    parser.add_argument("--target-delta", type=float, default=0.50)
    parser.add_argument("--strike-count", type=int, default=9)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/hedging_data/raw",
    )
    parser.add_argument(
        "--version2-daily-chains",
        action="store_true",
        help="Also fetch daily filtered chains for each hedge date.",
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


def data_to_frame(data: dict) -> pd.DataFrame:
    """
    MarketData returns column-wise JSON for many endpoints.
    This turns it into a normal DataFrame.
    """
    df = pd.DataFrame(data)
    if "updated" in df.columns:
        # updated can be unix seconds on many endpoints
        try:
            df["updated"] = pd.to_datetime(df["updated"], unit="s", utc=True)
        except Exception:
            pass
    return df


def listify_strikes(data: dict, expiration: str | None = None) -> list[float]:
    """
    Strikes endpoint response is keyed by expiration date, e.g.
    {
        "s": "ok",
        "updated": 1663704000,
        "2026-02-04": [ ... strikes ... ]
    }
    """
    if data.get("s") == "no_data":
        raise RuntimeError(f"No strike data returned: {data}")

    if expiration is not None and expiration in data:
        vals = data[expiration]
        if vals:
            return sorted(float(x) for x in vals)

    for k, v in data.items():
        if k in {"s", "updated", "nextTime", "prevTime"}:
            continue
        if isinstance(v, list):
            return sorted(float(x) for x in v)

    raise RuntimeError(f"No strikes returned. Full response keys: {list(data.keys())}")


def pick_nearby_strikes(strikes: list[float], spot: float, strike_count: int) -> list[float]:
    if strike_count <= 0:
        raise ValueError("strike_count must be positive.")
    ordered = sorted(strikes, key=lambda k: abs(k - spot))
    chosen = sorted(ordered[:strike_count])
    return chosen


def safe_mid(df: pd.DataFrame) -> pd.Series:
    if "mid" in df.columns:
        return pd.to_numeric(df["mid"], errors="coerce")
    if {"bid", "ask"}.issubset(df.columns):
        bid = pd.to_numeric(df["bid"], errors="coerce")
        ask = pd.to_numeric(df["ask"], errors="coerce")
        return 0.5 * (bid + ask)
    raise ValueError("No mid column and no bid/ask columns found.")


def choose_held_option(chain_df: pd.DataFrame, target_delta: float) -> pd.Series:
    df = chain_df.copy()

    for col in ["strike", "bid", "ask", "mid", "openInterest", "volume", "underlyingPrice"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["mid_price"] = safe_mid(df)
    df = df[df["mid_price"].notna() & (df["mid_price"] > 0.01)].copy()

    spot = float(df["underlyingPrice"].dropna().iloc[0])
    df["atm_distance"] = (df["strike"] - spot).abs()

    if "openInterest" not in df.columns:
        df["openInterest"] = 0
    if "volume" not in df.columns:
        df["volume"] = 0

    df = df.sort_values(
        ["atm_distance", "openInterest", "volume"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    return df.iloc[0]


def get_expiry_chain(
    token: str,
    symbol: str,
    date: str,
    expiration: str,
    side: str | None,
) -> pd.DataFrame:
    url = f"{BASE_URL}/options/chain/{symbol}/"
    params = {"date": date, "expiration": expiration}
    if side is not None:
        params["side"] = side

    print("Requesting chain with:")
    print("  URL   :", url)
    print("  params:", params)

    data = http_get(url, get_headers(token), params)
    return data_to_frame(data)


def get_strikes(
    token: str,
    symbol: str,
    date: str,
    expiration: str,
) -> list[float]:
    url = f"{BASE_URL}/options/strikes/{symbol}/"
    params = {"date": date, "expiration": expiration}

    print("Requesting strikes with:")
    print("  URL   :", url)
    print("  params:", params)

    data = http_get(url, get_headers(token), params)
    return listify_strikes(data, expiration=expiration)


def get_option_quotes_history(
    token: str,
    option_symbol: str,
    from_date: str,
    to_date: str,
) -> pd.DataFrame:
    url = f"{BASE_URL}/options/quotes/{option_symbol}/"
    params = {"from": from_date, "to": to_date}

    print("Requesting held-option quote history with:")
    print("  URL   :", url)
    print("  params:", params)

    data = http_get(url, get_headers(token), params)
    return data_to_frame(data)


def main() -> None:
    load_dotenv()
    args = parse_args()

    token = os.getenv("MARKETDATA_TOKEN")
    if not token:
        raise RuntimeError("MARKETDATA_TOKEN not found in environment or .env")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    hedge_dates = pd.bdate_range(start=args.start_date, periods=args.hedge_days)
    hedge_end = hedge_dates[-1].strftime("%Y-%m-%d")

    # - fetch all strikes for this date/expiry
    strikes = get_strikes(
        token=token,
        symbol=args.symbol,
        date=args.start_date,
        expiration=args.expiration,
    )

    # - fetch one same-expiry chain side to estimate spot and use for filtering
    seed_side = "call" if args.side == "both" else args.side
    seed_chain = get_expiry_chain(
        token=token,
        symbol=args.symbol,
        date=args.start_date,
        expiration=args.expiration,
        side=seed_side,
    )

    if seed_chain.empty:
        raise RuntimeError("Seed chain came back empty.")

    if "underlyingPrice" not in seed_chain.columns:
        raise RuntimeError("No underlyingPrice in chain response; cannot choose nearby strikes safely.")

    spot = float(pd.to_numeric(seed_chain["underlyingPrice"], errors="coerce").dropna().iloc[0])
    nearby_strikes = pick_nearby_strikes(strikes, spot, args.strike_count)

    print(f"\nEstimated spot on {args.start_date}: {spot:.4f}")
    print(f"Nearby strikes selected for calibration: {nearby_strikes}")

    # - filter locally to avoid another wide chain request
    min_strike = min(nearby_strikes)
    max_strike = max(nearby_strikes)

    def local_filter(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        if "strike" in out.columns:
            out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
            out = out[(out["strike"] >= min_strike) & (out["strike"] <= max_strike)]

        # keep rows with some price information
        if "mid" in out.columns:
            out["mid"] = pd.to_numeric(out["mid"], errors="coerce")
            out = out[out["mid"].notna() & (out["mid"] > 0)]
        elif "bid" in out.columns and "ask" in out.columns:
            out["bid"] = pd.to_numeric(out["bid"], errors="coerce")
            out["ask"] = pd.to_numeric(out["ask"], errors="coerce")
            out = out[out["bid"].notna() & out["ask"].notna()]
            out = out[(out["bid"] >= 0) & (out["ask"] > 0)]

        return out.reset_index(drop=True)

    if args.side == "both":
        call_chain = local_filter(seed_chain)
        put_chain = local_filter(
            get_expiry_chain(
                token=token,
                symbol=args.symbol,
                date=args.start_date,
                expiration=args.expiration,
                side="put",
            )
        )
        day0_chain = pd.concat([call_chain, put_chain], ignore_index=True)
    else:
        day0_chain = local_filter(seed_chain)

    if day0_chain.empty:
        raise RuntimeError("Day-0 filtered chain is empty. Try larger strike_count or different expiry.")

    day0_chain["snapshot_date"] = args.start_date
    day0_chain_path = outdir / "day0_chain.csv"
    day0_chain.to_csv(day0_chain_path, index=False)

    # diag
    print("\nDay-0 chain columns:")
    print(day0_chain.columns.tolist())

    print("\nDay-0 chain preview:")
    preview_cols = [c for c in [
        "optionSymbol", "side", "strike", "bid", "ask", "mid",
        "openInterest", "volume", "iv", "delta", "underlyingPrice"
    ] if c in day0_chain.columns]
    print(day0_chain[preview_cols].head(20).to_string(index=False))

    print("\nNon-null counts:")
    for c in ["bid", "ask", "mid", "openInterest", "volume", "iv", "delta"]:
        if c in day0_chain.columns:
            print(f"{c}: {day0_chain[c].notna().sum()} non-null out of {len(day0_chain)}")
            
    # - pick held option
    held = choose_held_option(day0_chain, args.target_delta)
    option_symbol = str(held["optionSymbol"])

    held_summary = pd.DataFrame([{
        "symbol": args.symbol,
        "start_date": args.start_date,
        "expiration": args.expiration,
        "held_option_symbol": option_symbol,
        "held_side": held.get("side"),
        "held_strike": held.get("strike"),
        "held_day0_bid": held.get("bid"),
        "held_day0_ask": held.get("ask"),
        "held_day0_mid": held.get("mid_price"),
        "held_day0_iv": held.get("iv"),
        "held_day0_delta": held.get("delta"),
        "held_day0_gamma": held.get("gamma"),
        "held_day0_theta": held.get("theta"),
        "held_day0_vega": held.get("vega"),
        "held_day0_underlying_price": held.get("underlyingPrice"),
    }])
    held_summary_path = outdir / "held_option_summary.csv"
    held_summary.to_csv(held_summary_path, index=False)

    # - quote history for held contract
    held_quotes = get_option_quotes_history(
        token=token,
        option_symbol=option_symbol,
        from_date=args.start_date,
        to_date=hedge_end,
    )
    held_quotes_path = outdir / "held_option_quotes.csv"
    held_quotes.to_csv(held_quotes_path, index=False)

    # - optional daily chain pulls for version 2
    if args.version2_daily_chains:
        daily_dir = outdir / "daily_chains"
        daily_dir.mkdir(exist_ok=True)

        index_rows = []
        for dt in hedge_dates:
            ds = dt.strftime("%Y-%m-%d")
            try:
                if args.side == "both":
                    c = local_filter(
                        get_expiry_chain(token, args.symbol, ds, args.expiration, "call")
                    )
                    p = local_filter(
                        get_expiry_chain(token, args.symbol, ds, args.expiration, "put")
                    )
                    cdf = pd.concat([c, p], ignore_index=True)
                else:
                    cdf = local_filter(
                        get_expiry_chain(token, args.symbol, ds, args.expiration, args.side)
                    )

                cdf["snapshot_date"] = ds
                fp = daily_dir / f"chain_{ds}.csv"
                cdf.to_csv(fp, index=False)
                index_rows.append({"snapshot_date": ds, "rows": len(cdf), "file": str(fp)})
            except Exception as exc:
                index_rows.append({"snapshot_date": ds, "rows": None, "file": None, "error": str(exc)})

        pd.DataFrame(index_rows).to_csv(outdir / "daily_chain_index.csv", index=False)

    print("\nSaved:")
    print(f"  {day0_chain_path}")
    print(f"  {held_summary_path}")
    print(f"  {held_quotes_path}")
    if args.version2_daily_chains:
        print(f"  {outdir / 'daily_chain_index.csv'}")
        print(f"  {outdir / 'daily_chains'}")

    print("\nHeld option selected:")
    print(held_summary.to_string(index=False))


if __name__ == "__main__":
    main()