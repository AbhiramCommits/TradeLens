import numpy as np
import pandas as pd


def generate_trades(n: int = 500_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    trade_ids = np.arange(1, n + 1, dtype=int)
    timestamps = pd.to_datetime("2024-01-01 09:30:00") + pd.to_timedelta(
        rng.uniform(0, 365 * 24 * 3600, n), unit="s"
    )
    asset_classes = rng.choice(["Equity", "FX", "Fixed Income", "Commodity", "Credit"], n)
    notional_usd = np.round(rng.lognormal(mean=14, sigma=3, size=n), 2)
    market_volatility = np.round(rng.beta(a=2, b=5, size=n), 4)
    spread_bps = np.round(rng.gamma(shape=2, scale=5, size=n), 2)
    time_of_day = rng.choice(["Open", "Mid", "Close"], n, p=[0.3, 0.4, 0.3])
    venue_liquidity_score = np.round(rng.beta(a=5, b=2, size=n), 4)

    algo_label = np.empty(n, dtype=object)

    for i in range(n):
        vol = market_volatility[i]
        spd = spread_bps[i]
        tod = time_of_day[i]
        liq = venue_liquidity_score[i]

        if vol > 0.5 and spd > 10:
            algo_label[i] = "Aggressive"
        elif vol > 0.4 and tod == "Close":
            algo_label[i] = "IS"
        elif spd < 5 and liq > 0.7:
            algo_label[i] = "Passive"
        elif tod == "Open" and vol > 0.3:
            algo_label[i] = "VWAP"
        elif spd < spd.mean() and vol < vol.mean():
            algo_label[i] = "TWAP"
        else:
            algo_label[i] = rng.choice(["VWAP", "TWAP", "IS", "Aggressive", "Passive"])

    df = pd.DataFrame(
        {
            "trade_id": trade_ids,
            "timestamp": timestamps,
            "asset_class": asset_classes,
            "notional_usd": notional_usd,
            "market_volatility": market_volatility,
            "spread_bps": spread_bps,
            "time_of_day": time_of_day,
            "venue_liquidity_score": venue_liquidity_score,
            "algo_label": algo_label,
        }
    )

    return df


if __name__ == "__main__":
    import os

    df = generate_trades(500_000)
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "trades.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows to {out_path}")
    for label in ["VWAP", "TWAP", "IS", "Aggressive", "Passive"]:
        count = (df["algo_label"] == label).sum()
        print(f"  {label}: {count:,} ({count / len(df) * 100:.1f}%)")
