import os
import time

import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "tradeLens_db",
    "user": "postgres",
    "password": "postgres",
}

BATCH_SIZE = 10_000

TIME_OF_DAY_MAP = {"Open": 0.0, "Mid": 1.0, "Close": 2.0}

INSERT_SQL = """
    INSERT INTO trade_features (
        trade_id, timestamp, asset_class, notional_usd,
        market_volatility, spread_bps, time_of_day,
        venue_liquidity_score, algo_label
    ) VALUES (
        %(trade_id)s, %(timestamp)s, %(asset_class)s, %(notional_usd)s,
        %(market_volatility)s, %(spread_bps)s, %(time_of_day)s,
        %(venue_liquidity_score)s, %(algo_label)s
    )
    ON CONFLICT (trade_id) DO NOTHING;
"""


def load_trades(csv_path: str):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    total_rows = 0
    t_start = time.perf_counter()

    chunk_iter = pd.read_csv(csv_path, chunksize=BATCH_SIZE)

    for batch_num, chunk in enumerate(chunk_iter, start=1):
        chunk["trade_id"] = chunk["trade_id"].astype(str)
        chunk["notional_usd"] = chunk["notional_usd"].astype(float)
        chunk["market_volatility"] = chunk["market_volatility"].astype(float)
        chunk["spread_bps"] = chunk["spread_bps"].astype(float)
        chunk["venue_liquidity_score"] = chunk["venue_liquidity_score"].astype(float)
        chunk["time_of_day"] = chunk["time_of_day"].map(TIME_OF_DAY_MAP).astype(float)

        records = chunk.to_dict(orient="records")
        execute_batch(cur, INSERT_SQL, records, page_size=BATCH_SIZE)
        conn.commit()

        total_rows += len(chunk)
        print(
            f"Batch {batch_num:>3d} inserted {len(chunk):,} rows "
            f"(cumulative: {total_rows:,})"
        )

    elapsed = time.perf_counter() - t_start
    cur.close()
    conn.close()
    print(
        f"Load complete: {total_rows:,} rows in {elapsed:.1f}s "
        f"({total_rows / elapsed:,.0f} rows/s)"
    )


if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trades.csv")
    load_trades(csv_path)
