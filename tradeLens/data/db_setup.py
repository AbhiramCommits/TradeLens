import psycopg2
from psycopg2 import sql

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "tradeLens_db",
    "user": "postgres",
    "password": "postgres",
}

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS trade_features (
    trade_id VARCHAR PRIMARY KEY,
    timestamp TIMESTAMP,
    asset_class VARCHAR,
    notional_usd FLOAT,
    market_volatility FLOAT,
    spread_bps FLOAT,
    time_of_day FLOAT,
    venue_liquidity_score FLOAT,
    algo_label VARCHAR
);
"""

CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_trade_features_asset_algo ON trade_features (asset_class, algo_label);",
    "CREATE INDEX IF NOT EXISTS idx_trade_features_timestamp ON trade_features (timestamp);",
]


def setup_database():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute(CREATE_TABLE_SQL)
    print("Table trade_features ready.")

    for idx_sql in CREATE_INDEXES_SQL:
        cur.execute(idx_sql)
        print(f"Index ensured: {idx_sql.split('ON')[1].strip().split(' ')[0]}")

    cur.close()
    conn.close()
    print("Database setup complete.")


if __name__ == "__main__":
    setup_database()
