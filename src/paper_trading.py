import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "marketmind.db"

INITIAL_CASH = 10_000.00
FEE_RATE = 0.001  # Illustrative 0.10% per transaction
MAX_POSITION_FRACTION = 0.20  # Maximum 20% of initial capital


def connect_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row

    return connection


def initialise_database():
    with connect_db() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS account (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                cash REAL NOT NULL
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                stop_price REAL NOT NULL,
                target_price REAL NOT NULL,
                entry_fee REAL NOT NULL,
                entry_time TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                exit_price REAL,
                exit_fee REAL,
                exit_time TEXT,
                status TEXT NOT NULL DEFAULT 'OPEN'
                    CHECK (status IN ('OPEN', 'CLOSED'))
            )
        """)

        connection.execute(
            """
            INSERT OR IGNORE INTO account (id, cash)
            VALUES (1, ?)
        """,
            (INITIAL_CASH,),
        )


def get_account():
    initialise_database()

    with connect_db() as connection:
        row = connection.execute("SELECT cash FROM account WHERE id = 1").fetchone()

    return float(row["cash"])


def get_open_positions():
    initialise_database()

    with connect_db() as connection:
        rows = connection.execute("""
            SELECT *
            FROM trades
            WHERE status = 'OPEN'
            ORDER BY entry_time DESC
        """).fetchall()

    return pd.DataFrame([dict(row) for row in rows])


def get_closed_trades():
    initialise_database()

    with connect_db() as connection:
        rows = connection.execute("""
            SELECT *
            FROM trades
            WHERE status = 'CLOSED'
            ORDER BY exit_time DESC
        """).fetchall()

    return pd.DataFrame([dict(row) for row in rows])


def open_position(symbol, quantity, entry_price, stop_price, target_price):
    initialise_database()

    symbol = symbol.strip().upper()
    quantity = int(quantity)
    entry_price = float(entry_price)
    stop_price = float(stop_price)
    target_price = float(target_price)

    if not symbol:
        raise ValueError("A stock symbol is required.")

    if quantity < 1:
        raise ValueError("Quantity must be at least one share.")

    if not all(
        pd.notna(value) and value > 0
        for value in [entry_price, stop_price, target_price]
    ):
        raise ValueError("All prices must be positive numbers.")

    if not stop_price < entry_price < target_price:
        raise ValueError(
            "For a long position, the stop must be below "
            "entry and the target must be above entry."
        )

    notional = quantity * entry_price
    entry_fee = notional * FEE_RATE
    total_cost = notional + entry_fee

    with connect_db() as connection:
        # Serialise account updates to avoid spending the same
        # virtual cash twice.
        connection.execute("BEGIN IMMEDIATE")

        account = connection.execute("SELECT cash FROM account WHERE id = 1").fetchone()

        cash = float(account["cash"])

        if notional > INITIAL_CASH * MAX_POSITION_FRACTION:
            raise ValueError(
                "Position exceeds the 20% initial-capital " "allocation limit."
            )

        if total_cost > cash:
            raise ValueError("Insufficient virtual cash, including fees.")

        connection.execute(
            """
            UPDATE account
            SET cash = cash - ?
            WHERE id = 1
        """,
            (total_cost,),
        )

        connection.execute(
            """
            INSERT INTO trades (
                symbol, quantity, entry_price,
                stop_price, target_price, entry_fee
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (symbol, quantity, entry_price, stop_price, target_price, entry_fee),
        )

    return True


def close_position(trade_id, exit_price):
    initialise_database()

    trade_id = int(trade_id)
    exit_price = float(exit_price)

    if pd.isna(exit_price) or exit_price <= 0:
        raise ValueError("Exit price must be positive.")

    with connect_db() as connection:
        connection.execute("BEGIN IMMEDIATE")

        trade = connection.execute(
            """
            SELECT *
            FROM trades
            WHERE id = ? AND status = 'OPEN'
        """,
            (trade_id,),
        ).fetchone()

        if trade is None:
            raise ValueError("Open position not found. Refresh the dashboard.")

        quantity = int(trade["quantity"])
        exit_fee = quantity * exit_price * FEE_RATE

        proceeds = quantity * exit_price - exit_fee

        connection.execute(
            """
            UPDATE account
            SET cash = cash + ?
            WHERE id = 1
        """,
            (proceeds,),
        )

        connection.execute(
            """
            UPDATE trades
            SET exit_price = ?,
                exit_fee = ?,
                exit_time = CURRENT_TIMESTAMP,
                status = 'CLOSED'
            WHERE id = ?
        """,
            (exit_price, exit_fee, trade_id),
        )

    return True


def calculate_portfolio(current_prices):
    """
    Calculate cash, market value and unrealised/realised P&L.

    current_prices should map symbols to their latest available
    prices, e.g. {"NVDA": 180.0, "MSFT": 500.0}.
    """
    cash = get_account()
    open_positions = get_open_positions()
    closed_trades = get_closed_trades()

    market_value = 0.0
    unrealised_pnl = 0.0
    missing_prices = []

    if not open_positions.empty:
        for _, trade in open_positions.iterrows():
            symbol = trade["symbol"]
            quantity = int(trade["quantity"])

            price = current_prices.get(symbol)

            if price is None or pd.isna(price) or float(price) <= 0:
                # Do not silently value a position at zero.
                missing_prices.append(symbol)
                continue

            price = float(price)
            market_value += quantity * price

            cost_basis = quantity * float(trade["entry_price"]) + float(
                trade["entry_fee"]
            )

            unrealised_pnl += quantity * price - cost_basis

    realised_pnl = 0.0

    if not closed_trades.empty:
        for _, trade in closed_trades.iterrows():
            cost_basis = int(trade["quantity"]) * float(trade["entry_price"]) + float(
                trade["entry_fee"]
            )

            net_proceeds = int(trade["quantity"]) * float(trade["exit_price"]) - float(
                trade["exit_fee"]
            )

            realised_pnl += net_proceeds - cost_basis

    # If a current price is missing, report only the known
    # marked value. Do not present this as complete equity.
    portfolio_value = cash + market_value

    return {
        "cash": cash,
        "market_value": market_value,
        "portfolio_value": portfolio_value,
        "unrealised_pnl": unrealised_pnl,
        "realised_pnl": realised_pnl,
        "missing_prices": sorted(set(missing_prices)),
        "open_positions": len(open_positions),
        "closed_trades": len(closed_trades),
    }


initialise_database()
