"""Quotex connection pool for handling multiple accounts.

When 100k users hit the server, a single Quotex connection becomes a
bottleneck. This module manages a pool of connections and round-robins
requests across them.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pyquotex')))

from pyquotex.stable_api import Quotex
from pyquotex.utils.account_type import AccountType

logger = logging.getLogger(__name__)


@dataclass
class AccountConfig:
    """Configuration for a single Quotex account."""
    email: str
    password: str
    is_demo: bool = True
    label: str = ""


@dataclass
class PooledConnection:
    """A managed connection in the pool."""
    client: Quotex
    config: AccountConfig
    connected: bool = False
    last_used: float = 0
    request_count: int = 0
    error_count: int = 0
    last_error: str = ""
    cooldown_until: float = 0


class ConnectionPool:
    """Manages multiple Quotex connections for load distribution."""

    def __init__(self, max_connections: int = 5):
        self._max_connections = max_connections
        self._connections: list[PooledConnection] = []
        self._lock = asyncio.Lock()
        self._round_robin_index = 0
        self._initialized = False

    async def initialize(self):
        """Initialize the connection pool from environment variables.

        Reads QUOTEX_EMAIL/QUOTEX_PASSWORD for the primary account,
        and QUOTEX_ACCOUNTS (JSON array) for additional accounts.
        """
        if self._initialized:
            return

        accounts = []

        # Primary account from env
        primary_email = os.getenv("QUOTEX_EMAIL")
        primary_password = os.getenv("QUOTEX_PASSWORD")
        if primary_email and primary_password:
            accounts.append(AccountConfig(
                email=primary_email,
                password=primary_password,
                is_demo=os.getenv("QUOTEX_DEMO", "true").lower() == "true",
                label="primary",
            ))

        # Additional accounts from QUOTEX_ACCOUNTS JSON
        import json
        extra_accounts_json = os.getenv("QUOTEX_ACCOUNTS", "[]")
        try:
            extra_accounts = json.loads(extra_accounts_json)
            for acc in extra_accounts:
                accounts.append(AccountConfig(
                    email=acc["email"],
                    password=acc["password"],
                    is_demo=acc.get("is_demo", True),
                    label=acc.get("label", f"account_{len(accounts)}"),
                ))
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to parse QUOTEX_ACCOUNTS: %s", e)

        # Create connections (up to max)
        for config in accounts[:self._max_connections]:
            conn = PooledConnection(
                client=None,
                config=config,
            )
            self._connections.append(conn)

        self._initialized = True
        logger.info("Connection pool initialized with %d accounts", len(self._connections))

    async def connect_all(self):
        """Connect all accounts in the pool."""
        tasks = [self._connect_one(i) for i in range(len(self._connections))]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        connected = sum(1 for r in results if r is True)
        logger.info("Pool connected: %d/%d accounts", connected, len(self._connections))
        return connected

    async def _connect_one(self, index: int) -> bool:
        """Connect a single account."""
        conn = self._connections[index]
        try:
            client = Quotex(
                email=conn.config.email,
                password=conn.config.password,
                lang="pt",
            )
            client.account_is_demo = AccountType.DEMO if conn.config.is_demo else AccountType.REAL
            client.debug_ws_enable = False
            client.trace_ws = False
            client.session_data = {"cookies": "", "token": "", "user_agent": ""}

            ok, reason = await client.connect()
            if ok:
                conn.client = client
                conn.connected = True
                conn.last_used = time.time()
                logger.info("Pool[%d] connected: %s (%s)", index, conn.config.label, reason)
                return True
            else:
                conn.last_error = reason
                logger.warning("Pool[%d] failed: %s — %s", index, conn.config.label, reason)
                return False
        except Exception as e:
            conn.last_error = str(e)
            logger.warning("Pool[%d] exception: %s", index, e)
            return False

    def _get_next(self) -> PooledConnection | None:
        """Get the next available connection (round-robin)."""
        active = [c for c in self._connections if c.connected and c.client is not None]
        if not active:
            return None

        # Skip connections in cooldown
        now = time.time()
        available = [c for c in active if c.cooldown_until < now]
        if not available:
            # All in cooldown, pick the one with earliest cooldown
            return min(active, key=lambda c: c.cooldown_until)

        # Round-robin with least-recently-used bias
        conn = min(available, key=lambda c: (c.request_count, c.last_used))
        conn.last_used = now
        conn.request_count += 1
        return conn

    async def get_connection(self) -> PooledConnection | None:
        """Get a connection from the pool."""
        async with self._lock:
            return self._get_next()

    async def mark_error(self, conn: PooledConnection, error: str, cooldown: float = 60):
        """Mark a connection as having an error (with cooldown)."""
        conn.error_count += 1
        conn.last_error = error
        if "429" in error or "rate limit" in error.lower():
            conn.cooldown_until = time.time() + cooldown
            logger.warning("Pool connection %s rate-limited, cooldown %ds", conn.config.label, cooldown)

    async def reconnect_one(self, index: int) -> bool:
        """Reconnect a single failed connection."""
        if index >= len(self._connections):
            return False
        conn = self._connections[index]
        conn.connected = False
        conn.client = None
        return await self._connect_one(index)

    async def get_candles(self, asset: str, count: int = 200, period: int = 60) -> list[dict]:
        """Fetch candles using a pooled connection."""
        conn = await self.get_connection()
        if conn is None:
            raise ConnectionError("No available connections in pool")

        try:
            candles = await conn.client.get_historical_candles(
                asset,
                amount_of_seconds=(count + 50) * period,
                period=period,
                timeout=30,
            )
            if candles is None:
                raise RuntimeError("No candles returned")

            result = []
            for c in candles:
                result.append({
                    "time": c.get("time", 0),
                    "open": float(c.get("open", 0)),
                    "high": float(c.get("high", 0)),
                    "low": float(c.get("low", 0)),
                    "close": float(c.get("close", 0)),
                })
            return result
        except Exception as e:
            await self.mark_error(conn, str(e))
            raise

    async def get_assets_live(self) -> list[dict]:
        """Fetch live assets using a pooled connection."""
        conn = await self.get_connection()
        if conn is None:
            raise ConnectionError("No available connections in pool")

        try:
            instruments = await conn.client.get_instruments()
            if not instruments:
                return []

            payment = conn.client.get_payment() or {}
            result = []

            for inst in instruments:
                if not isinstance(inst, (list, tuple)) or len(inst) < 19:
                    continue

                asset_id = inst[1]
                raw_name = inst[2].replace("\n", "")
                upstream_category = inst[3] if len(inst) > 3 else ""
                is_open_raw = inst[14]
                payment_pct = inst[5]
                turbo_pct = inst[18]

                is_open = bool(is_open_raw) and is_open_raw != 0
                if not is_open:
                    continue

                def _clean_pct(val) -> int:
                    if val is None:
                        return 0
                    try:
                        return int(round(float(val)))
                    except (TypeError, ValueError):
                        return 0

                payout = _clean_pct(payment_pct)
                turbo = _clean_pct(turbo_pct)
                info = payment.get(raw_name, {}) or {}
                profit = info.get("profit", {}) or {}
                profit_1m = _clean_pct(profit.get("1M"))
                profit_5m = _clean_pct(profit.get("5M"))

                if profit_1m == 0 and profit_5m == 0:
                    continue

                is_otc = "_otc" in asset_id.lower() or "otc" in raw_name.lower()

                result.append({
                    "asset_id": asset_id,
                    "description": raw_name,
                    "payout": payout,
                    "turbo_payment": turbo,
                    "profit_24h": _clean_pct(profit.get("24H")),
                    "profit_1m": profit_1m,
                    "profit_5m": profit_5m,
                    "is_otc": is_otc,
                    "is_open": True,
                    "upstream_category": upstream_category,
                })

            return result
        except Exception as e:
            await self.mark_error(conn, str(e))
            raise

    @property
    def stats(self) -> dict:
        """Return pool statistics."""
        return {
            "total": len(self._connections),
            "connected": sum(1 for c in self._connections if c.connected),
            "total_requests": sum(c.request_count for c in self._connections),
            "total_errors": sum(c.error_count for c in self._connections),
            "connections": [
                {
                    "label": c.config.label,
                    "connected": c.connected,
                    "requests": c.request_count,
                    "errors": c.error_count,
                    "last_error": c.last_error[:100] if c.last_error else "",
                }
                for c in self._connections
            ],
        }


# Global singleton
connection_pool = ConnectionPool()
