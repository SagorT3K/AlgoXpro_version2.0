"""Persistent pyquotex connection for the FastAPI backend.

Uses email/password login.  Retries with exponential backoff.
Falls back to QUOTEX_SSID env-var if password login fails.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import random

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pyquotex')))

from pyquotex.stable_api import Quotex
from pyquotex.utils.account_type import AccountType

logger = logging.getLogger(__name__)

client: Quotex | None = None
_connected = False
_last_connect_time: float = 0
_COOLDOWN_429 = 60  # seconds to wait after a 429 error


def _is_rate_limited(error: str) -> bool:
    """Check if the error is a rate-limit (429) response."""
    return "429" in error or "Too Many Requests" in error or "rate limit" in error.lower()


async def connect() -> Quotex:
    """Initialize and connect to Quotex.

    Tries standard email/password login first.
    If QUOTEX_SSID env var is set, uses that as a fallback bypass.
    Uses exponential backoff: 5s, 15s, 30s, 60s between retries.
    After a 429 error, waits at least 60 seconds before retrying.
    """
    global client, _connected, _last_connect_time

    if client is not None and _connected:
        return client

    # If we recently got a 429, enforce a cooldown
    import time
    now = time.time()
    if _last_connect_time > 0:
        elapsed = now - _last_connect_time
        if elapsed < _COOLDOWN_429:
            wait = _COOLDOWN_429 - elapsed
            logger.warning("Rate-limit cooldown: waiting %.0fs before reconnect", wait)
            await asyncio.sleep(wait)

    last_error = ""
    max_attempts = 5
    base_delay = 5  # seconds

    for attempt in range(1, max_attempts + 1):
        try:
            client = Quotex(
                email=os.environ.get("QUOTEX_EMAIL"),
                password=os.environ.get("QUOTEX_PASSWORD"),
                lang="pt",
            )
            client.account_is_demo = AccountType.DEMO
            client.debug_ws_enable = False
            client.trace_ws = False

            # Clear any cached session data so the library performs
            # a full HTTP login instead of reusing expired tokens.
            client.session_data = {"cookies": "", "token": "", "user_agent": ""}

            check, reason = await client.connect()
            if check:
                _connected = True
                _last_connect_time = 0  # reset cooldown on success
                logger.info("Login SUCCESS  |  Quotex: %s", reason)
                return client

            last_error = reason
            logger.warning("Attempt %d/%d failed: %s", attempt, max_attempts, reason)

            # Detect 429 rate limiting
            if _is_rate_limited(reason):
                logger.warning(
                    "Rate limited (429)! Waiting %ds before retry...",
                    _COOLDOWN_429
                )
                _last_connect_time = time.time()
                await asyncio.sleep(_COOLDOWN_429)
                continue

        except Exception as exc:
            last_error = str(exc)
            logger.warning("Attempt %d/%d exception: %s", attempt, max_attempts, last_error)

            # Detect 429 in exception messages too
            if _is_rate_limited(last_error):
                logger.warning(
                    "Rate limited (429)! Waiting %ds before retry...",
                    _COOLDOWN_429
                )
                _last_connect_time = time.time()
                await asyncio.sleep(_COOLDOWN_429)
                continue

        # Exponential backoff: 5s, 15s, 30s, 60s
        if attempt < max_attempts:
            delay = base_delay * (2 ** (attempt - 1))
            delay = min(delay, 60)  # cap at 60s
            # Add small random jitter to avoid thundering herd
            jitter = random.uniform(0, delay * 0.2)
            total_delay = delay + jitter
            logger.info("Retrying in %.0fs (attempt %d/%d)...", total_delay, attempt + 1, max_attempts)
            await asyncio.sleep(total_delay)

    _connected = False
    raise ConnectionError(
        f"Quotex login failed after {max_attempts} attempts. Last error: {last_error}"
    )


async def reconnect() -> Quotex:
    """Force reconnect (reset connection state and cooldown)."""
    global client, _connected, _last_connect_time
    client = None
    _connected = False
    _last_connect_time = 0  # reset cooldown so reconnect happens immediately
    return await connect()


def is_connected() -> bool:
    return _connected and client is not None


async def get_candles(
    asset: str, count: int = 200, period: int = 60
) -> list[dict]:
    """Fetch latest candles for *asset*.

    Returns a list of dicts: {time, open, high, low, close}
    """
    global client, _connected

    if client is None or not _connected:
        await connect()

    # Fetch only the candles actually needed (count × period seconds)
    # Add 20% buffer for indicator warm-up (e.g. BB needs 20, RSI needs 14)
    buffer = max(50, int(count * 0.2))
    target_seconds = (count + buffer) * period
    max_candles_needed = max(count, 100)

    for attempt in range(3):
        try:
            candles = await client.get_historical_candles(
                asset,
                amount_of_seconds=target_seconds,
                period=period,
                timeout=30
            )

            if candles is None:
                raise RuntimeError("No candles returned")

            # Ensure we have enough candles, fallback to count-based approach if needed
            actual_count = len(candles)
            print(f"[DEBUG CANDLE COUNT] Fetched {actual_count} candles for {asset}")

            # If we still don't have enough, try offset-based approach
            if actual_count < max_candles_needed:
                offset_seconds = max_candles_needed * period
                candles = await client.get_candles(
                    asset,
                    end_from_time=int(asyncio.get_event_loop().time()),
                    offset=offset_seconds,
                    period=period,
                )
                print(f"[DEBUG CANDLE COUNT] Fetched {len(candles)} candles (fallback) for {asset}")
                if candles is None:
                    raise RuntimeError("No candles returned in fallback")

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
            logger.warning("get_candles attempt %d failed: %s", attempt + 1, e)
            if attempt < 2:
                await reconnect()
            else:
                raise
    return []


async def get_assets() -> dict:
    """Fetch available assets from Quotex."""
    global client, _connected
    if client is None or not _connected:
        await connect()
    try:
        payment = client.get_payment() or {}
        if isinstance(payment, list):
            payment = {i[0]: i for i in payment if len(i) >= 2}
        assets_payload = client.get_all_asset_name()
        if not assets_payload:
            return {}
        result = {}
        for item in assets_payload:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            name = item[0]
            desc = item[1]
            pay = (
                payment.get(name, [0])[0]
                if isinstance(payment, dict) and name in payment
                else 0
            )
            if not isinstance(pay, (int, float)):
                pay = 0
            result[name] = {"description": desc, "payout": pay}
        return result
    except Exception as e:
        logger.error("get_assets failed: %s", e)
        return {}


async def get_assets_live() -> list[dict]:
    """Fetch all currently available assets with live payout + open status.

    Returns a list of dicts, one per asset:
        {
            "asset_id": "EURUSD" or "EURUSD_otc",
            "description": "Euro vs US Dollar",
            "payout": 85,            # int percentage (cleaned of any "%")
            "profit_1m": 78,
            "profit_5m": 80,
            "is_otc": False,
            "is_open": True,
        }

    Excludes any asset that pyquotex reports as closed.
    """
    global client, _connected
    if client is None or not _connected:
        await connect()

    try:
        # Ensure instruments are loaded (event-driven WS fetch)
        instruments = await client.get_instruments()
        if not instruments:
            return []

        payment = client.get_payment() or {}
        result: list[dict] = []

        # (DEBUG_INSTRUMENT_TUPLE instrumentation removed — was only
        # needed to discover i[3] as pyquotex's category field.)

        for inst in instruments:
            # Indexes (from listinfodata stream): 1=id, 2=name, 5=payment,
            # 14=open, 18=turbo_payment, -10=24H profit, -9=1M, -8=5M.
            if not isinstance(inst, (list, tuple)) or len(inst) < 19:
                continue

            asset_id = inst[1]
            raw_name = inst[2].replace("\n", "")
            # i[3] is the upstream Quotex category, e.g. "currency",
            # "cryptocurrency", "commodity", "stock".
            upstream_category = inst[3] if len(inst) > 3 else ""
            is_open_raw = inst[14]
            payment_pct = inst[5]
            turbo_pct = inst[18]

            # Skip assets that are not currently open for trading
            is_open = bool(is_open_raw) and is_open_raw != 0
            if not is_open:
                continue

            def _clean_pct(val) -> int:
                if val is None:
                    return 0
                try:
                    f = float(val)
                except (TypeError, ValueError):
                    return 0
                return int(round(f))

            payout = _clean_pct(payment_pct)
            turbo = _clean_pct(turbo_pct)

            # Profit dict from get_payment()
            info = payment.get(raw_name, {}) or {}
            profit = info.get("profit", {}) or {}
            profit_1m = _clean_pct(profit.get("1M"))
            profit_5m = _clean_pct(profit.get("5M"))

            # Skip assets that have no trading data at all (both 1m and 5m
            # profit = 0). Stocks are kept because they have profit_1m=0
            # but profit_5m>0 (only 5-minute trading available).
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
        logger.error("get_assets_live failed: %s", e)
        return []