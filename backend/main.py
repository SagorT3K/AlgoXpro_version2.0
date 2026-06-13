"""FastAPI backend — Consumer layer (Producer-Consumer Architecture).

This module contains ONLY FastAPI routes and consumer logic.  It NEVER
initiates direct Quotex API calls for candle data.  All candle reads go
through the centralized ``CandleCacheManager`` which is kept warm by the
``CandleProducer`` background worker.

Data flow:
    Quotex WebSocket
        --> CandleProducer (background, writes cache)
            --> CandleCacheManager (async dict + lock)
                --> /api/analyze-batch (reads cache, never hits Quotex)

Performance targets:
    - < 5ms API response latency (cache reads are lock-free snapshots)
    - 100k+ concurrent readers (no contention on the event loop)
    - CPU-bound analysis offloaded via asyncio.to_thread()
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pyquotex'))
sys.path.insert(0, os.path.dirname(__file__))

import analyzer
from cache_manager import candle_cache, CandleCacheManager
from account_pool import get_pool, AccountPool
from producer import CandleProducer, init_producer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Silence noisy third-party loggers
for _name in ("websockets", "pyquotex", "pyquotex.ws", "pyquotex.api", "pyquotex.ws.client"):
    logging.getLogger(_name).setLevel(logging.WARNING)


# ═══════════════════════════════════════════════════════════════════════════
#  Assets cache — populated by the CandleProducer, never by a competing loop
# ═══════════════════════════════════════════════════════════════════════════

_assets_cache: dict = {
    "data": {"Currencies": [], "Crypto": [], "Commodities": [], "Stocks": []},
    "updated_at": None,
}

# Display-name / categorization constants (unchanged)
_FX_BASES = ("USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD", "CHF", "CNY")
_CRYPTO_KEYS = ("BTC", "ETH", "LTC", "XRP", "BNB", "ADA", "DOT", "SOL", "DOGE")
_COMMODITY_KEYS = ("XAU", "XAG", "OIL", "BRENT", "WTI", "GOLD", "SILVER", "CRUDE", "UKBRENT", "USCRUDE")
_STOCK_KEYS = (
    "APPLE", "AMAZON", "GOOGLE", "TESLA", "FACEBOOK", "MICROSOFT",
    "NETFLIX", "NVIDIA", "META", "AAPL", "AMZN", "GOOGL", "TSLA", "MSFT",
)
_STOCK_INDEX_KEYS = (
    "S&P", "FTSE", "CAC", "DAX", "NIKKEI", "STOXX", "NASDAQ", "DOW",
    "RUSSELL", "HONG KONG", "ASX", "KOSPI", "SENSEX", "IBEX",
)


def _format_display_name(description: str, is_otc: bool) -> str:
    name = (description or "").strip()
    if is_otc and "(OTC)" not in name:
        name = f"{name} (OTC)"
    return name


def _categorize(asset_id: str, description: str, upstream_category: str = "") -> str:
    uc = (upstream_category or "").strip().lower()
    if uc == "currency":
        return "Currencies"
    if uc == "cryptocurrency":
        return "Crypto"
    if uc == "commodity":
        return "Commodities"
    if uc in ("stock", "stocks", "equity", "equities"):
        return "Stocks"

    aid = asset_id.upper().replace("_OTC", "")
    desc_u = description.upper()

    if any(k in aid for k in _STOCK_KEYS) or any(k in desc_u for k in _STOCK_KEYS):
        return "Stocks"
    if any(k in desc_u for k in _STOCK_INDEX_KEYS):
        return "Stocks"
    if any(k in aid for k in _COMMODITY_KEYS) or any(k in desc_u for k in _COMMODITY_KEYS):
        return "Commodities"
    if any(k in aid for k in _CRYPTO_KEYS) or any(k in desc_u for k in _CRYPTO_KEYS):
        return "Crypto"

    base = aid.replace("_otc", "")
    if any(b in base for b in _FX_BASES) and len(base) == 6 and base.isalpha():
        return "Currencies"
    if any(b in desc_u for b in _FX_BASES) and "/" in description:
        return "Currencies"

    return "Stocks"


async def _build_categories(raw: list[dict]) -> dict:
    categories: dict[str, list] = {
        "Currencies": [], "Crypto": [], "Commodities": [], "Stocks": [],
    }
    for a in raw:
        cat = _categorize(a["asset_id"], a["description"], a.get("upstream_category", ""))
        categories[cat].append({
            "name": _format_display_name(a["description"], a["is_otc"]),
            "asset_id": a["asset_id"],
            "payout": a["payout"],
            "profit_1m": a.get("profit_1m", 0),
            "profit_5m": a.get("profit_5m", 0),
            "is_otc": a["is_otc"],
            "is_open": a["is_open"],
        })

    def _sort_key(x):
        return (x["payout"], x["profit_1m"])

    for cat in categories:
        categories[cat].sort(key=_sort_key, reverse=True)
    return categories


async def update_assets_cache(raw_assets: list[dict]) -> None:
    """Update the assets cache from the producer's asset list.

    Called by CandleProducer at the end of each cycle.  This is the ONLY
    function that writes to _assets_cache — there is no background loop.
    """
    if not raw_assets:
        return
    _assets_cache["data"] = await _build_categories(raw_assets)
    _assets_cache["updated_at"] = datetime.now(timezone.utc).isoformat()
    logger.debug(
        "[ASSETS CACHE] Updated %d assets at %s",
        sum(len(v) for v in _assets_cache["data"].values()),
        _assets_cache["updated_at"],
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Lifespan — startup / shutdown
# ═══════════════════════════════════════════════════════════════════════════

_producer: CandleProducer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _producer

    pool = get_pool()

    # 1. Load and connect all Quotex accounts.
    num_accounts = pool.load_accounts()
    if num_accounts == 0:
        logger.error("[STARTUP] No accounts configured — check .env")
    else:
        connected = await pool.connect_all()
        logger.info("[STARTUP] %d/%d accounts connected", connected, num_accounts)

    # 2. One-shot warm-up: populate assets cache before the producer starts.
    #    Uses the pool (any healthy session) instead of a single client.
    try:
        raw = await pool.get_assets_live()
        if raw:
            _assets_cache["data"] = await _build_categories(raw)
            _assets_cache["updated_at"] = datetime.now(timezone.utc).isoformat()
            logger.info(
                "[STARTUP] Assets cache warmed: %d assets",
                sum(len(v) for v in _assets_cache["data"].values()),
            )
    except Exception as e:
        logger.warning("[STARTUP] Initial assets cache warm-up failed: %s", e)

    # 3. Start the producer: multi-account seed + live stream + monitor.
    #    - Seed: distributes assets across N accounts, each fetches ~11 assets
    #    - Live: polls ALL sessions' candle_generated_check for new candles
    #    - Monitor: reconnects failed sessions, seeds new assets
    _producer = init_producer(candle_cache, pool, update_assets_cache)
    _producer.start()

    yield

    # Shutdown
    if _producer is not None:
        await _producer.stop()
    logger.info("Shutting down")


# ═══════════════════════════════════════════════════════════════════════════
#  FastAPI app
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Quotex Signal Analyzer",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def serve_frontend():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


# ═══════════════════════════════════════════════════════════════════════════
#  Health / status endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    """Combined health check: pool + cache + producer state."""
    cache_stats = candle_cache.get_stats()
    producer_status = _producer.get_status() if _producer else {}
    pool = get_pool()
    pool_status = pool.get_status()

    # OK once: at least one healthy session + cache has data + seed complete
    healthy_sessions = pool_status.get("healthy_sessions", 0)
    seed_done = producer_status.get("seed_complete", False)
    all_ok = healthy_sessions > 0 and cache_stats["assets_cached"] > 0 and seed_done
    return {
        "status": "ok" if all_ok else "warming_up",
        "pool": pool_status,
        "cache": cache_stats,
        "producer": {
            k: v for k, v in producer_status.items() if k != "pool"
        },
    }


@app.get("/api/cache/stats")
async def cache_stats():
    """Detailed cache statistics for monitoring."""
    return candle_cache.get_stats()


@app.post("/api/reconnect")
async def reconnect():
    """Reconnect all pool sessions."""
    try:
        pool = get_pool()
        connected = await pool.connect_all()
        return {"status": "ok", "message": f"Reconnected {connected}/{len(pool.get_all_sessions())} sessions"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
#  Asset listing endpoints (read from assets cache)
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/assets")
async def assets():
    """List all available assets grouped by category (via pool)."""
    pool = get_pool()
    healthy = pool.get_healthy_sessions()
    if not healthy:
        return {"error": "No healthy sessions available"}

    # Use the first healthy session's client for the legacy assets endpoint.
    session = healthy[0]
    if session.client is None:
        return {"error": "Session client not available"}

    try:
        payment = session.client.get_payment() or {}
        if isinstance(payment, list):
            payment = {i[0]: i for i in payment if len(i) >= 2}
        assets_payload = session.client.get_all_asset_name()
        if not assets_payload:
            return {"error": "No assets returned"}
    except Exception as e:
        return {"error": str(e)}

    grouped: dict[str, list] = {"currencies": [], "crypto": [], "commodities": [], "stocks": []}
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

        market_type = "OTC" if "_otc" in name.lower() else "REAL"
        entry = {"name": name, "market_type": market_type, "payout": pay}
        nl = name.lower()
        if any(x in nl for x in ("usd", "eur", "gbp", "aud", "nzd", "jpy", "chf", "cad")):
            grouped["currencies"].append(entry)
        elif any(x in nl for x in ("btc", "eth", "ltc", "xrp", "doge", "sol", "ada", "atom")):
            grouped["crypto"].append(entry)
        elif any(x in nl for x in ("gold", "silver", "oil", "brent", "wti")):
            grouped["commodities"].append(entry)
        else:
            grouped["stocks"].append(entry)

    return grouped


@app.get("/api/assets-live")
async def assets_live():
    """Return all currently-open Quotex pairs from the in-memory cache.

    The cache is populated by the CandleProducer at startup and updated
    each cycle.  If the cache is still empty (server just started), returns
    a warming_up status — the frontend should poll after a few seconds.
    """
    data = _assets_cache["data"]
    total = sum(len(v) for v in data.values())

    if total == 0:
        return {
            "status": "warming_up",
            "message": "Assets cache not yet populated — retry in a few seconds",
            "retry_after": 3,
            "categories": {"Currencies": [], "Crypto": [], "Commodities": [], "Stocks": []},
        }

    return {
        "status": "ok",
        "updated_at": _assets_cache["updated_at"]
            or datetime.now(timezone.utc).isoformat(),
        "total": total,
        "categories": data,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Consumer: Analysis endpoints (NEVER hit Quotex directly)
# ═══════════════════════════════════════════════════════════════════════════

def _freshen_countdown(result: dict) -> dict:
    """Recalculate candle_seconds_remaining from the stored timestamp."""
    response = result.copy()
    candle_ts = response.pop("_candle_ts", None)

    now = int(time.time())
    if candle_ts:
        if candle_ts > 4102444800:
            candle_ts = candle_ts // 1000
        secs = 60 - (now - candle_ts)
        if not (0 <= secs <= 60):
            secs = 60 - (now % 60)
    else:
        secs = 60 - (now % 60)

    response["candle_seconds_remaining"] = secs
    response["entry_timing"] = "GOOD" if secs >= 20 else "WAIT"
    return response


def _analyze_cached(asset_name: str, candles: list[dict]) -> dict:
    """CPU-bound analysis — designed to run in a thread pool.

    This function is synchronous and pure: it takes candles, returns a dict.
    Called via ``asyncio.to_thread()`` so the event loop stays responsive.
    """
    return analyzer.analyze(asset_name, candles)


async def analyze_one(asset_name: str) -> dict:
    """Analyze a single pair using ONLY the candle cache.

    This method NEVER calls ``quotex_client.get_candles()``.  If the cache
    is empty (server still warming up), it returns a structured error with
    ``retry_after`` so the frontend can poll gracefully.
    """
    # 1. Read from cache (lock-free snapshot)
    candles = await candle_cache.get_candles(asset_name)
    if candles is None:
        return {
            "status": "warming_up",
            "asset": asset_name,
            "message": "Candle cache not yet populated — retry in a few seconds",
            "retry_after": 3,
        }

    if len(candles) < 30:
        return {
            "status": "warming_up",
            "asset": asset_name,
            "message": f"Insufficient candles ({len(candles)}/30) — cache still populating",
            "retry_after": 3,
        }

    # 2. Run CPU-bound analysis in a thread pool (non-blocking)
    try:
        result = await asyncio.to_thread(_analyze_cached, asset_name, candles)
        # Attach the candle timestamp for countdown recalculation
        if result.get("status") == "Result":
            result["_candle_ts"] = int(candles[-1]["time"])
        return _freshen_countdown(result)
    except Exception as e:
        logger.error("Analysis failed for %s: %s", asset_name, e)
        return {
            "status": "Error",
            "asset": asset_name,
            "message": str(e),
        }


class AnalyzeBatchRequest(BaseModel):
    pairs: list[str]


@app.post("/api/analyze-batch")
async def analyze_batch(req: AnalyzeBatchRequest):
    """Analyze multiple pairs — pure consumer, zero network calls.

    Reads candles from the centralized cache only.  If the cache is still
    warming up, returns ``status: "warming_up"`` with a ``retry_after``
    hint so the frontend can re-poll.

    Analysis is offloaded to threads via ``asyncio.to_thread()`` to keep
    the event loop responsive under heavy load.
    """
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_pairs: list[str] = []
    for p in req.pairs:
        if p not in seen:
            seen.add(p)
            unique_pairs.append(p)

    # Fire all analyses concurrently — each one is a cache read + to_thread
    tasks = [asyncio.create_task(analyze_one(p)) for p in unique_pairs]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return {
        "status": "ok",
        "results": {
            pair: (
                result
                if isinstance(result, dict)
                else {"status": "Error", "asset": pair, "message": str(result)}
            )
            for pair, result in zip(unique_pairs, results)
        },
    }


@app.get("/api/analyze/{asset_name}")
async def analyze(asset_name: str):
    """Single-asset analysis — reads from cache only."""
    return await analyze_one(asset_name)


# ═══════════════════════════════════════════════════════════════════════════
#  WebSocket endpoint
# ═══════════════════════════════════════════════════════════════════════════

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("[WS] Client connected")
    try:
        await ws.send_json({"event": "connected", "data": {"message": "Connected to Quotex Signal Server"}})

        data = _assets_cache["data"] or {"Currencies": [], "Crypto": [], "Commodities": [], "Stocks": []}
        await ws.send_json({
            "event": "assets",
            "data": {
                "categories": data,
                "updated_at": _assets_cache["updated_at"],
                "total": sum(len(v) for v in data.values()),
            },
        })

        while True:
            msg = await ws.receive_json()
            msg_type = msg.get("type")

            if msg_type == "ping":
                await ws.send_json({"event": "pong", "data": {}})

            elif msg_type == "analyze":
                asset = msg.get("asset", "")
                result = await analyze_one(asset)
                await ws.send_json({"event": "analysis", "data": result})

    except WebSocketDisconnect:
        logger.info("[WS] Client disconnected")
    except Exception as e:
        logger.error("[WS] Error: %s", e)
        try:
            await ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=4)
