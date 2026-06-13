"""Scaled FastAPI backend for 100k+ concurrent users.

Architecture:
- Redis pub/sub for cross-worker state sharing
- Connection pool for multiple Quotex accounts
- WebSocket to browsers for real-time updates
- Async analyzer (pandas in thread pool)
- Rate limiting middleware
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pyquotex'))
sys.path.insert(0, os.path.dirname(__file__))

import analyzer
from connection_pool import connection_pool
from redis_manager import redis_manager, CHANNEL_ASSETS, CHANNEL_SIGNALS, CHANNEL_STATUS, CHANNEL_PRICE
from rate_limiter import RateLimitMiddleware, can_open_ws, register_ws, unregister_ws

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

logging.getLogger('websockets').setLevel(logging.WARNING)
logging.getLogger('pyquotex').setLevel(logging.WARNING)

# Thread pool for CPU-bound analyzer
_analyzer_pool = ThreadPoolExecutor(max_workers=int(os.getenv("ANALYZER_WORKERS", "8")), thread_name_prefix="analyzer")

# ---------------------------------------------------------------------------
# In-memory caches (per-worker, shared via Redis)
# ---------------------------------------------------------------------------
_assets_cache: dict = {
    "data": {"Currencies": [], "Crypto": [], "Commodities": [], "Stocks": []},
    "updated_at": None,
}
_signal_cache: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------
class ConnectionManager:
    """Manages WebSocket connections from browsers."""

    def __init__(self):
        self._connections: dict[str, set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket, client_ip: str):
        await ws.accept()
        async with self._lock:
            if client_ip not in self._connections:
                self._connections[client_ip] = set()
            self._connections[client_ip].add(ws)

    async def disconnect(self, ws: WebSocket, client_ip: str):
        async with self._lock:
            if client_ip in self._connections:
                self._connections[client_ip].discard(ws)
                if not self._connections[client_ip]:
                    del self._connections[client_ip]

    @property
    def total_connections(self) -> int:
        return sum(len(conns) for conns in self._connections.values())

    @property
    def unique_ips(self) -> int:
        return len(self._connections)

    async def broadcast(self, event: str, data: dict):
        """Broadcast event to all connected browsers."""
        message = json.dumps({"event": event, "data": data}, default=str)
        disconnected = []
        for ip, conns in self._connections.items():
            for ws in conns:
                try:
                    await ws.send_text(message)
                except Exception:
                    disconnected.append((ip, ws))
        # Cleanup disconnected
        for ip, ws in disconnected:
            await self.disconnect(ws, ip)

    async def send_to(self, ws: WebSocket, event: str, data: dict):
        """Send event to a specific WebSocket."""
        try:
            await ws.send_text(json.dumps({"event": event, "data": data}, default=str))
        except Exception:
            pass


ws_manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Background tasks
# ---------------------------------------------------------------------------
_CACHE_REFRESH_SECONDS = 5
_SIGNAL_CACHE_CLEANUP_SECONDS = 120


def _get_signal_cache_key(asset_name: str) -> str:
    return f"{asset_name}:{int(time.time()) // 60}"


async def _build_categories(raw: list[dict]) -> dict:
    categories = {"Currencies": [], "Crypto": [], "Commodities": [], "Stocks": []}
    for a in raw:
        asset_id = a["asset_id"]
        description = a["description"]
        cat = _categorize(asset_id, description, a.get("upstream_category", ""))
        categories[cat].append({
            "name": _format_display_name(description, a["is_otc"]),
            "asset_id": asset_id,
            "payout": a["payout"],
            "profit_1m": a.get("profit_1m", 0),
            "profit_5m": a.get("profit_5m", 0),
            "is_otc": a["is_otc"],
            "is_open": a["is_open"],
        })
    for cat in categories:
        categories[cat].sort(key=lambda x: (x["payout"], x["profit_1m"]), reverse=True)
    return categories


async def _refresh_assets_cache_once():
    """Pull assets from connection pool and update cache + Redis."""
    try:
        raw = await connection_pool.get_assets_live()
        if not raw:
            return
        _assets_cache["data"] = await _build_categories(raw)
        _assets_cache["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Publish to Redis for other workers
        await redis_manager.publish_assets({
            "data": _assets_cache["data"],
            "updated_at": _assets_cache["updated_at"],
        })

        # Cache in Redis too
        await redis_manager.cache_set("assets_live", {
            "data": _assets_cache["data"],
            "updated_at": _assets_cache["updated_at"],
        }, ttl=10)

        logger.info("[CACHE] Refreshed: %d assets", sum(len(v) for v in _assets_cache["data"].values()))
    except Exception as e:
        logger.warning("[CACHE] Refresh error: %s", e)


async def _refresh_assets_cache_loop():
    await asyncio.sleep(2)
    while True:
        await _refresh_assets_cache_once()
        await asyncio.sleep(_CACHE_REFRESH_SECONDS)


async def _cleanup_signal_cache_loop():
    while True:
        await asyncio.sleep(_SIGNAL_CACHE_CLEANUP_SECONDS)
        now_bucket = int(time.time()) // 60
        stale = [k for k in _signal_cache if now_bucket - int(k.rsplit(":", 1)[1]) > 2]
        for k in stale:
            del _signal_cache[k]


async def _handle_redis_assets(data: dict):
    """Handle asset updates from Redis (other workers)."""
    if data.get("updated_at"):
        _assets_cache["data"] = data["data"]
        _assets_cache["updated_at"] = data["updated_at"]


async def _handle_redis_signals(data: dict):
    """Handle signal updates from Redis (other workers)."""
    asset = data.get("asset")
    result = data.get("result")
    if asset and result:
        cache_key = f"{asset}:{int(data.get('ts', time.time())) // 60}"
        _signal_cache[cache_key] = result


async def _handle_redis_price(data: dict):
    """Handle price ticks from Redis — broadcast to browsers."""
    await ws_manager.broadcast("price", data)


# ---------------------------------------------------------------------------
# App lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize connection pool
    await connection_pool.initialize()
    pool_ok = await connection_pool.connect_all()
    if pool_ok:
        logger.info("Connection pool ready")
    else:
        logger.warning("No pool connections — will retry on first request")

    # Initialize Redis
    redis_ok = await redis_manager.connect()
    if redis_ok:
        redis_manager.subscribe(CHANNEL_ASSETS, _handle_redis_assets)
        redis_manager.subscribe(CHANNEL_SIGNALS, _handle_redis_signals)
        redis_manager.subscribe(CHANNEL_PRICE, _handle_redis_price)
        await redis_manager.start_listening()
        logger.info("Redis pub/sub active")

    # Start background tasks
    cache_task = asyncio.create_task(_refresh_assets_cache_loop())
    cleanup_task = asyncio.create_task(_cleanup_signal_cache_loop())

    yield

    # Shutdown
    cache_task.cancel()
    cleanup_task.cancel()
    await redis_manager.disconnect()
    await connection_pool.connect_all()  # graceful close
    _analyzer_pool.shutdown(wait=False)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Quotex Signal Pro", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RateLimitMiddleware)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def serve_frontend():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    pool_stats = connection_pool.stats
    return {
        "status": "ok",
        "quotex_connected": pool_stats["connected"],
        "quotex_total": pool_stats["total"],
        "websocket_clients": ws_manager.total_connections,
        "unique_ips": ws_manager.unique_ips,
        "redis": redis_manager.is_connected,
        "pool_stats": pool_stats,
    }


@app.post("/api/reconnect")
async def reconnect():
    try:
        connected = await connection_pool.connect_all()
        return {"status": "ok", "connected": connected}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/assets-live")
async def assets_live():
    # Try Redis cache first (from another worker)
    cached = await redis_manager.cache_get("assets_live")
    if cached:
        return {
            "status": "ok",
            "updated_at": cached.get("updated_at"),
            "total": sum(len(v) for v in cached.get("data", {}).values()),
            "categories": cached.get("data", _assets_cache["data"]),
        }

    # Use local cache
    if _assets_cache["updated_at"] is None:
        try:
            await _refresh_assets_cache_once()
        except Exception as e:
            logger.warning("Inline cache refresh failed: %s", e)

    data = _assets_cache["data"] or {"Currencies": [], "Crypto": [], "Commodities": [], "Stocks": []}
    return {
        "status": "ok",
        "updated_at": _assets_cache["updated_at"] or datetime.now(timezone.utc).isoformat(),
        "total": sum(len(v) for v in data.values()),
        "categories": data,
    }


@app.get("/api/analyze/{asset_name}")
async def analyze(asset_name: str):
    # Check local signal cache
    cache_key = _get_signal_cache_key(asset_name)
    if cache_key in _signal_cache:
        return _freshen_countdown(_signal_cache[cache_key])

    # Check Redis cache (from another worker)
    redis_cached = await redis_manager.cache_get(f"signal:{cache_key}")
    if redis_cached:
        return _freshen_countdown(redis_cached)

    try:
        # Fetch candles from pool
        candles = await connection_pool.get_candles(asset_name, count=200, period=60)
        if not candles:
            return {"status": "Error", "asset": asset_name, "message": "No candle data available"}

        # Run analyzer in thread pool (non-blocking)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(_analyzer_pool, analyzer.analyze, asset_name, candles)

        if result.get("status") == "Result":
            cache_entry = result.copy()
            cache_entry["_candle_ts"] = int(candles[-1]["time"])
            _signal_cache[cache_key] = cache_entry

            # Publish to Redis for other workers
            await redis_manager.publish_signal(asset_name, cache_entry)
            await redis_manager.cache_set(f"signal:{cache_key}", cache_entry, ttl=60)

        return _freshen_countdown(result)

    except Exception as e:
        logger.error("Analysis failed for %s: %s", asset_name, e)
        return {"status": "Error", "asset": asset_name, "message": str(e)}


def _freshen_countdown(result: dict) -> dict:
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


# ---------------------------------------------------------------------------
# WebSocket endpoint for browsers
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    client_ip = ws.headers.get("X-Forwarded-For", "").split(",")[0].strip()
    if not client_ip:
        client_ip = ws.headers.get("X-Real-IP", "")
    if not client_ip:
        client_ip = ws.client.host if ws.client else "unknown"

    # Rate limit WebSocket connections
    if not can_open_ws(client_ip):
        await ws.close(code=1013, reason="Too many connections")
        return

    register_ws(client_ip)
    try:
        await ws_manager.connect(ws, client_ip)
        await ws_manager.send_to(ws, "connected", {
            "message": "Connected to Quotex Signal Pro",
            "worker_id": os.getenv("WORKER_ID", "0"),
        })

        # Send current assets immediately
        if _assets_cache["updated_at"]:
            await ws_manager.send_to(ws, "assets", {
                "categories": _assets_cache["data"],
                "updated_at": _assets_cache["updated_at"],
            })

        # Listen for client messages
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "analyze":
                asset = msg.get("asset")
                if asset:
                    # Run analysis and send result back
                    try:
                        candles = await connection_pool.get_candles(asset, count=200, period=60)
                        if candles:
                            loop = asyncio.get_event_loop()
                            result = await loop.run_in_executor(
                                _analyzer_pool, analyzer.analyze, asset, candles
                            )
                            if result.get("status") == "Result":
                                cache_key = _get_signal_cache_key(asset)
                                cache_entry = result.copy()
                                cache_entry["_candle_ts"] = int(candles[-1]["time"])
                                _signal_cache[cache_key] = cache_entry
                                await redis_manager.publish_signal(asset, cache_entry)
                            await ws_manager.send_to(ws, "analysis", result)
                        else:
                            await ws_manager.send_to(ws, "analysis", {
                                "status": "Error", "asset": asset, "message": "No data"
                            })
                    except Exception as e:
                        await ws_manager.send_to(ws, "analysis", {
                            "status": "Error", "asset": asset, "message": str(e)
                        })

            elif msg.get("type") == "ping":
                await ws_manager.send_to(ws, "pong", {})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug("WebSocket error: %s", e)
    finally:
        unregister_ws(client_ip)
        await ws_manager.disconnect(ws, client_ip)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
_FX_BASES = ("USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD", "CHF", "CNY")
_CRYPTO_KEYS = ("BTC", "ETH", "LTC", "XRP", "BNB", "ADA", "DOT", "SOL", "DOGE")
_COMMODITY_KEYS = ("XAU", "XAG", "OIL", "BRENT", "WTI", "GOLD", "SILVER", "CRUDE", "UKBRENT", "USCRUDE")
_STOCK_KEYS = ("APPLE", "AMAZON", "GOOGLE", "TESLA", "FACEBOOK", "MICROSOFT", "NETFLIX", "NVIDIA", "META", "AAPL", "AMZN", "GOOGL", "TSLA", "MSFT")
_STOCK_INDEX_KEYS = ("S&P", "FTSE", "CAC", "DAX", "NIKKEI", "STOXX", "NASDAQ", "DOW", "RUSSELL", "HONG KONG", "ASX", "KOSPI", "SENSEX", "IBEX")


def _format_display_name(description: str, is_otc: bool) -> str:
    name = (description or "").strip()
    if is_otc and "(OTC)" not in name:
        name = f"{name} (OTC)"
    return name


def _categorize(asset_id: str, description: str, upstream_category: str = "") -> str:
    uc = (upstream_category or "").strip().lower()
    if uc == "currency": return "Currencies"
    if uc == "cryptocurrency": return "Crypto"
    if uc == "commodity": return "Commodities"
    if uc in ("stock", "stocks", "equity", "equities"): return "Stocks"

    aid = asset_id.upper().replace("_OTC", "")
    desc_u = description.upper()
    if any(k in aid for k in _STOCK_KEYS) or any(k in desc_u for k in _STOCK_KEYS): return "Stocks"
    if any(k in desc_u for k in _STOCK_INDEX_KEYS): return "Stocks"
    if any(k in aid for k in _COMMODITY_KEYS) or any(k in desc_u for k in _COMMODITY_KEYS): return "Commodities"
    if any(k in aid for k in _CRYPTO_KEYS) or any(k in desc_u for k in _CRYPTO_KEYS): return "Crypto"
    base = aid.replace("_otc", "")
    if any(b in base for b in _FX_BASES) and len(base) == 6 and base.isalpha(): return "Currencies"
    if any(b in desc_u for b in _FX_BASES) and "/" in description: return "Currencies"
    return "Stocks"
