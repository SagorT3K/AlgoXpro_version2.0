"""Centralized Candle Cache Manager (Producer-Consumer Architecture).

Thread-safe, async-native cache store backed by asyncio.Lock.
Designed for 100k+ concurrent readers with zero blocking on the event loop.

Data flow:
    Producer (background worker) --> CandleCacheManager --> Consumer (FastAPI endpoints)
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)

# Maximum candles stored per asset.  250 candles × 60s = ~4.2 hours of
# 1-minute data — enough for all indicators (longest warmup: EMA-50).
MAX_CANDLES: int = 250


class CandleCacheManager:
    """Async-safe, in-memory candle store keyed by asset name.

    Every asset maps to a ``collections.deque(maxlen=MAX_CANDLES)`` of
    candle dicts ``{time, open, high, low, close}``.  Writes are serialized
    through an ``asyncio.Lock`` so concurrent producer batches never corrupt
    the deque.  Reads are lock-free — they return a *snapshot* list that the
    caller owns, so the producer can keep writing without races.

    Usage::

        cache = CandleCacheManager()

        # Producer side (background worker)
        await cache.set_candles("EURUSD", candle_list)

        # Consumer side (FastAPI endpoint)
        candles = await cache.get_candles("EURUSD")
        if candles is None:
            return {"status": "warming_up", ...}
    """

    def __init__(self) -> None:
        self._lock: asyncio.Lock = asyncio.Lock()
        self._store: dict[str, deque[dict]] = {}
        self._timestamps: dict[str, float] = {}  # last write epoch
        self._total_updates: int = 0
        self._total_misses: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def set_candles(self, asset: str, candles: list[dict]) -> None:
        """Replace the candle window for *asset* atomically.

        If *candles* is longer than ``MAX_CANDLES`` only the most recent
        entries are kept.  The write is serialized under ``self._lock``.
        """
        if not candles:
            return

        async with self._lock:
            dq: deque[dict] = self._store.get(asset)
            if dq is None:
                dq = deque(maxlen=MAX_CANDLES)
                self._store[asset] = dq

            # Atomic replace: clear then extend.
            dq.clear()
            # If we received more than MAX_CANDLES, keep only the tail.
            if len(candles) > MAX_CANDLES:
                candles = candles[-MAX_CANDLES:]
            dq.extend(candles)

            self._timestamps[asset] = time.time()
            self._total_updates += 1

        logger.debug(
            "[CACHE WRITE] %s: %d candles (total updates: %d)",
            asset, len(candles), self._total_updates,
        )

    async def append_candle(self, asset: str, candle: dict) -> bool:
        """Append a single closed candle to the cache for *asset*.

        Called by the live-stream listener when a ``candle-generated``
        WebSocket event arrives.  If the candle's timestamp already exists
        at the tail of the deque, it is skipped (deduplication).

        Returns ``True`` if the candle was appended, ``False`` if skipped.
        """
        candle_ts = candle.get("time", 0)
        if not candle_ts:
            return False

        appended = False
        async with self._lock:
            dq: deque[dict] = self._store.get(asset)
            if dq is None:
                # Asset not yet seeded — create a deque and append.
                dq = deque(maxlen=MAX_CANDLES)
                self._store[asset] = dq

            # Dedup: skip if the last candle has the same timestamp.
            if dq and dq[-1].get("time") == candle_ts:
                return False

            dq.append(candle)
            self._timestamps[asset] = time.time()
            self._total_updates += 1
            appended = True

        if appended:
            logger.debug(
                "[CACHE APPEND] %s: ts=%s (total updates: %d)",
                asset, candle_ts, self._total_updates,
            )
        return appended

    async def get_candles(self, asset: str) -> Optional[list[dict]]:
        """Return a *snapshot* of the latest candles for *asset*.

        Returns ``None`` if the asset has never been cached (e.g. during
        server warm-up).  The returned list is a plain Python list — safe
        to pass to the analyzer without holding any lock.

        This method is intentionally lock-free for maximum read throughput.
        Deque reads in CPython are atomic at the refcount level, and we
        only need a point-in-time snapshot.
        """
        dq = self._store.get(asset)
        if dq is None or len(dq) == 0:
            self._total_misses += 1
            return None
        # snapshot = independent list copy
        return list(dq)

    def get_last_update(self, asset: str) -> Optional[float]:
        """Return the epoch timestamp of the last cache write for *asset*."""
        return self._timestamps.get(asset)

    def get_stats(self) -> dict:
        """Return cache statistics for monitoring / health endpoints."""
        return {
            "assets_cached": len(self._store),
            "total_updates": self._total_updates,
            "total_misses": self._total_misses,
            "asset_counts": {
                name: len(dq) for name, dq in self._store.items()
            },
        }

    def get_all_assets(self) -> list[str]:
        """Return the list of all asset names currently in the cache."""
        return list(self._store.keys())

    async def prune(self, max_age_seconds: float = 600.0) -> int:
        """Remove assets whose last update is older than *max_age_seconds*.

        Called periodically by the producer to evict assets that went offline
        (e.g. OTC pairs that closed).  Returns the number of pruned entries.
        """
        now = time.time()
        pruned = 0
        async with self._lock:
            stale = [
                name for name, ts in self._timestamps.items()
                if now - ts > max_age_seconds
            ]
            for name in stale:
                self._store.pop(name, None)
                self._timestamps.pop(name, None)
                pruned += 1

        if pruned:
            logger.info("[CACHE PRUNE] Removed %d stale assets", pruned)
        return pruned


# ---------------------------------------------------------------------------
# Singleton instance shared across the application
# ---------------------------------------------------------------------------
candle_cache = CandleCacheManager()
