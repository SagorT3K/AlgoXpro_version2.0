"""Redis pub/sub manager for cross-worker state sharing.

When running multiple uvicorn workers, each worker has its own in-memory
state. Redis pub/sub allows them to share:
- Asset data updates
- Analysis results (signal cache)
- Connection status
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

# Redis configuration from environment
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "50"))

# Pub/Sub channels
CHANNEL_ASSETS = "quotex:assets"
CHANNEL_SIGNALS = "quotex:signals"
CHANNEL_STATUS = "quotex:status"
CHANNEL_PRICE = "quotex:price"


class RedisManager:
    """Manages Redis connections and pub/sub for cross-worker communication."""

    def __init__(self):
        self._pool: aioredis.ConnectionPool | None = None
        self._redis: aioredis.Redis | None = None
        self._pubsub: aioredis.client.PubSub | None = None
        self._subscriptions: dict[str, list[callable]] = {}
        self._listen_task: asyncio.Task | None = None
        self._connected = False

    async def connect(self) -> bool:
        """Initialize Redis connection pool."""
        try:
            self._pool = aioredis.ConnectionPool.from_url(
                REDIS_URL,
                max_connections=REDIS_MAX_CONNECTIONS,
                decode_responses=True,
            )
            self._redis = aioredis.Redis(connection_pool=self._pool)
            await self._redis.ping()
            self._connected = True
            logger.info("Redis connected: %s", REDIS_URL)
            return True
        except Exception as e:
            logger.warning("Redis connection failed: %s (running in standalone mode)", e)
            self._connected = False
            return False

    async def disconnect(self):
        """Close Redis connections."""
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        if self._pubsub:
            await self._pubsub.close()
        if self._redis:
            await self._redis.close()
        if self._pool:
            await self._pool.disconnect()
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    async def publish_assets(self, data: dict[str, Any]):
        """Publish asset data to all workers."""
        if not self._connected:
            return
        try:
            payload = json.dumps(data, default=str)
            await self._redis.publish(CHANNEL_ASSETS, payload)
        except Exception as e:
            logger.warning("Failed to publish assets: %s", e)

    async def publish_signal(self, asset: str, result: dict[str, Any]):
        """Publish analysis result to all workers."""
        if not self._connected:
            return
        try:
            payload = json.dumps({"asset": asset, "result": result, "ts": time.time()}, default=str)
            await self._redis.publish(CHANNEL_SIGNALS, payload)
        except Exception as e:
            logger.warning("Failed to publish signal: %s", e)

    async def publish_status(self, status: dict[str, Any]):
        """Publish connection status to all workers."""
        if not self._connected:
            return
        try:
            payload = json.dumps(status, default=str)
            await self._redis.publish(CHANNEL_STATUS, payload)
        except Exception as e:
            logger.warning("Failed to publish status: %s", e)

    async def publish_price(self, asset: str, price: float, timestamp: float):
        """Publish real-time price tick to all workers."""
        if not self._connected:
            return
        try:
            payload = json.dumps({"asset": asset, "price": price, "ts": timestamp})
            await self._redis.publish(CHANNEL_PRICE, payload)
        except Exception as e:
            logger.warning("Failed to publish price: %s", e)

    # ------------------------------------------------------------------
    # Caching (for cross-worker data sharing)
    # ------------------------------------------------------------------

    async def cache_set(self, key: str, value: Any, ttl: int = 60):
        """Set a cached value with TTL."""
        if not self._connected:
            return
        try:
            await self._redis.setex(f"quotex:cache:{key}", ttl, json.dumps(value, default=str))
        except Exception as e:
            logger.warning("Cache set failed: %s", e)

    async def cache_get(self, key: str) -> Any | None:
        """Get a cached value."""
        if not self._connected:
            return None
        try:
            val = await self._redis.get(f"quotex:cache:{key}")
            if val:
                return json.loads(val)
        except Exception as e:
            logger.warning("Cache get failed: %s", e)
        return None

    async def cache_delete(self, key: str):
        """Delete a cached value."""
        if not self._connected:
            return
        try:
            await self._redis.delete(f"quotex:cache:{key}")
        except Exception as e:
            logger.warning("Cache delete failed: %s", e)

    # ------------------------------------------------------------------
    # Pub/Sub listening
    # ------------------------------------------------------------------

    def subscribe(self, channel: str, callback):
        """Register a callback for a channel."""
        if channel not in self._subscriptions:
            self._subscriptions[channel] = []
        self._subscriptions[channel].append(callback)

    async def start_listening(self):
        """Start listening for pub/sub messages in background."""
        if not self._connected:
            return
        self._pubsub = self._redis.pubsub()
        for channel in self._subscriptions:
            await self._pubsub.subscribe(channel)
        self._listen_task = asyncio.create_task(self._listen_loop())

    async def _listen_loop(self):
        """Background loop that processes incoming pub/sub messages."""
        try:
            async for message in self._pubsub.listen():
                if message["type"] != "message":
                    continue
                channel = message["channel"]
                data = json.loads(message["data"])
                for callback in self._subscriptions.get(channel, []):
                    try:
                        await callback(data)
                    except Exception as e:
                        logger.warning("Subscription callback error on %s: %s", channel, e)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning("Pub/sub listen loop ended: %s", e)


# Global singleton
redis_manager = RedisManager()
