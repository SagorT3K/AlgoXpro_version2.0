"""Centralized Candle Producer — Multi-Account Seed + Live Stream.

v2: Fixes reconnection races and idle connection handling.

Changes from v1:
    - Seed loop skips sessions that are reconnecting or unhealthy
    - Monitor loop uses per-session backoff instead of global
    - Live listener only polls sessions that are truly healthy
    - Added connection health validation before each seed batch
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from account_pool import AccountPool, AccountSession
    from cache_manager import CandleCacheManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED_BATCH_PAUSE: float = 3.0          # pause between seed batches (let WS settle)
SEED_INTER_ASSET_DELAY: float = 1.5    # delay between individual seed calls
SEED_SUB_BATCH_SIZE: int = 5           # yield event loop after every N assets
SEED_SUB_BATCH_PAUSE: float = 2.0     # extra pause after every sub-batch
MONITOR_INTERVAL: float = 45.0         # seconds between monitor checks
LIVE_POLL_INTERVAL: float = 0.5        # seconds between live candle polls
CANDLE_COUNT: int = 250
CANDLE_PERIOD: int = 60
FETCH_TIMEOUT: float = 25.0            # per-asset hard timeout (< pyquotex 30s)


class CandleProducer:
    """Background worker: multi-account seed + live WebSocket stream.

    v2 fixes:
        - Skips unhealthy/reconnecting sessions during seed
        - Uses per-session backoff for reconnections
        - Validates connection health before each seed batch
    """

    def __init__(
        self,
        cache: CandleCacheManager,
        pool: AccountPool,
        assets_cache_callback: Callable | None = None,
    ) -> None:
        self._cache = cache
        self._pool = pool
        self._assets_cache_callback = assets_cache_callback
        self._tasks: list[asyncio.Task] = []
        self._running: bool = False
        self._warmed_assets: set[str] = set()
        self._total_seeded: int = 0
        self._total_live_appends: int = 0
        self._seed_complete: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            logger.warning("[PRODUCER] Already running — ignoring start()")
            return
        self._running = True
        self._tasks = [
            asyncio.create_task(self._seed_all_assets()),
            asyncio.create_task(self._live_stream_listener()),
            asyncio.create_task(self._monitor_loop()),
        ]
        logger.info("[PRODUCER] Started (multi-account seed + live stream + monitor)")

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()
        logger.info("[PRODUCER] Stopped")

    # ------------------------------------------------------------------
    # Task 1: One-shot seed — distributed across all accounts
    # ------------------------------------------------------------------

    async def _seed_all_assets(self) -> None:
        """Fetch historical candles for all assets, distributed across accounts.

        v2: Validates session health before each batch.  Skips sessions
        that are reconnecting.  Uses the pool's health checks.
        """
        logger.info("[SEED] Waiting 8s for pool connection + instruments...")
        await asyncio.sleep(8)

        assets = await self._pool.get_assets_live()
        if not assets:
            logger.warning("[SEED] No assets returned — will retry in monitor loop")
            return

        if self._assets_cache_callback is not None:
            try:
                await self._assets_cache_callback(assets)
            except Exception as e:
                logger.warning("[SEED] Assets cache callback failed: %s", e)

        asset_ids = [a["asset_id"] for a in assets]
        total = len(asset_ids)

        self._pool.distribute_assets(asset_ids)

        healthy = self._pool.get_healthy_sessions()
        logger.info(
            "[SEED] Seeding %d assets across %d accounts (~%d each)...",
            total, len(healthy), total // max(len(healthy), 1),
        )

        # Seed each session's assets, but validate health before each batch.
        seeded = 0
        failed = 0
        for session in healthy:
            if not self._running:
                break

            # Re-validate health right before starting this session's batch.
            if not session.is_healthy:
                logger.warning(
                    "[SEED] %s no longer healthy — skipping its %d assets",
                    session.profile.label, len(session.assigned_assets),
                )
                failed += len(session.assigned_assets)
                continue

            logger.info(
                "[SEED] %s seeding %d assets...",
                session.profile.label, len(session.assigned_assets),
            )

            session_seeded = 0
            assets_list = list(session.assigned_assets)
            for batch_start in range(0, len(assets_list), SEED_SUB_BATCH_SIZE):
                sub_batch = assets_list[batch_start: batch_start + SEED_SUB_BATCH_SIZE]

                for asset_id in sub_batch:
                    if not self._running:
                        break

                    if not session.is_healthy:
                        remaining = len(assets_list) - session_seeded
                        logger.warning(
                            "[SEED] %s became unhealthy after %d/%d — %d remaining",
                            session.profile.label, session_seeded, len(assets_list), remaining,
                        )
                        failed += remaining
                        break

                    try:
                        candles = await asyncio.wait_for(
                            self._pool.get_candles_for_session(
                                session, asset_id, count=CANDLE_COUNT, period=CANDLE_PERIOD,
                            ),
                            timeout=FETCH_TIMEOUT,
                        )
                    except asyncio.TimeoutError:
                        logger.warning("[SEED] %s: %s timed out after %.0fs", session.profile.label, asset_id, FETCH_TIMEOUT)
                        candles = []
                        failed += 1

                    if candles:
                        await self._cache.set_candles(asset_id, candles)
                        self._warmed_assets.add(asset_id)
                        session_seeded += 1
                        seeded += 1
                    else:
                        failed += 1

                    # Yield to event loop so keep-alive tasks can fire their ticks.
                    await asyncio.sleep(0)
                    await asyncio.sleep(SEED_INTER_ASSET_DELAY)

                if not self._running:
                    break

                # After every sub-batch, pause a bit longer — gives the WS
                # keep-alive task guaranteed breathing room.
                logger.debug(
                    "[SEED] %s: sub-batch done (%d/%d), pausing %.1fs...",
                    session.profile.label, session_seeded, len(assets_list), SEED_SUB_BATCH_PAUSE,
                )
                await asyncio.sleep(SEED_SUB_BATCH_PAUSE)

            logger.info(
                "[SEED] %s done: %d/%d seeded",
                session.profile.label, session_seeded, len(session.assigned_assets),
            )

            # Pause between sessions to let WS settle.
            if session != healthy[-1] and self._running:
                await asyncio.sleep(SEED_BATCH_PAUSE)

        self._total_seeded = seeded
        self._seed_complete = True
        logger.info(
            "[SEED] Complete: %d/%d assets seeded, %d failed",
            seeded, total, failed,
        )

    # ------------------------------------------------------------------
    # Task 2: Live stream listener — polls ALL sessions
    # ------------------------------------------------------------------

    async def _live_stream_listener(self) -> None:
        """Poll ALL healthy sessions' candle_generated_check for new candles.

        v2: Only polls sessions that are truly healthy (not reconnecting).
        Skips sessions that have dropped since the last poll.
        """
        last_seen_index: dict[str, int] = {}

        logger.info("[LIVE] Listener started (polling every %.1fs)", LIVE_POLL_INTERVAL)

        while self._running:
            await asyncio.sleep(LIVE_POLL_INTERVAL)

            for session in self._pool.get_healthy_sessions():
                # Double-check: skip if client is None or not fully connected.
                if (
                    session.client is None
                    or not hasattr(session.client, 'api')
                    or session._reconnecting
                ):
                    continue

                api = session.client.api
                if not hasattr(api, 'candle_generated_check'):
                    continue

                for asset_id in session.assigned_assets:
                    if asset_id not in self._warmed_assets:
                        continue

                    try:
                        candle_data = api.candle_generated_check.get(
                            asset_id, {}
                        ).get(CANDLE_PERIOD)

                        if not candle_data or not isinstance(candle_data, dict):
                            continue

                        candle_index = candle_data.get("index", 0)
                        if not candle_index:
                            continue

                        if last_seen_index.get(asset_id) == candle_index:
                            continue

                        last_seen_index[asset_id] = candle_index

                        candle = {
                            "time": candle_index,
                            "open": float(candle_data.get("open", 0)),
                            "high": float(candle_data.get("high", 0)),
                            "low": float(candle_data.get("low", 0)),
                            "close": float(candle_data.get("close", 0)),
                        }

                        appended = await self._cache.append_candle(asset_id, candle)
                        if appended:
                            self._total_live_appends += 1
                            session.total_live_appends += 1

                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        logger.debug(
                            "[LIVE] Error polling %s/%s: %s",
                            session.profile.label, asset_id, e,
                        )

    # ------------------------------------------------------------------
    # Task 3: Monitor loop — reconnect, seed new, prune
    # ------------------------------------------------------------------

    async def _monitor_loop(self) -> None:
        """Periodic health check with per-session reconnection.

        v2: Uses per-session backoff.  Only reconnects one session at a
        time to avoid reconnection storms.  Waits for reconnection to
        complete before attempting new operations.
        """
        logger.info("[MONITOR] Started (interval=%ss)", MONITOR_INTERVAL)

        while self._running:
            await asyncio.sleep(MONITOR_INTERVAL)

            # 1. Reconnect unhealthy sessions ONE AT A TIME.
            for session in self._pool.get_all_sessions():
                if not self._running:
                    break

                if (
                    not session.is_healthy
                    and not session.is_in_cooldown
                    and not session._reconnecting
                ):
                    logger.info("[MONITOR] Attempting reconnect: %s", session.profile.label)
                    success = await self._pool.reconnect_session(session)
                    if success:
                        # Re-distribute assets to include the reconnected session.
                        all_ids = list(self._warmed_assets)
                        if all_ids:
                            self._pool.distribute_assets(all_ids)

                    # Wait after each reconnect attempt to avoid burst.
                    await asyncio.sleep(2.0)

            # 2. Check for new assets that need seeding.
            try:
                assets = await self._pool.get_assets_live()
                if assets:
                    if self._assets_cache_callback is not None:
                        await self._assets_cache_callback(assets)

                    current_ids = {a["asset_id"] for a in assets}
                    new_assets = current_ids - self._warmed_assets

                    if new_assets:
                        logger.info(
                            "[MONITOR] %d new assets detected — seeding...",
                            len(new_assets),
                        )
                        # Only seed with currently healthy sessions.
                        healthy = self._pool.get_healthy_sessions()
                        for session in healthy:
                            for asset_id in list(new_assets):
                                if not self._running:
                                    break
                                if not session.is_healthy:
                                    break

                                candles = await self._pool.get_candles_for_session(
                                    session, asset_id,
                                    count=CANDLE_COUNT, period=CANDLE_PERIOD,
                                )
                                if candles:
                                    await self._cache.set_candles(asset_id, candles)
                                    self._warmed_assets.add(asset_id)
                                    self._total_seeded += 1
                                    new_assets.discard(asset_id)
                                await asyncio.sleep(SEED_INTER_ASSET_DELAY)

                    # Prune stale assets.
                    await self._cache.prune(max_age_seconds=3600.0)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("[MONITOR] Cycle error: %s", e)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        pool_status = self._pool.get_status()
        return {
            "running": self._running,
            "seed_complete": self._seed_complete,
            "assets_seeded": self._total_seeded,
            "warmed_assets": len(self._warmed_assets),
            "total_live_appends": self._total_live_appends,
            "pool": pool_status,
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
producer: CandleProducer | None = None


def init_producer(
    cache: CandleCacheManager,
    pool: AccountPool,
    assets_cache_callback: Callable | None = None,
) -> CandleProducer:
    global producer
    producer = CandleProducer(cache, pool, assets_cache_callback)
    return producer
