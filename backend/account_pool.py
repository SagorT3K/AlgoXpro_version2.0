"""Multi-Account Pool — distributes Quotex API load across N sessions.

v2: Fixes idle WS recycling, login burst 429, and reconnection races.

Root causes addressed:
    1. Idle recycling: After seeding, WS connections go silent for 60s+.
       The pyquotex watchdog detects this and force-closes the socket.
       Fix: Each session runs a keep-alive task that sends 42["tick"]
       every 20 seconds (well below the 60s idle threshold).

    2. Login burst 429: 6 accounts connecting from the same IP with 2s
       delay triggers Quotex's IP-based rate limiter.
       Fix: Increase inter-login delay to 8 seconds.

    3. Reconnection races: The monitor loop tries to fetch from sessions
       that are mid-reconnect, causing cascading failures.
       Fix: Add a _reconnecting flag; is_healthy returns False while
       reconnection is in progress.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pyquotex')))

from pyquotex.stable_api import Quotex
from pyquotex.utils.account_type import AccountType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ACCOUNT_CONNECT_DELAY: float = 8.0       # delay between account logins (prevent IP 429)
ACCOUNT_FETCH_DELAY: float = 1.5         # delay between asset fetches per account
ACCOUNT_COOLDOWN: float = 60.0           # cooldown after rate-limit (seconds)
ACCOUNT_MAX_RETRIES: int = 3             # retries per account before marking unhealthy
KEEP_ALIVE_INTERVAL: float = 15.0        # send 42["tick"] every 15s (below 60s idle)
RECONNECT_BACKOFF_BASE: float = 5.0      # base backoff for reconnections
RECONNECT_BACKOFF_MAX: float = 120.0     # max backoff cap
CANDLE_COUNT: int = 250
CANDLE_PERIOD: int = 60


# ---------------------------------------------------------------------------
# Account profile
# ---------------------------------------------------------------------------
@dataclass
class AccountProfile:
    """Credentials for a single Quotex account."""
    email: str
    password: str
    is_demo: bool = True
    label: str = ""

    @classmethod
    def from_env(cls) -> list[AccountProfile]:
        """Parse all accounts from .env (primary + QUOTEX_ACCOUNTS JSON)."""
        profiles: list[AccountProfile] = []

        primary_email = os.environ.get("QUOTEX_EMAIL", "")
        primary_pass = os.environ.get("QUOTEX_PASSWORD", "")
        primary_demo = os.environ.get("QUOTEX_DEMO", "true").lower() == "true"

        if primary_email and primary_pass:
            profiles.append(cls(
                email=primary_email,
                password=primary_pass,
                is_demo=primary_demo,
                label="primary",
            ))

        raw_json = os.environ.get("QUOTEX_ACCOUNTS", "[]")
        try:
            accounts_list = json.loads(raw_json)
            if isinstance(accounts_list, list):
                for acc in accounts_list:
                    if isinstance(acc, dict) and acc.get("email") and acc.get("password"):
                        profiles.append(cls(
                            email=acc["email"],
                            password=acc["password"],
                            is_demo=acc.get("is_demo", True),
                            label=acc.get("label", f"account{len(profiles)+1}"),
                        ))
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("[POOL] Failed to parse QUOTEX_ACCOUNTS: %s", e)

        return profiles


# ---------------------------------------------------------------------------
# Account session — one Quotex client + keep-alive + assigned assets
# ---------------------------------------------------------------------------
@dataclass
class AccountSession:
    """Manages a single Quotex connection and its assigned asset subset."""
    profile: AccountProfile
    client: Optional[Quotex] = field(default=None, repr=False)
    connected: bool = False
    _reconnecting: bool = field(default=False, repr=False)
    assigned_assets: list[str] = field(default_factory=list)
    consecutive_failures: int = 0
    last_failure_time: float = 0.0
    cooldown_until: float = 0.0
    total_fetched: int = 0
    total_failed: int = 0
    total_live_appends: int = 0
    _last_connect_time: float = field(default=0.0, repr=False)
    _keep_alive_task: Optional[asyncio.Task] = field(default=None, repr=False)
    _reconnect_backoff: float = field(default=RECONNECT_BACKOFF_BASE, repr=False)

    @property
    def is_healthy(self) -> bool:
        """True if connected, not in cooldown, and not mid-reconnect."""
        return (
            self.connected
            and not self._reconnecting
            and time.time() >= self.cooldown_until
            and self.client is not None
        )

    @property
    def is_in_cooldown(self) -> bool:
        """True if in a post-error cooldown."""
        return time.time() < self.cooldown_until

    def record_failure(self, is_rate_limit: bool = False) -> None:
        """Record a fetch failure.  Triggers cooldown on rate limits."""
        self.consecutive_failures += 1
        self.total_failed += 1
        self.last_failure_time = time.time()
        if is_rate_limit:
            self.cooldown_until = time.time() + ACCOUNT_COOLDOWN
            logger.warning(
                "[POOL] %s: rate-limited — cooling down for %ds",
                self.profile.label, ACCOUNT_COOLDOWN,
            )

    def record_success(self) -> None:
        """Reset failure counter on successful fetch."""
        self.consecutive_failures = 0
        self._reconnect_backoff = RECONNECT_BACKOFF_BASE

    def mark_unhealthy(self) -> None:
        """Mark session as disconnected — assets will be reassigned."""
        self.connected = False
        self.cooldown_until = time.time() + ACCOUNT_COOLDOWN
        self._stop_keep_alive()
        logger.warning(
            "[POOL] %s: marked unhealthy — assets reassigned",
            self.profile.label,
        )

    # ------------------------------------------------------------------
    # Keep-alive: send 42["tick"] every 20s to prevent idle recycling
    # ------------------------------------------------------------------

    def _start_keep_alive(self) -> None:
        """Start the background keep-alive task for this session."""
        self._stop_keep_alive()
        self._keep_alive_task = asyncio.create_task(self._keep_alive_loop())

    def _stop_keep_alive(self) -> None:
        """Stop the keep-alive task if running."""
        if self._keep_alive_task is not None:
            self._keep_alive_task.cancel()
            self._keep_alive_task = None

    async def _keep_alive_loop(self) -> None:
        """Send keep-alive tick every 15s and manually reset pyquotex idle timer.

        The pyquotex watchdog fires at 60s of idle. We send at 15s intervals
        to stay well under that threshold. We also forcibly update the client's
        internal last_message_at so the watchdog never triggers even if the
        outgoing tick frame is not counted as inbound activity.
        """
        while True:
            await asyncio.sleep(KEEP_ALIVE_INTERVAL)
            try:
                if not (self.client and hasattr(self.client, 'websocket')):
                    break
                ws = self.client.websocket
                if ws is None:
                    break
                await ws.send('42["tick"]')
                # Manually reset pyquotex's internal idle timer so the watchdog
                # never sees us as idle, regardless of whether the server echoes.
                for obj in (ws, self.client, getattr(self.client, 'api', None)):
                    if obj is None:
                        continue
                    for attr in ('_last_message_at', 'last_message_at', '_last_ping_time'):
                        if hasattr(obj, attr):
                            setattr(obj, attr, time.time())
                logger.debug("[KEEPALIVE] %s: tick sent + idle timer reset", self.profile.label)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug("[KEEPALIVE] %s: tick failed (%s) — connection likely dead", self.profile.label, e)
                break


# ---------------------------------------------------------------------------
# Account Pool — manages all sessions + distributes assets
# ---------------------------------------------------------------------------
class AccountPool:
    """Multi-account connection pool with round-robin asset distribution.

    Fixes applied:
        - 8s delay between account logins (prevents IP 429)
        - Keep-alive ticks every 20s per session (prevents idle recycling)
        - Per-session reconnection backoff (prevents reconnection storms)
        - _reconnecting flag (prevents fetch attempts during reconnect)
    """

    def __init__(self) -> None:
        self._sessions: list[AccountSession] = []

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def load_accounts(self) -> int:
        """Parse accounts from .env and create sessions.  Returns count."""
        profiles = AccountProfile.from_env()
        self._sessions = [AccountSession(profile=p) for p in profiles]
        logger.info(
            "[POOL] Loaded %d accounts: %s",
            len(self._sessions),
            [s.profile.label for s in self._sessions],
        )
        return len(self._sessions)

    async def connect_all(self) -> int:
        """Connect all accounts sequentially with 8s delay between each.

        The extended delay prevents Quotex's IP-based rate limiter from
        blocking connections when multiple accounts connect from the same IP.
        """
        connected = 0
        for i, session in enumerate(self._sessions):
            if i > 0:
                logger.info(
                    "[POOL] Waiting %.0fs before connecting next account...",
                    ACCOUNT_CONNECT_DELAY,
                )
                await asyncio.sleep(ACCOUNT_CONNECT_DELAY)

            success = await self._connect_session(session)
            if success:
                connected += 1

        logger.info("[POOL] %d/%d accounts connected", connected, len(self._sessions))
        return connected

    async def _connect_session(self, session: AccountSession) -> bool:
        """Connect a single session and start its keep-alive task."""
        try:
            client = Quotex(
                email=session.profile.email,
                password=session.profile.password,
                lang="pt",
            )
            client.account_is_demo = (
                AccountType.DEMO if session.profile.is_demo
                else AccountType.REAL
            )
            client.debug_ws_enable = False
            client.trace_ws = False
            # Disable native pyquotex auto-reconnect so ALL reconnections
            # route through AccountPool.reconnect_session which enforces the
            # 8s ACCOUNT_CONNECT_DELAY throttle.
            for attr in ('auto_reconnect', 'reconnect', '_auto_reconnect', 'enable_reconnect'):
                try:
                    setattr(client, attr, False)
                except Exception:
                    pass  # attribute doesn't exist on this pyquotex version
            client.session_data = {"cookies": "", "token": "", "user_agent": ""}

            check, reason = await client.connect()
            if check:
                session.client = client
                session.connected = True
                session._reconnecting = False
                session.consecutive_failures = 0
                session._reconnect_backoff = RECONNECT_BACKOFF_BASE
                session._start_keep_alive()
                logger.info("[POOL] %s connected (%s)", session.profile.label, reason)
                return True
            else:
                logger.warning("[POOL] %s login failed: %s", session.profile.label, reason)
                return False
        except Exception as e:
            logger.warning("[POOL] %s connection error: %s", session.profile.label, e)
            return False

    async def reconnect_session(self, session: AccountSession) -> bool:
        """Reconnect a single failed session with exponential backoff.

        Sets _reconnecting=True to prevent fetch attempts during reconnect.
        Uses per-session backoff to avoid reconnection storms.
        """
        if session._reconnecting:
            return False  # already in progress

        session._reconnecting = True
        session._stop_keep_alive()

        logger.info(
            "[POOL] Reconnecting %s (backoff=%.0fs)...",
            session.profile.label, session._reconnect_backoff,
        )

        # Exponential backoff before reconnecting
        await asyncio.sleep(session._reconnect_backoff)
        session._reconnect_backoff = min(
            session._reconnect_backoff * 2, RECONNECT_BACKOFF_MAX
        )

        try:
            success = await self._connect_session(session)
            if success:
                logger.info("[POOL] %s reconnected", session.profile.label)
                return True
            else:
                session._reconnecting = False
                session.mark_unhealthy()
                return False
        except Exception as e:
            logger.warning("[POOL] %s reconnect error: %s", session.profile.label, e)
            session._reconnecting = False
            session.mark_unhealthy()
            return False

    # ------------------------------------------------------------------
    # Asset distribution
    # ------------------------------------------------------------------

    def distribute_assets(self, asset_ids: list[str]) -> None:
        """Distribute assets round-robin across healthy sessions."""
        for session in self._sessions:
            session.assigned_assets = []

        healthy = [s for s in self._sessions if s.is_healthy]
        if not healthy:
            logger.warning("[POOL] No healthy sessions — cannot distribute assets")
            return

        for i, asset_id in enumerate(asset_ids):
            session = healthy[i % len(healthy)]
            session.assigned_assets.append(asset_id)

        for s in healthy:
            logger.info(
                "[POOL] %s assigned %d assets: %s...",
                s.profile.label,
                len(s.assigned_assets),
                s.assigned_assets[:3],
            )

    def reassign_assets(self) -> None:
        """Reassign assets from unhealthy sessions to healthy ones."""
        healthy = [s for s in self._sessions if s.is_healthy]
        if not healthy:
            return

        unassigned: list[str] = []
        for session in self._sessions:
            if not session.is_healthy and session.assigned_assets:
                unassigned.extend(session.assigned_assets)
                session.assigned_assets = []

        if not unassigned:
            return

        for i, asset_id in enumerate(unassigned):
            session = healthy[i % len(healthy)]
            session.assigned_assets.append(asset_id)

        logger.info(
            "[POOL] Reassigned %d assets across %d healthy sessions",
            len(unassigned), len(healthy),
        )

    # ------------------------------------------------------------------
    # Candle fetching (per-session)
    # ------------------------------------------------------------------

    async def get_candles_for_session(
        self,
        session: AccountSession,
        asset_id: str,
        count: int = CANDLE_COUNT,
        period: int = CANDLE_PERIOD,
    ) -> list[dict]:
        """Fetch candles using a specific session's client.

        Returns empty list if session is unhealthy, mid-reconnect, or
        encounters an error.  Triggers cooldown on rate limits.
        """
        if not session.is_healthy:
            return []

        try:
            buffer = max(50, int(count * 0.2))
            target_seconds = (count + buffer) * period

            candles = await session.client.get_historical_candles(
                asset_id,
                amount_of_seconds=target_seconds,
                period=period,
                timeout=30,
            )

            if candles is None or len(candles) == 0:
                session.record_failure()
                return []

            result = [
                {
                    "time": c.get("time", 0),
                    "open": float(c.get("open", 0)),
                    "high": float(c.get("high", 0)),
                    "low": float(c.get("low", 0)),
                    "close": float(c.get("close", 0)),
                }
                for c in candles
            ]
            session.record_success()
            return result

        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = (
                "429" in error_str
                or "too many requests" in error_str
                or "rate limit" in error_str
                or "batch fetch timeout" in error_str
            )
            session.record_failure(is_rate_limit=is_rate_limit)
            logger.debug(
                "[POOL] %s fetch %s failed: %s (failures=%d)",
                session.profile.label, asset_id, e, session.consecutive_failures,
            )
            return []

    async def get_assets_live(self) -> list[dict]:
        """Fetch the live asset list using the first healthy session."""
        for session in self._sessions:
            if not session.is_healthy or session.client is None:
                continue

            try:
                instruments = await session.client.get_instruments()
                if not instruments:
                    continue

                payment = session.client.get_payment() or {}
                result: list[dict] = []

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

                if result:
                    return result

            except Exception as e:
                logger.warning(
                    "[POOL] %s get_assets_live failed: %s",
                    session.profile.label, e,
                )
                continue

        return []

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_all_sessions(self) -> list[AccountSession]:
        return list(self._sessions)

    def get_healthy_sessions(self) -> list[AccountSession]:
        return [s for s in self._sessions if s.is_healthy]

    def get_session_for_asset(self, asset_id: str) -> Optional[AccountSession]:
        for session in self._sessions:
            if asset_id in session.assigned_assets:
                return session
        return None

    def get_status(self) -> dict:
        return {
            "total_sessions": len(self._sessions),
            "healthy_sessions": len(self.get_healthy_sessions()),
            "sessions": [
                {
                    "label": s.profile.label,
                    "connected": s.connected,
                    "healthy": s.is_healthy,
                    "reconnecting": s._reconnecting,
                    "in_cooldown": s.is_in_cooldown,
                    "assigned_assets": len(s.assigned_assets),
                    "total_fetched": s.total_fetched,
                    "total_failed": s.total_failed,
                    "consecutive_failures": s.consecutive_failures,
                }
                for s in self._sessions
            ],
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
account_pool: AccountPool | None = None


def get_pool() -> AccountPool:
    """Get or create the module-level account pool singleton."""
    global account_pool
    if account_pool is None:
        account_pool = AccountPool()
    return account_pool
