"""Routing and request-dedup bridge extracted from plugin.py."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, Optional, Tuple


class RequestRouter:
    """Encapsulates routing mode parsing and short-ttl request dedup behavior."""

    def __init__(self, plugin: Any):
        self._plugin = plugin

    def get_routing_mode_value(self, key: str, default: str) -> str:
        value = str(self._plugin.get_config(f"routing.{key}", default) or default).strip().lower()
        return value or default

    def get_search_owner(self) -> str:
        owner = self.get_routing_mode_value("search_owner", "action")
        if owner not in {"action", "tool", "dual"}:
            return "action"
        return owner

    def get_tool_search_mode(self) -> str:
        mode = self.get_routing_mode_value("tool_search_mode", "forward")
        if mode not in {"forward", "disabled"}:
            raise ValueError(
                "routing.tool_search_mode 非法，仅允许 forward|disabled。"
                " 请执行 scripts/release_vnext_migrate.py migrate。"
            )
        return mode

    def is_request_dedup_enabled(self) -> bool:
        return bool(self._plugin.get_config("routing.enable_request_dedup", True))

    def get_request_dedup_ttl_seconds(self) -> float:
        try:
            ttl = float(self._plugin.get_config("routing.request_dedup_ttl_seconds", 2))
        except (TypeError, ValueError):
            ttl = 2.0
        return max(0.1, ttl)

    def cleanup_request_dedup_cache_locked(self, now_ts: Optional[float] = None) -> None:
        now_ts = now_ts if now_ts is not None else time.time()
        stale_keys = [
            key
            for key, entry in self._plugin._request_dedup_cache.items()
            if float(entry.get("expires_at", 0.0)) <= now_ts
        ]
        for key in stale_keys:
            self._plugin._request_dedup_cache.pop(key, None)

    async def execute_request_with_dedup(
        self,
        request_key: str,
        executor: Callable[[], Awaitable[Any]],
    ) -> Tuple[bool, Any]:
        if not self.is_request_dedup_enabled():
            result = await executor()
            return False, result

        wait_future: Optional[asyncio.Future] = None
        is_owner = False
        now_ts = time.time()

        async with self._plugin._request_dedup_lock:
            self.cleanup_request_dedup_cache_locked(now_ts)

            cached = self._plugin._request_dedup_cache.get(request_key)
            if cached and float(cached.get("expires_at", 0.0)) > now_ts:
                return True, cached.get("result")

            inflight = self._plugin._request_dedup_inflight.get(request_key)
            if inflight is not None:
                wait_future = inflight
            else:
                loop = asyncio.get_running_loop()
                new_future: asyncio.Future = loop.create_future()
                self._plugin._request_dedup_inflight[request_key] = new_future
                wait_future = new_future
                is_owner = True

        if not is_owner and wait_future is not None:
            result = await wait_future
            return True, result

        assert wait_future is not None
        try:
            result = await executor()
            ttl = self.get_request_dedup_ttl_seconds()
            expires_at = time.time() + ttl
            async with self._plugin._request_dedup_lock:
                self._plugin._request_dedup_cache[request_key] = {
                    "result": result,
                    "expires_at": expires_at,
                }
                future = self._plugin._request_dedup_inflight.pop(request_key, None)
                if future is not None and not future.done():
                    future.set_result(result)
            return False, result
        except Exception as e:
            async with self._plugin._request_dedup_lock:
                future = self._plugin._request_dedup_inflight.pop(request_key, None)
                if future is not None and not future.done():
                    future.set_exception(e)
            raise
