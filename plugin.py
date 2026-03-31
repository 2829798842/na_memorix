"""na_memorix 的 Nekro 插件入口。"""

import asyncio
import copy
import hashlib
import importlib
import json
import sys
from contextlib import asynccontextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, AsyncIterator, Literal, Optional

from fastapi import APIRouter, Depends, Request
from pydantic import Field

from nekro_agent.api import i18n
from nekro_agent.api.plugin import ConfigBase, ExtraField, NekroPlugin, SandboxMethodType
from nekro_agent.core.config import config as app_config
from nekro_agent.models.db_chat_message import DBChatMessage
from nekro_agent.schemas.agent_ctx import AgentCtx

from amemorix.services import ImportService, QueryService, SummaryService
from amemorix.settings import AppSettings, DEFAULT_CONFIG, _deep_merge

# 为复制过来的 A_Memorix 模块保留旧的绝对导入兼容。
sys.modules.setdefault("server", importlib.import_module(f"{__package__}.server"))

plugin = NekroPlugin(
    name="na_memorix",
    module_name="na_memorix",
    description="A_Memorix-style memory graph plugin",
    version="0.1.0",
    author="litroenade",
    url="https://github.com/2829798842/na_memorix",
    i18n_name=i18n.i18n_text(
        zh_CN="na_memorix 记忆图谱",
        en_US="na_memorix Memory Graph",
    ),
    i18n_description=i18n.i18n_text(
        zh_CN="在 Nekro Agent 中提供 **知识库** 与 **长期记忆** 能力",
        en_US="Provides knowledge base and long-term memory capabilities inside Nekro Agent",
    ),
    allow_sleep=True,
    sleep_brief="用于长期记忆检索、图谱知识导入与当前聊天上下文的自动记忆注入。",
)


@plugin.mount_config()
class NaMemorixConfig(ConfigBase):
    """插件配置面。"""

    GLOBAL_MEMORY_ENABLED: bool = Field(
        default=True,
        title="全局记忆开关",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="全局记忆开关", en_US="Global Memory Switch"),
            i18n_description=i18n.i18n_text(
                zh_CN="关闭后暂停自动注入和后台维护，但仍可手动查看与搜索记忆。",
                en_US="Disables auto injection and background maintenance while keeping manual browsing and search available.",
            ),
        ).model_dump(),
    )
    AUTO_INJECT_ENABLED: bool = Field(default=True, title="自动注入记忆")
    AUTO_INJECT_TOP_K: int = Field(default=6, ge=1, le=20, title="自动注入 Top-K")
    AUTO_INJECT_MIN_SCORE: float = Field(default=0.32, ge=0.0, le=1.0, title="自动注入最低分数")
    CHAT_FILTER_ENABLED: bool = Field(default=False, title="启用聊天过滤")
    CHAT_FILTER_MODE: Literal["whitelist", "blacklist"] = Field(default="whitelist", title="聊天过滤模式")
    CHAT_FILTER_CHATS: str = Field(
        default="",
        title="聊天过滤列表",
        description="每行一个 chat key，支持 stream:/group:/user: 前缀。",
    )
    EMBEDDING_MODEL_GROUP: str = Field(default="text-embedding", title="Embedding 模型组")
    EMBEDDING_DIMENSION: int = Field(
        default=int(getattr(app_config, "MEMORY_EMBEDDING_DIMENSION", 1024) or 1024),
        ge=1,
        title="Embedding 维度",
    )
    EMBEDDING_TIMEOUT_SECONDS: int = Field(default=30, ge=5, le=600, title="Embedding 超时秒数")
    SUMMARIZATION_MODEL_GROUP: str = Field(default="default", title="总结模型组")
    SUMMARIZATION_CONTEXT_LENGTH: int = Field(default=50, ge=4, le=200, title="总结上下文长度")
    PERSON_PROFILE_ENABLED: bool = Field(default=True, title="启用人物画像")
    PERSON_PROFILE_TTL_MINUTES: int = Field(default=360, ge=10, title="人物画像缓存时间")
    PERSON_PROFILE_REFRESH_INTERVAL_MINUTES: int = Field(default=30, ge=1, title="人物画像刷新周期")
    WEB_READ_ONLY: bool = Field(default=False, title="前端只读模式")
    TOP_K_PARAGRAPHS: int = Field(default=20, ge=1, le=200, title="段落召回数量")
    TOP_K_RELATIONS: int = Field(default=10, ge=1, le=200, title="关系召回数量")
    TOP_K_FINAL: int = Field(default=10, ge=1, le=100, title="最终结果数量")
    RETRIEVAL_ALPHA: float = Field(default=0.5, ge=0.0, le=1.0, title="向量/关系融合权重")
    SPARSE_ENABLED: bool = Field(default=True, title="启用稀疏检索")
    CHUNK_COLLECTION_NAME: str = Field(default="na_memorix_chunks", title="Qdrant 段落集合名")
    RELATION_COLLECTION_NAME: str = Field(default="na_memorix_relations", title="Qdrant 关系集合名")
    TABLE_PREFIX: str = Field(default="na_memorix", title="PostgreSQL 表前缀")
    AUTO_SAVE_INTERVAL_MINUTES: int = Field(default=5, ge=1, title="自动保存周期")


@dataclass(frozen=True)
class RuntimeSnapshot:
    """记录哪些配置变化需要重建 runtime。"""

    settings_config: dict[str, Any]
    rebuild_fingerprint: str
    live_fingerprint: str


@dataclass
class RuntimeHandle:
    """持有具体 runtime 实例及其长生命周期后台对象。"""

    ctx: Any
    snapshot: RuntimeSnapshot
    task_manager: Optional[Any] = None
    task_manager_started: bool = False
    retain_count: int = 0
    retired: bool = False
    closing: bool = False
    closed: bool = False


STRUCTURAL_PATHS: tuple[str, ...] = (
    "storage",
    "embedding",
    "retrieval",
    "summarization",
    "threshold",
    "graph",
    "qdrant",
    "tasks",
    "advanced.debug",
)

LIVE_PATHS: tuple[str, ...] = (
    "advanced.enable_auto_save",
    "advanced.auto_save_interval_minutes",
    "memory",
    "person_profile",
    "filter",
    "routing",
    "web",
)


_runtime_handle: Optional[RuntimeHandle] = None
_runtime_lock = asyncio.Lock()
_bound_runtime_context: ContextVar[Optional[Any]] = ContextVar("na_memorix_bound_runtime_context", default=None)
_runtime_sync_task: Optional[asyncio.Task[Any]] = None
_RUNTIME_SYNC_INTERVAL_SECONDS = 2.0


def _parse_chat_filter_chats(raw_value: Any) -> list[str]:
    """解析插件配置界面里的多行聊天过滤输入。"""

    if raw_value is None:
        return []

    text = str(raw_value or "")
    items: list[str] = []
    for line in text.replace(",", "\n").splitlines():
        item = line.strip()
        if item:
            items.append(item)
    return items


def _get_nested(config: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = config
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current


def _set_nested(config: dict[str, Any], path: str, value: Any) -> None:
    current = config
    parts = path.split(".")
    for part in parts[:-1]:
        existing = current.get(part)
        if not isinstance(existing, dict):
            existing = {}
            current[part] = existing
        current = existing
    current[parts[-1]] = value


def _snapshot_payload(config: dict[str, Any], paths: tuple[str, ...]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for path in paths:
        _set_nested(payload, path, copy.deepcopy(_get_nested(config, path)))
    return payload


def _stable_fingerprint(payload: dict[str, Any]) -> str:
    dumped = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(dumped.encode("utf-8")).hexdigest()


def _build_runtime_snapshot(settings_config: dict[str, Any]) -> RuntimeSnapshot:
    rebuild_payload = _snapshot_payload(settings_config, STRUCTURAL_PATHS)
    live_payload = _snapshot_payload(settings_config, LIVE_PATHS)
    return RuntimeSnapshot(
        settings_config=settings_config,
        rebuild_fingerprint=_stable_fingerprint(rebuild_payload),
        live_fingerprint=_stable_fingerprint(live_payload),
    )


def _build_settings_dict(config_obj: NaMemorixConfig) -> dict[str, Any]:
    plugin_dir = plugin.get_plugin_data_dir()
    runtime_dir = plugin_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    overlay: dict[str, Any] = {
        "storage": {
            "data_dir": str(runtime_dir),
            "table_prefix": str(config_obj.TABLE_PREFIX or "na_memorix").strip() or "na_memorix",
        },
        "embedding": {
            "dimension": int(config_obj.EMBEDDING_DIMENSION),
            "model_name": str(config_obj.EMBEDDING_MODEL_GROUP),
            "model_group": str(config_obj.EMBEDDING_MODEL_GROUP),
            "timeout_seconds": int(config_obj.EMBEDDING_TIMEOUT_SECONDS),
        },
        "retrieval": {
            "top_k_paragraphs": int(config_obj.TOP_K_PARAGRAPHS),
            "top_k_relations": int(config_obj.TOP_K_RELATIONS),
            "top_k_final": int(config_obj.TOP_K_FINAL),
            "alpha": float(config_obj.RETRIEVAL_ALPHA),
            "sparse": {
                "enabled": bool(config_obj.SPARSE_ENABLED),
                "backend": "postgres",
            },
        },
        "advanced": {
            "enable_auto_save": True,
            "auto_save_interval_minutes": int(config_obj.AUTO_SAVE_INTERVAL_MINUTES),
        },
        "memory": {
            "enabled": bool(config_obj.GLOBAL_MEMORY_ENABLED),
        },
        "filter": {
            "enabled": bool(config_obj.CHAT_FILTER_ENABLED),
            "mode": str(config_obj.CHAT_FILTER_MODE or "whitelist").strip().lower() or "whitelist",
            "chats": _parse_chat_filter_chats(config_obj.CHAT_FILTER_CHATS),
        },
        "routing": {
            "auto_inject_enabled": bool(config_obj.AUTO_INJECT_ENABLED),
            "auto_inject_top_k": int(config_obj.AUTO_INJECT_TOP_K),
            "auto_inject_min_score": float(config_obj.AUTO_INJECT_MIN_SCORE),
        },
        "summarization": {
            "enabled": bool(config_obj.GLOBAL_MEMORY_ENABLED),
            "context_length": int(config_obj.SUMMARIZATION_CONTEXT_LENGTH),
            "model_name": str(config_obj.SUMMARIZATION_MODEL_GROUP or "default"),
            "model_group": str(config_obj.SUMMARIZATION_MODEL_GROUP or "default"),
        },
        "person_profile": {
            "enabled": bool(config_obj.PERSON_PROFILE_ENABLED),
            "profile_ttl_minutes": int(config_obj.PERSON_PROFILE_TTL_MINUTES),
            "refresh_interval_minutes": int(config_obj.PERSON_PROFILE_REFRESH_INTERVAL_MINUTES),
        },
        "web": {
            "read_only": bool(config_obj.WEB_READ_ONLY),
        },
        "qdrant": {
            "chunk_collection": str(config_obj.CHUNK_COLLECTION_NAME),
            "relation_collection": str(config_obj.RELATION_COLLECTION_NAME),
        },
    }
    return _deep_merge(DEFAULT_CONFIG, overlay)


def build_settings() -> AppSettings:
    """根据插件配置面构建 runtime 的最终配置。"""

    settings_config = _build_settings_dict(plugin.get_config(NaMemorixConfig))
    return AppSettings(config=settings_config, config_path=None)


def _apply_live_config_updates(ctx: Any, settings: AppSettings) -> None:
    """刷新那些无需重建存储即可热更新的字段。"""

    ctx.settings = settings
    ctx.config = settings.config
    if getattr(ctx, "person_profile_service", None) is not None:
        ctx.person_profile_service.plugin_config = settings.config


def _get_active_runtime_handle() -> Optional[RuntimeHandle]:
    return _runtime_handle


def _sync_runtime_locked(*, retain: bool = False) -> tuple[RuntimeHandle, Optional[RuntimeHandle], bool]:
    """在共享锁内构建、刷新或切换当前活动 runtime。"""

    global _runtime_handle

    settings = build_settings()
    snapshot = _build_runtime_snapshot(settings.config)
    retired_handle: Optional[RuntimeHandle] = None
    should_restart_task_manager = False

    if _runtime_handle is None:
        from amemorix.bootstrap import build_context

        _runtime_handle = RuntimeHandle(
            ctx=build_context(settings),
            snapshot=snapshot,
        )
    elif snapshot.rebuild_fingerprint != _runtime_handle.snapshot.rebuild_fingerprint:
        from amemorix.bootstrap import build_context

        retired_handle = _runtime_handle
        retired_handle.retired = True
        should_restart_task_manager = bool(retired_handle.task_manager_started)
        _runtime_handle = RuntimeHandle(
            ctx=build_context(settings),
            snapshot=snapshot,
        )
    elif snapshot.live_fingerprint != _runtime_handle.snapshot.live_fingerprint:
        _apply_live_config_updates(_runtime_handle.ctx, settings)
        _runtime_handle.snapshot = snapshot

    assert _runtime_handle is not None
    if retain:
        _runtime_handle.retain_count += 1
    return _runtime_handle, retired_handle, should_restart_task_manager


async def _ensure_active_task_manager_started() -> Any:
    """按需启动当前活动 runtime 的 task manager。"""

    async with _runtime_lock:
        handle = _runtime_handle
        if handle is None:
            raise RuntimeError("runtime not initialized")

        if handle.task_manager is None:
            from amemorix.task_manager import TaskManager

            handle.task_manager = TaskManager(handle.ctx)
        if not handle.task_manager_started:
            await handle.task_manager.start()
            handle.task_manager_started = True
        return handle.task_manager


async def _finalize_runtime_handle(handle: RuntimeHandle) -> None:
    """当无人再使用时，停止后台任务并关闭已退役的 runtime。"""

    if handle.closed:
        return

    if handle.task_manager is not None and handle.task_manager_started:
        try:
            await handle.task_manager.stop()
        finally:
            handle.task_manager_started = False

    should_close = False
    async with _runtime_lock:
        if handle.closed:
            return
        if handle.retain_count <= 0 and not handle.closing:
            handle.closing = True
            should_close = True

    if not should_close:
        return

    try:
        await handle.ctx.close()
    finally:
        async with _runtime_lock:
            handle.task_manager = None
            handle.closed = True
            handle.closing = False


async def _retire_runtime_handle(handle: Optional[RuntimeHandle]) -> None:
    """在旧 runtime 失去活动身份后，平滑地将其退役。"""

    if handle is None:
        return
    handle.retired = True
    await _finalize_runtime_handle(handle)


async def _synchronize_runtime(*, retain: bool = False, require_task_manager: bool = False) -> RuntimeHandle:
    """让活动 runtime 与最新插件配置保持同步。"""

    retired_handle: Optional[RuntimeHandle] = None
    should_restart_task_manager = False

    async with _runtime_lock:
        handle, retired_handle, should_restart_task_manager = _sync_runtime_locked(retain=retain)

    if retired_handle is not None:
        await _retire_runtime_handle(retired_handle)

    if require_task_manager or should_restart_task_manager:
        await _ensure_active_task_manager_started()
        active = _get_active_runtime_handle()
        if active is not None:
            return active

    return handle


async def _release_runtime_handle(handle: RuntimeHandle) -> None:
    """在请求结束后释放已保留的 runtime handle。"""

    close_after_release = False

    async with _runtime_lock:
        if handle.retain_count > 0:
            handle.retain_count -= 1
        close_after_release = bool(handle.retired and handle.retain_count <= 0 and not handle.closed)

    if close_after_release:
        await _finalize_runtime_handle(handle)


async def ensure_runtime_ready() -> Any:
    """返回一个与当前插件配置一致的 runtime。"""

    bound_ctx = _bound_runtime_context.get()
    if bound_ctx is not None:
        return bound_ctx
    return (await _synchronize_runtime()).ctx


@asynccontextmanager
async def runtime_scope() -> AsyncIterator[Any]:
    """在当前请求或任务的作用域内绑定一个稳定的 runtime。"""

    handle = await _synchronize_runtime(retain=True)
    token = bind_runtime_context(handle.ctx)
    try:
        yield handle.ctx
    finally:
        reset_runtime_context(token)
        await _release_runtime_handle(handle)


def bind_runtime_context(ctx: Any) -> Token:
    """把稳定的 runtime context 绑定到当前请求或任务作用域。"""

    return _bound_runtime_context.set(ctx)


def reset_runtime_context(token: Token) -> None:
    """在请求处理结束后重置作用域内的 runtime context。"""

    _bound_runtime_context.reset(token)


def get_runtime_context() -> Optional[Any]:
    """在不强制初始化的前提下返回当前 runtime context。"""

    bound_ctx = _bound_runtime_context.get()
    if bound_ctx is not None:
        return bound_ctx
    handle = _get_active_runtime_handle()
    return handle.ctx if handle is not None else None


def get_task_manager() -> Optional[Any]:
    """在不强制初始化的前提下返回当前 task manager。"""

    handle = _get_active_runtime_handle()
    if handle is None or not handle.task_manager_started:
        return None
    return handle.task_manager


def get_runtime_status() -> dict[str, Any]:
    """返回轻量的就绪状态信息，而不触发惰性初始化。"""

    handle = _get_active_runtime_handle()
    if handle is None:
        return {
            "ready": True,
            "runtime_initialized": False,
            "task_manager_started": False,
            "mode": "lazy",
        }

    ctx = handle.ctx
    runtime_ready = all(
        [
            getattr(ctx, "metadata_store", None) is not None,
            getattr(ctx, "graph_store", None) is not None,
            getattr(ctx, "vector_store", None) is not None,
            getattr(ctx, "retriever", None) is not None,
        ]
    )
    return {
        "ready": runtime_ready,
        "runtime_initialized": True,
        "task_manager_started": bool(handle.task_manager_started),
        "mode": "active",
    }


async def ensure_task_manager_started() -> Any:
    """惰性创建并启动 task manager。"""

    await _synchronize_runtime(require_task_manager=True)
    manager = get_task_manager()
    if manager is None:
        raise RuntimeError("task manager not initialized")
    return manager


async def _compat_runtime_dependency(request: Request) -> AsyncIterator[None]:
    """为兼容层 `/api/*` 路由施加请求级 runtime 绑定。"""

    path = str(request.url.path or "")
    if not path.startswith("/api") or path == "/api/config":
        yield
        return

    async with runtime_scope():
        yield


async def _v1_runtime_dependency() -> AsyncIterator[None]:
    """为 `/v1/*` 路由施加请求级 runtime 绑定。"""

    async with runtime_scope():
        yield


async def _runtime_sync_loop() -> None:
    """让长生命周期 runtime 对象持续跟随配置变化。"""

    while True:
        try:
            await asyncio.sleep(_RUNTIME_SYNC_INTERVAL_SECONDS)
            if _get_active_runtime_handle() is None:
                continue
            await _synchronize_runtime()
        except asyncio.CancelledError:
            break
        except Exception:
            plugin.logger.exception("na_memorix runtime sync loop failed")


class RuntimeProxy:
    """向旧版 A_Memorix 兼容代码暴露当前 runtime。"""

    @staticmethod
    def _ctx() -> Optional[Any]:
        return _bound_runtime_context.get() or get_runtime_context()

    def get_config(self, key: str, default: Any = None) -> Any:
        ctx = self._ctx()
        if ctx is not None:
            return ctx.get_config(key, default)
        return build_settings().get(key, default)

    @property
    def settings(self) -> AppSettings:
        ctx = self._ctx()
        if ctx is not None:
            return ctx.settings
        return build_settings()

    @property
    def config(self) -> dict[str, Any]:
        ctx = self._ctx()
        if ctx is not None:
            return ctx.config
        return build_settings().config

    @property
    def metadata_store(self) -> Any:
        ctx = self._ctx()
        return ctx.metadata_store if ctx is not None else None

    @property
    def graph_store(self) -> Any:
        ctx = self._ctx()
        return ctx.graph_store if ctx is not None else None

    @property
    def vector_store(self) -> Any:
        ctx = self._ctx()
        return ctx.vector_store if ctx is not None else None

    @property
    def embedding_manager(self) -> Any:
        ctx = self._ctx()
        return ctx.embedding_manager if ctx is not None else None

    @property
    def sparse_index(self) -> Any:
        ctx = self._ctx()
        return ctx.sparse_index if ctx is not None else None

    @property
    def retriever(self) -> Any:
        ctx = self._ctx()
        return ctx.retriever if ctx is not None else None

    @property
    def person_profile_service(self) -> Any:
        ctx = self._ctx()
        return ctx.person_profile_service if ctx is not None else None

    @property
    def _runtime_auto_save(self) -> Optional[bool]:
        ctx = self._ctx()
        return getattr(ctx, "_runtime_auto_save", None) if ctx is not None else None

    @_runtime_auto_save.setter
    def _runtime_auto_save(self, value: Optional[bool]) -> None:
        ctx = self._ctx()
        if ctx is not None:
            ctx._runtime_auto_save = value


runtime_proxy = RuntimeProxy()


@plugin.mount_router()
def create_router() -> APIRouter:
    """挂载兼容 Web UI 路由与 `/v1` API 路由。"""

    from .server import MemorixServer
    from amemorix.routers.v1_router import router as v1_router

    compat = MemorixServer(plugin_instance=runtime_proxy)
    router = APIRouter()
    router.include_router(compat.app.router, dependencies=[Depends(_compat_runtime_dependency)])
    router.include_router(v1_router, dependencies=[Depends(_v1_runtime_dependency)])
    return router


async def _build_recent_query_text(_ctx: AgentCtx, limit: int = 6) -> str:
    chat_key = _ctx.from_chat_key or ""
    if not chat_key:
        return ""
    messages = (
        await DBChatMessage.filter(chat_key=chat_key)
        .order_by("-send_timestamp")
        .limit(max(1, int(limit)))
        .all()
    )
    items = [str(msg.content_text or "").strip() for msg in reversed(messages) if str(msg.content_text or "").strip()]
    return "\n".join(items[-limit:])


@plugin.mount_prompt_inject_method("na_memorix_prompt")
async def na_memorix_prompt(_ctx: AgentCtx) -> str:
    cfg = plugin.get_config(NaMemorixConfig)
    if not bool(cfg.GLOBAL_MEMORY_ENABLED) or not bool(cfg.AUTO_INJECT_ENABLED):
        return ""

    async with runtime_scope() as ctx:
        if not ctx.is_chat_enabled(
            stream_id=_ctx.from_chat_key,
            group_id=_ctx.channel_id if (_ctx.channel_type or "") == "group" else None,
            user_id=_ctx.from_platform_userid,
        ):
            return ""

        query_text = await _build_recent_query_text(_ctx)
        if not query_text:
            return ""

        service = QueryService(ctx)
        result = await service.search(query=query_text, top_k=int(cfg.AUTO_INJECT_TOP_K))
        items = [
            item
            for item in result.get("results", [])
            if float(item.get("score", 0.0) or 0.0) >= float(cfg.AUTO_INJECT_MIN_SCORE)
        ]
        if not items:
            return ""

        lines = ["[Memorix Context]"]
        for idx, item in enumerate(items[: int(cfg.AUTO_INJECT_TOP_K)], start=1):
            content = str(item.get("content", "") or "").strip()
            source = str(item.get("source", "") or "").strip()
            score = float(item.get("score", 0.0) or 0.0)
            if not content:
                continue
            suffix = f" | score={score:.3f}"
            if source:
                suffix += f" | source={source}"
            lines.append(f"{idx}. {content[:280]}{suffix}")
        return "\n".join(lines) if len(lines) > 1 else ""


@plugin.mount_sandbox_method(SandboxMethodType.TOOL, "搜索记忆", description="在 na_memorix 记忆库中搜索相关知识。")
async def memorix_search(_ctx: AgentCtx, query: str, top_k: int = 5):
    del _ctx
    async with runtime_scope() as ctx:
        return await QueryService(ctx).search(query=query, top_k=top_k)


@plugin.mount_sandbox_method(SandboxMethodType.TOOL, "导入记忆文本", description="向 na_memorix 导入一段文本知识。")
async def memorix_import_text(_ctx: AgentCtx, text: str, source: str = ""):
    async with runtime_scope() as ctx:
        src = source.strip() or f"chat:{_ctx.from_chat_key or 'manual'}"
        return await ImportService(ctx).import_text(text=text, source=src)


@plugin.mount_sandbox_method(
    SandboxMethodType.TOOL,
    "导入当前聊天总结",
    description="将当前聊天最近消息总结后写入 na_memorix。",
)
async def memorix_import_current_chat_summary(_ctx: AgentCtx, context_length: int = 50):
    async with runtime_scope() as ctx:
        chat_key = _ctx.from_chat_key or ""
        if not chat_key:
            raise ValueError("missing chat key")
        messages = (
            await DBChatMessage.filter(chat_key=chat_key)
            .order_by("-send_timestamp")
            .limit(max(4, int(context_length)))
            .all()
        )
        payload = [
            {
                "role": "assistant" if str(msg.sender_id or "") == "-1" else "user",
                "content": str(msg.content_text or ""),
            }
            for msg in reversed(messages)
            if str(msg.content_text or "").strip()
        ]
        return await SummaryService(ctx).import_from_transcript(
            session_id=chat_key,
            messages=payload,
            source=f"chat_summary:{chat_key}",
            context_length=max(4, int(context_length)),
        )


@plugin.mount_sandbox_method(SandboxMethodType.TOOL, "记忆状态", description="查看 na_memorix 当前状态统计。")
async def memorix_status(_ctx: AgentCtx):
    del _ctx
    async with runtime_scope() as ctx:
        return await QueryService(ctx).stats()


@plugin.mount_sandbox_method(
    SandboxMethodType.TOOL,
    "memorix_reindex",
    description="Rebuild na_memorix Qdrant vectors from PostgreSQL",
)
async def memorix_reindex(_ctx: AgentCtx, batch_size: int = 32):
    del _ctx
    manager = await ensure_task_manager_started()
    task = await manager.enqueue_reindex_task({"batch_size": max(1, int(batch_size))})
    return {
        "task_id": task.get("task_id"),
        "status": task.get("status"),
        "created_at": task.get("created_at"),
    }


@plugin.mount_init_method()
async def _init_plugin():
    """保持插件初始化轻量且惰性。"""

    global _runtime_sync_task

    if _runtime_sync_task is None or _runtime_sync_task.done():
        _runtime_sync_task = asyncio.create_task(_runtime_sync_loop(), name="na-memorix-runtime-sync")
    return None


@plugin.mount_cleanup_method()
async def _cleanup_plugin():
    global _runtime_handle, _runtime_sync_task

    if _runtime_sync_task is not None:
        _runtime_sync_task.cancel()
        try:
            await _runtime_sync_task
        except asyncio.CancelledError:
            pass
        finally:
            _runtime_sync_task = None

    async with _runtime_lock:
        handle = _runtime_handle
        _runtime_handle = None

    await _retire_runtime_handle(handle)
