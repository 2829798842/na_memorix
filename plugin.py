"""na_memorix 的 Nekro 插件入口。"""

import asyncio
import copy
import datetime
import hashlib
import importlib
import json
import sys
import threading
from contextlib import asynccontextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import Field

from nekro_agent.api import i18n
from nekro_agent.api.plugin import ConfigBase, ExtraField, NekroPlugin, SandboxMethodType
from nekro_agent.core.config import config as app_config
from nekro_agent.models.db_chat_channel import DBChatChannel
from nekro_agent.models.db_chat_message import DBChatMessage
from nekro_agent.schemas.agent_ctx import AgentCtx

from amemorix.common.logging import bind_plugin_logger
from amemorix.services import ImportService, QueryService, SummaryService
from amemorix.settings import AppSettings, DEFAULT_CONFIG, _deep_merge

from .builtin_memory_sync import (
    filter_results_for_scope,
    get_chat_import_cursor,
    set_chat_import_cursor,
    sync_builtin_memories,
)
from .core.utils.runtime_dependencies import get_runtime_dependency_report

sys.modules.setdefault("server", importlib.import_module(f"{__package__}.server"))

plugin = NekroPlugin(
    name="na_memorix",
    module_name="na_memorix",
    description="A_Memorix-style memory graph plugin",
    version="0.2.0",
    author="litroenade",
    url="",
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
bind_plugin_logger(plugin)

_PROJECT_REPO_URL = "https://github.com/litroenade/na_memorix"
_WEB_PANEL_ROOT = f"/plugins/{plugin.key}/"
plugin.url = _PROJECT_REPO_URL
_WEB_PANEL_LINK_STYLE_PRIMARY = (
    "display:inline-flex;align-items:center;justify-content:center;"
    "padding:6px 12px;border-radius:999px;text-decoration:none;"
    "font-weight:700;background:linear-gradient(135deg,#0ea5e9,#22d3ee);"
    "color:#03131d;"
)
_WEB_PANEL_LINK_STYLE_SECONDARY = (
    "display:inline-flex;align-items:center;justify-content:center;"
    "padding:6px 12px;border-radius:999px;text-decoration:none;"
    "font-weight:600;border:1px solid rgba(14,165,233,0.35);"
    "color:#38bdf8;"
)


def _panel_link(label: str, href: str, *, primary: bool = False) -> str:
    """生成配置提示里使用的入口链接按钮。"""

    style = _WEB_PANEL_LINK_STYLE_PRIMARY if primary else _WEB_PANEL_LINK_STYLE_SECONDARY
    return (
        f"<a href='{href}' target='_blank' rel='noopener noreferrer' style='{style}'>"
        f"{label}</a>"
    )


_WEB_PANEL_LINKS_ZH = (
    "界面入口："
    "<div style='display:flex;flex-wrap:wrap;gap:8px;margin-top:8px;'>"
    f"{_panel_link('主面板', _WEB_PANEL_ROOT, primary=True)}"
    f"{_panel_link('导入中心', f'{_WEB_PANEL_ROOT}import')}"
    f"{_panel_link('检索调优', f'{_WEB_PANEL_ROOT}tuning')}"
    f"{_panel_link('项目仓库', _PROJECT_REPO_URL)}"
    "</div>"
)
_WEB_PANEL_LINKS_EN = (
    "Panel entry: "
    "<div style='display:flex;flex-wrap:wrap;gap:8px;margin-top:8px;'>"
    f"{_panel_link('Main Panel', _WEB_PANEL_ROOT, primary=True)}"
    f"{_panel_link('Import Center', f'{_WEB_PANEL_ROOT}import')}"
    f"{_panel_link('Retrieval Tuning', f'{_WEB_PANEL_ROOT}tuning')}"
    f"{_panel_link('Repository', _PROJECT_REPO_URL)}"
    "</div>"
)


def _config_extra(
    *,
    category_zh: str,
    category_en: str,
    title_zh: str,
    title_en: str,
    description_zh: str,
    description_en: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """生成带有 Nekro WebUI 元数据的配置字段扩展定义。"""

    return ExtraField(
        i18n_category=i18n.i18n_text(
            zh_CN=category_zh,
            en_US=category_en,
        ),
        i18n_title=i18n.i18n_text(
            zh_CN=title_zh,
            en_US=title_en,
        ),
        i18n_description=i18n.i18n_text(
            zh_CN=description_zh,
            en_US=description_en,
        ),
        **kwargs,
    ).model_dump()


def _config_field(
    default: Any,
    *,
    category_zh: str,
    category_en: str,
    title_zh: str,
    title_en: str,
    description_zh: str,
    description_en: str,
    extra_kwargs: Optional[dict[str, Any]] = None,
    **field_kwargs: Any,
) -> Any:
    """生成符合 Nekro 配置面板风格的字段定义。"""

    return Field(
        default=default,
        title=title_zh,
        description=description_zh,
        json_schema_extra=_config_extra(
            category_zh=category_zh,
            category_en=category_en,
            title_zh=title_zh,
            title_en=title_en,
            description_zh=description_zh,
            description_en=description_en,
            **(extra_kwargs or {}),
        ),
        **field_kwargs,
    )


@plugin.mount_config()
class NaMemorixConfig(ConfigBase):
    """na_memorix 插件配置面。"""

    GLOBAL_MEMORY_ENABLED: bool = _config_field(
        True,
        category_zh="基础开关",
        category_en="General",
        title_zh="启用全局记忆",
        title_en="Enable Global Memory",
        description_zh="关闭后会暂停自动注入、人物画像刷新与后台维护，但仍可手动导入、检索和查看记忆。",
        description_en="Disables auto injection, profile refresh, and background maintenance while keeping manual import, search, and browsing available.",
    )
    AUTO_INJECT_ENABLED: bool = _config_field(
        True,
        category_zh="自动注入",
        category_en="Auto Injection",
        title_zh="启用自动记忆注入",
        title_en="Enable Auto Memory Injection",
        description_zh="在对话处理中自动检索相关记忆并注入上下文，关闭后只保留手动搜索与导入能力。",
        description_en="Automatically retrieves related memories during conversations and injects them into context. When disabled, only manual search and import remain available.",
    )
    AUTO_INJECT_TOP_K: int = _config_field(
        6,
        category_zh="自动注入",
        category_en="Auto Injection",
        title_zh="自动注入召回数量",
        title_en="Auto Injection Top K",
        description_zh="每次自动注入时，最多保留多少条候选记忆写入上下文。",
        description_en="Maximum number of recalled memory candidates kept for each automatic injection.",
        ge=1,
        le=20,
    )
    AUTO_INJECT_MIN_SCORE: float = _config_field(
        0.32,
        category_zh="自动注入",
        category_en="Auto Injection",
        title_zh="自动注入最低分数",
        title_en="Auto Injection Minimum Score",
        description_zh="自动注入时仅保留相似度不低于该阈值的候选记忆。",
        description_en="Only memory candidates with similarity scores above this threshold will be injected automatically.",
        ge=0.0,
        le=1.0,
    )
    BUILTIN_MEMORY_SYNC_ENABLED: bool = _config_field(
        True,
        category_zh="原生记忆同步",
        category_en="Native Memory Sync",
        title_zh="启用原生记忆同步",
        title_en="Enable Native Memory Sync",
        description_zh="自动把 Nekro Agent 原生记忆表中的新增段落与关系镜像导入到 na_memorix，并在当前工作区内参与检索。",
        description_en="Automatically mirrors newly consolidated Nekro Agent native memories into na_memorix and makes them retrievable inside the current workspace.",
    )
    BUILTIN_MEMORY_SYNC_BATCH_SIZE: int = _config_field(
        64,
        category_zh="原生记忆同步",
        category_en="Native Memory Sync",
        title_zh="原生记忆同步批量大小",
        title_en="Native Memory Sync Batch Size",
        description_zh="每次自动同步时，最多补入多少条原生记忆记录。数值越大，单次补齐越快，但耗时也会增加。",
        description_en="Maximum number of native memory records mirrored per automatic sync run. Higher values catch up faster but increase per-run cost.",
        ge=1,
        le=500,
    )
    BUILTIN_MEMORY_SYNC_INCLUDE_RELATIONS: bool = _config_field(
        True,
        category_zh="原生记忆同步",
        category_en="Native Memory Sync",
        title_zh="同步原生关系图",
        title_en="Mirror Native Relations",
        description_zh="开启后会同步 Nekro 原生记忆中的实体关系，不仅导入段落文本，也导入图关系。",
        description_en="When enabled, mirrors entity relations from Nekro native memory in addition to paragraph text.",
    )
    CHAT_FILTER_ENABLED: bool = _config_field(
        False,
        category_zh="聊天过滤",
        category_en="Chat Filter",
        title_zh="启用聊天过滤",
        title_en="Enable Chat Filtering",
        description_zh="启用后，可通过白名单或黑名单限制插件仅在指定聊天内生效。",
        description_en="When enabled, the plugin can be limited to specific chats through whitelist or blacklist mode.",
    )
    CHAT_FILTER_MODE: Literal["whitelist", "blacklist"] = _config_field(
        "whitelist",
        category_zh="聊天过滤",
        category_en="Chat Filter",
        title_zh="聊天过滤模式",
        title_en="Chat Filter Mode",
        description_zh="白名单模式只允许列表中的聊天使用，黑名单模式则会排除列表中的聊天。",
        description_en="Whitelist mode only allows configured chats, while blacklist mode excludes the configured chats.",
    )
    CHAT_FILTER_CHATS: str = _config_field(
        "",
        category_zh="聊天过滤",
        category_en="Chat Filter",
        title_zh="聊天过滤列表",
        title_en="Chat Filter List",
        description_zh="每行一个 chat key，支持填写 stream:/group:/user: 前缀，用于限定插件在哪些聊天中工作。",
        description_en="One chat key per line. Supports stream:/group:/user: prefixes to control which chats the plugin applies to.",
        extra_kwargs={
            "is_textarea": True,
            "placeholder": "stream:public\ngroup:123456\nuser:admin",
        },
    )
    EMBEDDING_MODEL_GROUP: str = _config_field(
        "text-embedding",
        category_zh="模型配置",
        category_en="Model Configuration",
        title_zh="记忆 Embedding 模型组",
        title_en="Memory Embedding Model Group",
        description_zh="用于向量化段落与关系的 Embedding 模型组，应选择系统内 MODEL_TYPE 为 embedding 的模型组，并与下方维度保持一致。",
        description_en="Embedding model group used to vectorize chunks and relations. It should use a model group with MODEL_TYPE set to embedding and match the dimension below.",
        extra_kwargs={
            "ref_model_groups": True,
            "model_type": "embedding",
            "placeholder": "text-embedding",
        },
    )
    EMBEDDING_DIMENSION: int = _config_field(
        int(getattr(app_config, "MEMORY_EMBEDDING_DIMENSION", 1024) or 1024),
        category_zh="模型配置",
        category_en="Model Configuration",
        title_zh="记忆 Embedding 维度",
        title_en="Memory Embedding Dimension",
        description_zh="向量维度需要与所选 Embedding 模型输出以及既有 Qdrant 索引保持一致。",
        description_en="The vector dimension must match both the selected embedding model output and the existing Qdrant index.",
        ge=1,
    )
    EMBEDDING_TIMEOUT_SECONDS: int = _config_field(
        30,
        category_zh="模型配置",
        category_en="Model Configuration",
        title_zh="Embedding 超时（秒）",
        title_en="Embedding Timeout (s)",
        description_zh="单次 Embedding 请求的超时时间，模型响应较慢时可适当调大。",
        description_en="Timeout for a single embedding request. Increase it if the model responds slowly.",
        ge=5,
        le=600,
    )
    SUMMARIZATION_MODEL_GROUP: str = _config_field(
        "default",
        category_zh="模型配置",
        category_en="Model Configuration",
        title_zh="记忆总结模型组",
        title_en="Memory Summarization Model Group",
        description_zh="用于总结近期对话并写入长期记忆的聊天模型组。",
        description_en="Chat model group used to summarize recent conversations into long-term memory.",
        extra_kwargs={
            "ref_model_groups": True,
            "model_type": "chat",
            "placeholder": "default",
        },
    )
    SUMMARIZATION_CONTEXT_LENGTH: int = _config_field(
        50,
        category_zh="模型配置",
        category_en="Model Configuration",
        title_zh="总结上下文长度",
        title_en="Summarization Context Length",
        description_zh="每次总结时回看最近多少条聊天消息；越大越完整，但模型开销也越高。",
        description_en="How many recent chat messages are included in each summary. Larger values provide more context but increase model cost.",
        ge=4,
        le=200,
    )
    SCHEDULED_SUMMARY_ENABLED: bool = _config_field(
        False,
        category_zh="定时总结",
        category_en="Scheduled Summary",
        title_zh="启用定时批量总结",
        title_en="Enable Scheduled Summary",
        description_zh="按每日固定时间自动检查活跃频道，把新增聊天记录总结写入 na_memorix。会调用总结模型，可能消耗 token，默认关闭。",
        description_en="Checks active chats at configured daily times and summarizes new messages into na_memorix. This calls the summarization model and may consume tokens, so it is disabled by default.",
    )
    SCHEDULED_SUMMARY_TIMES: str = _config_field(
        "04:00",
        category_zh="定时总结",
        category_en="Scheduled Summary",
        title_zh="定时总结时间",
        title_en="Scheduled Summary Times",
        description_zh="每日触发时间，每行一个 HH:MM，例如 04:00。插件每分钟检查一次是否跨过目标时间。",
        description_en="Daily trigger times, one HH:MM value per line, for example 04:00. The plugin checks once per minute whether a target time was crossed.",
        extra_kwargs={
            "is_textarea": True,
            "placeholder": "04:00\n23:30",
        },
    )
    SCHEDULED_SUMMARY_CHAT_LIMIT: int = _config_field(
        20,
        category_zh="定时总结",
        category_en="Scheduled Summary",
        title_zh="单次频道上限",
        title_en="Chat Limit Per Run",
        description_zh="每次定时批量总结最多处理多少个有新增文本消息的活跃频道，用于控制 token 和运行时长。",
        description_en="Maximum number of active chats with new text messages processed in one scheduled run, limiting token cost and runtime.",
        ge=1,
        le=200,
    )
    PERSON_PROFILE_ENABLED: bool = _config_field(
        True,
        category_zh="人物画像",
        category_en="Person Profiles",
        title_zh="启用人物画像",
        title_en="Enable Person Profiles",
        description_zh="启用后会为活跃用户维护人物画像，并在检索与理解上下文时提供辅助信息。",
        description_en="Maintains person profiles for active users and uses them to assist retrieval and context understanding.",
    )
    PERSON_PROFILE_TTL_MINUTES: int = _config_field(
        360,
        category_zh="人物画像",
        category_en="Person Profiles",
        title_zh="人物画像缓存时间（分钟）",
        title_en="Profile Cache TTL (min)",
        description_zh="人物画像缓存的保留时长，超过该时长后会重新生成或刷新。",
        description_en="How long person profiles remain cached before they are regenerated or refreshed.",
        ge=10,
    )
    PERSON_PROFILE_REFRESH_INTERVAL_MINUTES: int = _config_field(
        30,
        category_zh="人物画像",
        category_en="Person Profiles",
        title_zh="人物画像刷新周期（分钟）",
        title_en="Profile Refresh Interval (min)",
        description_zh="后台刷新人物画像的最小间隔，用于避免过于频繁地重复生成。",
        description_en="Minimum interval between background profile refreshes to avoid regenerating profiles too often.",
        ge=1,
    )
    WEB_READ_ONLY: bool = _config_field(
        False,
        category_zh="Web 面板",
        category_en="Web Panel",
        title_zh="Web 面板只读模式",
        title_en="Web Panel Read-only Mode",
        description_zh=(
            "启用后，前端面板仅允许查看记忆数据，禁止写入和修改操作。<br/>"
            f"{_WEB_PANEL_LINKS_ZH}"
        ),
        description_en=(
            "When enabled, the web panel becomes read-only and disallows write or edit operations.<br/>"
            f"{_WEB_PANEL_LINKS_EN}"
        ),
    )
    TOP_K_PARAGRAPHS: int = _config_field(
        20,
        category_zh="检索配置",
        category_en="Retrieval",
        title_zh="段落召回数量",
        title_en="Paragraph Recall Count",
        description_zh="检索阶段先从段落索引中召回的候选数量。",
        description_en="Number of paragraph candidates recalled from the chunk index during retrieval.",
        ge=1,
        le=200,
    )
    TOP_K_RELATIONS: int = _config_field(
        10,
        category_zh="检索配置",
        category_en="Retrieval",
        title_zh="关系召回数量",
        title_en="Relation Recall Count",
        description_zh="检索阶段先从关系索引中召回的候选数量。",
        description_en="Number of relation candidates recalled from the relation index during retrieval.",
        ge=1,
        le=200,
    )
    TOP_K_FINAL: int = _config_field(
        10,
        category_zh="检索配置",
        category_en="Retrieval",
        title_zh="最终结果数量",
        title_en="Final Result Count",
        description_zh="融合排序完成后最终返回给插件和前端的结果条数。",
        description_en="Number of final results returned to the plugin and frontend after fusion ranking.",
        ge=1,
        le=100,
    )
    RETRIEVAL_ALPHA: float = _config_field(
        0.5,
        category_zh="检索配置",
        category_en="Retrieval",
        title_zh="向量/关系融合权重",
        title_en="Vector/Relation Fusion Weight",
        description_zh="控制向量检索与关系检索的融合权重；越接近 1 越偏向向量结果。",
        description_en="Controls the fusion weight between vector retrieval and relation retrieval. Values closer to 1 favor vector results.",
        ge=0.0,
        le=1.0,
    )
    SPARSE_ENABLED: bool = _config_field(
        True,
        category_zh="检索配置",
        category_en="Retrieval",
        title_zh="启用稀疏检索",
        title_en="Enable Sparse Retrieval",
        description_zh="启用后会同时使用稀疏检索辅助召回，通常能改善关键词类查询效果。",
        description_en="Enables sparse retrieval alongside vector retrieval, which usually improves keyword-oriented queries.",
    )
    CHUNK_COLLECTION_NAME: str = _config_field(
        "na_memorix_chunks",
        category_zh="存储配置",
        category_en="Storage",
        title_zh="Qdrant 段落集合名",
        title_en="Qdrant Chunk Collection Name",
        description_zh="Qdrant 中用于存储段落向量的集合名。修改后会切换到新的段落索引。",
        description_en="Name of the Qdrant collection used for chunk vectors. Changing it switches the plugin to a different chunk index.",
        extra_kwargs={"placeholder": "na_memorix_chunks"},
    )
    RELATION_COLLECTION_NAME: str = _config_field(
        "na_memorix_relations",
        category_zh="存储配置",
        category_en="Storage",
        title_zh="Qdrant 关系集合名",
        title_en="Qdrant Relation Collection Name",
        description_zh="Qdrant 中用于存储关系向量的集合名。修改后会切换到新的关系索引。",
        description_en="Name of the Qdrant collection used for relation vectors. Changing it switches the plugin to a different relation index.",
        extra_kwargs={"placeholder": "na_memorix_relations"},
    )
    TABLE_PREFIX: str = _config_field(
        "na_memorix",
        category_zh="存储配置",
        category_en="Storage",
        title_zh="PostgreSQL 表前缀",
        title_en="PostgreSQL Table Prefix",
        description_zh="PostgreSQL 存储表前缀。变更后会使用另一套数据表，通常只建议在初始化阶段调整。",
        description_en="Prefix used for PostgreSQL tables. Changing it switches the plugin to a different set of tables and is usually only recommended during initial setup.",
        extra_kwargs={"placeholder": "na_memorix"},
    )
    AUTO_SAVE_INTERVAL_MINUTES: int = _config_field(
        5,
        category_zh="存储配置",
        category_en="Storage",
        title_zh="自动保存周期（分钟）",
        title_en="Auto Save Interval (min)",
        description_zh="后台自动保存运行时状态与缓存的周期，单位为分钟。",
        description_en="Interval in minutes for automatically saving runtime state and caches in the background.",
        ge=1,
    )


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
    "schedule",
    "web",
)


_runtime_handle: Optional[RuntimeHandle] = None
_runtime_lock = asyncio.Lock()
_bound_runtime_context: ContextVar[Optional[Any]] = ContextVar("na_memorix_bound_runtime_context", default=None)
_runtime_sync_task: Optional[asyncio.Task[Any]] = None
_scheduled_summary_task: Optional[asyncio.Task[Any]] = None
_runtime_tuning_overlay_lock = threading.RLock()
_runtime_tuning_overlay: dict[str, Any] = {}
_runtime_tuning_overlay_history: list[dict[str, Any]] = []
_RUNTIME_SYNC_INTERVAL_SECONDS = 2.0
_RUNTIME_FREE_COMPAT_PATHS = frozenset({"/api/config", "/api/ui_capabilities"})


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


def _parse_schedule_times_config(raw_value: Any) -> list[str]:
    """解析每日定时总结时间，保留合法 HH:MM 项。"""

    text = str(raw_value or "")
    times: list[str] = []
    seen: set[str] = set()
    for line in text.replace(",", "\n").splitlines():
        item = line.strip()
        if not item:
            continue
        try:
            hour_text, minute_text = item.split(":", 1)
            hour = int(hour_text)
            minute = int(minute_text)
        except (TypeError, ValueError):
            plugin.logger.warning("忽略非法 na_memorix 定时总结时间: %s", item)
            continue
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            plugin.logger.warning("忽略越界 na_memorix 定时总结时间: %s", item)
            continue
        normalized = f"{hour:02d}:{minute:02d}"
        if normalized in seen:
            continue
        seen.add(normalized)
        times.append(normalized)
    return times or ["04:00"]


def _parse_schedule_time_item(value: str) -> tuple[int, int] | None:
    try:
        hour_text, minute_text = str(value or "").strip().split(":", 1)
        hour = int(hour_text)
        minute = int(minute_text)
    except (TypeError, ValueError):
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour, minute


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
            # 插件模式从 Nekro 模型组解析 endpoint，避免继承 standalone OpenAI 默认端点。
            "openai": None,
            "openapi": None,
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
        "schedule": {
            "enabled": bool(config_obj.SCHEDULED_SUMMARY_ENABLED),
            "import_times": _parse_schedule_times_config(config_obj.SCHEDULED_SUMMARY_TIMES),
            "chat_limit": int(config_obj.SCHEDULED_SUMMARY_CHAT_LIMIT),
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


def get_runtime_tuning_overlay() -> dict[str, Any]:
    """返回当前运行时调优覆盖层。"""

    with _runtime_tuning_overlay_lock:
        return copy.deepcopy(_runtime_tuning_overlay)


def apply_runtime_tuning_profile_patch(patch: dict[str, Any], *, push_history: bool = True) -> dict[str, Any]:
    """把调优 patch 叠加到当前运行时覆盖层。"""

    normalized_patch = copy.deepcopy(patch or {})
    with _runtime_tuning_overlay_lock:
        global _runtime_tuning_overlay
        if push_history:
            _runtime_tuning_overlay_history.append(copy.deepcopy(_runtime_tuning_overlay))
            if len(_runtime_tuning_overlay_history) > 20:
                _runtime_tuning_overlay_history.pop(0)
        _runtime_tuning_overlay = _deep_merge(_runtime_tuning_overlay, normalized_patch)
        return copy.deepcopy(_runtime_tuning_overlay)


def rollback_runtime_tuning_profile() -> Optional[dict[str, Any]]:
    """回滚到上一份运行时调优覆盖层。"""

    with _runtime_tuning_overlay_lock:
        global _runtime_tuning_overlay
        if not _runtime_tuning_overlay_history:
            return None
        _runtime_tuning_overlay = _runtime_tuning_overlay_history.pop()
        return copy.deepcopy(_runtime_tuning_overlay)


def build_settings() -> AppSettings:
    """根据插件配置面构建 runtime 的最终配置。"""

    settings_config = _build_settings_dict(plugin.get_config(NaMemorixConfig))
    runtime_overlay = get_runtime_tuning_overlay()
    if runtime_overlay:
        settings_config = _deep_merge(settings_config, runtime_overlay)
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


def _extract_compat_api_path(path: str) -> Optional[str]:
    """从挂载后的请求路径中提取兼容层 `/api` 子路径。"""

    normalized = str(path or "").strip()
    if not normalized:
        return None

    marker = "/api"
    marker_index = normalized.find(marker)
    if marker_index < 0:
        return None

    compat_path = normalized[marker_index:]
    if compat_path == marker or compat_path.startswith(f"{marker}/"):
        return compat_path
    return None


def _build_runtime_boot_error_detail(exc: Exception) -> str:
    """将 runtime 初始化异常转换为面向 Web 的可读错误。"""

    dependency_report = get_runtime_dependency_report()
    if not dependency_report.get("ready", True):
        detail = str(dependency_report.get("detail") or "").strip()
        if detail:
            return detail
    text = str(exc).strip()
    return text or "na_memorix 运行时初始化失败"


async def _compat_runtime_dependency(request: Request) -> AsyncIterator[None]:
    """为兼容层 `/api/*` 路由施加请求级 runtime 绑定。"""

    compat_path = _extract_compat_api_path(str(request.url.path or ""))
    if compat_path is None or compat_path in _RUNTIME_FREE_COMPAT_PATHS:
        yield
        return

    try:
        async with runtime_scope():
            yield
    except (ImportError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=_build_runtime_boot_error_detail(exc)) from exc


async def _v1_runtime_dependency() -> AsyncIterator[None]:
    """为 `/v1/*` 路由施加请求级 runtime 绑定。"""

    try:
        async with runtime_scope():
            yield
    except (ImportError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=_build_runtime_boot_error_detail(exc)) from exc


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

    def get_plugin_data_dir(self) -> Path:
        return plugin.get_plugin_data_dir()

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

_PLUGIN_PACKAGE_DIR = Path(__file__).resolve().parent


def _module_belongs_to_plugin_dir(module: Any) -> bool:
    """判断模块对象是否来自当前插件目录。"""

    file_path = getattr(module, "__file__", None)
    if not file_path:
        return False
    try:
        return Path(file_path).resolve().is_relative_to(_PLUGIN_PACKAGE_DIR)
    except Exception:
        return False


def _purge_plugin_module_cache() -> None:
    """主动清理插件模块缓存，确保宿主热重载能吃到新代码。"""

    package_prefix = f"{__package__}."
    stale_modules = [name for name in list(sys.modules) if name == __package__ or name.startswith(package_prefix)]

    for alias_prefix in ("amemorix", "core"):
        for name, module in list(sys.modules.items()):
            if (name == alias_prefix or name.startswith(f"{alias_prefix}.")) and _module_belongs_to_plugin_dir(module):
                stale_modules.append(name)

    for name in sorted(set(stale_modules), reverse=True):
        sys.modules.pop(name, None)


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


def _scheduled_summary_time_crossed(
    *,
    last_check: datetime.datetime,
    now: datetime.datetime,
    time_text: str,
) -> bool:
    parsed = _parse_schedule_time_item(time_text)
    if parsed is None:
        return False
    hour, minute = parsed
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return last_check < target <= now


def _channel_filter_ids(channel: DBChatChannel) -> tuple[str | None, str | None]:
    channel_type = str(channel.channel_type or "").strip().lower()
    channel_id = str(channel.channel_id or "").strip()
    group_id = channel_id if channel_type == "group" else None
    user_id = channel_id if channel_type in {"private", "c2c"} else None
    return group_id, user_id


async def _run_scheduled_summary_once(trigger_time: str) -> dict[str, int]:
    """按活跃频道批量导入新增聊天总结。"""

    cfg = plugin.get_config(NaMemorixConfig)
    if not bool(cfg.GLOBAL_MEMORY_ENABLED) or not bool(cfg.SCHEDULED_SUMMARY_ENABLED):
        return {"success": 0, "failed": 0, "skipped": 0}

    context_length = max(4, min(200, int(cfg.SUMMARIZATION_CONTEXT_LENGTH)))
    chat_limit = max(1, min(200, int(cfg.SCHEDULED_SUMMARY_CHAT_LIMIT)))
    success_count = 0
    failed_count = 0
    skipped_count = 0
    seen_chat_keys: set[str] = set()

    async with runtime_scope() as ctx:
        service = SummaryService(ctx)
        channels = await DBChatChannel.filter(is_active=True).order_by("-update_time", "id").all()
        for channel in channels:
            if success_count >= chat_limit:
                break

            chat_key = str(channel.chat_key or "").strip()
            if not chat_key or chat_key in seen_chat_keys:
                continue
            seen_chat_keys.add(chat_key)

            group_id, user_id = _channel_filter_ids(channel)
            if not ctx.is_chat_enabled(stream_id=chat_key, group_id=group_id, user_id=user_id):
                skipped_count += 1
                continue

            cursor = await get_chat_import_cursor(plugin.store, chat_key)
            rows = (
                await DBChatMessage.filter(chat_key=chat_key, id__gt=cursor)
                .exclude(content_text="")
                .order_by("id")
                .limit(context_length)
                .all()
            )
            messages = [
                {
                    "message_id": int(msg.id),
                    "role": "assistant" if str(msg.sender_id or "") == "-1" else "user",
                    "content": str(msg.content_text or "").strip(),
                }
                for msg in rows
                if str(msg.content_text or "").strip()
            ]
            if len(messages) < 4:
                skipped_count += 1
                continue

            last_message_id = int(messages[-1]["message_id"])
            try:
                result = await service.import_from_transcript(
                    session_id=chat_key,
                    messages=messages,
                    source=f"chat_summary:{chat_key}",
                    context_length=context_length,
                )
                if bool(result.get("success")):
                    await set_chat_import_cursor(plugin.store, chat_key, last_message_id)
                    success_count += 1
                else:
                    failed_count += 1
                    plugin.logger.warning(
                        "na_memorix scheduled summary failed: chat_key=%s message=%s",
                        chat_key,
                        result.get("message"),
                    )
            except Exception:
                failed_count += 1
                plugin.logger.exception("na_memorix scheduled summary crashed: chat_key=%s", chat_key)

    plugin.logger.info(
        "na_memorix scheduled summary finished: trigger=%s success=%s failed=%s skipped=%s",
        trigger_time,
        success_count,
        failed_count,
        skipped_count,
    )
    return {"success": success_count, "failed": failed_count, "skipped": skipped_count}


async def _scheduled_summary_loop() -> None:
    """原版 A_Memorix 风格的每日定时批量总结循环。"""

    last_check = datetime.datetime.now()
    while True:
        try:
            await asyncio.sleep(60)
            cfg = plugin.get_config(NaMemorixConfig)
            if not bool(cfg.GLOBAL_MEMORY_ENABLED) or not bool(cfg.SCHEDULED_SUMMARY_ENABLED):
                last_check = datetime.datetime.now()
                continue

            now = datetime.datetime.now()
            for time_text in _parse_schedule_times_config(cfg.SCHEDULED_SUMMARY_TIMES):
                if _scheduled_summary_time_crossed(last_check=last_check, now=now, time_text=time_text):
                    plugin.logger.info("na_memorix scheduled summary triggered: %s", time_text)
                    await _run_scheduled_summary_once(time_text)
            last_check = now
        except asyncio.CancelledError:
            break
        except Exception:
            plugin.logger.exception("na_memorix scheduled summary loop failed")
            last_check = datetime.datetime.now()


async def _get_workspace_id_for_ctx(_ctx: AgentCtx) -> int | None:
    chat_key = str(_ctx.from_chat_key or "").strip()
    if not chat_key:
        return None
    channel = await DBChatChannel.get_or_none(chat_key=chat_key)
    if channel is None or channel.workspace_id is None:
        return None
    return int(channel.workspace_id)


async def _sync_builtin_memory_for_ctx(_ctx: AgentCtx, ctx: Any, cfg: NaMemorixConfig) -> int | None:
    workspace_id = await _get_workspace_id_for_ctx(_ctx)
    if workspace_id is None or not bool(cfg.BUILTIN_MEMORY_SYNC_ENABLED):
        return workspace_id
    await sync_builtin_memories(
        ctx=ctx,
        plugin_store=plugin.store,
        workspace_id=workspace_id,
        batch_size=int(cfg.BUILTIN_MEMORY_SYNC_BATCH_SIZE),
        include_relations=bool(cfg.BUILTIN_MEMORY_SYNC_INCLUDE_RELATIONS),
        logger=plugin.logger,
    )
    return workspace_id


async def _filter_results_for_ctx(
    _ctx: AgentCtx,
    ctx: Any,
    results: list[dict[str, Any]],
    workspace_id: int | None,
) -> list[dict[str, Any]]:
    return await filter_results_for_scope(
        ctx=ctx,
        results=list(results or []),
        workspace_id=workspace_id,
        chat_key=str(_ctx.from_chat_key or "").strip(),
    )


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

        workspace_id = await _sync_builtin_memory_for_ctx(_ctx, ctx, cfg)

        query_text = await _build_recent_query_text(_ctx)
        if not query_text:
            return ""

        service = QueryService(ctx)
        result = await service.search(query=query_text, top_k=int(cfg.AUTO_INJECT_TOP_K))
        filtered_results = await _filter_results_for_ctx(
            _ctx,
            ctx,
            list(result.get("results", [])),
            workspace_id,
        )
        items = [
            item
            for item in filtered_results
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
    async with runtime_scope() as ctx:
        cfg = plugin.get_config(NaMemorixConfig)
        workspace_id = await _sync_builtin_memory_for_ctx(_ctx, ctx, cfg)
        result = await QueryService(ctx).search(query=query, top_k=top_k)
        result["results"] = await _filter_results_for_ctx(
            _ctx,
            ctx,
            list(result.get("results", [])),
            workspace_id,
        )
        result["count"] = len(result["results"])
        return result


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

    global _runtime_sync_task, _scheduled_summary_task

    if _runtime_sync_task is None or _runtime_sync_task.done():
        _runtime_sync_task = asyncio.create_task(_runtime_sync_loop(), name="na-memorix-runtime-sync")
    if _scheduled_summary_task is None or _scheduled_summary_task.done():
        _scheduled_summary_task = asyncio.create_task(
            _scheduled_summary_loop(),
            name="na-memorix-scheduled-summary",
        )
    return None


@plugin.mount_cleanup_method()
async def _cleanup_plugin():
    global _runtime_handle, _runtime_sync_task, _scheduled_summary_task

    if _runtime_sync_task is not None:
        _runtime_sync_task.cancel()
        try:
            await _runtime_sync_task
        except asyncio.CancelledError:
            pass
        finally:
            _runtime_sync_task = None

    if _scheduled_summary_task is not None:
        _scheduled_summary_task.cancel()
        try:
            await _scheduled_summary_task
        except asyncio.CancelledError:
            pass
        finally:
            _scheduled_summary_task = None

    async with _runtime_lock:
        handle = _runtime_handle
        _runtime_handle = None

    await _retire_runtime_handle(handle)
    _purge_plugin_module_cache()
