"""Orchestrator helpers for KnowledgeQueryTool."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.common.logger import get_logger

from ...core.utils.aggregate_query_service import AggregateQueryService
from ...core.utils.episode_retrieval_service import EpisodeRetrievalService
from ...core.utils.search_execution_service import (
    SearchExecutionRequest,
    SearchExecutionService,
)
from ...core.utils.time_parser import parse_query_time_range
from .query_modes_entity import query_entity
from .query_modes_person import query_person
from .query_modes_relation import query_relation

logger = get_logger("A_Memorix.QueryToolOrchestrator")


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))

    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "on", "y"}:
        return True
    if token in {"0", "false", "no", "off", "n"}:
        return False
    raise ValueError(f"布尔参数格式错误: {value}")

def get_search_owner(tool) -> str:
    owner = str(tool.get_config("routing.search_owner", "action") or "action").strip().lower()
    if owner not in {"action", "tool", "dual"}:
        return "action"
    return owner

def get_tool_search_mode(tool) -> str:
    mode = str(tool.get_config("routing.tool_search_mode", "forward") or "forward").strip().lower()
    if mode not in {"forward", "disabled"}:
        raise ValueError(
            "routing.tool_search_mode 非法，仅允许 forward|disabled。"
            " 请执行 scripts/release_vnext_migrate.py migrate。"
        )
    return mode

def resolve_search_context(
    tool,
    function_args: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    stream_id = function_args.get("stream_id") or tool.chat_id
    group_id = function_args.get("group_id")
    user_id = function_args.get("user_id")

    if group_id is None and tool.chat_stream and getattr(tool.chat_stream, "group_info", None):
        group_id = getattr(tool.chat_stream.group_info, "group_id", None)
    if user_id is None and tool.chat_stream and getattr(tool.chat_stream, "user_info", None):
        user_id = getattr(tool.chat_stream.user_info, "user_id", None)

    stream_id_text = str(stream_id).strip() if stream_id is not None else None
    group_id_text = str(group_id).strip() if group_id is not None else None
    user_id_text = str(user_id).strip() if user_id is not None else None

    return (
        stream_id_text or None,
        group_id_text or None,
        user_id_text or None,
    )

def build_forward_search_content(tool, results: List[Dict[str, Any]]) -> str:
    if not results:
        return "未找到相关结果。"

    summary_lines = [f"找到 {len(results)} 条相关信息："]
    for i, item in enumerate(results[:5], 1):
        result_type = item.get("type", "")
        icon = "📄" if result_type == "paragraph" else "🔗"
        content_text = item.get("content", "N/A")
        summary_lines.append(f"{i}. {icon} {content_text}")
    return "\n".join(summary_lines)

def build_forward_time_content(tool, results: List[Dict[str, Any]]) -> str:
    if not results:
        return "未找到符合时间条件的结果。"

    lines = [f"找到 {len(results)} 条时间相关信息："]
    for i, item in enumerate(results[:5], 1):
        time_meta = item.get("metadata", {}).get("time_meta", {})
        s_text = time_meta.get("effective_start_text", "N/A")
        e_text = time_meta.get("effective_end_text", "N/A")
        basis = time_meta.get("match_basis", "none")
        lines.append(f"{i}. {item.get('content', 'N/A')}")
        lines.append(f"   时间: {s_text} ~ {e_text} ({basis})")
    return "\n".join(lines)

def _build_episode_content(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "未找到匹配的 episode。"
    lines = [f"找到 {len(results)} 条 episode："]
    for idx, item in enumerate(results[:5], 1):
        title = str(item.get("title", "") or "Untitled")
        summary = str(item.get("summary", "") or "").strip()
        if len(summary) > 90:
            summary = summary[:90] + "..."
        lines.append(f"{idx}. 🧠 {title}")
        if summary:
            lines.append(f"   {summary}")
    return "\n".join(lines)

def _serialize_episode_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "episode",
        "episode_id": str(row.get("episode_id", "") or ""),
        "title": str(row.get("title", "") or ""),
        "summary": str(row.get("summary", "") or ""),
        "source": str(row.get("source", "") or ""),
        "time_meta": {
            "event_time_start": row.get("event_time_start"),
            "event_time_end": row.get("event_time_end"),
            "time_granularity": row.get("time_granularity"),
            "time_confidence": row.get("time_confidence"),
        },
        "participants": list(row.get("participants", []) or []),
        "keywords": list(row.get("keywords", []) or []),
        "paragraph_count": int(row.get("paragraph_count") or 0),
        "evidence_ids": list(row.get("evidence_ids", []) or []),
        "llm_confidence": row.get("llm_confidence"),
        "segmentation_model": row.get("segmentation_model"),
        "segmentation_version": row.get("segmentation_version"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }

async def _query_episode(
    tool,
    *,
    query: str,
    top_k: int,
    time_from: Optional[str],
    time_to: Optional[str],
    person: Optional[str],
    source: Optional[str],
    include_paragraphs: bool,
) -> Dict[str, Any]:
    if tool.metadata_store is None:
        return {
            "success": False,
            "query_type": "episode",
            "error": "MetadataStore 未初始化",
            "content": "❌ MetadataStore 未初始化",
            "results": [],
        }
    if not bool(tool.get_config("episode.enabled", True)):
        return {
            "success": False,
            "query_type": "episode",
            "error": "episode.enabled=false",
            "content": "❌ Episode 模块未启用",
            "results": [],
        }
    if not bool(tool.get_config("episode.query_enabled", True)):
        return {
            "success": False,
            "query_type": "episode",
            "error": "episode.query_enabled=false",
            "content": "❌ Episode 查询已禁用",
            "results": [],
        }

    try:
        ts_from, ts_to = parse_query_time_range(time_from, time_to)
    except ValueError as e:
        return {
            "success": False,
            "query_type": "episode",
            "error": f"时间参数错误: {e}",
            "content": f"❌ 时间参数错误: {e}",
            "results": [],
        }

    episode_service = EpisodeRetrievalService(
        metadata_store=tool.metadata_store,
        retriever=getattr(tool, "retriever", None),
    )
    rows = await episode_service.query(
        query=str(query or "").strip(),
        top_k=max(1, int(top_k)),
        time_from=ts_from,
        time_to=ts_to,
        person=str(person).strip() if person else None,
        source=str(source).strip() if source else None,
        include_paragraphs=False,
    )
    results = [_serialize_episode_row(row) for row in rows]
    if include_paragraphs:
        for item in results:
            paragraphs = tool.metadata_store.get_episode_paragraphs(
                episode_id=str(item.get("episode_id") or ""),
                limit=50,
            )
            item["paragraphs"] = paragraphs

    return {
        "success": True,
        "query_type": "episode",
        "query": str(query or "").strip(),
        "results": results,
        "count": len(results),
        "time_from": time_from,
        "time_to": time_to,
        "person": person,
        "source": source,
        "content": _build_episode_content(results),
    }


def _disabled_forward_branch_result(*, query_type: str, route_hint: str) -> Dict[str, Any]:
    return {
        "success": False,
        "query_type": query_type,
        "error": "knowledge_query 的 search/time 已被禁用",
        "content": f"❌ knowledge_query 的 search/time 已被禁用，请改用 {route_hint}",
        "results": [],
        "count": 0,
    }


async def _query_aggregate(
    tool,
    *,
    query: str,
    top_k: int,
    use_threshold: bool,
    mix: bool,
    mix_top_k: Optional[int],
    time_from: Optional[str],
    time_to: Optional[str],
    person: Optional[str],
    source: Optional[str],
    include_paragraphs: bool,
    function_args: Dict[str, Any],
) -> Dict[str, Any]:
    aggregate_service = AggregateQueryService(plugin_config=tool.plugin_config)
    tool_search_mode = tool._get_tool_search_mode()
    search_owner = tool._get_search_owner()

    route_hint = "knowledge_search Action"
    if search_owner == "tool":
        route_hint = "调整 routing.tool_search_mode 为 forward"

    async def _search_runner() -> Dict[str, Any]:
        if tool_search_mode == "disabled":
            return _disabled_forward_branch_result(
                query_type="search",
                route_hint=route_hint,
            )
        return await tool._execute_forward_search_or_time(
            query_type="search",
            query=str(query or "").strip(),
            top_k=top_k,
            use_threshold=bool(use_threshold),
            time_from=str(time_from) if time_from is not None else None,
            time_to=str(time_to) if time_to is not None else None,
            person=str(person) if person is not None else None,
            source=str(source) if source is not None else None,
            function_args=function_args,
        )

    async def _time_runner() -> Dict[str, Any]:
        if tool_search_mode == "disabled":
            return _disabled_forward_branch_result(
                query_type="time",
                route_hint=route_hint,
            )
        return await tool._execute_forward_search_or_time(
            query_type="time",
            query=str(query or "").strip(),
            top_k=top_k,
            use_threshold=bool(use_threshold),
            time_from=str(time_from) if time_from is not None else None,
            time_to=str(time_to) if time_to is not None else None,
            person=str(person) if person is not None else None,
            source=str(source) if source is not None else None,
            function_args=function_args,
        )

    async def _episode_runner() -> Dict[str, Any]:
        return await _query_episode(
            tool,
            query=str(query or "").strip(),
            top_k=top_k,
            time_from=str(time_from) if time_from is not None else None,
            time_to=str(time_to) if time_to is not None else None,
            person=str(person).strip() if person else None,
            source=str(source).strip() if source else None,
            include_paragraphs=include_paragraphs,
        )

    return await aggregate_service.execute(
        query=str(query or "").strip(),
        top_k=top_k,
        mix=bool(mix),
        mix_top_k=mix_top_k,
        time_from=str(time_from) if time_from is not None else None,
        time_to=str(time_to) if time_to is not None else None,
        search_runner=_search_runner,
        time_runner=_time_runner,
        episode_runner=_episode_runner,
    )

async def execute_forward_search_or_time(
    tool,
    *,
    query_type: str,
    query: str,
    top_k: int,
    use_threshold: bool,
    time_from: Optional[str],
    time_to: Optional[str],
    person: Optional[str],
    source: Optional[str],
    function_args: Dict[str, Any],
) -> Dict[str, Any]:
    stream_id, group_id, user_id = tool._resolve_search_context(function_args)
    enable_ppr = bool(tool.get_config("retrieval.enable_ppr", True))

    execution = await SearchExecutionService.execute(
        retriever=tool.retriever,
        threshold_filter=tool.threshold_filter,
        plugin_config=tool.plugin_config,
        request=SearchExecutionRequest(
            caller="tool",
            stream_id=stream_id,
            group_id=group_id,
            user_id=user_id,
            query_type=query_type,
            query=str(query or "").strip(),
            top_k=top_k,
            time_from=str(time_from) if time_from is not None else None,
            time_to=str(time_to) if time_to is not None else None,
            person=str(person).strip() if person else None,
            source=str(source).strip() if source else None,
            use_threshold=bool(use_threshold),
            enable_ppr=enable_ppr,
        ),
        enforce_chat_filter=True,
        reinforce_access=True,
    )

    if not execution.success:
        return {
            "success": False,
            "query_type": query_type,
            "error": execution.error,
            "content": f"❌ {execution.error}",
            "results": [],
        }

    if execution.chat_filtered:
        return {
            "success": True,
            "query_type": query_type,
            "content": "",
            "results": [],
            "count": 0,
            "elapsed_ms": execution.elapsed_ms,
            "chat_filtered": True,
            "dedup_hit": execution.dedup_hit,
        }

    serialized_results = SearchExecutionService.to_serializable_results(execution.results)
    content = (
        tool._build_forward_search_content(serialized_results)
        if query_type == "search"
        else tool._build_forward_time_content(serialized_results)
    )
    result = {
        "success": True,
        "query_type": query_type,
        "query": query,
        "results": serialized_results,
        "count": len(serialized_results),
        "elapsed_ms": execution.elapsed_ms,
        "content": content,
        "dedup_hit": execution.dedup_hit,
    }
    if query_type == "time":
        result.update(
            {
                "time_from": time_from,
                "time_to": time_to,
                "person": person,
                "source": source,
            }
        )
    return result

async def execute_tool(tool, function_args: dict[str, Any]) -> dict[str, Any]:
    """执行工具函数（供LLM调用）

    Args:
        function_args: 工具调用参数
            - query_type: 查询类型
            - query: 查询内容
            - top_k: 返回结果数量
            - use_threshold: 是否使用阈值过滤

    Returns:
        dict: 工具执行结果
    """
    # 解析参数
    query_type = str(function_args.get("query_type", "search") or "search").strip().lower()
    if not tool.retriever and query_type in {"search", "time"}:
        return {
            "success": False,
            "query_type": query_type,
            "error": "知识查询Tool未初始化",
            "content": "❌ 知识查询Tool未初始化",
            "results": [],
        }

    query = function_args.get("query", "")
    top_k_raw = function_args.get("top_k")
    default_top_k = (
        int(tool.get_config("retrieval.temporal.default_top_k", 10))
        if query_type == "time"
        else int(tool.get_config("retrieval.aggregate.default_top_k", 5))
        if query_type == "aggregate"
        else int(tool.get_config("episode.default_top_k", 5)) if query_type == "episode" else 10
    )
    if top_k_raw is None:
        top_k = default_top_k
    else:
        try:
            top_k = max(1, int(top_k_raw))
        except (TypeError, ValueError):
            return {
                "success": False,
                "error": "top_k 必须是整数",
                "content": "❌ top_k 必须是整数",
                "results": [],
            }
    use_threshold_raw = function_args.get("use_threshold", True)
    time_from = function_args.get("time_from")
    time_to = function_args.get("time_to")
    person = function_args.get("person")
    source = function_args.get("source")
    person_id = function_args.get("person_id")
    include_paragraphs_raw = function_args.get("include_paragraphs", False)
    for_injection_raw = function_args.get("for_injection", False)
    force_refresh_raw = function_args.get("force_refresh", False)
    mix_raw = function_args.get("mix", tool.get_config("retrieval.aggregate.default_mix", False))
    mix_top_k_raw = function_args.get("mix_top_k")
    stream_id = function_args.get("stream_id")
    user_id = function_args.get("user_id")

    try:
        use_threshold = _coerce_bool(use_threshold_raw, default=True)
        include_paragraphs = _coerce_bool(include_paragraphs_raw, default=False)
        for_injection = _coerce_bool(for_injection_raw, default=False)
        force_refresh = _coerce_bool(force_refresh_raw, default=False)
        mix = _coerce_bool(mix_raw, default=False)
    except ValueError as e:
        return {
            "success": False,
            "error": str(e),
            "content": f"❌ {e}",
            "results": [],
        }

    mix_top_k: Optional[int] = None
    if mix_top_k_raw is not None:
        try:
            mix_top_k = max(1, int(mix_top_k_raw))
        except (TypeError, ValueError):
            return {
                "success": False,
                "error": "mix_top_k 必须是整数",
                "content": "❌ mix_top_k 必须是整数",
                "results": [],
            }

    logger.info(
        f"{tool.log_prefix} LLM调用: query_type={query_type}, "
        f"query='{query}', top_k={top_k}, time_from={time_from}, time_to={time_to}"
    )

    if tool.debug_enabled:
        logger.info(f"{tool.log_prefix} [DEBUG] 工具完整参数: {function_args}")

    try:
        # 根据查询类型执行
        if query_type in {"search", "time"}:
            tool_search_mode = tool._get_tool_search_mode()
            search_owner = tool._get_search_owner()
            if tool_search_mode == "disabled":
                route_hint = "knowledge_search Action"
                if search_owner == "tool":
                    route_hint = "调整 routing.tool_search_mode 为 forward"
                result = {
                    "success": False,
                    "query_type": query_type,
                    "error": "knowledge_query 的 search/time 已被禁用",
                    "content": f"❌ knowledge_query 的 search/time 已被禁用，请改用 {route_hint}",
                    "results": [],
                }
            else:
                result = await tool._execute_forward_search_or_time(
                    query_type=query_type,
                    query=str(query or "").strip(),
                    top_k=top_k,
                    use_threshold=bool(use_threshold),
                    time_from=str(time_from) if time_from is not None else None,
                    time_to=str(time_to) if time_to is not None else None,
                    person=str(person) if person is not None else None,
                    source=str(source) if source is not None else None,
                    function_args=function_args,
                )
        elif query_type == "entity":
            result = await query_entity(tool, query)
        elif query_type == "relation":
            result = await query_relation(tool, query)
        elif query_type == "person":
            result = await query_person(
                tool,
                query=query,
                person_id=person_id,
                top_k=top_k,
                for_injection=for_injection,
                force_refresh=force_refresh,
                stream_id=stream_id,
                user_id=user_id,
            )
        elif query_type == "episode":
            result = await _query_episode(
                tool,
                query=str(query or "").strip(),
                top_k=top_k,
                time_from=str(time_from) if time_from is not None else None,
                time_to=str(time_to) if time_to is not None else None,
                person=str(person).strip() if person else None,
                source=str(source).strip() if source else None,
                include_paragraphs=include_paragraphs,
            )
        elif query_type == "aggregate":
            result = await _query_aggregate(
                tool,
                query=str(query or "").strip(),
                top_k=top_k,
                use_threshold=bool(use_threshold),
                mix=bool(mix),
                mix_top_k=mix_top_k,
                time_from=str(time_from) if time_from is not None else None,
                time_to=str(time_to) if time_to is not None else None,
                person=str(person).strip() if person else None,
                source=str(source).strip() if source else None,
                include_paragraphs=include_paragraphs,
                function_args=function_args,
            )
        elif query_type == "stats":
            result = tool._get_stats()
        else:
            result = {
                "success": False,
                "error": f"未知的查询类型: {query_type}",
                "content": f"❌ 未知的查询类型: {query_type}",
                "results": [],
            }

        return result

    except Exception as e:
        error_msg = f"查询失败: {str(e)}"
        logger.error(f"{tool.log_prefix} {error_msg}")
        return {
            "success": False,
            "error": error_msg,
            "content": f"❌ 查询发生错误: {error_msg}",
            "results": [],
        }

async def direct_execute_tool(
    tool,
    query_type: str = "search",
    query: str = "",
    top_k: int = 10,
    use_threshold: bool = True,
    mix: bool = False,
    mix_top_k: Optional[int] = None,
    time_from: Optional[str] = None,
    time_to: Optional[str] = None,
    person: Optional[str] = None,
    source: Optional[str] = None,
    person_id: Optional[str] = None,
    include_paragraphs: bool = False,
    for_injection: bool = False,
    force_refresh: bool = False,
    stream_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """直接执行工具函数（供插件调用）

    Args:
        query_type: 查询类型
        query: 查询内容
        top_k: 返回结果数量
        use_threshold: 是否使用阈值过滤

    Returns:
        Dict: 执行结果
    """
    function_args = {
        "query_type": query_type,
        "query": query,
        "top_k": top_k,
        "use_threshold": use_threshold,
        "mix": mix,
        "mix_top_k": mix_top_k,
        "time_from": time_from,
        "time_to": time_to,
        "person": person,
        "source": source,
        "person_id": person_id,
        "include_paragraphs": include_paragraphs,
        "for_injection": for_injection,
        "force_refresh": force_refresh,
        "stream_id": stream_id,
        "user_id": user_id,
    }

    return await execute_tool(tool, function_args)

