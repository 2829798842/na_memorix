"""Person-mode handlers for KnowledgeQueryTool."""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.common.logger import get_logger
from src.plugin_system.apis import person_api

from ...core.utils.person_profile_service import PersonProfileService

logger = get_logger("A_Memorix.QueryModesPerson")

def is_person_profile_injection_enabled(tool, stream_id: Optional[str], user_id: Optional[str]) -> bool:
    if not bool(tool.get_config("person_profile.enabled", True)):
        return False

    opt_in_required = bool(tool.get_config("person_profile.opt_in_required", True))
    default_enabled = bool(tool.get_config("person_profile.default_injection_enabled", False))

    if not opt_in_required:
        return default_enabled

    s_id = str(stream_id or "").strip()
    u_id = str(user_id or "").strip()
    if not s_id or not u_id or tool.metadata_store is None:
        return False
    return bool(tool.metadata_store.get_person_profile_switch(s_id, u_id, default=default_enabled))

async def query_person(
    tool,
    query: str,
    person_id: Optional[str],
    top_k: int,
    for_injection: bool = False,
    force_refresh: bool = False,
    stream_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """查询人物画像。"""
    if not bool(tool.get_config("person_profile.enabled", True)):
        if for_injection:
            return {
                "success": True,
                "query_type": "person",
                "content": "",
                "results": [],
                "disabled_reason": "person_profile_module_disabled",
            }
        return {
            "success": False,
            "query_type": "person",
            "error": "人物画像功能未启用（person_profile.enabled=false）",
            "content": "❌ 人物画像功能未启用（person_profile.enabled=false）",
            "results": [],
        }

    resolved_stream_id = str(stream_id or tool.chat_id or "").strip()
    resolved_user_id = str(user_id or "").strip()
    if not resolved_user_id and tool.chat_stream and getattr(tool.chat_stream, "user_info", None):
        resolved_user_id = str(getattr(tool.chat_stream.user_info, "user_id", "") or "").strip()

    if for_injection and not tool._is_person_profile_injection_enabled(resolved_stream_id, resolved_user_id):
        return {
            "success": True,
            "query_type": "person",
            "content": "",
            "results": [],
            "disabled_reason": "person_profile_not_opted_in",
        }

    service = PersonProfileService(
        metadata_store=tool.metadata_store,
        graph_store=tool.graph_store,
        vector_store=tool.vector_store,
        embedding_manager=tool.embedding_manager,
        sparse_index=tool.sparse_index,
        plugin_config=tool.plugin_config,
        retriever=tool.retriever,
    )

    pid = str(person_id or "").strip()
    if not pid and resolved_user_id and tool.platform:
        try:
            pid = person_api.get_person_id(tool.platform, resolved_user_id)
        except Exception:
            pid = ""
    if not pid and query:
        pid = service.resolve_person_id(str(query))

    if not pid:
        if for_injection:
            return {
                "success": True,
                "query_type": "person",
                "content": "",
                "results": [],
                "disabled_reason": "person_id_unresolved",
            }
        return {
            "success": False,
            "query_type": "person",
            "error": "未能解析 person_id，请提供 person_id 或有效的人名/别名",
            "content": "❌ 未能解析 person_id，请提供 person_id 或有效的人名/别名",
            "results": [],
        }

    ttl_minutes = float(tool.get_config("person_profile.profile_ttl_minutes", 360))
    ttl_seconds = max(60.0, ttl_minutes * 60.0)

    profile = await service.query_person_profile(
        person_id=pid,
        person_keyword=str(query or "").strip(),
        top_k=max(4, top_k),
        ttl_seconds=ttl_seconds,
        force_refresh=bool(force_refresh),
        source_note="knowledge_query:person",
    )

    if not profile.get("success", False):
        if for_injection:
            return {
                "success": True,
                "query_type": "person",
                "content": "",
                "results": [],
                "error": profile.get("error", "unknown"),
            }
        return {
            "success": False,
            "query_type": "person",
            "error": profile.get("error", "unknown"),
            "content": "❌ 人物画像查询失败",
            "results": [],
        }

    if resolved_stream_id and resolved_user_id and tool.metadata_store is not None:
        try:
            tool.metadata_store.mark_person_profile_active(resolved_stream_id, resolved_user_id, pid)
        except Exception as e:
            logger.warning(f"{tool.log_prefix} 记录活跃人物失败: {e}")

    persona_block = PersonProfileService.format_persona_profile_block(profile)
    if not persona_block and not for_injection:
        persona_block = "暂无足够证据形成该人物画像。"

    return {
        "success": True,
        "query_type": "person",
        "person_id": pid,
        "person_name": profile.get("person_name", ""),
        "profile_version": profile.get("profile_version"),
        "updated_at": profile.get("updated_at"),
        "expires_at": profile.get("expires_at"),
        "evidence_ids": profile.get("evidence_ids", []),
        "aliases": profile.get("aliases", []),
        "relation_edges": profile.get("relation_edges", []),
        "vector_evidence": profile.get("vector_evidence", []),
        "profile_source": profile.get("profile_source", "auto_snapshot"),
        "has_manual_override": bool(profile.get("has_manual_override", False)),
        "manual_override_text": profile.get("manual_override_text", ""),
        "auto_profile_text": profile.get("auto_profile_text", profile.get("profile_text", "")),
        "override_updated_at": profile.get("override_updated_at"),
        "override_updated_by": profile.get("override_updated_by", ""),
        "profile_text": profile.get("profile_text", ""),
        "content": persona_block,
        "results": [],
    }

