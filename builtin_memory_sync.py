import json
from typing import Any, Optional

from nekro_agent.models.db_mem_entity import DBMemEntity
from nekro_agent.models.db_mem_paragraph import DBMemParagraph
from nekro_agent.models.db_mem_relation import DBMemRelation

from amemorix.services import ImportService

BUILTIN_MEMORY_SYNC_STATE_PREFIX = "builtin_memory_sync_state:"
AUTO_MEMORY_IMPORT_CHAT_CURSOR_PREFIX = "auto_memory_import_chat_cursor:"
BUILTIN_SOURCE_PREFIX = "builtin_memory:"
CHAT_SUMMARY_SOURCE_PREFIX = "chat_summary:"


def build_builtin_sync_state_key(workspace_id: int) -> str:
    return f"{BUILTIN_MEMORY_SYNC_STATE_PREFIX}{int(workspace_id)}"


def build_chat_import_cursor_key(chat_key: str) -> str:
    return f"{AUTO_MEMORY_IMPORT_CHAT_CURSOR_PREFIX}{str(chat_key or '').strip()}"


def build_builtin_paragraph_source(workspace_id: int, paragraph_id: int) -> str:
    return f"{BUILTIN_SOURCE_PREFIX}{int(workspace_id)}:paragraph:{int(paragraph_id)}"


def build_builtin_relation_source(workspace_id: int, relation_id: int) -> str:
    return f"{BUILTIN_SOURCE_PREFIX}{int(workspace_id)}:relation:{int(relation_id)}"


def parse_builtin_workspace_id(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value or "").strip()
    if not text.startswith(BUILTIN_SOURCE_PREFIX):
        return None
    parts = text.split(":")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except (TypeError, ValueError):
        return None


def parse_chat_summary_chat_key(value: Any) -> str:
    text = str(value or "").strip()
    if not text.startswith(CHAT_SUMMARY_SOURCE_PREFIX):
        return ""
    return text[len(CHAT_SUMMARY_SOURCE_PREFIX) :].strip()


async def get_chat_import_cursor(plugin_store: Any, chat_key: str) -> int:
    raw = await plugin_store.get(store_key=build_chat_import_cursor_key(chat_key))
    if raw is None:
        return 0
    try:
        return max(0, int(str(raw).strip() or "0"))
    except (TypeError, ValueError):  
        return 0


async def set_chat_import_cursor(plugin_store: Any, chat_key: str, message_id: int) -> None:
    await plugin_store.set(
        store_key=build_chat_import_cursor_key(chat_key),
        value=str(max(0, int(message_id))),
    )


def _paragraph_knowledge_type(paragraph: DBMemParagraph) -> str:
    value = getattr(paragraph, "knowledge_type", "")
    if hasattr(value, "value"):
        return str(value.value or "").strip().lower()
    return str(value or "").strip().lower()


def _paragraph_time_meta(paragraph: DBMemParagraph) -> dict[str, Any]:
    if paragraph.event_time is None:
        return {}
    return {"event_time": paragraph.event_time}


def _builtin_paragraph_metadata(paragraph: DBMemParagraph) -> dict[str, Any]:
    source = build_builtin_paragraph_source(paragraph.workspace_id, paragraph.id)
    return {
        "builtin_workspace_id": int(paragraph.workspace_id),
        "builtin_record_type": "paragraph",
        "builtin_record_id": int(paragraph.id),
        "builtin_origin_chat_key": str(paragraph.origin_chat_key or "").strip(),
        "builtin_anchor_msg_id": str(paragraph.anchor_msg_id or "").strip(),
        "record_source": source,
    }


def _builtin_relation_metadata(
    relation: DBMemRelation,
    *,
    source_paragraph_hash: str = "",
) -> dict[str, Any]:
    source = build_builtin_relation_source(relation.workspace_id, relation.id)
    metadata = {
        "builtin_workspace_id": int(relation.workspace_id),
        "builtin_record_type": "relation",
        "builtin_record_id": int(relation.id),
        "record_source": source,
    }
    if source_paragraph_hash:
        metadata["source_paragraph"] = source_paragraph_hash
    return metadata


async def _read_builtin_sync_state(plugin_store: Any, workspace_id: int) -> dict[str, int]:
    raw = await plugin_store.get(store_key=build_builtin_sync_state_key(workspace_id))
    if raw is None:
        return {"last_paragraph_id": 0, "last_relation_id": 0}

    text = str(raw or "").strip()
    if not text:
        return {"last_paragraph_id": 0, "last_relation_id": 0}

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            cursor = max(0, int(text))
        except (TypeError, ValueError):
            return {"last_paragraph_id": 0, "last_relation_id": 0}
        return {"last_paragraph_id": cursor, "last_relation_id": 0}

    if not isinstance(payload, dict):
        return {"last_paragraph_id": 0, "last_relation_id": 0}

    def _coerce(*keys: str) -> int:
        for key in keys:
            if key in payload:
                try:
                    return max(0, int(payload[key] or 0))
                except (TypeError, ValueError):
                    return 0
        return 0

    return {
        "last_paragraph_id": _coerce("last_paragraph_id", "paragraph_id", "last_paragraph_cursor"),
        "last_relation_id": _coerce("last_relation_id", "relation_id", "last_relation_cursor"),
    }


async def _write_builtin_sync_state(
    plugin_store: Any,
    workspace_id: int,
    *,
    last_paragraph_id: int,
    last_relation_id: int,
) -> None:
    await plugin_store.set(
        store_key=build_builtin_sync_state_key(workspace_id),
        value=json.dumps(
            {
                "last_paragraph_id": max(0, int(last_paragraph_id)),
                "last_relation_id": max(0, int(last_relation_id)),
            },
            ensure_ascii=False,
        ),
    )


async def _resolve_imported_paragraph_hash(
    ctx: Any,
    workspace_id: int,
    paragraph_id: int,
    cache: dict[int, str],
) -> str:
    cached = cache.get(int(paragraph_id))
    if cached:
        return cached

    source_name = build_builtin_paragraph_source(workspace_id, paragraph_id)

    def _lookup() -> str:
        paragraphs = ctx.metadata_store.get_paragraphs_by_source(source_name)
        if not paragraphs:
            return ""
        return str(paragraphs[0].get("hash", "") or "")

    resolved = str(await ctx.run_blocking(_lookup) or "")
    if resolved:
        cache[int(paragraph_id)] = resolved
    return resolved


async def sync_builtin_memories(
    *,
    ctx: Any,
    plugin_store: Any,
    workspace_id: int,
    batch_size: int = 64,
    include_relations: bool = True,
    logger: Any | None = None,
) -> dict[str, int]:
    normalized_workspace_id = int(workspace_id)
    normalized_batch_size = max(1, int(batch_size))
    state = await _read_builtin_sync_state(plugin_store, normalized_workspace_id)
    last_paragraph_id = int(state.get("last_paragraph_id", 0) or 0)
    last_relation_id = int(state.get("last_relation_id", 0) or 0)
    import_service = ImportService(ctx)

    imported_paragraphs = 0
    imported_relations = 0
    paragraph_hash_cache: dict[int, str] = {}

    paragraphs = (
        await DBMemParagraph.filter(
            workspace_id=normalized_workspace_id,
            is_inactive=False,
            id__gt=last_paragraph_id,
        )
        .order_by("id")
        .limit(normalized_batch_size)
        .all()
    )

    for paragraph in paragraphs:
        result = await import_service.import_paragraph(
            content=str(paragraph.content or ""),
            source=build_builtin_paragraph_source(normalized_workspace_id, paragraph.id),
            knowledge_type=_paragraph_knowledge_type(paragraph),
            time_meta=_paragraph_time_meta(paragraph),
            metadata=_builtin_paragraph_metadata(paragraph),
        )
        paragraph_hash_cache[int(paragraph.id)] = str(result.get("hash", "") or "")
        last_paragraph_id = max(last_paragraph_id, int(paragraph.id))
        imported_paragraphs += 1

    if include_relations:
        relations = (
            await DBMemRelation.filter(
                workspace_id=normalized_workspace_id,
                is_inactive=False,
                id__gt=last_relation_id,
            )
            .order_by("id")
            .limit(normalized_batch_size)
            .all()
        )
        entity_ids = {
            int(entity_id)
            for relation in relations
            for entity_id in (relation.subject_entity_id, relation.object_entity_id)
            if entity_id is not None
        }
        entity_map = {
            int(item.id): str(item.name or item.canonical_name or "").strip()
            for item in await DBMemEntity.filter(id__in=list(entity_ids)).all()
        }

        for relation in relations:
            subject = entity_map.get(int(relation.subject_entity_id or 0), "").strip()
            obj = entity_map.get(int(relation.object_entity_id or 0), "").strip()
            predicate = str(relation.predicate or "").strip()
            if not (subject and predicate and obj):
                if logger is not None:
                    logger.warning(
                        "跳过原生关系导入，实体信息不完整: workspace=%s relation_id=%s",
                        normalized_workspace_id,
                        relation.id,
                    )
                last_relation_id = max(last_relation_id, int(relation.id))
                continue

            source_paragraph_hash = ""
            if relation.paragraph_id is not None:
                source_paragraph_hash = paragraph_hash_cache.get(int(relation.paragraph_id), "")
                if not source_paragraph_hash:
                    source_paragraph_hash = await _resolve_imported_paragraph_hash(
                        ctx,
                        normalized_workspace_id,
                        int(relation.paragraph_id),
                        paragraph_hash_cache,
                    )

            await import_service.import_relation(
                subject=subject,
                predicate=predicate,
                obj=obj,
                confidence=float(relation.base_weight or 1.0),
                source_paragraph=source_paragraph_hash,
                metadata=_builtin_relation_metadata(
                    relation,
                    source_paragraph_hash=source_paragraph_hash,
                ),
            )
            last_relation_id = max(last_relation_id, int(relation.id))
            imported_relations += 1

    if imported_paragraphs or imported_relations:
        await ctx.save_all()

    await _write_builtin_sync_state(
        plugin_store,
        normalized_workspace_id,
        last_paragraph_id=last_paragraph_id,
        last_relation_id=last_relation_id,
    )

    if logger is not None and (imported_paragraphs or imported_relations):
        logger.info(
            "原生记忆同步完成: workspace=%s paragraphs=%s relations=%s",
            normalized_workspace_id,
            imported_paragraphs,
            imported_relations,
        )

    return {
        "workspace_id": normalized_workspace_id,
        "imported_paragraphs": imported_paragraphs,
        "imported_relations": imported_relations,
        "last_paragraph_id": last_paragraph_id,
        "last_relation_id": last_relation_id,
    }


def _extract_scope_markers_from_item(item: dict[str, Any]) -> tuple[int | None, str, str]:
    metadata = dict(item.get("metadata") or {})

    workspace_raw = metadata.get("builtin_workspace_id")
    workspace_id: int | None = None
    if workspace_raw is not None:
        try:
            workspace_id = int(workspace_raw)
        except (TypeError, ValueError):
            workspace_id = None

    chat_key = str(metadata.get("chat_summary_chat_key", "") or "").strip()
    record_source = str(metadata.get("record_source", "") or "").strip()

    if workspace_id is None and record_source:
        workspace_id = parse_builtin_workspace_id(record_source)
    if not chat_key and record_source:
        chat_key = parse_chat_summary_chat_key(record_source)

    return workspace_id, chat_key, record_source


async def _hydrate_relation_scope_markers(
    ctx: Any,
    item: dict[str, Any],
) -> tuple[int | None, str, str]:
    relation_hash = str(item.get("hash", "") or "").strip()
    if not relation_hash:
        return None, "", ""

    def _lookup() -> tuple[int | None, str, str]:
        relation = ctx.metadata_store.get_relation(relation_hash)
        if relation is None:
            return None, "", ""

        metadata = dict(relation.get("metadata", {}) or {})
        record_source = str(metadata.get("record_source", "") or "").strip()
        workspace_id = metadata.get("builtin_workspace_id")
        chat_key = str(metadata.get("chat_summary_chat_key", "") or "").strip()

        try:
            normalized_workspace_id = int(workspace_id) if workspace_id is not None else None
        except (TypeError, ValueError):
            normalized_workspace_id = None

        if not record_source and relation.get("source_paragraph"):
            paragraph = ctx.metadata_store.get_paragraph(str(relation.get("source_paragraph", "") or ""))
            if paragraph is not None:
                record_source = str(paragraph.get("source", "") or "").strip()
                paragraph_metadata = dict(paragraph.get("metadata", {}) or {})
                if normalized_workspace_id is None:
                    paragraph_workspace = paragraph_metadata.get("builtin_workspace_id")
                    try:
                        normalized_workspace_id = (
                            int(paragraph_workspace) if paragraph_workspace is not None else None
                        )
                    except (TypeError, ValueError):
                        normalized_workspace_id = None
                if not chat_key:
                    chat_key = str(paragraph_metadata.get("chat_summary_chat_key", "") or "").strip()

        if normalized_workspace_id is None and record_source:
            normalized_workspace_id = parse_builtin_workspace_id(record_source)
        if not chat_key and record_source:
            chat_key = parse_chat_summary_chat_key(record_source)

        return normalized_workspace_id, chat_key, record_source

    return await ctx.run_blocking(_lookup)


async def filter_results_for_scope(
    *,
    ctx: Any,
    results: list[dict[str, Any]],
    workspace_id: int | None,
    chat_key: str,
) -> list[dict[str, Any]]:
    normalized_workspace_id = int(workspace_id) if workspace_id is not None else None
    normalized_chat_key = str(chat_key or "").strip()
    filtered: list[dict[str, Any]] = []

    for item in results:
        current = dict(item)
        item_workspace_id, item_chat_key, _record_source = _extract_scope_markers_from_item(current)

        if (item_workspace_id is None and not item_chat_key) and str(current.get("type", "") or "") == "relation":
            item_workspace_id, item_chat_key, hydrated_source = await _hydrate_relation_scope_markers(ctx, current)
            metadata = dict(current.get("metadata") or {})
            if item_workspace_id is not None:
                metadata["builtin_workspace_id"] = item_workspace_id
            if item_chat_key:
                metadata["chat_summary_chat_key"] = item_chat_key
            if hydrated_source:
                metadata["record_source"] = hydrated_source
            current["metadata"] = metadata

        if item_workspace_id is not None:
            if normalized_workspace_id is None or int(item_workspace_id) != normalized_workspace_id:
                continue

        if item_chat_key:
            if not normalized_chat_key or item_chat_key != normalized_chat_key:
                continue

        filtered.append(current)

    return filtered
