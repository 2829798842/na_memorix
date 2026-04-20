"""宿主导入中心桥接后端实现。"""

import asyncio
import copy
import json
import time
import uuid
from pathlib import Path
from typing import Any

from nekro_agent.models.db_chat_channel import DBChatChannel
from nekro_agent.models.db_chat_message import DBChatMessage

from amemorix.services import SummaryService
from amemorix.common.logging import get_logger
from core.utils.hash import compute_paragraph_hash
from core.utils.time_parser import normalize_time_meta

from .builtin_memory_sync import (
    get_chat_import_cursor,
    set_chat_import_cursor,
    sync_builtin_memories,
)

logger = get_logger("A_Memorix.ImportBackend")

TASK_TYPE_IMPORT_BACKEND = "import_backend"
DEFAULT_IMPORT_POLL_INTERVAL_MS = 1000
DEFAULT_MAX_FILE_CONCURRENCY = 6
DEFAULT_MAX_CHUNK_CONCURRENCY = 12
DEFAULT_FILE_CONCURRENCY = 2
DEFAULT_CHUNK_CONCURRENCY = 4
AUTO_MIGRATE_MIN_MESSAGES = 4
AUTO_MIGRATE_DEFAULT_CHAT_LIMIT = 20
AUTO_MIGRATE_DEFAULT_MESSAGE_WINDOW = 50
SUPPORTED_TEXT_SUFFIXES = {".txt", ".md"}
SUPPORTED_JSON_SUFFIXES = {".json"}
SUPPORTED_IMPORT_SUFFIXES = SUPPORTED_TEXT_SUFFIXES | SUPPORTED_JSON_SUFFIXES
TIME_META_KEYS = (
    "event_time",
    "event_time_start",
    "event_time_end",
    "time_range",
    "time_granularity",
    "time_confidence",
)


def resolve_local_plugin_data_dir(module_file: str | Path) -> Path:
    """根据当前插件模块文件路径定位插件数据目录。"""

    return Path(module_file).resolve().parent


def resolve_path_aliases(plugin_data_dir: Path, workdir_dir: Path) -> dict[str, Path]:
    """生成导入页使用的路径别名。"""

    plugin_root = plugin_data_dir.resolve()
    runtime_dir = (plugin_root / "runtime").resolve()

    def _pick_existing(*candidates: Path) -> Path:
        for item in candidates:
            if item.exists():
                return item.resolve()
        return candidates[0].resolve()

    return {
        "plugin_data": plugin_root,
        "runtime": runtime_dir,
        "raw": _pick_existing(plugin_root / "raw", runtime_dir / "raw", plugin_root),
        "workdir": workdir_dir.resolve(),
    }


def resolve_alias_path(alias_map: dict[str, Path], alias: str, relative_path: str = "") -> Path:
    """解析路径别名并保证访问不越界。"""

    normalized_alias = str(alias or "").strip()
    if not normalized_alias:
        raise ValueError("路径别名不能为空")
    base_dir = alias_map.get(normalized_alias)
    if base_dir is None:
        raise ValueError(f"未知路径别名: {normalized_alias}")
    base_dir = base_dir.resolve()
    relative = str(relative_path or "").strip()
    if not relative:
        return base_dir
    target = (base_dir / relative).resolve()
    try:
        target.relative_to(base_dir)
    except ValueError as exc:
        raise ValueError("相对路径越界，禁止访问别名目录之外的路径") from exc
    return target


def discover_candidate_files(
    base_path: Path,
    *,
    pattern: str = "*",
    recursive: bool = True,
    suffixes: set[str] | None = None,
) -> list[Path]:
    """按 glob 规则发现可导入文件。"""

    normalized_base = base_path.resolve()
    allowed_suffixes = {item.lower() for item in (suffixes or SUPPORTED_IMPORT_SUFFIXES)}
    if not normalized_base.exists():
        raise FileNotFoundError(f"路径不存在: {normalized_base}")

    if normalized_base.is_file():
        return [normalized_base] if normalized_base.suffix.lower() in allowed_suffixes else []

    iterator = normalized_base.rglob(pattern) if recursive else normalized_base.glob(pattern)
    files = [
        item.resolve()
        for item in iterator
        if item.is_file() and item.suffix.lower() in allowed_suffixes
    ]
    files.sort(key=lambda item: str(item))
    return files

def _preview_text(text: str, limit: int = 120) -> str:
    normalized = " ".join(str(text or "").strip().split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit].rstrip()}..."


def _split_text(text: str, max_length: int = 500) -> list[str]:
    paragraphs = str(text or "").split("\n\n")
    out: list[str] = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_length:
            out.append(paragraph)
            continue

        sentences = []
        current = ""
        for char in paragraph:
            current += char
            if char in "。！？.!?":
                sentences.append(current.strip())
                current = ""
        if current.strip():
            sentences.append(current.strip())

        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += sentence
            else:
                if current_chunk:
                    out.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk:
            out.append(current_chunk.strip())
    return out


def _serialize_time_meta(raw_value: Any) -> dict[str, Any]:
    normalized = normalize_time_meta(raw_value or {})
    return {
        key: value
        for key, value in normalized.items()
        if value is not None
    }


def _build_chunk_state(
    *,
    index: int,
    chunk_type: str,
    content_preview: str,
    plan: dict[str, Any],
) -> dict[str, Any]:
    return {
        "index": index,
        "chunk_type": chunk_type,
        "status": "queued",
        "step": "queued",
        "content_preview": content_preview,
        "error": "",
        "_plan": copy.deepcopy(plan),
    }


def _build_text_chunks(
    text: str,
    *,
    source_name: str,
    strategy_override: str = "",
    time_meta: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    paragraphs = _split_text(text)
    if not paragraphs:
        raise ValueError("文本内容为空")
    chunk_type = str(strategy_override or "text").strip() or "text"
    serialized_time_meta = _serialize_time_meta(time_meta or {})
    chunks: list[dict[str, Any]] = []
    for index, paragraph in enumerate(paragraphs):
        chunks.append(
            _build_chunk_state(
                index=index,
                chunk_type=chunk_type,
                content_preview=_preview_text(paragraph),
                plan={
                    "kind": "paragraph",
                    "content": paragraph,
                    "source": source_name,
                    "knowledge_type": chunk_type if chunk_type in {"narrative", "factual", "quote"} else "",
                    "time_meta": serialized_time_meta,
                },
            )
        )
    return chunks


def _is_relation_payload(raw_value: Any) -> bool:
    if not isinstance(raw_value, dict):
        return False
    subject = str(raw_value.get("subject", raw_value.get("head", ""))).strip()
    predicate = str(raw_value.get("predicate", raw_value.get("relation", ""))).strip()
    obj = str(raw_value.get("object", raw_value.get("tail", ""))).strip()
    return bool(subject and predicate and obj)


def _relation_plan_from_payload(raw_value: dict[str, Any], *, source_name: str) -> dict[str, Any]:
    subject = str(raw_value.get("subject", raw_value.get("head", ""))).strip()
    predicate = str(raw_value.get("predicate", raw_value.get("relation", ""))).strip()
    obj = str(raw_value.get("object", raw_value.get("tail", ""))).strip()
    if not (subject and predicate and obj):
        raise ValueError("关系导入缺少 subject/predicate/object")
    confidence = raw_value.get("confidence", raw_value.get("score", 1.0))
    source_paragraph = str(raw_value.get("source_paragraph", "")).strip()
    return {
        "kind": "relation",
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "confidence": float(confidence or 1.0),
        "source_paragraph": source_paragraph,
        "source": source_name,
    }


def _paragraph_plan_from_payload(raw_value: Any, *, source_name: str) -> dict[str, Any]:
    if isinstance(raw_value, str):
        content = raw_value
        payload = {}
    elif isinstance(raw_value, dict):
        payload = raw_value
        content = str(
            payload.get("content", payload.get("text", payload.get("paragraph", "")))
        )
    else:
        raise ValueError("段落导入仅支持字符串或对象")

    normalized_content = str(content or "").strip()
    if not normalized_content:
        raise ValueError("段落内容为空")
    time_meta = payload.get("time_meta")
    if not isinstance(time_meta, dict):
        time_meta = {
            key: payload.get(key)
            for key in TIME_META_KEYS
            if payload.get(key) is not None
        }
    return {
        "kind": "paragraph",
        "content": normalized_content,
        "source": str(payload.get("source", source_name) or source_name),
        "knowledge_type": str(payload.get("knowledge_type", "")).strip(),
        "time_meta": _serialize_time_meta(time_meta),
    }


def extract_json_import_chunks(payload: Any, *, source_name: str) -> tuple[list[dict[str, Any]], str]:
    """从 JSON 负载中提取导入计划。"""

    chunks: list[dict[str, Any]] = []
    index = 0

    def _append_paragraph(raw_value: Any) -> None:
        nonlocal index
        plan = _paragraph_plan_from_payload(raw_value, source_name=source_name)
        chunks.append(
            _build_chunk_state(
                index=index,
                chunk_type="json",
                content_preview=_preview_text(str(plan["content"])),
                plan=plan,
            )
        )
        index += 1

    def _append_relation(raw_value: dict[str, Any], *, source_paragraph_chunk_index: int | None = None) -> None:
        nonlocal index
        plan = _relation_plan_from_payload(raw_value, source_name=source_name)
        if source_paragraph_chunk_index is not None:
            plan["source_paragraph_chunk_index"] = source_paragraph_chunk_index
        preview = f"{plan['subject']} | {plan['predicate']} | {plan['object']}"
        chunks.append(
            _build_chunk_state(
                index=index,
                chunk_type="json",
                content_preview=_preview_text(preview),
                plan=plan,
            )
        )
        index += 1

    if isinstance(payload, dict) and (
        isinstance(payload.get("paragraphs"), list) or isinstance(payload.get("relations"), list)
    ):
        for item in payload.get("paragraphs") or []:
            _append_paragraph(item)
        for item in payload.get("relations") or []:
            if not isinstance(item, dict):
                raise ValueError("relations 列表中的元素必须是对象")
            _append_relation(item)
        if not chunks:
            raise ValueError("JSON 导入缺少可写入的 paragraphs/relations")
        return chunks, "web_json"

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, (str, dict)) and not _is_relation_payload(item):
                if isinstance(item, dict) and any(
                    isinstance(item.get(key), list)
                    for key in ("relations", "triples", "spo_list", "openie")
                ):
                    nested_chunks = extract_openie_import_chunks(item, source_name=source_name)
                    index_mapping: dict[int, int] = {}
                    for nested_chunk in nested_chunks:
                        old_index = int(nested_chunk.get("index", 0) or 0)
                        index_mapping[old_index] = index
                        nested_chunk["index"] = index
                        nested_plan = dict(nested_chunk.get("_plan") or {})
                        source_index = nested_plan.get("source_paragraph_chunk_index")
                        if source_index is not None:
                            nested_plan["source_paragraph_chunk_index"] = index_mapping.get(
                                int(source_index or 0),
                                int(source_index or 0),
                            )
                            nested_chunk["_plan"] = nested_plan
                        chunks.append(nested_chunk)
                        index += 1
                    continue
                _append_paragraph(item)
                continue
            if isinstance(item, dict) and _is_relation_payload(item):
                _append_relation(item)
                continue
            raise ValueError("JSON 数组导入仅支持段落对象、段落字符串或关系对象")
        if not chunks:
            raise ValueError("JSON 数组中没有可导入的内容")
        return chunks, "script_json"

    if isinstance(payload, dict):
        if _is_relation_payload(payload):
            _append_relation(payload)
            return chunks, "script_json"
        if any(
            isinstance(payload.get(key), list)
            for key in ("relations", "triples", "spo_list", "openie")
        ):
            chunks = extract_openie_import_chunks(payload, source_name=source_name)
            return chunks, "script_json"
        _append_paragraph(payload)
        return chunks, "script_json"

    raise ValueError("JSON 导入仅支持对象或数组")


def extract_openie_import_chunks(payload: Any, *, source_name: str) -> list[dict[str, Any]]:
    """从 OpenIE 风格 JSON 中提取导入计划。"""

    chunks: list[dict[str, Any]] = []
    index = 0
    queue: list[Any] = [payload]

    while queue:
        current = queue.pop(0)
        if isinstance(current, list):
            queue.extend(current)
            continue
        if not isinstance(current, dict):
            continue

        if _is_relation_payload(current):
            plan = _relation_plan_from_payload(current, source_name=source_name)
            preview = f"{plan['subject']} | {plan['predicate']} | {plan['object']}"
            chunks.append(
                _build_chunk_state(
                    index=index,
                    chunk_type="json",
                    content_preview=_preview_text(preview),
                    plan=plan,
                )
            )
            index += 1
            continue

        text_value = str(
            current.get("content", current.get("text", current.get("paragraph", current.get("sentence", ""))))
        ).strip()
        relations_value = current.get("relations")
        if not isinstance(relations_value, list):
            relations_value = current.get("triples")
        if not isinstance(relations_value, list):
            relations_value = current.get("spo_list")
        if not isinstance(relations_value, list):
            relations_value = current.get("openie")

        paragraph_chunk_index: int | None = None
        if text_value:
            paragraph_plan = _paragraph_plan_from_payload(
                {
                    "content": text_value,
                    "source": current.get("source", source_name) or source_name,
                    "time_meta": current.get("time_meta"),
                    "knowledge_type": current.get("knowledge_type", "factual"),
                },
                source_name=source_name,
            )
            chunks.append(
                _build_chunk_state(
                    index=index,
                    chunk_type="json",
                    content_preview=_preview_text(text_value),
                    plan=paragraph_plan,
                )
            )
            paragraph_chunk_index = index
            index += 1

        if isinstance(relations_value, list):
            for relation_item in relations_value:
                if not isinstance(relation_item, dict):
                    continue
                plan = _relation_plan_from_payload(relation_item, source_name=source_name)
                if paragraph_chunk_index is not None:
                    plan["source_paragraph_chunk_index"] = paragraph_chunk_index
                preview = f"{plan['subject']} | {plan['predicate']} | {plan['object']}"
                chunks.append(
                    _build_chunk_state(
                        index=index,
                        chunk_type="json",
                        content_preview=_preview_text(preview),
                        plan=plan,
                    )
                )
                index += 1
            continue

        for key in ("items", "records", "data", "chunks", "messages"):
            nested = current.get(key)
            if isinstance(nested, list):
                queue.extend(nested)

    if not chunks:
        raise ValueError("未从 OpenIE 文件中提取到可导入内容")
    return chunks


def extract_temporal_backfill_chunks(payload: Any) -> list[dict[str, Any]]:
    """从 JSON 负载中提取时序回填计划。"""

    chunks: list[dict[str, Any]] = []
    queue: list[Any] = [payload]
    seen_keys: set[str] = set()
    index = 0

    while queue:
        current = queue.pop(0)
        if isinstance(current, list):
            queue.extend(current)
            continue
        if not isinstance(current, dict):
            continue

        time_meta = current.get("time_meta")
        if not isinstance(time_meta, dict):
            time_meta = {
                key: current.get(key)
                for key in TIME_META_KEYS
                if current.get(key) is not None
            }
        normalized_time_meta = _serialize_time_meta(time_meta)

        paragraph_hash = str(
            current.get("paragraph_hash", current.get("hash", ""))
        ).strip()
        if not paragraph_hash:
            content = str(current.get("content", current.get("text", ""))).strip()
            if content:
                paragraph_hash = compute_paragraph_hash(content)
        if paragraph_hash and normalized_time_meta:
            dedupe_key = json.dumps(
                [paragraph_hash, normalized_time_meta],
                sort_keys=True,
                ensure_ascii=False,
            )
            if dedupe_key not in seen_keys:
                seen_keys.add(dedupe_key)
                preview = current.get("content", current.get("text", paragraph_hash))
                chunks.append(
                    _build_chunk_state(
                        index=index,
                        chunk_type="json",
                        content_preview=_preview_text(str(preview)),
                        plan={
                            "kind": "backfill",
                            "paragraph_hash": paragraph_hash,
                            "time_meta": normalized_time_meta,
                        },
                    )
                )
                index += 1

        for key in ("paragraphs", "items", "records", "data", "chunks", "messages"):
            nested = current.get(key)
            if isinstance(nested, list):
                queue.extend(nested)

    return chunks


def _public_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": int(chunk.get("index", 0) or 0),
        "chunk_type": str(chunk.get("chunk_type", "") or ""),
        "status": str(chunk.get("status", "") or ""),
        "step": str(chunk.get("step", "") or ""),
        "content_preview": str(chunk.get("content_preview", "") or ""),
        "error": str(chunk.get("error", "") or ""),
    }


def _refresh_file_state(file_state: dict[str, Any]) -> None:
    chunks = list(file_state.get("chunks") or [])
    total = len(chunks)
    done = len([item for item in chunks if item.get("status") == "completed"])
    failed = len([item for item in chunks if item.get("status") == "failed"])
    cancelled = len([item for item in chunks if item.get("status") == "cancelled"])
    running = len([item for item in chunks if item.get("status") == "running"])
    cancel_requested = len([item for item in chunks if item.get("status") == "cancel_requested"])
    finished = done + failed + cancelled

    status = "queued"
    current_step = "queued"
    if running:
        status = "running"
        current_step = next(
            (str(item.get("step") or "writing") for item in chunks if item.get("status") == "running"),
            "writing",
        )
    elif cancel_requested:
        status = "cancel_requested"
        current_step = "cancel_requested"
    elif total and finished == total:
        if failed and done:
            status = "completed_with_errors"
            current_step = "completed_with_errors"
        elif failed and not done:
            status = "failed"
            current_step = "failed"
        elif cancelled and not done and not failed:
            status = "cancelled"
            current_step = "cancelled"
        elif cancelled or failed:
            status = "completed_with_errors"
            current_step = "completed_with_errors"
        else:
            status = "completed"
            current_step = "completed"

    file_state["status"] = status
    file_state["current_step"] = current_step
    file_state["progress"] = (finished / total) if total else 1.0
    file_state["total_chunks"] = total
    file_state["done_chunks"] = done
    file_state["failed_chunks"] = failed
    file_state["cancelled_chunks"] = cancelled


def _refresh_task_state(task_state: dict[str, Any]) -> None:
    files = list(task_state.get("files") or [])
    for file_state in files:
        _refresh_file_state(file_state)

    total = sum(int(item.get("total_chunks", 0) or 0) for item in files)
    done = sum(int(item.get("done_chunks", 0) or 0) for item in files)
    failed = sum(int(item.get("failed_chunks", 0) or 0) for item in files)
    cancelled = sum(int(item.get("cancelled_chunks", 0) or 0) for item in files)
    running = any(str(item.get("status") or "") == "running" for item in files)
    cancel_requested = any(str(item.get("status") or "") == "cancel_requested" for item in files)
    finished = done + failed + cancelled

    if running:
        task_state["status"] = "running"
        task_state["current_step"] = next(
            (
                str(item.get("current_step") or "writing")
                for item in files
                if str(item.get("status") or "") == "running"
            ),
            "writing",
        )
    elif cancel_requested:
        task_state["status"] = "cancel_requested"
        task_state["current_step"] = "cancel_requested"
    elif total and finished == total:
        if failed and done:
            task_state["status"] = "completed_with_errors"
            task_state["current_step"] = "completed_with_errors"
        elif failed and not done:
            task_state["status"] = "failed"
            task_state["current_step"] = "failed"
        elif cancelled and not done and not failed:
            task_state["status"] = "cancelled"
            task_state["current_step"] = "cancelled"
        elif cancelled or failed:
            task_state["status"] = "completed_with_errors"
            task_state["current_step"] = "completed_with_errors"
        else:
            task_state["status"] = "completed"
            task_state["current_step"] = "completed"

    task_state["progress"] = (finished / total) if total else 1.0
    task_state["total_chunks"] = total
    task_state["done_chunks"] = done
    task_state["failed_chunks"] = failed
    task_state["cancelled_chunks"] = cancelled
    task_state["updated_at"] = time.time()


def _public_file(file_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "file_id": str(file_state.get("file_id", "") or ""),
        "name": str(file_state.get("name", "") or ""),
        "detected_strategy_type": str(file_state.get("detected_strategy_type", "") or ""),
        "status": str(file_state.get("status", "") or ""),
        "current_step": str(file_state.get("current_step", "") or ""),
        "progress": float(file_state.get("progress", 0.0) or 0.0),
        "total_chunks": int(file_state.get("total_chunks", 0) or 0),
        "done_chunks": int(file_state.get("done_chunks", 0) or 0),
        "failed_chunks": int(file_state.get("failed_chunks", 0) or 0),
        "cancelled_chunks": int(file_state.get("cancelled_chunks", 0) or 0),
    }


def _public_task(task_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": str(task_state.get("task_id", "") or ""),
        "status": str(task_state.get("status", "") or ""),
        "progress": float(task_state.get("progress", 0.0) or 0.0),
        "current_step": str(task_state.get("current_step", "") or ""),
        "source": str(task_state.get("source", "") or ""),
        "task_kind": str(task_state.get("task_kind", "") or ""),
        "schema_detected": str(task_state.get("schema_detected", "") or ""),
        "error": str(task_state.get("error", "") or ""),
        "retry_parent_task_id": str(task_state.get("retry_parent_task_id", "") or ""),
        "retry_summary": copy.deepcopy(task_state.get("retry_summary") or {}),
        "artifact_paths": copy.deepcopy(task_state.get("artifact_paths") or {}),
        "rollback_info": copy.deepcopy(task_state.get("rollback_info") or {}),
        "created_at": task_state.get("created_at"),
        "updated_at": task_state.get("updated_at"),
        "started_at": task_state.get("started_at"),
        "finished_at": task_state.get("finished_at"),
        "total_chunks": int(task_state.get("total_chunks", 0) or 0),
        "done_chunks": int(task_state.get("done_chunks", 0) or 0),
        "failed_chunks": int(task_state.get("failed_chunks", 0) or 0),
        "cancelled_chunks": int(task_state.get("cancelled_chunks", 0) or 0),
        "files": [_public_file(item) for item in task_state.get("files") or []],
    }


class ImportBackend:
    """为导入中心页面提供宿主桥接后端。"""

    def __init__(self, plugin_instance: Any):
        self.plugin = plugin_instance
        self._bg_tasks: dict[str, asyncio.Task[Any]] = {}
        self._bg_tasks_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._workdir_dir = Path(__file__).resolve().parent

    async def get_guide(self) -> dict[str, Any]:
        alias_map = self._path_aliases()
        content = "\n".join(
            [
                "# 导入中心说明",
                "",
                "宿主已挂载导入页所需的 `/api/import/*` 桥接后端，当前页面可以直接创建并查看任务。",
                "",
                "## 当前可用任务",
                "",
                "- 上传文件：支持 `.txt` / `.md` / `.json`",
                "- 粘贴导入：支持 text / json",
                "- 自动迁移记忆：一键迁移宿主原生记忆与历史聊天总结",
                "- 时序回填：从 JSON 时间字段或段落创建时间回填 event_time",
                "",
                "## 当前路径别名",
                "",
                *[f"- `{key}` -> `{value}`" for key, value in alias_map.items()],
                "",
                "## 已知约束",
                "",
                "- `dedupe_policy`、`llm_enabled` 等参数目前仅做兼容接收，不会改变底层 ImportService 的写入语义。",
                "- 自动迁移中的聊天总结会调用总结模型，首次迁移或积压较多时可能耗费较多 token。",
                "- 时序回填优先消费提供的 JSON 时间字段；若没有可用字段且未禁用 created fallback，会回退到段落的 created_at。",
            ]
        )
        return {
            "success": True,
            "source": "local",
            "path": str(self._state_root() / "guide.md"),
            "content": content,
        }

    async def list_tasks(self, *, limit: int = 80) -> dict[str, Any]:
        from .plugin import ensure_runtime_ready

        ctx = await ensure_runtime_ready()
        records = await ctx.run_blocking(
            ctx.metadata_store.list_async_tasks,
            task_type=TASK_TYPE_IMPORT_BACKEND,
            limit=max(1, min(200, int(limit))),
        )
        items: list[dict[str, Any]] = []
        for record in records:
            state = await self._read_task_state_from_record(record)
            if state is not None:
                items.append(_public_task(state))
                continue
            result = dict(record.get("result") or {})
            items.append(
                {
                    "task_id": str(record.get("task_id", "") or ""),
                    "status": str(record.get("status", "") or ""),
                    "progress": float(result.get("progress", 0.0) or 0.0),
                    "current_step": str(result.get("current_step", "") or ""),
                    "source": str(result.get("source", "") or ""),
                    "task_kind": str(result.get("task_kind", "") or ""),
                    "total_chunks": int(result.get("total_chunks", 0) or 0),
                    "done_chunks": int(result.get("done_chunks", 0) or 0),
                    "failed_chunks": int(result.get("failed_chunks", 0) or 0),
                    "cancelled_chunks": int(result.get("cancelled_chunks", 0) or 0),
                    "created_at": record.get("created_at"),
                    "updated_at": record.get("updated_at"),
                    "started_at": record.get("started_at"),
                    "finished_at": record.get("finished_at"),
                    "files": [],
                    "error": str(record.get("error_message", "") or ""),
                    "schema_detected": str(result.get("schema_detected", "") or ""),
                    "retry_parent_task_id": "",
                    "retry_summary": {},
                    "artifact_paths": {},
                    "rollback_info": {},
                }
            )
        return {
            "success": True,
            "items": items,
            "settings": self._settings_payload(),
        }

    async def get_task(self, task_id: str) -> dict[str, Any]:
        from .plugin import ensure_runtime_ready

        ctx = await ensure_runtime_ready()
        record = await self._get_task_record(ctx, task_id)
        if record is None:
            raise ValueError("Task not found")
        state = await self._read_task_state_from_record(record)
        if state is None:
            raise ValueError("Task state not found")
        return {
            "success": True,
            "task": _public_task(state),
        }

    async def get_task_chunks(
        self,
        task_id: str,
        file_id: str,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        from .plugin import ensure_runtime_ready

        ctx = await ensure_runtime_ready()
        record = await self._get_task_record(ctx, task_id)
        if record is None:
            raise ValueError("Task not found")
        state = await self._read_task_state_from_record(record)
        if state is None:
            raise ValueError("Task state not found")
        file_state = next(
            (item for item in state.get("files") or [] if str(item.get("file_id", "") or "") == str(file_id or "").strip()),
            None,
        )
        if file_state is None:
            raise ValueError("File not found")
        chunks = list(file_state.get("chunks") or [])
        normalized_offset = max(0, int(offset))
        normalized_limit = max(1, min(500, int(limit)))
        sliced = chunks[normalized_offset : normalized_offset + normalized_limit]
        return {
            "success": True,
            "items": [_public_chunk(item) for item in sliced],
            "total": len(chunks),
        }

    async def create_upload_task(self, *, saved_files: list[dict[str, Any]], payload: dict[str, Any]) -> dict[str, Any]:
        if not saved_files:
            raise ValueError("上传文件列表不能为空")
        input_mode = str(payload.get("input_mode", "text") or "text").strip().lower()
        prepared_files: list[dict[str, Any]] = []
        schema_detected = "plain_text"
        for item in saved_files:
            file_path = Path(str(item["path"]))
            file_state, current_schema = await asyncio.to_thread(
                self._prepare_file_from_path,
                file_path=file_path,
                display_name=str(item.get("name", file_path.name)),
                source_name=f"upload:{item.get('name', file_path.name)}",
                input_mode=input_mode,
                strategy_override=str(payload.get("strategy_override", "") or ""),
            )
            prepared_files.append(file_state)
            if current_schema == "web_json":
                schema_detected = current_schema
        return await self._create_task(
            task_kind="upload",
            schema_detected=schema_detected,
            files=prepared_files,
            task_payload={
                "input_mode": input_mode,
                "options": copy.deepcopy(payload),
            },
        )

    async def create_paste_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        content = str(payload.get("content", "") or "")
        if not content.strip():
            raise ValueError("粘贴内容不能为空")
        input_mode = str(payload.get("input_mode", "text") or "text").strip().lower()
        task_id = uuid.uuid4().hex
        paste_name = str(payload.get("name", "") or "").strip() or f"paste_{task_id[:8]}"
        task_dir = self._task_temp_dir(task_id)
        await asyncio.to_thread(task_dir.mkdir, parents=True, exist_ok=True)
        suffix = ".json" if input_mode == "json" else ".txt"
        file_path = task_dir / f"{paste_name}{suffix}"
        await asyncio.to_thread(file_path.write_text, content, "utf-8")
        file_state, schema_detected = await asyncio.to_thread(
            self._prepare_file_from_path,
            file_path=file_path,
            display_name=file_path.name,
            source_name=f"paste:{paste_name}",
            input_mode=input_mode,
            strategy_override=str(payload.get("strategy_override", "") or ""),
        )
        return await self._create_task(
            task_kind="paste",
            schema_detected=schema_detected,
            files=[file_state],
            task_payload={
                "input_mode": input_mode,
                "options": copy.deepcopy(payload),
            },
            forced_task_id=task_id,
        )

    async def create_auto_migrate_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        from .plugin import NaMemorixConfig, plugin

        import_builtin_memory = bool(payload.get("import_builtin_memory", True))
        import_chat_summary = bool(payload.get("import_chat_summary", True))
        if not import_builtin_memory and not import_chat_summary:
            raise ValueError("至少需要启用一种自动迁移来源")

        cfg = plugin.get_config(NaMemorixConfig)
        chat_limit = max(1, min(200, int(payload.get("chat_limit", AUTO_MIGRATE_DEFAULT_CHAT_LIMIT) or AUTO_MIGRATE_DEFAULT_CHAT_LIMIT)))
        message_window = max(4, min(200, int(payload.get("message_window", AUTO_MIGRATE_DEFAULT_MESSAGE_WINDOW) or AUTO_MIGRATE_DEFAULT_MESSAGE_WINDOW)))
        prepared_files: list[dict[str, Any]] = []

        if import_builtin_memory:
            workspace_channels = (
                await DBChatChannel.filter(workspace_id__not_isnull=True)
                .order_by("-update_time")
                .all()
            )
            seen_workspaces: set[int] = set()
            for channel in workspace_channels:
                if channel.workspace_id is None:
                    continue
                workspace_id = int(channel.workspace_id)
                if workspace_id in seen_workspaces:
                    continue
                seen_workspaces.add(workspace_id)
                prepared_files.append(
                    self._build_auto_migrate_builtin_file(
                        workspace_id=workspace_id,
                        batch_size=int(cfg.BUILTIN_MEMORY_SYNC_BATCH_SIZE),
                        include_relations=bool(cfg.BUILTIN_MEMORY_SYNC_INCLUDE_RELATIONS),
                    )
                )

        if import_chat_summary:
            chat_channels = (
                await DBChatChannel.filter(is_active=True)
                .order_by("-update_time")
                .limit(chat_limit)
                .all()
            )
            seen_chat_keys: set[str] = set()
            for channel in chat_channels:
                chat_key = str(channel.chat_key or "").strip()
                if not chat_key or chat_key in seen_chat_keys:
                    continue
                seen_chat_keys.add(chat_key)
                cursor = await get_chat_import_cursor(plugin.store, chat_key)
                rows = (
                    await DBChatMessage.filter(chat_key=chat_key, id__gt=cursor)
                    .exclude(content_text="")
                    .order_by("id")
                    .limit(message_window)
                    .all()
                )
                messages = [
                    {
                        "message_id": int(item.id),
                        "role": "assistant" if str(item.sender_id or "") == "-1" else "user",
                        "content": str(item.content_text or "").strip(),
                    }
                    for item in rows
                    if str(item.content_text or "").strip()
                ]
                if not messages:
                    continue
                prepared_files.append(
                    self._build_auto_migrate_chat_file(
                        chat_key=chat_key,
                        messages=messages,
                        last_message_id=int(messages[-1]["message_id"]),
                        message_window=message_window,
                    )
                )

        if not prepared_files:
            raise ValueError("没有发现可自动迁移的记忆或聊天记录")

        return await self._create_task(
            task_kind="auto_migrate",
            schema_detected="plain_text",
            files=prepared_files,
            task_payload={
                "options": copy.deepcopy(payload),
                "import_builtin_memory": import_builtin_memory,
                "import_chat_summary": import_chat_summary,
                "chat_limit": chat_limit,
                "message_window": message_window,
            },
        )

    async def create_temporal_backfill_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        from .plugin import ensure_runtime_ready

        ctx = await ensure_runtime_ready()
        alias_map = self._path_aliases_paths()
        alias = str(payload.get("alias", "") or "").strip()
        relative_path = str(payload.get("relative_path", "") or "").strip()
        dry_run = bool(payload.get("dry_run", False))
        no_created_fallback = bool(payload.get("no_created_fallback", False))
        limit = max(1, min(200000, int(payload.get("limit", 100000) or 100000)))

        prepared_files: list[dict[str, Any]] = []
        target_path: Path | None = None
        if alias:
            target_path = resolve_alias_path(alias_map, alias, relative_path)
            json_files = await asyncio.to_thread(
                discover_candidate_files,
                target_path,
                pattern="*.json",
                recursive=True,
                suffixes=SUPPORTED_JSON_SUFFIXES,
            )
            for file_path in json_files:
                relative_name = self._safe_display_path(target_path, file_path)
                file_state = await asyncio.to_thread(
                    self._prepare_backfill_file_from_json,
                    file_path=file_path,
                    display_name=relative_name,
                )
                for chunk in file_state.get("chunks") or []:
                    plan = dict(chunk.get("_plan") or {})
                    plan["dry_run"] = dry_run
                    chunk["_plan"] = plan
                if file_state["chunks"]:
                    prepared_files.append(file_state)

        if not prepared_files:
            fallback_file = await self._prepare_created_time_backfill_file(
                ctx,
                dry_run=dry_run,
                no_created_fallback=no_created_fallback,
                limit=limit,
            )
            if fallback_file["chunks"]:
                prepared_files.append(fallback_file)

        if not prepared_files:
            raise ValueError("没有发现可回填的时间数据")

        return await self._create_task(
            task_kind="temporal_backfill",
            schema_detected="plain_text",
            files=prepared_files,
            task_payload={
                "dry_run": dry_run,
                "no_created_fallback": no_created_fallback,
                "limit": limit,
                "root_path": str(target_path) if target_path is not None else "",
                "options": copy.deepcopy(payload),
            },
        )

    async def cancel_task(self, task_id: str) -> dict[str, Any]:
        from .plugin import ensure_runtime_ready

        ctx = await ensure_runtime_ready()
        record = await self._get_task_record(ctx, task_id)
        if record is None:
            raise ValueError("Task not found")
        state = await self._read_task_state_from_record(record)
        if state is None:
            raise ValueError("Task state not found")

        current_status = str(state.get("status", "") or "")
        if current_status not in {"queued", "preparing", "running", "cancel_requested"}:
            return {"success": True, "task": _public_task(state)}

        state["status"] = "cancel_requested"
        state["current_step"] = "cancel_requested"
        state["updated_at"] = time.time()
        await self._write_task_state(state)
        await self._update_task_record(ctx, state, cancel_requested=True)
        return {"success": True, "task": _public_task(state)}

    async def retry_failed(self, task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        from .plugin import ensure_runtime_ready

        ctx = await ensure_runtime_ready()
        record = await self._get_task_record(ctx, task_id)
        if record is None:
            raise ValueError("Task not found")
        state = await self._read_task_state_from_record(record)
        if state is None:
            raise ValueError("Task state not found")

        retry_files: list[dict[str, Any]] = []
        retry_chunks = 0
        for file_state in state.get("files") or []:
            failed_chunks = [
                copy.deepcopy(chunk)
                for chunk in file_state.get("chunks") or []
                if str(chunk.get("status", "") or "") == "failed"
            ]
            if not failed_chunks:
                continue
            retry_chunks += len(failed_chunks)
            for index, chunk in enumerate(failed_chunks):
                chunk["index"] = index
                chunk["status"] = "queued"
                chunk["step"] = "queued"
                chunk["error"] = ""
            new_file = copy.deepcopy(file_state)
            new_file["file_id"] = uuid.uuid4().hex
            new_file["status"] = "queued"
            new_file["current_step"] = "queued"
            new_file["chunks"] = failed_chunks
            retry_files.append(new_file)

        if not retry_files:
            raise ValueError("当前任务没有可重试的失败分块")

        retry_summary = {
            "chunk_retry_files": len(retry_files),
            "chunk_retry_chunks": retry_chunks,
            "file_fallback_files": 0,
        }
        created = await self._create_task(
            task_kind=str(state.get("task_kind", "") or "retry"),
            schema_detected=str(state.get("schema_detected", "") or ""),
            files=retry_files,
            task_payload={
                "retry_parent_task_id": str(task_id or "").strip(),
                "retry_overrides": copy.deepcopy(payload),
            },
            retry_parent_task_id=str(task_id or "").strip(),
            retry_summary=retry_summary,
        )
        return {
            "success": True,
            "task": created["task"],
            "retry_summary": retry_summary,
        }

    def _prepare_file_from_path(
        self,
        *,
        file_path: Path,
        display_name: str,
        source_name: str,
        input_mode: str,
        strategy_override: str,
    ) -> tuple[dict[str, Any], str]:
        file_content = file_path.read_text(encoding="utf-8")
        normalized_input_mode = str(input_mode or "text").strip().lower()
        schema_detected = "plain_text"
        if normalized_input_mode == "json":
            payload = json.loads(file_content)
            chunks, schema_detected = extract_json_import_chunks(payload, source_name=source_name)
            detected_strategy_type = "json"
        else:
            chunks = _build_text_chunks(
                file_content,
                source_name=source_name,
                strategy_override=strategy_override,
            )
            detected_strategy_type = str(strategy_override or "text").strip() or "text"

        file_state = {
            "file_id": uuid.uuid4().hex,
            "name": display_name,
            "source_name": source_name,
            "local_path": str(file_path.resolve()),
            "detected_strategy_type": detected_strategy_type,
            "status": "queued",
            "current_step": "queued",
            "progress": 0.0,
            "total_chunks": len(chunks),
            "done_chunks": 0,
            "failed_chunks": 0,
            "cancelled_chunks": 0,
            "chunks": chunks,
        }
        _refresh_file_state(file_state)
        file_state["status"] = "queued"
        file_state["current_step"] = "queued"
        file_state["progress"] = 0.0
        return file_state, schema_detected

    def _prepare_backfill_file_from_json(self, *, file_path: Path, display_name: str) -> dict[str, Any]:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        chunks = extract_temporal_backfill_chunks(payload)
        file_state = {
            "file_id": uuid.uuid4().hex,
            "name": display_name,
            "source_name": f"temporal_backfill:{display_name}",
            "local_path": str(file_path.resolve()),
            "detected_strategy_type": "json",
            "status": "queued",
            "current_step": "queued",
            "progress": 0.0,
            "total_chunks": len(chunks),
            "done_chunks": 0,
            "failed_chunks": 0,
            "cancelled_chunks": 0,
            "chunks": chunks,
        }
        _refresh_file_state(file_state)
        file_state["status"] = "queued"
        file_state["current_step"] = "queued"
        file_state["progress"] = 0.0
        return file_state

    def _build_auto_migrate_builtin_file(
        self,
        *,
        workspace_id: int,
        batch_size: int,
        include_relations: bool,
    ) -> dict[str, Any]:
        preview = f"同步工作区 {workspace_id} 的原生记忆增量"
        chunks = [
            _build_chunk_state(
                index=0,
                chunk_type="builtin_sync",
                content_preview=preview,
                plan={
                    "kind": "builtin_sync",
                    "workspace_id": int(workspace_id),
                    "batch_size": max(1, int(batch_size)),
                    "include_relations": bool(include_relations),
                },
            )
        ]
        file_state = {
            "file_id": uuid.uuid4().hex,
            "name": f"workspace_{int(workspace_id)}",
            "source_name": f"auto_migrate_builtin:{int(workspace_id)}",
            "local_path": "",
            "detected_strategy_type": "auto_migrate",
            "status": "queued",
            "current_step": "queued",
            "progress": 0.0,
            "total_chunks": len(chunks),
            "done_chunks": 0,
            "failed_chunks": 0,
            "cancelled_chunks": 0,
            "chunks": chunks,
        }
        _refresh_file_state(file_state)
        file_state["status"] = "queued"
        file_state["current_step"] = "queued"
        file_state["progress"] = 0.0
        return file_state

    def _build_auto_migrate_chat_file(
        self,
        *,
        chat_key: str,
        messages: list[dict[str, Any]],
        last_message_id: int,
        message_window: int,
    ) -> dict[str, Any]:
        message_count = len(messages)
        preview = f"迁移聊天 {chat_key} 的 {message_count} 条增量消息"
        chunks = [
            _build_chunk_state(
                index=0,
                chunk_type="chat_summary",
                content_preview=preview,
                plan={
                    "kind": "chat_summary",
                    "chat_key": str(chat_key or "").strip(),
                    "messages": copy.deepcopy(messages),
                    "last_message_id": int(last_message_id),
                    "message_window": max(4, int(message_window)),
                    "source": f"chat_summary:{str(chat_key or '').strip()}",
                },
            )
        ]
        file_state = {
            "file_id": uuid.uuid4().hex,
            "name": str(chat_key or "").strip(),
            "source_name": f"auto_migrate_chat:{str(chat_key or '').strip()}",
            "local_path": "",
            "detected_strategy_type": "auto_migrate",
            "status": "queued",
            "current_step": "queued",
            "progress": 0.0,
            "total_chunks": len(chunks),
            "done_chunks": 0,
            "failed_chunks": 0,
            "cancelled_chunks": 0,
            "chunks": chunks,
        }
        _refresh_file_state(file_state)
        file_state["status"] = "queued"
        file_state["current_step"] = "queued"
        file_state["progress"] = 0.0
        return file_state

    async def _prepare_created_time_backfill_file(
        self,
        ctx: Any,
        *,
        dry_run: bool,
        no_created_fallback: bool,
        limit: int,
    ) -> dict[str, Any]:
        if no_created_fallback:
            return {
                "file_id": uuid.uuid4().hex,
                "name": "created_at_fallback",
                "source_name": "temporal_backfill:created_at_fallback",
                "local_path": "",
                "detected_strategy_type": "json",
                "status": "queued",
                "current_step": "queued",
                "progress": 0.0,
                "total_chunks": 0,
                "done_chunks": 0,
                "failed_chunks": 0,
                "cancelled_chunks": 0,
                "chunks": [],
            }

        def _collect_missing_time_rows() -> list[dict[str, Any]]:
            return ctx.metadata_store.query(
                """
                SELECT hash, content, created_at
                FROM paragraphs
                WHERE COALESCE(is_deleted, 0) = 0
                  AND event_time IS NULL
                  AND event_time_start IS NULL
                  AND event_time_end IS NULL
                  AND created_at IS NOT NULL
                ORDER BY updated_at DESC NULLS LAST, created_at DESC
                LIMIT %s
                """,
                (int(limit),),
            )

        rows = await ctx.run_blocking(_collect_missing_time_rows)
        chunks: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            created_at = row.get("created_at")
            if created_at is None:
                continue
            chunks.append(
                _build_chunk_state(
                    index=index,
                    chunk_type="json",
                    content_preview=_preview_text(str(row.get("content", row.get("hash", "")))),
                    plan={
                        "kind": "backfill",
                        "paragraph_hash": str(row.get("hash", "")).strip(),
                        "time_meta": _serialize_time_meta({"event_time": created_at}),
                        "dry_run": dry_run,
                        "fallback_source": "created_at",
                    },
                )
            )

        file_state = {
            "file_id": uuid.uuid4().hex,
            "name": "created_at_fallback",
            "source_name": "temporal_backfill:created_at_fallback",
            "local_path": "",
            "detected_strategy_type": "json",
            "status": "queued",
            "current_step": "queued",
            "progress": 0.0,
            "total_chunks": len(chunks),
            "done_chunks": 0,
            "failed_chunks": 0,
            "cancelled_chunks": 0,
            "chunks": chunks,
        }
        _refresh_file_state(file_state)
        file_state["status"] = "queued"
        file_state["current_step"] = "queued"
        file_state["progress"] = 0.0
        return file_state

    async def _create_task(
        self,
        *,
        task_kind: str,
        schema_detected: str,
        files: list[dict[str, Any]],
        task_payload: dict[str, Any],
        forced_task_id: str | None = None,
        retry_parent_task_id: str = "",
        retry_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from .plugin import ensure_runtime_ready

        task_id = forced_task_id or uuid.uuid4().hex
        state = {
            "task_id": task_id,
            "status": "queued",
            "progress": 0.0,
            "current_step": "queued",
            "source": task_kind,
            "task_kind": task_kind,
            "schema_detected": schema_detected,
            "error": "",
            "retry_parent_task_id": retry_parent_task_id,
            "retry_summary": copy.deepcopy(retry_summary or {}),
            "artifact_paths": {},
            "rollback_info": {},
            "created_at": time.time(),
            "updated_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "files": copy.deepcopy(files),
        }
        _refresh_task_state(state)
        state["status"] = "queued"
        state["current_step"] = "queued"
        state["progress"] = 0.0
        state["updated_at"] = state["created_at"]
        state_path = self._task_state_path(task_id)
        task_payload = {
            **copy.deepcopy(task_payload),
            "task_kind": task_kind,
            "schema_detected": schema_detected,
            "state_path": str(state_path),
        }

        ctx = await ensure_runtime_ready()
        await self._write_task_state(state)
        await ctx.run_blocking(
            ctx.metadata_store.create_async_task,
            task_id=task_id,
            task_type=TASK_TYPE_IMPORT_BACKEND,
            payload=task_payload,
        )
        await self._update_task_record(ctx, state)

        background = asyncio.create_task(
            self._run_task(task_id),
            name=f"import-backend-{task_id[:8]}",
        )
        async with self._bg_tasks_lock:
            self._bg_tasks[task_id] = background
        background.add_done_callback(lambda _: asyncio.create_task(self._cleanup_bg_task(task_id)))
        return {
            "success": True,
            "task": _public_task(state),
        }

    async def _cleanup_bg_task(self, task_id: str) -> None:
        async with self._bg_tasks_lock:
            self._bg_tasks.pop(task_id, None)

    async def _run_task(self, task_id: str) -> None:
        from amemorix.services import ImportService

        from .plugin import runtime_scope

        try:
            async with runtime_scope() as ctx:
                record = await self._get_task_record(ctx, task_id)
                if record is None:
                    return
                state = await self._read_task_state_from_record(record)
                if state is None:
                    raise ValueError("Task state not found")
                state["status"] = "preparing"
                state["current_step"] = "preparing"
                state["started_at"] = time.time()
                state["updated_at"] = state["started_at"]
                await self._write_task_state(state)
                await self._update_task_record(ctx, state)

                import_service = ImportService(ctx)
                should_save = False
                for file_state in state.get("files") or []:
                    if await self._cancel_requested(ctx, task_id):
                        self._mark_remaining_chunks_cancelled(state)
                        state["finished_at"] = time.time()
                        await self._write_task_state(state)
                        await self._update_task_record(ctx, state, cancel_requested=True)
                        if should_save:
                            await ctx.save_all()
                        return

                    file_state["status"] = "running"
                    file_state["current_step"] = "preparing"
                    _refresh_task_state(state)
                    await self._write_task_state(state)
                    await self._update_task_record(ctx, state)

                    paragraph_hash_by_chunk_index: dict[int, str] = {}
                    for chunk in file_state.get("chunks") or []:
                        if str(chunk.get("status", "") or "") not in {"queued", "failed"}:
                            continue
                        if await self._cancel_requested(ctx, task_id):
                            self._mark_remaining_chunks_cancelled(state)
                            state["finished_at"] = time.time()
                            await self._write_task_state(state)
                            await self._update_task_record(ctx, state, cancel_requested=True)
                            if should_save:
                                await ctx.save_all()
                            return

                        plan = dict(chunk.get("_plan") or {})
                        step = self._plan_step(plan)
                        chunk["status"] = "running"
                        chunk["step"] = step
                        file_state["current_step"] = step
                        _refresh_task_state(state)
                        await self._write_task_state(state)
                        await self._update_task_record(ctx, state)

                        try:
                            result = await self._execute_chunk_plan(
                                import_service,
                                ctx,
                                plan,
                                paragraph_hash_by_chunk_index,
                            )
                            if result.get("hash"):
                                paragraph_hash_by_chunk_index[int(chunk.get("index", 0) or 0)] = str(result["hash"])
                            chunk["status"] = "completed"
                            chunk["step"] = "completed"
                            chunk["error"] = ""
                            saved_flag = result.get("saved")
                            if saved_flag is None:
                                should_save = should_save or not bool(plan.get("dry_run", False))
                            else:
                                should_save = should_save or bool(saved_flag)
                        except Exception as exc:
                            chunk["status"] = "failed"
                            chunk["step"] = "failed"
                            chunk["error"] = str(exc)

                        _refresh_task_state(state)
                        await self._write_task_state(state)
                        await self._update_task_record(ctx, state)

                if should_save:
                    await ctx.save_all()
                state["finished_at"] = time.time()
                _refresh_task_state(state)
                await self._write_task_state(state)
                await self._update_task_record(ctx, state)
        except Exception as exc:
            logger.error("import backend task failed: %s", exc, exc_info=True)
            try:
                from .plugin import ensure_runtime_ready

                ctx = await ensure_runtime_ready()
                record = await self._get_task_record(ctx, task_id)
                state = await self._read_task_state_from_record(record or {})
                if state is not None:
                    state["status"] = "failed"
                    state["current_step"] = "failed"
                    state["error"] = str(exc)
                    state["finished_at"] = time.time()
                    _refresh_task_state(state)
                    state["status"] = "failed"
                    state["current_step"] = "failed"
                    await self._write_task_state(state)
                    await self._update_task_record(ctx, state, error_message=str(exc))
            except Exception:
                logger.error("failed to persist import backend failure state", exc_info=True)

    async def _execute_chunk_plan(
        self,
        import_service: Any,
        ctx: Any,
        plan: dict[str, Any],
        paragraph_hash_by_chunk_index: dict[int, str],
    ) -> dict[str, Any]:
        
        from .plugin import plugin

        kind = str(plan.get("kind", "") or "").strip()
        if kind == "paragraph":
            return await import_service.import_paragraph(
                content=str(plan.get("content", "") or ""),
                source=str(plan.get("source", "web_import") or "web_import"),
                knowledge_type=str(plan.get("knowledge_type", "") or ""),
                time_meta=dict(plan.get("time_meta") or {}),
            )
        if kind == "relation":
            source_paragraph = str(plan.get("source_paragraph", "") or "").strip()
            if not source_paragraph and plan.get("source_paragraph_chunk_index") is not None:
                source_paragraph = paragraph_hash_by_chunk_index.get(
                    int(plan.get("source_paragraph_chunk_index") or 0),
                    "",
                )
            return await import_service.import_relation(
                subject=str(plan.get("subject", "") or ""),
                predicate=str(plan.get("predicate", "") or ""),
                obj=str(plan.get("object", "") or ""),
                confidence=float(plan.get("confidence", 1.0) or 1.0),
                source_paragraph=source_paragraph,
            )
        if kind == "backfill":
            paragraph_hash = str(plan.get("paragraph_hash", "") or "").strip()
            if not paragraph_hash:
                raise ValueError("回填计划缺少 paragraph_hash")
            time_meta = dict(plan.get("time_meta") or {})
            if not time_meta:
                raise ValueError("回填计划缺少 time_meta")
            paragraph = await ctx.run_blocking(ctx.metadata_store.get_paragraph, paragraph_hash)
            if paragraph is None:
                raise ValueError("目标段落不存在")
            if bool(plan.get("dry_run", False)):
                return {
                    "hash": paragraph_hash,
                    "updated": True,
                    "dry_run": True,
                }
            updated = await ctx.run_blocking(
                ctx.metadata_store.update_paragraph_time_meta,
                paragraph_hash,
                time_meta,
            )
            if not updated:
                raise ValueError("段落时间元数据未发生变化")
            return {"hash": paragraph_hash, "updated": True, "saved": True}
        if kind == "builtin_sync":
            result = await sync_builtin_memories(
                ctx=ctx,
                plugin_store=plugin.store,
                workspace_id=int(plan.get("workspace_id", 0) or 0),
                batch_size=max(1, int(plan.get("batch_size", 64) or 64)),
                include_relations=bool(plan.get("include_relations", True)),
                logger=logger,
            )
            imported_total = int(result.get("imported_paragraphs", 0) or 0) + int(
                result.get("imported_relations", 0) or 0
            )
            return {
                "saved": imported_total > 0,
                "imported_paragraphs": int(result.get("imported_paragraphs", 0) or 0),
                "imported_relations": int(result.get("imported_relations", 0) or 0),
            }
        if kind == "chat_summary":
            chat_key = str(plan.get("chat_key", "") or "").strip()
            messages = [
                {
                    "role": str(item.get("role", "user") or "user"),
                    "content": str(item.get("content", "") or "").strip(),
                    "message_id": int(item.get("message_id", 0) or 0),
                }
                for item in list(plan.get("messages") or [])
                if str(item.get("content", "") or "").strip()
            ]
            if len(messages) < AUTO_MIGRATE_MIN_MESSAGES:
                return {
                    "saved": False,
                    "skipped": True,
                    "reason": "messages_below_threshold",
                    "message_count": len(messages),
                }
            payload_messages = [
                {"role": str(item["role"]), "content": str(item["content"])}
                for item in messages
            ]
            result = await SummaryService(ctx).import_from_transcript(
                session_id=chat_key,
                messages=payload_messages,
                source=str(plan.get("source", f"chat_summary:{chat_key}") or f"chat_summary:{chat_key}"),
                context_length=max(AUTO_MIGRATE_MIN_MESSAGES, int(plan.get("message_window", len(payload_messages)) or len(payload_messages))),
            )
            if not bool(result.get("success", False)):
                raise ValueError(str(result.get("message", "聊天总结导入失败") or "聊天总结导入失败"))
            await set_chat_import_cursor(
                plugin.store,
                chat_key,
                int(plan.get("last_message_id", 0) or 0),
            )
            return {
                "saved": True,
                "imported_messages": len(payload_messages),
                "last_message_id": int(plan.get("last_message_id", 0) or 0),
            }
        raise ValueError(f"未知导入计划类型: {kind}")

    async def _cancel_requested(self, ctx: Any, task_id: str) -> bool:
        record = await self._get_task_record(ctx, task_id)
        return bool(record and record.get("cancel_requested"))

    def _mark_remaining_chunks_cancelled(self, state: dict[str, Any]) -> None:
        for file_state in state.get("files") or []:
            for chunk in file_state.get("chunks") or []:
                if str(chunk.get("status", "") or "") in {"queued", "running", "cancel_requested"}:
                    chunk["status"] = "cancelled"
                    chunk["step"] = "cancelled"
                    chunk["error"] = ""
            _refresh_file_state(file_state)
        _refresh_task_state(state)
        state["status"] = "cancelled"
        state["current_step"] = "cancelled"

    def _plan_step(self, plan: dict[str, Any]) -> str:
        kind = str(plan.get("kind", "") or "").strip()
        if kind == "backfill":
            return "backfilling"
        if kind in {"builtin_sync", "chat_summary"}:
            return "migrating"
        if kind == "relation":
            return "extracting"
        return "writing"

    async def _get_task_record(self, ctx: Any, task_id: str) -> dict[str, Any] | None:
        return await ctx.run_blocking(
            ctx.metadata_store.get_async_task,
            str(task_id or "").strip(),
        )

    async def _update_task_record(
        self,
        ctx: Any,
        state: dict[str, Any],
        *,
        cancel_requested: bool | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any] | None:
        result_payload = {
            "progress": float(state.get("progress", 0.0) or 0.0),
            "current_step": str(state.get("current_step", "") or ""),
            "source": str(state.get("source", "") or ""),
            "task_kind": str(state.get("task_kind", "") or ""),
            "schema_detected": str(state.get("schema_detected", "") or ""),
            "total_chunks": int(state.get("total_chunks", 0) or 0),
            "done_chunks": int(state.get("done_chunks", 0) or 0),
            "failed_chunks": int(state.get("failed_chunks", 0) or 0),
            "cancelled_chunks": int(state.get("cancelled_chunks", 0) or 0),
        }
        return await ctx.run_blocking(
            ctx.metadata_store.update_async_task,
            task_id=str(state.get("task_id", "") or ""),
            status=str(state.get("status", "") or ""),
            result=result_payload,
            error_message=error_message if error_message is not None else str(state.get("error", "") or ""),
            started_at=state.get("started_at"),
            finished_at=state.get("finished_at"),
            cancel_requested=cancel_requested,
        )

    async def _read_task_state_from_record(self, record: dict[str, Any]) -> dict[str, Any] | None:
        payload = dict(record.get("payload") or {})
        state_path = str(payload.get("state_path", "") or "").strip()
        if not state_path:
            return None
        return await self._read_task_state(Path(state_path))

    async def _read_task_state(self, state_path: Path) -> dict[str, Any] | None:
        def _read() -> dict[str, Any] | None:
            if not state_path.exists():
                return None
            return json.loads(state_path.read_text(encoding="utf-8"))

        async with self._state_lock:
            return await asyncio.to_thread(_read)

    async def _write_task_state(self, state: dict[str, Any]) -> None:
        state_path = self._task_state_path(str(state.get("task_id", "") or ""))
        state_path.parent.mkdir(parents=True, exist_ok=True)

        def _write() -> None:
            temp_path = state_path.with_suffix(".tmp")
            temp_path.write_text(
                json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temp_path.replace(state_path)

        async with self._state_lock:
            await asyncio.to_thread(_write)

    def _settings_payload(self) -> dict[str, Any]:
        return {
            "poll_interval_ms": DEFAULT_IMPORT_POLL_INTERVAL_MS,
            "default_file_concurrency": DEFAULT_FILE_CONCURRENCY,
            "default_chunk_concurrency": DEFAULT_CHUNK_CONCURRENCY,
            "max_file_concurrency": DEFAULT_MAX_FILE_CONCURRENCY,
            "max_chunk_concurrency": DEFAULT_MAX_CHUNK_CONCURRENCY,
            "path_aliases": self._path_aliases(),
        }

    def _path_aliases_paths(self) -> dict[str, Path]:
        plugin_data_dir = resolve_local_plugin_data_dir(__file__)
        return resolve_path_aliases(plugin_data_dir, self._workdir_dir)

    def _path_aliases(self) -> dict[str, str]:
        return {
            key: str(value)
            for key, value in self._path_aliases_paths().items()
        }

    def _state_root(self) -> Path:
        plugin_data_dir = resolve_local_plugin_data_dir(__file__)
        return plugin_data_dir / "runtime" / "import_backend"

    def _task_state_path(self, task_id: str) -> Path:
        return self._state_root() / "tasks" / f"{task_id}.json"

    def _task_temp_dir(self, task_id: str) -> Path:
        return self._state_root() / "uploads" / task_id

    def _safe_display_path(self, root_path: Path, file_path: Path) -> str:
        try:
            return str(file_path.resolve().relative_to(root_path.resolve()))
        except ValueError:
            return file_path.name
