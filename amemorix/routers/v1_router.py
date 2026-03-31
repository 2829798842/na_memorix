"""Define the standalone `/v1` API routes for na_memorix."""

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from amemorix.services import (
    DeleteService,
    MemoryService,
    PersonProfileApiService,
    QueryService,
)

router = APIRouter(prefix="/v1", tags=["v1"])


class ImportTaskCreateRequest(BaseModel):
    """Request body for creating an import task."""

    mode: str = Field(default="text")
    payload: Any
    options: Dict[str, Any] = Field(default_factory=dict)


class SummaryTaskCreateRequest(BaseModel):
    """Request body for creating a summary task."""

    session_id: Optional[str] = None
    source: str = ""
    messages: list[dict[str, Any]] = Field(default_factory=list)
    context_length: int = 50


class ReindexTaskCreateRequest(BaseModel):
    """Request body for creating a reindex task."""

    batch_size: int = Field(default=32, ge=1, le=256)


class QuerySearchRequest(BaseModel):
    """Semantic search request."""

    query: str
    top_k: Optional[int] = None


class QueryTimeRequest(BaseModel):
    """Temporal search request."""

    query: str = ""
    time_from: Optional[str] = None
    time_to: Optional[str] = None
    person: Optional[str] = None
    source: Optional[str] = None
    top_k: Optional[int] = None


class QueryEntityRequest(BaseModel):
    """Entity detail request."""

    entity_name: str


class QueryRelationRequest(BaseModel):
    """Relation query request."""

    subject: str = ""
    predicate: str = ""
    object: str = ""


class DeleteParagraphRequest(BaseModel):
    """Paragraph deletion request."""

    paragraph_hash: str


class DeleteEntityRequest(BaseModel):
    """Entity deletion request."""

    entity_name: str


class DeleteRelationRequest(BaseModel):
    """Relation deletion request."""

    relation: str


class MemoryStatusRequest(BaseModel):
    """Placeholder request body for status endpoint."""


class MemoryProtectRequest(BaseModel):
    """Memory protect request."""

    id: str
    hours: float = 24.0


class MemoryReinforceRequest(BaseModel):
    """Memory reinforce request."""

    id: str


class MemoryRestoreRequest(BaseModel):
    """Memory restore request."""

    hash: str
    type: str = "relation"


class PersonQueryRequest(BaseModel):
    """Person profile query request."""

    person_id: str = ""
    person_keyword: str = ""
    top_k: int = 12
    force_refresh: bool = False


class PersonOverrideRequest(BaseModel):
    """Person profile override upsert request."""

    person_id: str
    override_text: str
    updated_by: str = "v1"


class PersonOverrideDeleteRequest(BaseModel):
    """Person profile override delete request."""

    person_id: str


class PersonRegistryUpsertRequest(BaseModel):
    """Person registry upsert request."""

    person_id: str
    person_name: str = ""
    nickname: str = ""
    user_id: str = ""
    platform: str = ""
    group_nick_name: Any = None
    memory_points: Any = None
    last_know: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


async def _ctx(request: Request):
    """Get the live runtime context for the current request."""

    del request
    from ...plugin import ensure_runtime_ready

    return await ensure_runtime_ready()


def _task_manager(request: Request):
    """Get the already-started task manager."""

    del request
    from ...plugin import get_task_manager

    manager = get_task_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="Task manager not initialized")
    return manager


def _task_or_404(task):
    """Ensure the task exists."""

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


async def _ensure_task_manager(request: Request):
    """Ensure the background task manager is available."""

    del request
    from ...plugin import ensure_task_manager_started

    return await ensure_task_manager_started()


@router.post("/import/tasks")
async def create_import_task(request: Request, body: ImportTaskCreateRequest):
    """Create an async import task."""

    manager = await _ensure_task_manager(request)
    task = await manager.enqueue_import_task(body.model_dump())
    return {
        "task_id": task.get("task_id"),
        "status": task.get("status"),
        "created_at": task.get("created_at"),
    }


@router.get("/import/tasks/{task_id}")
async def get_import_task(request: Request, task_id: str):
    """Get an async import task."""

    manager = _task_manager(request)
    return _task_or_404(manager.get_task(task_id))


@router.post("/query/search")
async def query_search(request: Request, body: QuerySearchRequest):
    """Execute semantic search."""

    service = QueryService(await _ctx(request))
    try:
        return await service.search(query=body.query, top_k=body.top_k)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/query/time")
async def query_time(request: Request, body: QueryTimeRequest):
    """Execute temporal search."""

    service = QueryService(await _ctx(request))
    try:
        return await service.time_search(
            query=body.query,
            time_from=body.time_from,
            time_to=body.time_to,
            person=body.person,
            source=body.source,
            top_k=body.top_k,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/query/entity")
async def query_entity(request: Request, body: QueryEntityRequest):
    """Query entity details."""

    service = QueryService(await _ctx(request))
    try:
        return await service.entity(entity_name=body.entity_name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/query/relation")
async def query_relation(request: Request, body: QueryRelationRequest):
    """Query relation details."""

    service = QueryService(await _ctx(request))
    try:
        return await service.relation(subject=body.subject, predicate=body.predicate, obj=body.object)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/query/stats")
async def query_stats(request: Request):
    """Return store statistics."""

    service = QueryService(await _ctx(request))
    return await service.stats()


@router.post("/delete/paragraph")
async def delete_paragraph(request: Request, body: DeleteParagraphRequest):
    """Delete a paragraph."""

    service = DeleteService(await _ctx(request))
    try:
        return await service.paragraph(body.paragraph_hash)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/delete/entity")
async def delete_entity(request: Request, body: DeleteEntityRequest):
    """Delete an entity."""

    service = DeleteService(await _ctx(request))
    try:
        return await service.entity(body.entity_name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/delete/relation")
async def delete_relation(request: Request, body: DeleteRelationRequest):
    """Delete a relation."""

    service = DeleteService(await _ctx(request))
    try:
        return await service.relation(body.relation)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/delete/clear")
async def delete_clear(request: Request):
    """Clear all memorix data."""

    service = DeleteService(await _ctx(request))
    try:
        return await service.clear()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/memory/status")
async def memory_status(request: Request, body: MemoryStatusRequest):
    """Return memory maintenance status."""

    del body
    service = MemoryService(await _ctx(request))
    return await service.status()


@router.post("/memory/protect")
async def memory_protect(request: Request, body: MemoryProtectRequest):
    """Protect a memory relation."""

    service = MemoryService(await _ctx(request))
    return await service.protect(query_or_hash=body.id, hours=body.hours)


@router.post("/memory/reinforce")
async def memory_reinforce(request: Request, body: MemoryReinforceRequest):
    """Reinforce a memory relation."""

    service = MemoryService(await _ctx(request))
    return await service.reinforce(query_or_hash=body.id)


@router.post("/memory/restore")
async def memory_restore(request: Request, body: MemoryRestoreRequest):
    """Restore a deleted or frozen memory."""

    service = MemoryService(await _ctx(request))
    return await service.restore(hash_value=body.hash, restore_type=body.type)


@router.post("/person/query")
async def person_query(request: Request, body: PersonQueryRequest):
    """Query or refresh a person profile."""

    service = PersonProfileApiService(await _ctx(request))
    try:
        return await service.query(
            person_id=body.person_id,
            person_keyword=body.person_keyword,
            top_k=body.top_k,
            force_refresh=body.force_refresh,
            source_note="v1:person_query",
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/person/override")
async def person_override(request: Request, body: PersonOverrideRequest):
    """Save a manual person profile override."""

    service = PersonProfileApiService(await _ctx(request))
    try:
        return await service.set_override(
            person_id=body.person_id,
            override_text=body.override_text,
            updated_by=body.updated_by,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/person/override")
async def person_override_delete(request: Request, body: PersonOverrideDeleteRequest):
    """Delete a manual person profile override."""

    service = PersonProfileApiService(await _ctx(request))
    try:
        return await service.delete_override(person_id=body.person_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/person/registry/upsert")
async def person_registry_upsert(request: Request, body: PersonRegistryUpsertRequest):
    """Upsert a person registry record."""

    service = PersonProfileApiService(await _ctx(request))
    try:
        data = await service.upsert_registry(body.model_dump())
        return {"success": True, "item": data}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/person/registry/list")
async def person_registry_list(
    request: Request,
    keyword: str = Query("", description="keyword"),
    page: int = Query(1, ge=1),
    page_size: Optional[int] = Query(None, ge=1, le=200),
):
    """List person registry records."""

    service = PersonProfileApiService(await _ctx(request))
    return await service.list_registry(keyword=keyword, page=page, page_size=page_size)


@router.post("/summary/tasks")
async def create_summary_task(request: Request, body: SummaryTaskCreateRequest):
    """Create an async summary task."""

    manager = await _ensure_task_manager(request)
    task = await manager.enqueue_summary_task(body.model_dump())
    return {
        "task_id": task.get("task_id"),
        "status": task.get("status"),
        "created_at": task.get("created_at"),
    }


@router.get("/summary/tasks/{task_id}")
async def get_summary_task(request: Request, task_id: str):
    """Get an async summary task."""

    manager = _task_manager(request)
    return _task_or_404(manager.get_task(task_id))


@router.post("/reindex")
async def create_reindex_task(request: Request, body: Optional[ReindexTaskCreateRequest] = None):
    """Create an async reindex task."""

    manager = await _ensure_task_manager(request)
    payload = body.model_dump() if body is not None else {}
    task = await manager.enqueue_reindex_task(payload)
    return {
        "task_id": task.get("task_id"),
        "status": task.get("status"),
        "created_at": task.get("created_at"),
    }


@router.get("/reindex/tasks/{task_id}")
async def get_reindex_task(request: Request, task_id: str):
    """Get an async reindex task."""

    manager = _task_manager(request)
    return _task_or_404(manager.get_task(task_id))
