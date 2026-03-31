"""统一调度导入、摘要、重建索引与维护任务。"""

import asyncio
import datetime
import uuid
from typing import Any, Dict, List, Optional, Tuple

from amemorix.common.logging import get_logger
from amemorix.context import AppContext
from amemorix.services import (
    ImportService,
    MemoryService,
    PersonProfileApiService,
    SummaryService,
)

logger = get_logger("A_Memorix.TaskManager")

TASK_STATUS_QUEUED = "queued"
TASK_STATUS_RUNNING = "running"
TASK_STATUS_SUCCEEDED = "succeeded"
TASK_STATUS_FAILED = "failed"
TASK_STATUS_CANCELED = "canceled"


class TaskManager:
    """管理后台任务队列与周期维护协程。

    Attributes:
        ctx (AppContext): 共享运行时上下文。
        import_service (ImportService): 导入任务服务。
        summary_service (SummaryService): 摘要任务服务。
        memory_service (MemoryService): 记忆维护服务。
        person_service (PersonProfileApiService): 人物画像服务。
        import_queue (asyncio.Queue[Tuple[str, Dict[str, Any]]]): 导入任务队列。
        summary_queue (asyncio.Queue[Tuple[str, Dict[str, Any]]]): 摘要任务队列。
        reindex_queue (asyncio.Queue[Tuple[str, Dict[str, Any]]]): 重建索引任务队列。
        _workers (List[asyncio.Task]): 已启动的后台工作协程。
        _stopping (bool): 是否处于停止流程中。
        _reindex_enqueue_lock (asyncio.Lock): 保证重建索引任务串行入队的锁。
    """

    def __init__(self, ctx: AppContext):
        """初始化任务管理器及其队列。

        Args:
            ctx: 共享运行时上下文。
        """
        self.ctx = ctx
        self.import_service = ImportService(ctx)
        self.summary_service = SummaryService(ctx)
        self.memory_service = MemoryService(ctx)
        self.person_service = PersonProfileApiService(ctx)

        queue_maxsize = int(self.ctx.get_config("tasks.queue_maxsize", 1024))
        self.import_queue: asyncio.Queue[Tuple[str, Dict[str, Any]]] = asyncio.Queue(maxsize=max(1, queue_maxsize))
        self.summary_queue: asyncio.Queue[Tuple[str, Dict[str, Any]]] = asyncio.Queue(maxsize=max(1, queue_maxsize))
        self.reindex_queue: asyncio.Queue[Tuple[str, Dict[str, Any]]] = asyncio.Queue(maxsize=1)
        self._workers: List[asyncio.Task] = []
        self._stopping = False
        self._reindex_enqueue_lock = asyncio.Lock()

    async def start(self) -> None:
        """启动任务工作协程与周期维护循环。"""
        self._stopping = False
        import_workers = max(1, int(self.ctx.get_config("tasks.import_workers", 1)))
        summary_workers = max(1, int(self.ctx.get_config("tasks.summary_workers", 1)))

        for idx in range(import_workers):
            self._workers.append(asyncio.create_task(self._import_worker(idx), name=f"import-worker-{idx}"))
        for idx in range(summary_workers):
            self._workers.append(asyncio.create_task(self._summary_worker(idx), name=f"summary-worker-{idx}"))
        self._workers.append(asyncio.create_task(self._reindex_worker(0), name="reindex-worker-0"))

        self._workers.append(asyncio.create_task(self._auto_save_loop(), name="auto-save-loop"))
        self._workers.append(asyncio.create_task(self._memory_maintenance_loop(), name="memory-maint-loop"))
        self._workers.append(asyncio.create_task(self._person_profile_refresh_loop(), name="person-profile-loop"))
        logger.info("TaskManager started with %s workers", len(self._workers))

    async def stop(self) -> None:
        """停止全部后台任务并等待协程退出。"""
        self._stopping = True
        for task in self._workers:
            task.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.info("TaskManager stopped")

    async def enqueue_import_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """创建并排队一个导入任务。

        Args:
            payload: 导入任务的请求参数。

        Returns:
            Dict[str, Any]: 创建后的任务记录。
        """
        task_id = uuid.uuid4().hex
        task = self.ctx.metadata_store.create_async_task(task_id=task_id, task_type="import", payload=payload)
        await self.import_queue.put((task_id, payload))
        return task

    async def enqueue_summary_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """创建并排队一个摘要任务。

        Args:
            payload: 摘要任务的请求参数。

        Returns:
            Dict[str, Any]: 创建后的任务记录。
        """
        task_id = uuid.uuid4().hex
        task = self.ctx.metadata_store.create_async_task(task_id=task_id, task_type="summary", payload=payload)
        await self.summary_queue.put((task_id, payload))
        return task

    async def enqueue_reindex_task(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """创建并排队一个重建向量索引任务。

        Args:
            payload: 重建任务的可选参数。

        Returns:
            Dict[str, Any]: 已存在的活跃任务或新建任务记录。
        """
        async with self._reindex_enqueue_lock:
            active = self._find_active_task("reindex")
            if active is not None:
                return active

            task_id = uuid.uuid4().hex
            task_payload = dict(payload or {})
            task = self.ctx.metadata_store.create_async_task(task_id=task_id, task_type="reindex", payload=task_payload)
            await self.reindex_queue.put((task_id, task_payload))
            return task

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """按任务 ID 查询任务详情。

        Args:
            task_id: 任务唯一标识。

        Returns:
            Optional[Dict[str, Any]]: 命中的任务记录；不存在时返回 ``None``。
        """
        return self.ctx.metadata_store.get_async_task(task_id)

    def _find_active_task(self, task_type: str) -> Optional[Dict[str, Any]]:
        """查找指定类型仍在排队或执行中的任务。

        Args:
            task_type: 任务类型。

        Returns:
            Optional[Dict[str, Any]]: 首个活跃任务记录；不存在时返回 ``None``。
        """
        for item in self.ctx.metadata_store.list_async_tasks(task_type=task_type, limit=20):
            if str(item.get("status") or "") in {TASK_STATUS_QUEUED, TASK_STATUS_RUNNING}:
                return item
        return None

    async def _import_worker(self, worker_idx: int) -> None:
        """循环消费导入任务队列。

        Args:
            worker_idx: 工作协程编号。
        """
        while not self._stopping:
            task_id = ""
            try:
                task_id, payload = await self.import_queue.get()
                existing = self.ctx.metadata_store.get_async_task(task_id)
                if existing and existing.get("cancel_requested"):
                    self.ctx.metadata_store.update_async_task(
                        task_id=task_id,
                        status=TASK_STATUS_CANCELED,
                        finished_at=datetime.datetime.now().timestamp(),
                    )
                    continue

                now = datetime.datetime.now().timestamp()
                self.ctx.metadata_store.update_async_task(task_id=task_id, status=TASK_STATUS_RUNNING, started_at=now)
                mode = str(payload.get("mode", "text"))
                body = payload.get("payload")
                options = payload.get("options") if isinstance(payload.get("options"), dict) else {}
                result = await self.import_service.run_import(mode=mode, payload=body, options=options)
                self.ctx.metadata_store.update_async_task(
                    task_id=task_id,
                    status=TASK_STATUS_SUCCEEDED,
                    result=result,
                    finished_at=datetime.datetime.now().timestamp(),
                )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if task_id:
                    self.ctx.metadata_store.update_async_task(
                        task_id=task_id,
                        status=TASK_STATUS_FAILED,
                        error_message=str(exc),
                        finished_at=datetime.datetime.now().timestamp(),
                    )
                logger.error("Import worker %s failed task %s: %s", worker_idx, task_id, exc, exc_info=True)
            finally:
                if task_id:
                    self.import_queue.task_done()

    async def _summary_worker(self, worker_idx: int) -> None:
        """循环消费摘要任务队列。

        Args:
            worker_idx: 工作协程编号。
        """
        while not self._stopping:
            task_id = ""
            try:
                task_id, payload = await self.summary_queue.get()
                existing = self.ctx.metadata_store.get_async_task(task_id)
                if existing and existing.get("cancel_requested"):
                    self.ctx.metadata_store.update_async_task(
                        task_id=task_id,
                        status=TASK_STATUS_CANCELED,
                        finished_at=datetime.datetime.now().timestamp(),
                    )
                    continue

                self.ctx.metadata_store.update_async_task(
                    task_id=task_id,
                    status=TASK_STATUS_RUNNING,
                    started_at=datetime.datetime.now().timestamp(),
                )
                session_id = str(payload.get("session_id", "")).strip() or uuid.uuid4().hex
                messages = payload.get("messages")
                if not isinstance(messages, list):
                    messages = []
                source = str(payload.get("source", f"summary:{session_id}"))
                context_length = int(payload.get("context_length", self.ctx.get_config("summarization.context_length", 50)))
                result = await self.summary_service.import_from_transcript(
                    session_id=session_id,
                    messages=messages,
                    source=source,
                    context_length=context_length,
                )
                status = TASK_STATUS_SUCCEEDED if result.get("success") else TASK_STATUS_FAILED
                self.ctx.metadata_store.update_async_task(
                    task_id=task_id,
                    status=status,
                    result=result,
                    error_message="" if result.get("success") else str(result.get("message", "")),
                    finished_at=datetime.datetime.now().timestamp(),
                )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if task_id:
                    self.ctx.metadata_store.update_async_task(
                        task_id=task_id,
                        status=TASK_STATUS_FAILED,
                        error_message=str(exc),
                        finished_at=datetime.datetime.now().timestamp(),
                    )
                logger.error("Summary worker %s failed task %s: %s", worker_idx, task_id, exc, exc_info=True)
            finally:
                if task_id:
                    self.summary_queue.task_done()

    async def _reindex_worker(self, worker_idx: int) -> None:
        """串行执行重建索引任务。

        Args:
            worker_idx: 工作协程编号，占位以保持工作协程签名一致。
        """
        del worker_idx
        while not self._stopping:
            task_id = ""
            try:
                task_id, payload = await self.reindex_queue.get()
                existing = self.ctx.metadata_store.get_async_task(task_id)
                if existing and existing.get("cancel_requested"):
                    self.ctx.metadata_store.update_async_task(
                        task_id=task_id,
                        status=TASK_STATUS_CANCELED,
                        finished_at=datetime.datetime.now().timestamp(),
                    )
                    continue

                started_at = datetime.datetime.now().timestamp()
                self.ctx.metadata_store.update_async_task(
                    task_id=task_id,
                    status=TASK_STATUS_RUNNING,
                    started_at=started_at,
                )
                batch_size = max(1, int((payload or {}).get("batch_size", 32)))
                await asyncio.to_thread(
                    self.ctx.vector_store.rebuild_index,
                    embedding_manager=self.ctx.embedding_manager,
                    batch_size=batch_size,
                )
                result = {
                    "success": True,
                    "batch_size": batch_size,
                    "paragraph_count": self.ctx.metadata_store.count_paragraphs(),
                    "relation_count": self.ctx.metadata_store.count_relations(),
                    "vector_count": int(self.ctx.vector_store.num_vectors),
                }
                self.ctx.metadata_store.update_async_task(
                    task_id=task_id,
                    status=TASK_STATUS_SUCCEEDED,
                    result=result,
                    finished_at=datetime.datetime.now().timestamp(),
                )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if task_id:
                    self.ctx.metadata_store.update_async_task(
                        task_id=task_id,
                        status=TASK_STATUS_FAILED,
                        error_message=str(exc),
                        finished_at=datetime.datetime.now().timestamp(),
                    )
                logger.error("Reindex worker failed task %s: %s", task_id, exc, exc_info=True)
            finally:
                if task_id:
                    self.reindex_queue.task_done()

    async def _auto_save_loop(self) -> None:
        """按配置周期触发全量持久化。"""
        while not self._stopping:
            try:
                interval_min = float(self.ctx.get_config("advanced.auto_save_interval_minutes", 5))
                await asyncio.sleep(max(30.0, interval_min * 60.0))
                if bool(self.ctx.get_config("advanced.enable_auto_save", True)):
                    await self.ctx.save_all()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Auto-save loop error: %s", exc)

    async def _memory_maintenance_loop(self) -> None:
        """周期执行记忆衰减、冻结与关系裁剪。"""
        while not self._stopping:
            try:
                interval_h = float(self.ctx.get_config("memory.base_decay_interval_hours", 1.0))
                interval_s = max(60.0, interval_h * 3600.0)
                await asyncio.sleep(interval_s)
                if not bool(self.ctx.get_config("memory.enabled", True)):
                    continue

                half_life = float(self.ctx.get_config("memory.half_life_hours", 24.0))
                if half_life > 0:
                    factor = 0.5 ** (interval_h / half_life)
                    self.ctx.graph_store.decay(factor)

                prune_threshold = float(self.ctx.get_config("memory.prune_threshold", 0.1))
                low_edges = self.ctx.graph_store.get_low_weight_edges(prune_threshold)
                hashes_to_freeze: List[str] = []
                edges_to_deactivate = []
                for src, tgt in low_edges:
                    src_c = self.ctx.graph_store._canonicalize(src)  # noqa: SLF001
                    tgt_c = self.ctx.graph_store._canonicalize(tgt)  # noqa: SLF001
                    s_idx = self.ctx.graph_store._node_to_idx.get(src_c)  # noqa: SLF001
                    t_idx = self.ctx.graph_store._node_to_idx.get(tgt_c)  # noqa: SLF001
                    if s_idx is None or t_idx is None:
                        continue
                    edge_hashes = list(self.ctx.graph_store._edge_hash_map.get((s_idx, t_idx), set()))  # noqa: SLF001
                    if not edge_hashes:
                        continue
                    status = self.ctx.metadata_store.get_relation_status_batch(edge_hashes)
                    protected = any(
                        (v.get("is_pinned") or ((v.get("protected_until") or 0) > datetime.datetime.now().timestamp()))
                        for v in status.values()
                    )
                    if not protected:
                        hashes_to_freeze.extend(edge_hashes)
                        edges_to_deactivate.append((src, tgt))

                if hashes_to_freeze:
                    self.ctx.metadata_store.mark_relations_inactive(hashes_to_freeze)
                    self.ctx.graph_store.deactivate_edges(edges_to_deactivate)

                freeze_hours = float(self.ctx.get_config("memory.freeze_duration_hours", 24.0))
                cutoff = datetime.datetime.now().timestamp() - max(1.0, freeze_hours * 3600.0)
                expired = self.ctx.metadata_store.get_prune_candidates(cutoff)
                if expired:
                    cursor = self.ctx.metadata_store._conn.cursor()
                    placeholders = ",".join(["?"] * len(expired))
                    cursor.execute(
                        f"SELECT hash, subject, object FROM relations WHERE hash IN ({placeholders})",
                        expired,
                    )
                    ops = [(str(row[1]), str(row[2]), str(row[0])) for row in cursor.fetchall()]
                    if ops:
                        self.ctx.graph_store.prune_relation_hashes(ops)
                    self.ctx.metadata_store.backup_and_delete_relations(expired)
                self.ctx.graph_store.save()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Memory maintenance loop error: %s", exc, exc_info=True)

    async def _person_profile_refresh_loop(self) -> None:
        """周期刷新活跃人物画像快照。"""
        while not self._stopping:
            try:
                interval_min = int(self.ctx.get_config("person_profile.refresh_interval_minutes", 30))
                await asyncio.sleep(max(60, interval_min * 60))
                if not bool(self.ctx.get_config("person_profile.enabled", True)):
                    continue

                active_window_h = float(self.ctx.get_config("person_profile.active_window_hours", 72.0))
                active_after = datetime.datetime.now().timestamp() - max(1.0, active_window_h * 3600.0)
                limit = int(self.ctx.get_config("person_profile.max_refresh_per_cycle", 50))
                top_k = int(self.ctx.get_config("person_profile.top_k_evidence", 12))

                pids = self.ctx.metadata_store.get_active_person_ids_for_enabled_switches(
                    active_after=active_after,
                    limit=limit,
                )
                for pid in pids:
                    try:
                        await self.person_service.query(
                            person_id=pid,
                            top_k=top_k,
                            force_refresh=False,
                            source_note="task:person_profile_refresh",
                        )
                    except Exception:
                        continue
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Person profile refresh loop error: %s", exc)

