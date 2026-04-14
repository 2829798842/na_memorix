"""宿主检索调优后端实现。"""

import asyncio
import copy
import time
import uuid
from typing import Any

import toml

from amemorix.common.logging import get_logger
from core.retrieval.dual_path import DualPathRetriever, DualPathRetrieverConfig, FusionConfig, RetrievalStrategy
from core.retrieval.sparse_bm25 import SparseBM25Config, SparseBM25Index
from core.utils.search_execution_service import SearchExecutionRequest, SearchExecutionService

from .retrieval_tuning_core import (
    DEFAULT_INTENSITY,
    DEFAULT_OBJECTIVE,
    DEFAULT_POLL_INTERVAL_MS,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_TOP_K_EVAL,
    EvaluationCase,
    calculate_metrics,
    deep_merge_dict,
    extract_tuning_profile,
    generate_candidate_profiles,
    make_paragraph_query,
    make_relation_query,
    normalize_tuning_profile_patch,
    render_markdown_report,
    resolve_intensity,
    resolve_objective,
    score_round_metrics,
)

logger = get_logger("A_Memorix.RetrievalTuningBackend")


class RetrievalTuningBackend:
    """为宿主调优页提供兼容后端。"""

    def __init__(self, plugin_instance: Any):
        self.plugin = plugin_instance
        self._bg_tasks: dict[str, asyncio.Task[Any]] = {}
        self._bg_tasks_lock = asyncio.Lock()

    async def get_profile(self) -> dict[str, Any]:
        from .plugin import ensure_runtime_ready

        ctx = await ensure_runtime_ready()
        profile = extract_tuning_profile(ctx.config)
        return {
            "success": True,
            "profile": profile,
            "settings": {
                "default_objective": DEFAULT_OBJECTIVE,
                "default_intensity": DEFAULT_INTENSITY,
                "default_sample_size": DEFAULT_SAMPLE_SIZE,
                "default_top_k_eval": DEFAULT_TOP_K_EVAL,
                "poll_interval_ms": DEFAULT_POLL_INTERVAL_MS,
            },
        }

    async def apply_profile(self, profile_patch: dict[str, Any], *, reason: str = "web_manual_apply") -> dict[str, Any]:
        from .plugin import apply_runtime_tuning_profile_patch, ensure_runtime_ready

        normalized_patch = normalize_tuning_profile_patch(profile_patch)
        if not normalized_patch:
            raise ValueError("调优 patch 不能为空")
        apply_runtime_tuning_profile_patch(normalized_patch)
        ctx = await ensure_runtime_ready()
        return {
            "success": True,
            "reason": str(reason or "web_manual_apply"),
            "profile": extract_tuning_profile(ctx.config),
        }

    async def rollback_profile(self) -> dict[str, Any]:
        from .plugin import ensure_runtime_ready, rollback_runtime_tuning_profile

        rolled_back = rollback_runtime_tuning_profile()
        if rolled_back is None:
            raise ValueError("当前没有可回滚的调优配置历史")
        ctx = await ensure_runtime_ready()
        return {
            "success": True,
            "overlay": rolled_back,
            "profile": extract_tuning_profile(ctx.config),
        }

    async def export_profile_toml(self) -> dict[str, Any]:
        from .plugin import ensure_runtime_ready

        ctx = await ensure_runtime_ready()
        profile = extract_tuning_profile(ctx.config)
        return {
            "success": True,
            "profile": profile,
            "toml": toml.dumps(profile).strip() + "\n",
        }

    async def list_tasks(self, *, limit: int = 100) -> dict[str, Any]:
        from .plugin import ensure_runtime_ready

        ctx = await ensure_runtime_ready()
        records = await ctx.run_blocking(
            ctx.metadata_store.list_async_tasks,
            task_type="retrieval_tuning",
            limit=max(1, min(200, int(limit))),
        )
        return {
            "success": True,
            "items": [self._serialize_task_summary(record) for record in records],
        }

    async def get_task(self, task_id: str) -> dict[str, Any]:
        from .plugin import ensure_runtime_ready

        ctx = await ensure_runtime_ready()
        record = await self._get_task_record(ctx, task_id)
        if record is None:
            raise ValueError("Task not found")
        return {
            "success": True,
            "task": self._serialize_task_detail(record),
        }

    async def get_task_rounds(self, task_id: str, *, offset: int = 0, limit: int = 100) -> dict[str, Any]:
        from .plugin import ensure_runtime_ready

        ctx = await ensure_runtime_ready()
        record = await self._get_task_record(ctx, task_id)
        if record is None:
            raise ValueError("Task not found")
        rounds = list((record.get("result") or {}).get("rounds") or [])
        sliced = rounds[max(0, int(offset)) : max(0, int(offset)) + max(1, int(limit))]
        return {
            "success": True,
            "items": sliced,
            "total": len(rounds),
        }

    async def get_task_report(self, task_id: str, *, report_format: str = "md") -> dict[str, Any]:
        from .plugin import ensure_runtime_ready

        ctx = await ensure_runtime_ready()
        record = await self._get_task_record(ctx, task_id)
        if record is None:
            raise ValueError("Task not found")
        report = str((record.get("result") or {}).get("report_markdown") or "")
        if str(report_format or "md").strip().lower() != "md":
            raise ValueError("仅支持 format=md")
        return {
            "success": True,
            "content": report,
        }

    async def cancel_task(self, task_id: str) -> dict[str, Any]:
        from .plugin import ensure_runtime_ready

        ctx = await ensure_runtime_ready()
        record = await self._get_task_record(ctx, task_id)
        if record is None:
            raise ValueError("Task not found")
        status = str(record.get("status") or "")
        if status not in {"queued", "running"}:
            return {
                "success": True,
                "task": self._serialize_task_detail(record),
            }
        updated = await self._update_task_record(
            ctx,
            task_id,
            cancel_requested=True,
            result_patch={
                "status_note": "cancel_requested",
            },
        )
        return {
            "success": True,
            "task": self._serialize_task_detail(updated or record),
        }

    async def apply_best_profile(self, task_id: str) -> dict[str, Any]:
        from .plugin import apply_runtime_tuning_profile_patch, ensure_runtime_ready

        ctx = await ensure_runtime_ready()
        record = await self._get_task_record(ctx, task_id)
        if record is None:
            raise ValueError("Task not found")
        result = dict(record.get("result") or {})
        best_profile = dict(result.get("best_profile") or {})
        if not best_profile:
            raise ValueError("当前任务尚未产出最优参数")
        apply_runtime_tuning_profile_patch(best_profile)
        refreshed_ctx = await ensure_runtime_ready()
        return {
            "success": True,
            "task_id": task_id,
            "profile": extract_tuning_profile(refreshed_ctx.config),
        }

    async def create_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        from .plugin import ensure_runtime_ready

        ctx = await ensure_runtime_ready()
        base_profile = extract_tuning_profile(ctx.config)
        objective = resolve_objective(payload.get("objective"))
        intensity = resolve_intensity(payload.get("intensity"))
        sample_size = max(4, min(500, int(payload.get("sample_size", DEFAULT_SAMPLE_SIZE))))
        top_k_eval = max(5, min(100, int(payload.get("top_k_eval", DEFAULT_TOP_K_EVAL))))
        llm_enabled = bool(payload.get("llm_enabled", True))
        requested_rounds = payload.get("rounds")
        candidates = generate_candidate_profiles(base_profile, objective, intensity, requested_rounds)
        task_payload = {
            "objective": objective,
            "intensity": intensity,
            "sample_size": sample_size,
            "top_k_eval": top_k_eval,
            "llm_enabled": llm_enabled,
            "requested_rounds": requested_rounds,
            "rounds_total": len(candidates),
            "base_profile": base_profile,
        }
        task_id = uuid.uuid4().hex
        created = await ctx.run_blocking(
            ctx.metadata_store.create_async_task,
            task_id=task_id,
            task_type="retrieval_tuning",
            payload=task_payload,
        )
        background = asyncio.create_task(
            self._run_task(
                task_id=task_id,
                task_payload=task_payload,
                candidate_profiles=candidates,
            ),
            name=f"retrieval-tuning-{task_id[:8]}",
        )
        async with self._bg_tasks_lock:
            self._bg_tasks[task_id] = background
        background.add_done_callback(lambda _: asyncio.create_task(self._cleanup_bg_task(task_id)))
        return {
            "success": True,
            "task": self._serialize_task_detail(created),
        }

    async def _cleanup_bg_task(self, task_id: str) -> None:
        async with self._bg_tasks_lock:
            self._bg_tasks.pop(task_id, None)

    async def _run_task(
        self,
        *,
        task_id: str,
        task_payload: dict[str, Any],
        candidate_profiles: list[dict[str, Any]],
    ) -> None:
        from .plugin import runtime_scope

        try:
            async with runtime_scope() as ctx:
                started_at = time.time()
                await self._update_task_record(
                    ctx,
                    task_id,
                    status="running",
                    started_at=started_at,
                    result_patch={
                        "progress": 0.0,
                        "rounds_total": len(candidate_profiles),
                        "rounds_done": 0,
                        "best_score": 0.0,
                        "status_note": (
                            "llm_enabled 已退化为宿主自举样本评估"
                            if bool(task_payload.get("llm_enabled"))
                            else "宿主自举样本评估"
                        ),
                    },
                )
                cases = await self._build_eval_cases(ctx, sample_size=int(task_payload["sample_size"]), seed=task_id)
                sparse_cache: dict[str, SparseBM25Index] = {}
                rounds: list[dict[str, Any]] = []
                best_round: dict[str, Any] | None = None
                baseline_metrics: dict[str, Any] | None = None

                for round_index, candidate in enumerate(candidate_profiles, start=1):
                    current = await self._get_task_record(ctx, task_id)
                    if current and bool(current.get("cancel_requested")):
                        await self._update_task_record(
                            ctx,
                            task_id,
                            status="canceled",
                            finished_at=time.time(),
                            result_patch={
                                "rounds": rounds,
                                "rounds_done": len(rounds),
                                "progress": len(rounds) / max(1, len(candidate_profiles)),
                            },
                        )
                        return

                    metrics, avg_latency_ms = await self._evaluate_candidate(
                        ctx,
                        candidate["profile"],
                        cases,
                        top_k_eval=int(task_payload["top_k_eval"]),
                        sparse_cache=sparse_cache,
                    )
                    score = score_round_metrics(metrics, str(task_payload["objective"]))
                    round_result = {
                        "round_index": round_index,
                        "label": candidate["label"],
                        "profile": candidate["profile"],
                        "metrics": metrics,
                        "score": score,
                        "latency_ms": round(avg_latency_ms, 4),
                    }
                    rounds.append(round_result)
                    if baseline_metrics is None:
                        baseline_metrics = metrics
                    if best_round is None or float(round_result["score"]) > float(best_round["score"]):
                        best_round = round_result

                    await self._update_task_record(
                        ctx,
                        task_id,
                        result_patch={
                            "rounds": rounds,
                            "rounds_done": len(rounds),
                            "rounds_total": len(candidate_profiles),
                            "progress": len(rounds) / max(1, len(candidate_profiles)),
                            "baseline_metrics": baseline_metrics or {},
                            "best_metrics": dict((best_round or {}).get("metrics") or {}),
                            "best_profile": dict((best_round or {}).get("profile") or {}),
                            "best_score": float((best_round or {}).get("score", 0.0)),
                        },
                    )

                best_round = best_round or {
                    "metrics": {},
                    "profile": {},
                    "score": 0.0,
                }
                final_task = {
                    "task_id": task_id,
                    "objective": task_payload["objective"],
                    "intensity": task_payload["intensity"],
                    "sample_size": task_payload["sample_size"],
                    "top_k_eval": task_payload["top_k_eval"],
                    "rounds": rounds,
                    "rounds_done": len(rounds),
                    "rounds_total": len(candidate_profiles),
                    "baseline_metrics": baseline_metrics or {},
                    "best_metrics": dict(best_round.get("metrics") or {}),
                    "best_profile": dict(best_round.get("profile") or {}),
                }
                report_markdown = render_markdown_report(final_task)
                await self._update_task_record(
                    ctx,
                    task_id,
                    status="completed",
                    finished_at=time.time(),
                    result_patch={
                        "rounds": rounds,
                        "rounds_done": len(rounds),
                        "rounds_total": len(candidate_profiles),
                        "progress": 1.0,
                        "baseline_metrics": baseline_metrics or {},
                        "best_metrics": dict(best_round.get("metrics") or {}),
                        "best_profile": dict(best_round.get("profile") or {}),
                        "best_score": float(best_round.get("score", 0.0)),
                        "report_markdown": report_markdown,
                    },
                )
        except Exception as exc:
            logger.error("retrieval tuning task failed: %s", exc, exc_info=True)
            try:
                from .plugin import ensure_runtime_ready

                ctx = await ensure_runtime_ready()
                await self._update_task_record(
                    ctx,
                    task_id,
                    status="failed",
                    finished_at=time.time(),
                    error_message=str(exc),
                )
            except Exception:
                logger.error("failed to persist retrieval tuning failure state", exc_info=True)

    async def _build_eval_cases(self, ctx: Any, *, sample_size: int, seed: str) -> list[EvaluationCase]:
        def _collect() -> list[EvaluationCase]:
            import random

            randomizer = random.Random(seed)
            paragraph_limit = max(40, sample_size * 5)
            relation_limit = max(20, sample_size * 4)
            paragraph_rows = ctx.metadata_store.query(
                """
                SELECT hash, content, source
                FROM paragraphs
                WHERE COALESCE(is_deleted, 0) = 0
                  AND LENGTH(COALESCE(content, '')) >= %s
                ORDER BY updated_at DESC NULLS LAST, created_at DESC
                LIMIT %s
                """,
                (32, paragraph_limit),
            )
            relation_rows = ctx.metadata_store.query(
                """
                SELECT hash, subject, predicate, object
                FROM relations
                ORDER BY last_accessed DESC NULLS LAST, created_at DESC
                LIMIT %s
                """,
                (relation_limit,),
            )

            randomizer.shuffle(paragraph_rows)
            randomizer.shuffle(relation_rows)

            relation_target = min(len(relation_rows), max(4, sample_size // 3))
            paragraph_target = max(0, sample_size - relation_target)

            cases: list[EvaluationCase] = []
            for row in paragraph_rows:
                if len([case for case in cases if case.expected_type == "paragraph"]) >= paragraph_target:
                    break
                query = make_paragraph_query(str(row.get("content") or ""))
                if not query:
                    continue
                cases.append(
                    EvaluationCase(
                        case_id=f"paragraph:{row['hash']}",
                        category="paragraph",
                        query=query,
                        expected_hash=str(row["hash"]),
                        expected_type="paragraph",
                        metadata={"source": str(row.get("source") or "")},
                    )
                )

            for row in relation_rows:
                if len([case for case in cases if case.expected_type == "relation"]) >= relation_target:
                    break
                query = make_relation_query(
                    str(row.get("subject") or ""),
                    str(row.get("predicate") or ""),
                    str(row.get("object") or ""),
                    seed_text=str(row.get("hash") or ""),
                )
                if not query:
                    continue
                cases.append(
                    EvaluationCase(
                        case_id=f"relation:{row['hash']}",
                        category="relation",
                        query=query,
                        expected_hash=str(row["hash"]),
                        expected_type="relation",
                    )
                )

            if not cases:
                raise ValueError("当前记忆库缺少可评估样本，至少需要若干段落或关系数据")

            randomizer.shuffle(cases)
            return cases[:sample_size]

        return await ctx.run_blocking(_collect)

    async def _evaluate_candidate(
        self,
        ctx: Any,
        profile: dict[str, Any],
        cases: list[EvaluationCase],
        *,
        top_k_eval: int,
        sparse_cache: dict[str, SparseBM25Index],
    ) -> tuple[dict[str, Any], float]:
        ranked_items_by_case: dict[str, list[dict[str, Any]]] = {}
        total_latency_ms = 0.0
        candidate_retriever = self._build_candidate_retriever(ctx, profile, sparse_cache=sparse_cache)
        plugin_config = copy.deepcopy(ctx.config)
        plugin_config["graph_store"] = ctx.graph_store
        plugin_config["metadata_store"] = ctx.metadata_store

        for case in cases:
            started = time.perf_counter()
            result = await SearchExecutionService.execute(
                retriever=candidate_retriever,
                threshold_filter=ctx.threshold_filter,
                plugin_config=plugin_config,
                request=SearchExecutionRequest(
                    caller="retrieval_tuning.evaluate",
                    query_type="search",
                    query=case.query,
                    top_k=top_k_eval,
                    use_threshold=True,
                    enable_ppr=bool(profile["retrieval"]["enable_ppr"]),
                ),
                enforce_chat_filter=False,
                reinforce_access=False,
            )
            total_latency_ms += (time.perf_counter() - started) * 1000.0
            if not result.success:
                raise ValueError(result.error or "候选 profile 评估失败")
            ranked_items_by_case[case.case_id] = [
                {
                    "hash": getattr(item, "hash_value", ""),
                    "type": getattr(item, "result_type", ""),
                    "score": float(getattr(item, "score", 0.0)),
                }
                for item in result.results
            ]

        metrics = calculate_metrics(cases, ranked_items_by_case, top_k=top_k_eval)
        avg_latency_ms = total_latency_ms / max(1, len(cases))
        return metrics, avg_latency_ms

    def _build_candidate_retriever(
        self,
        ctx: Any,
        profile: dict[str, Any],
        *,
        sparse_cache: dict[str, SparseBM25Index],
    ) -> DualPathRetriever:
        retrieval = extract_tuning_profile(profile)["retrieval"]
        sparse_cfg_raw = dict(retrieval.get("sparse") or {})
        fusion_cfg_raw = dict(retrieval.get("fusion") or {})
        sparse_key = toml.dumps({"sparse": sparse_cfg_raw})
        sparse_index = sparse_cache.get(sparse_key)
        if sparse_index is None:
            sparse_index = SparseBM25Index(
                metadata_store=ctx.metadata_store,
                config=SparseBM25Config(**sparse_cfg_raw),
            )
            sparse_cache[sparse_key] = sparse_index

        return DualPathRetriever(
            vector_store=ctx.vector_store,
            graph_store=ctx.graph_store,
            metadata_store=ctx.metadata_store,
            embedding_manager=ctx.embedding_manager,
            sparse_index=sparse_index,
            config=DualPathRetrieverConfig(
                top_k_paragraphs=int(retrieval["top_k_paragraphs"]),
                top_k_relations=int(retrieval["top_k_relations"]),
                top_k_final=int(retrieval["top_k_final"]),
                alpha=float(retrieval["alpha"]),
                enable_ppr=bool(retrieval["enable_ppr"]),
                ppr_alpha=float(ctx.get_config("retrieval.ppr_alpha", 0.85)),
                ppr_concurrency_limit=int(ctx.get_config("retrieval.ppr_concurrency_limit", 4)),
                enable_parallel=bool(retrieval["enable_parallel"]),
                retrieval_strategy=RetrievalStrategy.DUAL_PATH,
                debug=bool(ctx.get_config("advanced.debug", False)),
                sparse=SparseBM25Config(**sparse_cfg_raw),
                fusion=FusionConfig(**fusion_cfg_raw),
            ),
        )

    async def _get_task_record(self, ctx: Any, task_id: str) -> dict[str, Any] | None:
        return await ctx.run_blocking(ctx.metadata_store.get_async_task, str(task_id or "").strip())

    async def _update_task_record(
        self,
        ctx: Any,
        task_id: str,
        *,
        status: str | None = None,
        result_patch: dict[str, Any] | None = None,
        error_message: str | None = None,
        started_at: float | None = None,
        finished_at: float | None = None,
        cancel_requested: bool | None = None,
    ) -> dict[str, Any] | None:
        current = await self._get_task_record(ctx, task_id)
        merged_result: dict[str, Any] | None = None
        if result_patch is not None:
            merged_result = deep_merge_dict(dict((current or {}).get("result") or {}), result_patch)
        return await ctx.run_blocking(
            ctx.metadata_store.update_async_task,
            task_id=str(task_id or "").strip(),
            status=status,
            result=merged_result,
            error_message=error_message,
            started_at=started_at,
            finished_at=finished_at,
            cancel_requested=cancel_requested,
        )

    def _serialize_task_summary(self, record: dict[str, Any]) -> dict[str, Any]:
        payload = dict(record.get("payload") or {})
        result = dict(record.get("result") or {})
        return {
            "task_id": str(record.get("task_id") or ""),
            "status": str(record.get("status") or ""),
            "objective": str(payload.get("objective") or DEFAULT_OBJECTIVE),
            "intensity": str(payload.get("intensity") or DEFAULT_INTENSITY),
            "progress": float(result.get("progress", 0.0) or 0.0),
            "rounds_done": int(result.get("rounds_done", 0) or 0),
            "rounds_total": int(result.get("rounds_total", payload.get("rounds_total", 0)) or 0),
            "best_score": float(result.get("best_score", 0.0) or 0.0),
            "created_at": record.get("created_at"),
            "updated_at": record.get("updated_at"),
        }

    def _serialize_task_detail(self, record: dict[str, Any]) -> dict[str, Any]:
        summary = self._serialize_task_summary(record)
        payload = dict(record.get("payload") or {})
        result = dict(record.get("result") or {})
        return {
            **summary,
            "sample_size": int(payload.get("sample_size", DEFAULT_SAMPLE_SIZE) or DEFAULT_SAMPLE_SIZE),
            "top_k_eval": int(payload.get("top_k_eval", DEFAULT_TOP_K_EVAL) or DEFAULT_TOP_K_EVAL),
            "llm_enabled": bool(payload.get("llm_enabled", True)),
            "cancel_requested": bool(record.get("cancel_requested", False)),
            "started_at": record.get("started_at"),
            "finished_at": record.get("finished_at"),
            "error_message": str(record.get("error_message") or ""),
            "baseline_metrics": dict(result.get("baseline_metrics") or {}),
            "best_metrics": dict(result.get("best_metrics") or {}),
            "best_profile": copy.deepcopy(result.get("best_profile") or {}),
        }
