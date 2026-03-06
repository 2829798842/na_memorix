"""Maintenance/runtime loops extracted from plugin.py."""

from __future__ import annotations

import asyncio
import datetime
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

from src.common.logger import get_logger

from ..utils.io import atomic_write

logger = get_logger("A_Memorix.MaintenanceTasks")

async def scheduled_import_loop(plugin):
    """定时总结导入循环"""
    import asyncio
    import datetime
    
    logger.info("A_Memorix 定时总结导入任务已启动")
    
    # 记录上次检查的时间，用于跨越时间点检测
    last_check_now = datetime.datetime.now()
    
    while True:
        try:
            # 每分钟检查一次
            await asyncio.sleep(60)
            
            # 检查总开关和定时开关
            if not plugin.get_config("summarization.enabled", True) or not plugin.get_config("schedule.enabled", True):
                continue
            
            now = datetime.datetime.now()
            import_times = plugin.get_config("schedule.import_times", ["04:00"])
            
            for t_str in import_times:
                try:
                    # 解析配置的时间点 (HH:MM)
                    h, m = map(int, t_str.split(":"))
                    # 构造今天的该时间点
                    target_time = now.replace(hour=h, minute=m, second=0, microsecond=0)
                    
                    # 如果当前时间刚跨过目标时间点
                    if last_check_now < target_time <= now:
                        logger.info(f"触发 A_Memorix 定时导入任务: {t_str}")
                        await plugin._perform_bulk_summary_import()
                except (ValueError, Exception) as e:
                    logger.error(f"解析定时配置 '{t_str}' 出错: {e}")
            
            last_check_now = now
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"定时导入循环发生未知错误: {e}")
            await asyncio.sleep(60)

async def person_profile_refresh_loop(plugin):
    """按需刷新人物画像快照（仅针对已开启范围内活跃人物）。"""
    logger.info("A_Memorix 人物画像定时刷新任务已启动")
    try:
        while True:
            interval_minutes = int(plugin.get_config("person_profile.refresh_interval_minutes", 30))
            await asyncio.sleep(max(60, interval_minutes * 60))

            if not bool(plugin.get_config("person_profile.enabled", True)):
                continue

            await plugin._refresh_person_profiles_for_enabled_switches()
    except asyncio.CancelledError:
        logger.info("人物画像定时刷新任务已取消")
    except Exception as e:
        logger.error(f"人物画像定时刷新循环异常: {e}")

async def refresh_person_profiles_for_enabled_switches(plugin):
    """刷新已开启范围内活跃人物画像。"""
    if plugin.metadata_store is None:
        return

    active_window_hours = float(plugin.get_config("person_profile.active_window_hours", 72.0))
    active_after = time.time() - max(0.0, active_window_hours) * 3600.0
    max_refresh = int(plugin.get_config("person_profile.max_refresh_per_cycle", 50))
    top_k_evidence = int(plugin.get_config("person_profile.top_k_evidence", 12))
    ttl_minutes = float(plugin.get_config("person_profile.profile_ttl_minutes", 360.0))
    ttl_seconds = max(60.0, ttl_minutes * 60.0)

    try:
        person_ids = plugin.metadata_store.get_active_person_ids_for_enabled_switches(
            active_after=active_after,
            limit=max_refresh,
        )
    except Exception as e:
        logger.warning(f"获取待刷新人物集合失败: {e}")
        return

    if not person_ids:
        logger.debug("人物画像刷新跳过：暂无已开启范围内活跃人物")
        return

    from ..utils.person_profile_service import PersonProfileService

    service = PersonProfileService(
        metadata_store=plugin.metadata_store,
        graph_store=plugin.graph_store,
        vector_store=plugin.vector_store,
        embedding_manager=plugin.embedding_manager,
        sparse_index=plugin.sparse_index,
        plugin_config=plugin.config,
    )

    refreshed = 0
    for person_id in person_ids:
        try:
            result = await service.query_person_profile(
                person_id=person_id,
                top_k=top_k_evidence,
                ttl_seconds=ttl_seconds,
                force_refresh=True,
                source_note="schedule_refresh",
            )
            if result.get("success"):
                refreshed += 1
        except Exception as e:
            logger.warning(f"刷新人物画像失败: person_id={person_id}, err={e}")

    logger.info(f"人物画像按需刷新完成: refreshed={refreshed}, candidates={len(person_ids)}")

async def perform_bulk_summary_import(plugin):
    """为所有活跃聊天执行总结导入"""
    import asyncio
    from ..utils.summary_importer import SummaryImporter
    from src.common.database.database_model import ChatStreams
    
    # 实例化导入器
    importer = SummaryImporter(
        vector_store=plugin.vector_store,
        graph_store=plugin.graph_store,
        metadata_store=plugin.metadata_store,
        embedding_manager=plugin.embedding_manager,
        plugin_config=plugin.config
    )
    
    # 获取所有已知的聊天流 ID, Group ID 和 User ID
    def _get_all_streams():
        try:
            # 获取 stream_id, group_id, user_id
            query = ChatStreams.select(ChatStreams.stream_id, ChatStreams.group_id, ChatStreams.user_id)
            return [{
                "stream_id": s.stream_id, 
                "group_id": s.group_id,
                "user_id": s.user_id
            } for s in query]
        except Exception as e:
            logger.error(f"获取聊天流列表失败: {e}")
            return []
        
    streams = await asyncio.to_thread(_get_all_streams)
    
    if not streams:
        logger.info("未发现可总结的聊天流")
        return
        
    logger.info(f"开始为 {len(streams)} 个聊天流执行批量总结检查...")
    
    success_count = 0
    skipped_count = 0
    
    for s in streams:
        s_id = s["stream_id"]
        g_id = s.get("group_id")
        u_id = s.get("user_id")
        
        # 过滤检查
        if not plugin.is_chat_enabled(stream_id=s_id, group_id=g_id, user_id=u_id):
            skipped_count += 1
            continue
            
        try:
            # 执行总结导入 (SummaryImporter 内部会处理无新消息的情况)
            success, msg = await importer.import_from_stream(s_id)
            if success:
                success_count += 1
                logger.info(f"聊天流 {s_id} 自动总结成功")
        except Exception as e:
            logger.error(f"处理聊天流 {s_id} 自动总结时出错: {e}")
            
    logger.info(f"批量总结任务完成，成功: {success_count}，跳过: {skipped_count}")

async def relation_vector_backfill_loop(plugin):
    """后台分批回填关系向量。"""
    logger.info("A_Memorix 关系向量回填任务已启动")
    try:
        while True:
            cfg = plugin.get_config("retrieval.relation_vectorization", {}) or {}
            if not isinstance(cfg, dict):
                cfg = {}
            interval_seconds = max(1, int(cfg.get("backfill_interval_seconds", 5)))
            await asyncio.sleep(interval_seconds)

            if not bool(cfg.get("enabled", False)) or not bool(cfg.get("backfill_enabled", False)):
                continue
            if not plugin.relation_write_service or not plugin.metadata_store:
                continue
            plugin._cleanup_orphan_relation_vectors(limit=200)

            batch_size = max(1, int(cfg.get("backfill_batch_size", 200)))
            max_retry = max(1, int(cfg.get("max_retry", 3)))

            start = time.perf_counter()
            rows = plugin.metadata_store.list_relations_by_vector_state(
                states=["none", "failed", "pending"],
                limit=batch_size,
                max_retry=max_retry,
            )
            if not rows:
                continue

            success = 0
            failed = 0
            skipped = 0
            for row in rows:
                res = await plugin.relation_write_service.ensure_relation_vector(
                    hash_value=str(row["hash"]),
                    subject=str(row.get("subject", "")),
                    predicate=str(row.get("predicate", "")),
                    obj=str(row.get("object", "")),
                )
                if res.vector_state == "ready":
                    if res.vector_written:
                        success += 1
                    else:
                        skipped += 1
                else:
                    failed += 1

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            state_stats = plugin.metadata_store.count_relations_by_vector_state()
            remaining = (
                int(state_stats.get("none", 0))
                + int(state_stats.get("failed", 0))
                + int(state_stats.get("pending", 0))
            )
            logger.info(
                "metric.relation_backfill_batch_latency=%.2f metric.relation_backfill_batch_latency_ms=%.2f processed=%s success=%s failed=%s skipped=%s remaining=%s",
                elapsed_ms,
                elapsed_ms,
                len(rows),
                success,
                failed,
                skipped,
                remaining,
            )
    except asyncio.CancelledError:
        logger.info("关系向量回填任务已取消")
    except Exception as e:
        logger.error(f"关系向量回填循环发生错误: {e}")

async def episode_generation_loop(plugin):
    """Episode 异步生成循环。"""
    logger.info("A_Memorix Episode 生成任务已启动")
    try:
        while True:
            interval_seconds = max(1, int(plugin.get_config("episode.generation_interval_seconds", 30)))
            await asyncio.sleep(interval_seconds)

            if not bool(plugin.get_config("episode.enabled", True)):
                continue
            if not bool(plugin.get_config("episode.generation_enabled", True)):
                continue
            if plugin.metadata_store is None:
                continue

            batch_size = max(1, int(plugin.get_config("episode.generation_batch_size", 20)))
            max_retry = max(0, int(plugin.get_config("episode.max_retry", 3)))

            start = time.perf_counter()
            pending_rows = plugin.metadata_store.fetch_episode_source_rebuild_batch(
                limit=batch_size,
                max_retry=max_retry,
            )
            if not pending_rows:
                continue

            from ..utils.episode_service import EpisodeService

            service = EpisodeService(
                metadata_store=plugin.metadata_store,
                plugin_config=getattr(plugin, "config", {}) or {},
            )

            started_sources: List[str] = []
            done_sources: List[str] = []
            failed_sources: Dict[str, str] = {}
            episode_count = 0
            fallback_count = 0
            group_count = 0
            paragraph_count = 0

            for row in pending_rows:
                source = str(row.get("source", "") or "").strip()
                if not source:
                    continue
                requested_at = row.get("requested_at")
                try:
                    requested_at = float(requested_at) if requested_at is not None else None
                except Exception:
                    requested_at = None

                started = plugin.metadata_store.mark_episode_source_running(
                    source,
                    requested_at=requested_at,
                )
                if not started:
                    continue

                started_sources.append(source)
                try:
                    result = await service.rebuild_source(source)
                    episode_count += int(result.get("episode_count") or 0)
                    fallback_count += int(result.get("fallback_count") or 0)
                    group_count += int(result.get("group_count") or 0)
                    paragraph_count += int(result.get("paragraph_count") or 0)
                    plugin.metadata_store.mark_episode_source_done(
                        source,
                        requested_at=requested_at,
                    )
                    done_sources.append(source)
                except Exception as e:
                    err = str(e)[:500]
                    failed_sources[source] = err
                    plugin.metadata_store.mark_episode_source_failed(
                        source,
                        err,
                        requested_at=requested_at,
                    )

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            processed_count = len(done_sources) + len(failed_sources)
            avg_ms = elapsed_ms / processed_count if processed_count > 0 else 0.0
            logger.info(
                "metric.episode_generation_batch_latency_ms=%.2f metric.episode_generation_avg_latency_ms=%.2f "
                "processed=%s done=%s failed=%s started=%s episodes=%s fallback=%s groups=%s paragraphs=%s",
                elapsed_ms,
                avg_ms,
                processed_count,
                len(done_sources),
                len(failed_sources),
                len(started_sources),
                episode_count,
                fallback_count,
                group_count,
                paragraph_count,
            )
    except asyncio.CancelledError:
        logger.info("Episode 生成任务已取消")
    except Exception as e:
        logger.error(f"Episode 生成循环发生错误: {e}")

def cleanup_orphan_relation_vectors(plugin, limit: int = 200) -> int:
    """清理关系孤儿向量（deleted_relations 存在且 relations 不存在）。"""
    if not plugin.metadata_store or not plugin.vector_store:
        return 0
    try:
        candidate_hashes = plugin.metadata_store.get_orphan_deleted_relation_hashes(
            limit=max(1, int(limit)),
        )
    except Exception as e:
        logger.debug(f"孤儿向量清理跳过（查询失败）: {e}")
        return 0
    if not candidate_hashes:
        return 0
    orphan_hashes = [h for h in candidate_hashes if h in plugin.vector_store]
    if not orphan_hashes:
        return 0

    deleted = plugin.vector_store.delete(orphan_hashes)
    if deleted > 0:
        logger.info(
            "metric.orphan_vector_cleanup_count=%s scanned=%s",
            deleted,
            len(candidate_hashes),
        )
    return int(deleted)

def get_relation_vector_stats(plugin) -> Dict[str, Any]:
    """返回关系向量状态与覆盖统计。"""
    if not plugin.metadata_store:
        return {}

    state_stats = plugin.metadata_store.count_relations_by_vector_state()
    relation_hashes = set(plugin.metadata_store.list_hashes("relations"))
    paragraph_hashes = set(plugin.metadata_store.list_hashes("paragraphs"))
    entity_hashes = set(plugin.metadata_store.list_hashes("entities"))

    live_vector_hashes = set()
    orphan_count = 0
    relation_vector_hits = 0
    ready_but_missing_vector = 0
    if plugin.vector_store:
        known_hashes = set(getattr(plugin.vector_store, "_known_hashes", set()))
        live_vector_hashes = {h for h in known_hashes if h in plugin.vector_store}
        relation_vector_hits = len(relation_hashes & live_vector_hashes)
        orphan_count = len(live_vector_hashes - relation_hashes - paragraph_hashes - entity_hashes)

        if relation_hashes:
            ready_rows = plugin.metadata_store.list_relations_by_vector_state(
                states=["ready"],
                limit=max(1, len(relation_hashes)),
            )
            ready_but_missing_vector = sum(
                1 for row in ready_rows if str(row.get("hash")) not in live_vector_hashes
            )

    relation_total = max(0, int(state_stats.get("total", len(relation_hashes))))
    ready_total = max(0, int(state_stats.get("ready", 0)))
    ready_coverage = (ready_total / relation_total) if relation_total > 0 else 0.0
    vector_coverage = (
        relation_vector_hits / len(relation_hashes)
        if relation_hashes
        else 0.0
    )

    return {
        "states": state_stats,
        "orphan_vectors": orphan_count,
        "relation_total": len(relation_hashes),
        "relation_vector_hits": relation_vector_hits,
        "relation_ready_coverage": ready_coverage,
        "relation_vector_coverage": vector_coverage,
        "ready_but_missing_vector": ready_but_missing_vector,
    }

async def save_all(plugin):
    """统一保存所有数据 (Unified Persistence)"""
    if not plugin.vector_store or not plugin.graph_store:
        return

    commit_id = str(uuid.uuid4())
    logger.info(f"开始统一保存 (Commit ID: {commit_id})...")
    
    try:
        # 并行保存各组件
        # VectorStore 和 GraphStore 的 save 方法现在已经是线程安全的(或使用原子写)
        # 但为了减少IO阻塞，最好在线程池运行
        await asyncio.gather(
            asyncio.to_thread(plugin.vector_store.save),
            asyncio.to_thread(plugin.graph_store.save)
            # MetadataStore 是 SQLite，通常实时写入，无需显式 save
        )
        
        # 更新 Manifest，标志着一次完整的持久化状态
        await plugin._update_manifest(commit_id)
        logger.info(f"统一保存完成 (Commit ID: {commit_id})")
        
    except Exception as e:
        logger.error(f"统一保存失败: {e}")

async def update_manifest(plugin, commit_id: str):
    """更新持久化清单"""
    manifest = {
        "last_commit_id": commit_id,
        "timestamp": time.time(),
        "iso_timestamp": datetime.datetime.now().isoformat(),
        "version": plugin.plugin_version
    }
    
    data_dir = Path(plugin.get_config("storage.data_dir", "./plugins/A_memorix/data"))
    manifest_path = data_dir / "persistence_manifest.json"
    
    try:
        # 使用原子写入更新 Manifest
        with atomic_write(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    except Exception as e:
        logger.error(f"更新 Manifest 失败: {e}")

async def auto_save_loop(plugin):
    """自动保存循环"""
    logger.info("自动保存任务已启动")
    try:
        while True:
            # 获取配置的间隔时间 (分钟)
            interval = plugin.get_config("advanced.auto_save_interval_minutes", 5)
            if interval <= 0:
                interval = 5
            
            await asyncio.sleep(interval * 60)
            
            if plugin.get_config("advanced.enable_auto_save", True):
                await plugin.save_all()
                
    except asyncio.CancelledError:
        logger.info("自动保存任务已取消")
    except Exception as e:
        logger.error(f"自动保存循环发生错误: {e}")

async def reinforce_access(plugin, relation_hashes: List[str]):
    """
    触发记忆强化 (Thread-safe push to buffer)
    """
    if not plugin.get_config("memory.enable_auto_reinforce", True):
        return
        
    async with plugin._memory_lock:
        plugin.reinforce_buffer.update(relation_hashes)

async def memory_maintenance_loop(plugin):
    """
    记忆维护循环 (Decay, Reinforce, Freeze, Prune)
    """
    logger.info("A_Memorix 记忆维护循环已启动 (V5)")
    
    while True:
        try:
            # 获取间隔 (默认1小时)
            interval_hours = plugin.get_config("memory.base_decay_interval_hours", 1.0)
            interval_seconds = max(60, int(interval_hours * 3600))
            
            await asyncio.sleep(interval_seconds)
            
            if not plugin.metadata_store or not plugin.graph_store:
                continue
                
            # Master Switch Check
            if not plugin.get_config("memory.enabled", True):
                continue
                
            async with plugin._memory_lock:
                # 1. Process Reinforce Buffer
                current_buffer = list(plugin.reinforce_buffer)
                plugin.reinforce_buffer.clear()
                
                if current_buffer:
                    await plugin._process_reinforce_batch(current_buffer)
                    
                # 2. 全局衰减 (Global Decay)
                half_life = plugin.get_config("memory.half_life_hours", 24.0)
                if half_life > 0:
                    # factor = (1/2) ^ (dt / half_life)
                    factor = 0.5 ** (interval_hours / half_life)
                    # 保护地板值由 prune 逻辑处理，decay 只负责乘法
                    plugin.graph_store.decay(factor)
                    logger.debug(f"执行记忆衰减: factor={factor:.4f}")
                    
                # 3. 冷冻与修剪 (Freeze & Prune) (检查候选记忆)
                await plugin._process_freeze_and_prune()
                
                # 4. 孤儿节点回收 (Orphan GC) (标记与清除)
                await plugin._orphan_gc_phase()

        except asyncio.CancelledError:
            logger.info("记忆维护循环已取消")
            break
        except Exception as e:
            logger.error(f"记忆维护循环发生错误: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(60)

async def process_reinforce_batch(plugin, hashes: List[str]):
    """处理强化批次"""
    try:
        # 获取当前状态
        status_map = plugin.metadata_store.get_relation_status_batch(hashes)
        
        now = datetime.datetime.now().timestamp()
        cooldown = plugin.get_config("memory.reinforce_cooldown_hours", 1.0) * 3600
        max_weight = plugin.get_config("memory.max_weight", 10.0)
        revive_boost = plugin.get_config("memory.revive_boost_weight", 0.5)
        auto_protect = plugin.get_config("memory.auto_protect_ttl_hours", 24.0) * 3600
        
        hashes_to_update = []
        hashes_to_revive = []
        updates_protect = []
        
        relation_info = plugin.metadata_store.get_relations_subject_object_map(hashes)
        
        for h in hashes:
            if h not in status_map: continue
            s = status_map[h]
            info = relation_info.get(h)
            if not info: continue
            
            src, tgt = info
            
            # 冷却检查 (Cooldown Check)
            last_re = s.get("last_reinforced") or 0
            if (now - last_re) < cooldown and not s["is_inactive"]:
                continue # 如果仍在冷却中且处于活跃状态，则跳过
                
            # 计算增量权重 (Calculate Delta)
            current_w = s["weight"]
            # Delta = amount * (1 - w/max)
            delta = 1.0 * (1.0 - (current_w / max_weight))
            if delta < 0: delta = 0
            
            # 逻辑:
            # 1. 更新图权重 (Update Graph Weight)
            plugin.graph_store.update_edge_weight(src, tgt, delta, max_weight=max_weight)
            
            # 2. 元数据更新 (Metadata Updates)
            # 如果是不活跃状态，复活需要进行显式处理？
            # 实际上 update_edge_weight 会添加缺失的边，但我们需要更新元数据标志。
            if s["is_inactive"]:
                hashes_to_revive.append(h)
            else:
                hashes_to_update.append(h)
                
        # 批量更新元数据 (Batch update Metadata)
        if hashes_to_revive:
            plugin.metadata_store.mark_relations_active(hashes_to_revive, boost_weight=revive_boost)
            plugin.metadata_store.update_relations_protection(
                hashes_to_revive, 
                protected_until=now + auto_protect, 
                last_reinforced=now
            )
            logger.info(f"复活记忆: {len(hashes_to_revive)} 条")
            
        if hashes_to_update:
            plugin.metadata_store.update_relations_protection(
                hashes_to_update, 
                protected_until=now + auto_protect, 
                last_reinforced=now
            )
            
    except Exception as e:
        logger.error(f"处理强化批次失败: {e}")

async def process_freeze_and_prune(plugin):
    """处理冷冻与修剪"""
    try:
        prune_threshold = plugin.get_config("memory.prune_threshold", 0.1)
        freeze_duration = plugin.get_config("memory.freeze_duration_hours", 24.0) * 3600
        now = datetime.datetime.now().timestamp()
        
        # 1. 冷冻阶段 (FREEZE PASS) (不活跃逻辑)
        # 策略：如果一条边权重过低，且其下所有关系均无保护，则冻结该边。
        # "冻结" = 在元数据中标记为不活跃 + 从邻接矩阵中移除 (但保留在 Map 中)。
        # 只有当边被移除，该记忆才不会参与 PageRank，符合 "不活跃" 定义。
        
        # 从图中获取低权重边 (邻居矩阵)
        low_edges = plugin.graph_store.get_low_weight_edges(prune_threshold)
        
        hashes_to_freeze = [] # 元数据更新列表
        edges_to_deactivate = [] # 图邻域更新列表
        
        for src, tgt in low_edges:
            associated_hashes = plugin.graph_store.get_relation_hashes_for_edge(src, tgt)
            if not associated_hashes:
                continue

            # 检查保护状态 (Check Protection)
            statuses = plugin.metadata_store.get_relation_status_batch(list(associated_hashes))

            is_edge_protected = False
            current_edge_hashes = []

            for h, st in statuses.items():
                # 保护规则: 已置顶 (Pinned) 或 TTL 有效
                if st["is_pinned"] or (st["protected_until"] or 0) > now:
                    is_edge_protected = True
                    break
                # 如果已是不活跃状态则跳过 (虽然在已停用的低权重边中不应出现，但为了安全进行检查)
                if st["is_inactive"]:
                    pass
                current_edge_hashes.append(h)

            if not is_edge_protected and current_edge_hashes:
                # Freeze the whole edge
                hashes_to_freeze.extend(current_edge_hashes)
                edges_to_deactivate.append((src, tgt))
                    
        if hashes_to_freeze:
            plugin.metadata_store.mark_relations_inactive(hashes_to_freeze, inactive_since=now)
            # 仅从矩阵中移除 (保留在 Map 中)
            plugin.graph_store.deactivate_edges(edges_to_deactivate)
            logger.info(f"冷冻记忆: {len(hashes_to_freeze)} 条关系, 冻结 {len(edges_to_deactivate)} 条边")

        # 2. 修剪阶段 (PRUNE PASS) (删除逻辑)
        # 从元数据和 Map 中移除过期的不活跃关系。
        cutoff = now - freeze_duration
        expired_hashes = plugin.metadata_store.get_prune_candidates(cutoff)
        
        if expired_hashes:
            ops_to_prune = [] # List[(src, tgt, hash)] for GraphStore
            actually_deleted_hashes = []

            relation_info = plugin.metadata_store.get_relations_subject_object_map(expired_hashes)
            for h, pair in relation_info.items():
                s, o = pair
                # We need to remove this specific hash from map
                ops_to_prune.append((s, o, h))
                actually_deleted_hashes.append(h)
            
            # Update GraphStore (Map -> if empty -> Matrix)
            # Note: Matrix entry should be already gone via Freeze, but prune_relation_hashes handles that safety.
            if ops_to_prune:
                plugin.graph_store.prune_relation_hashes(ops_to_prune)
                
            # 从元数据中备份并删除 (Backup and Delete in Metadata)
            count = plugin.metadata_store.backup_and_delete_relations(actually_deleted_hashes)
            # 同步删除关系向量，避免孤儿向量污染召回
            if plugin.vector_store and actually_deleted_hashes:
                deleted_vec = plugin.vector_store.delete(actually_deleted_hashes)
                logger.info(
                    "metric.orphan_vector_cleanup_count=%s prune_hashes=%s",
                    deleted_vec,
                    len(actually_deleted_hashes),
                )
            logger.info(f"物理修剪: {count} 条记忆 (已清理映射)")

    except Exception as e:
        logger.error(f"处理冷冻与修剪失败: {e}")

async def orphan_gc_phase(plugin):
    """
    孤儿节点回收阶段 (Orphan GC Phase)
    策略: Mark & Sweep (标记-清除)
    逻辑:
    1. Mark: 找出孤儿(Active Degree=0 & 未冻结)，同时满足 Retention 要求的，标记为 is_deleted=1.
    2. Sweep: 找出 is_deleted=1 且 deleted_at < now - grace 的，物理删除.
    """
    # Feature Toggle
    orphan_config = plugin.get_config("memory.orphan", {})
    if not orphan_config.get("enable_soft_delete", True):
        return

    try:
        logger.debug("开始孤儿节点回收阶段 (GC Phase)...")
        
        # Configs
        entity_retention = orphan_config.get("entity_retention_days", 7.0) * 86400
        para_retention = orphan_config.get("paragraph_retention_days", 7.0) * 86400
        grace_period = orphan_config.get("sweep_grace_hours", 24.0) * 3600
        
        # ==========================================================
        # 1. MARK PHASE (标记)
        # ==========================================================
        
        # 1.1 标记实体 (Mark Entities)
        # 从图中获取孤儿候选者 (活跃但孤立)
        # 注意: include_inactive=True (默认) 会排除掉那些虽然度为 0 但参与了冻结边的节点 -> 保护冻结节点不被删除
        isolated_candidates = plugin.graph_store.get_isolated_nodes(include_inactive=True)
        
        if isolated_candidates:
            # 通过元数据过滤 (保留时长与引用检查)
            final_entity_candidates = plugin.metadata_store.get_entity_gc_candidates(
                isolated_candidates, 
                retention_seconds=entity_retention
            )
            
            if final_entity_candidates:
                cnt = plugin.metadata_store.mark_as_deleted(final_entity_candidates, "entity")
                if cnt > 0:
                    logger.info(f"[GC-Mark] 标记删除实体: {cnt} 个")

        # 1.2 标记段落 (Mark Paragraphs)
        # 通过元数据过滤 (保留时长 & 无关系 & 无实体)
        para_candidates = plugin.metadata_store.get_paragraph_gc_candidates(retention_seconds=para_retention)
        if para_candidates:
            cnt = plugin.metadata_store.mark_as_deleted(para_candidates, "paragraph")
            if cnt > 0:
                logger.info(f"[GC-Mark] 标记删除段落: {cnt} 个")
                
        # ==========================================================
        # 2. SWEEP PHASE (物理清理)
        # ==========================================================
        
        # 2.1 清理段落 (Sweep Paragraphs)
        dead_paragraphs_tuples = plugin.metadata_store.sweep_deleted_items("paragraph", grace_period)
        if dead_paragraphs_tuples:
            dead_para_hashes = [t[0] for t in dead_paragraphs_tuples]
            count = plugin.metadata_store.physically_delete_paragraphs(dead_para_hashes)
            if count > 0:
                logger.info(f"[GC-Sweep] 物理删除段落: {count} 个")

        # 2.2 清理实体 (Sweep Entities)
        dead_entities_tuples = plugin.metadata_store.sweep_deleted_items("entity", grace_period)
        if dead_entities_tuples:
            dead_entity_hashes = [t[0] for t in dead_entities_tuples]
            dead_entity_names = [t[1] for t in dead_entities_tuples]
            
            # 关键顺序：先从图存储中删除 (内存/矩阵)，然后再从元数据中删除。
            
            # 1. 图存储删除 (需要名称) (GraphStore Delete)
            plugin.graph_store.delete_nodes(dead_entity_names)
            
            # 2. 元数据存储删除 (需要哈希) (MetadataStore Delete)
            count = plugin.metadata_store.physically_delete_entities(dead_entity_hashes)
            if count > 0:
               logger.info(f"[GC-Sweep] 物理删除实体: {count} 个")

    except Exception as e:
        logger.error(f"孤儿节点回收失败: {e}")
        import traceback
        traceback.print_exc()

