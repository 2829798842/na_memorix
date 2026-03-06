"""
记忆维护指令组件
"""

import time
import datetime
from typing import Tuple, Optional, List, Dict, Any

from src.common.logger import get_logger
from src.plugin_system.base.base_command import BaseCommand
from src.chat.message_receive.message import MessageRecv

from ...core import (
    DualPathRetriever,
    RetrievalStrategy,
    DualPathRetrieverConfig,
    SparseBM25Config,
    FusionConfig,
    GraphRelationRecallConfig,
    RelationIntentConfig,
)

logger = get_logger("A_Memorix.MemoryCommand")

class MemoryMaintenanceCommand(BaseCommand):
    """记忆维护指令
    
    支持:
    - /memory status: 查看记忆健康状态
    - /memory protect <hash|query> [hours]: 保护记忆 (Pin if hours=0, else TTL)
    - /memory reinforce <hash|query>: 手动强化记忆 (绕过冷却)
    - /memory restore <hash>: 从回收站恢复记忆
    """
    
    command_name = "memory"
    command_description = "记忆系统维护指令 (Status, Protect, Reinforce, Restore)"
    command_pattern = r"^\/memory(?:\s+(?P<action>\w+))?(?:\s+(?P<args>.+))?$"
    
    def __init__(self, message: MessageRecv, plugin_config: Optional[dict] = None):
        super().__init__(message, plugin_config)
        self.retriever: Optional[DualPathRetriever] = None
        self._initialize_stores()
        self._initialize_retriever()
        
    def _initialize_stores(self):
        # 类似 QueryCommand 获取实例
        self.vector_store = self.plugin_config.get("vector_store")
        self.graph_store = self.plugin_config.get("graph_store")
        self.metadata_store = self.plugin_config.get("metadata_store")
        self.relation_write_service = self.plugin_config.get("relation_write_service")
        self.sparse_index = self.plugin_config.get("sparse_index")
        
        if not all([self.vector_store, self.graph_store, self.metadata_store]):
            from ...plugin import A_MemorixPlugin
            instances = A_MemorixPlugin.get_storage_instances()
            if instances:
                self.graph_store = self.graph_store or instances.get("graph_store")
                self.metadata_store = self.metadata_store or instances.get("metadata_store")
                self.relation_write_service = self.relation_write_service or instances.get("relation_write_service")
                self.sparse_index = self.sparse_index or instances.get("sparse_index")
        
        self.embedding_manager = self.plugin_config.get("embedding_manager")
        if not self.embedding_manager:
            from ...plugin import A_MemorixPlugin
            instances = A_MemorixPlugin.get_storage_instances()
            if instances:
                self.embedding_manager = instances.get("embedding_manager")

    def _initialize_retriever(self):
        """初始化检索器用于语义解析"""
        try:
            if not all([self.vector_store, self.graph_store, self.metadata_store, self.embedding_manager]):
                return

            sparse_cfg_raw = self.get_config("retrieval.sparse", {}) or {}
            if not isinstance(sparse_cfg_raw, dict):
                sparse_cfg_raw = {}
            fusion_cfg_raw = self.get_config("retrieval.fusion", {}) or {}
            if not isinstance(fusion_cfg_raw, dict):
                fusion_cfg_raw = {}
            graph_recall_cfg_raw = self.get_config("retrieval.search.graph_recall", {}) or {}
            if not isinstance(graph_recall_cfg_raw, dict):
                graph_recall_cfg_raw = {}
            relation_intent_cfg_raw = self.get_config("retrieval.search.relation_intent", {}) or {}
            if not isinstance(relation_intent_cfg_raw, dict):
                relation_intent_cfg_raw = {}

            try:
                sparse_cfg = SparseBM25Config(**sparse_cfg_raw)
            except Exception:
                sparse_cfg = SparseBM25Config()
            try:
                fusion_cfg = FusionConfig(**fusion_cfg_raw)
            except Exception:
                fusion_cfg = FusionConfig()
            try:
                graph_recall_cfg = GraphRelationRecallConfig(**graph_recall_cfg_raw)
            except Exception:
                graph_recall_cfg = GraphRelationRecallConfig()
            try:
                relation_intent_cfg = RelationIntentConfig(**relation_intent_cfg_raw)
            except Exception:
                relation_intent_cfg = RelationIntentConfig()

            config = DualPathRetrieverConfig(
                retrieval_strategy=RetrievalStrategy.DUAL_PATH,
                top_k_paragraphs=self.get_config("retrieval.top_k_paragraphs", 20),
                top_k_relations=self.get_config("retrieval.top_k_relations", 10),
                top_k_final=10,
                alpha=self.get_config("retrieval.alpha", 0.5),
                enable_ppr=bool(self.get_config("retrieval.enable_ppr", True)),
                ppr_alpha=self.get_config("retrieval.ppr_alpha", 0.85),
                ppr_concurrency_limit=self.get_config("retrieval.ppr_concurrency_limit", 4),
                enable_parallel=self.get_config("retrieval.enable_parallel", True),
                sparse=sparse_cfg,
                fusion=fusion_cfg,
                graph_recall=graph_recall_cfg,
                relation_intent=relation_intent_cfg,
            )
            
            self.retriever = DualPathRetriever(
                vector_store=self.vector_store,
                graph_store=self.graph_store,
                metadata_store=self.metadata_store,
                embedding_manager=self.embedding_manager,
                sparse_index=self.sparse_index,
                config=config,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize retriever for MemoryCommand: {e}")

    @staticmethod
    def _is_hash_like(value: str) -> bool:
        v = (value or "").strip()
        return len(v) in (32, 64) and all(c in "0123456789abcdefABCDEF" for c in v)

    async def execute(self) -> Tuple[bool, Optional[str], int]:
        if not self.metadata_store or not self.graph_store:
            return False, "❌ 存储组件未初始化", 1
            
        action = self.matched_groups.get("action", "status")
        args = self.matched_groups.get("args", "")
        
        # 默认 status
        if not action: 
            action = "status"
            
        action = action.lower()
        
        try:
            result = (False, None, 1)
            
            if action == "status":
                result = await self._handle_status()
            elif action == "protect":
                result = await self._handle_protect(args)
            elif action == "reinforce":
                result = await self._handle_reinforce(args)
            elif action == "restore":
                result = await self._handle_restore(args)
            elif action == "help":
                result = (True, self._get_help(), 1)
            else:
                result = (True, self._get_help(), 1)
                
            # 显式发送消息以确保用户可见
            if result[1]:
                await self.send_text(result[1])
                
            return result
        except Exception as e:
            logger.error(f"Command execution error: {e}")
            import traceback
            traceback.print_exc()
            return False, f"❌ 执行出错: {e}", 1

    async def _handle_status(self) -> Tuple[bool, str, int]:
        """查看记忆状态"""
        now = datetime.datetime.now().timestamp()
        summary = self.metadata_store.get_memory_status_summary(now)
        active_count = summary["active_count"]
        inactive_count = summary["inactive_count"]
        deleted_count = summary["deleted_count"]
        pinned_count = summary["pinned_count"]
        temp_protected_count = summary["temp_protected_count"]
        
        # Get Configs
        mem_conf = self.get_config("memory", {})
        half_life = mem_conf.get("half_life_hours", 24.0)
        prune_thresh = mem_conf.get("prune_threshold", 0.1)
        
        lines = [
            "🧠 **记忆系统状态 (System Status)**",
            "",
            f"📊 **统计**: ",
            f"  - 🟢 活跃记忆 (Active): {active_count}",
            f"  - 🔵 冷冻记忆 (Inactive): {inactive_count}",
            f"  - 🛡️ 受保护 (Pinned/TTL): {pinned_count} / {temp_protected_count}",
            f"  - 🗑️ 回收站 (Deleted): {deleted_count}",
            "",
            f"⚙️ **参数**: ",
            f"  - 半衰期: {half_life}h",
            f"  - 冷冻阈值: {prune_thresh}",
        ]
        return True, "\n".join(lines), 1

    async def _handle_protect(self, args: str) -> Tuple[bool, str, int]:
        """保护记忆 /memory protect query [hours]"""
        if not args:
            return False, "用法: /memory protect <内容|Hash> [小时数, 0=永久]", 1
            
        parts = args.rsplit(" ", 1)
        duration = 24.0
        query = args
        
        # Try parse duration from last part
        if len(parts) > 1:
            try:
                duration = float(parts[1])
                query = parts[0]
            except ValueError:
                pass # Last part is not number, assume full string is query
        
        # Identify relations
        hashes = await self._resolve_relations(query)
        if not hashes:
            return True, f"未找到与 '{query}' 相关的关系。", 1
            
        now = datetime.datetime.now().timestamp()
        
        if duration <= 0:
            # Permanent Pin
            self.metadata_store.update_relations_protection(hashes, is_pinned=True)
            msg = f"🔒 已永久锁定 {len(hashes)} 条相关记忆。"
        else:
            # TTL Protect
            until = now + duration * 3600
            self.metadata_store.update_relations_protection(hashes, protected_until=until)
            msg = f"🛡️ 已保护 {len(hashes)} 条相关记忆 ({duration}小时)。"
            
        return True, msg, 1

    async def _handle_reinforce(self, args: str) -> Tuple[bool, str, int]:
        """手动强化 /memory reinforce query"""
        if not args:
            return False, "用法: /memory reinforce <内容|Hash>", 1
            
        hashes = await self._resolve_relations(args)
        if not hashes:
            return True, f"未找到与 '{args}' 相关的关系。", 1
            
        # Manual reinforce bypasses cleanup loop logic? 
        # Ideally we reuse plugin logic but with force=True.
        # But here we can just update directly.
        
        now = datetime.datetime.now().timestamp()
        revive_boost = self.get_config("memory.revive_boost_weight", 0.5)
        max_weight = self.get_config("memory.max_weight", 10.0)
        
        # Check active status
        status_map = self.metadata_store.get_relation_status_batch(hashes)
        
        revived = []
        reinforced = []

        relation_map = self.metadata_store.get_relations_subject_object_map(hashes)
        for h, pair in relation_map.items():
            u, v = pair
            st = status_map.get(h)
            if not st: continue
            
            # Boost
            # Manual reinforce gives a solid boost, say +1.0 or reset to max?
            # Let's give +1.0 but cap at max
            delta = 1.0
            
            self.graph_store.update_edge_weight(u, v, delta, max_weight=max_weight)
            
            if st["is_inactive"]:
                revived.append(h)
            else:
                reinforced.append(h)
                
        # Update Metadata
        if revived:
            self.metadata_store.mark_relations_active(revived, boost_weight=revive_boost)
        
        # Update timestamps for all
        self.metadata_store.update_relations_protection(
            hashes, 
            last_reinforced=now,
            # Manual reinforce also protects for auto_protect time?
            protected_until=now + self.get_config("memory.auto_protect_ttl_hours", 24.0)*3600
        )
        
        return True, f"💪 已强化 {len(hashes)} 条记忆 (复活: {len(revived)})", 1

    async def _handle_restore(self, args: str) -> Tuple[bool, str, int]:
        """恢复记忆 /memory restore <hash>"""
        if not args:
            return False, "用法: /memory restore <Hash|Query>", 1
            
        r_hash = args.strip().lower()
        
        # Try resolve if not direct hash
        target_hashes = [r_hash]
        if self._is_hash_like(r_hash):
            resolved = self.metadata_store.resolve_relation_hash_alias(
                r_hash,
                include_deleted=True,
            )
            if resolved:
                target_hashes = resolved
        else:
            # 非 hash 输入：按内容在回收站检索
            found = self.metadata_store.search_deleted_relation_hashes_by_text(r_hash, limit=5)
            if found:
                target_hashes = found
            else:
                return False, "❌ 未能通过关键词找到回收站中的记忆，请尝试确切的 Hash。", 1

        restored_count = 0
        msgs = []
        
        for h in target_hashes:
            # 1. Restore Metadata
            data = self.metadata_store.restore_relation_metadata(h)
            if not data:
                continue
                
            # 2. Restore Graph Edge
            subject = data["subject"]
            obj = data["object"]
            conf = data["confidence"]
            
            graph_restored = False
            if self.graph_store.has_node(subject) and self.graph_store.has_node(obj):
                self.graph_store.add_edges(
                    edges=[(subject, obj)], 
                    weights=[conf], 
                    relation_hashes=[h]
                )
                graph_restored = True
                
            msg = f"[{subject}]->[{obj}]"
            if not graph_restored:
                msg += "(仅元数据)"
            else:
                rv_cfg = self.get_config("retrieval.relation_vectorization", {}) or {}
                rv_enabled = bool(rv_cfg.get("enabled", False)) if isinstance(rv_cfg, dict) else False
                if rv_enabled and self.metadata_store:
                    has_vector = bool(self.vector_store and h in self.vector_store)
                    if has_vector:
                        try:
                            self.metadata_store.set_relation_vector_state(h, "ready")
                        except Exception:
                            pass
                    else:
                        # 不阻塞 restore，交给后台回填任务处理
                        try:
                            self.metadata_store.set_relation_vector_state(h, "pending")
                        except Exception:
                            pass
                        msg += "(待回填向量)"
            msgs.append(msg)
            restored_count += 1
            
        if restored_count == 0:
             return True, f"❌ 未找到可恢复的记忆 (Hash: {r_hash})", 1
        
        return True, f"♻️ 已恢复 {restored_count} 条记忆: " + ", ".join(msgs), 1

    async def _resolve_relations(self, query: str) -> List[str]:
        """解析查询为关系哈希列表"""
        query = (query or "").strip()

        # 1. If matches hash format (兼容 32/64；优先 64)
        if self._is_hash_like(query):
            query = query.lower()
            resolved = self.metadata_store.resolve_relation_hash_alias(query, include_deleted=False)
            if resolved:
                return resolved
                
        # 2. Semantic Search with Retriever
        if self.retriever:
            # Use top_k=5 relations
            results = await self.retriever.retrieve(query, top_k=10)
            # Filter for relations
            rel_results = [r for r in results if r.result_type == "relation"]
            if rel_results:
                 # Take top 3 or those with high score?
                 # Let's take top 3
                 return [r.hash_value for r in rel_results[:3]]
                 
        # 3. Fallback to SQL LIKE
        hashes = self.metadata_store.search_relation_hashes_by_text(query, limit=5)
        
        return hashes

    def _get_help(self) -> str:
        return self.command_description + "\n" + self.command_pattern
