"""
删除知识 Command 组件

提供知识库删除功能，支持段落、实体和关系的删除。
"""

import re
import time
from typing import Tuple, Optional, List, Dict, Any

from src.common.logger import get_logger
from src.plugin_system.base.base_command import BaseCommand
from src.chat.message_receive.message import MessageRecv

from ...core import VectorStore, GraphStore, MetadataStore
from ...core.utils.hash import compute_hash, normalize_text

logger = get_logger("A_Memorix.DeleteCommand")


class DeleteCommand(BaseCommand):
    """删除知识 Command"""

    command_name = "delete"
    command_description = "删除知识库内容，支持段落、实体、关系和清空"
    command_pattern = r"^\/delete(?:\s+(?P<mode>\w+))?(?:\s+(?P<content>.+))?$"

    def __init__(self, message: MessageRecv, plugin_config: Optional[dict] = None):
        super().__init__(message, plugin_config)

        self.vector_store: Optional[VectorStore] = self.plugin_config.get("vector_store")
        self.graph_store: Optional[GraphStore] = self.plugin_config.get("graph_store")
        self.metadata_store: Optional[MetadataStore] = self.plugin_config.get("metadata_store")

        # 兜底：当配置里没有实例时，从插件全局实例获取
        if not all([
            self.vector_store is not None,
            self.graph_store is not None,
            self.metadata_store is not None,
        ]):
            from ...plugin import A_MemorixPlugin

            instances = A_MemorixPlugin.get_storage_instances()
            if instances:
                self.vector_store = self.vector_store or instances.get("vector_store")
                self.graph_store = self.graph_store or instances.get("graph_store")
                self.metadata_store = self.metadata_store or instances.get("metadata_store")

        if self.message and self.message.chat_stream:
            self.log_prefix = f"[DeleteCommand-{self.message.chat_stream.stream_id}]"
        else:
            self.log_prefix = "[DeleteCommand]"

    async def execute(self) -> Tuple[bool, Optional[str], int]:
        """执行删除命令"""
        if not all([
            self.vector_store is not None,
            self.graph_store is not None,
            self.metadata_store is not None,
        ]):
            return False, "❌ 知识库未初始化，无法执行删除", 0

        mode = (self.matched_groups.get("mode") or "help").lower()
        content = (self.matched_groups.get("content") or "").strip()

        mode_alias = {
            "p": "paragraph",
            "e": "entity",
            "r": "relation",
            "s": "stats",
            "h": "help",
            "?": "help",
        }
        mode = mode_alias.get(mode, mode)

        try:
            if mode in ["help", ""]:
                return True, self._get_help_message(), 0

            if mode == "stats":
                ok, msg = self._get_deletion_stats()
                return ok, msg, 0

            if mode == "clear":
                ok, msg = await self._clear_knowledge_base()
            elif mode == "paragraph":
                if not content:
                    return False, "❌ 用法: /delete paragraph <hash或内容>", 0
                ok, msg = await self._delete_paragraph(content)
            elif mode == "entity":
                if not content:
                    return False, "❌ 用法: /delete entity <实体名称>", 0
                ok, msg = await self._delete_entity(content)
            elif mode == "relation":
                if not content:
                    return False, "❌ 用法: /delete relation <hash或subject|predicate|object>", 0
                ok, msg = await self._delete_relation(content)
            else:
                return False, f"❌ 未知删除模式: {mode}\n\n{self._get_help_message()}", 0

            if ok:
                try:
                    self.vector_store.save()
                    self.graph_store.save()
                except Exception as save_e:  # noqa: BLE001
                    logger.warning(f"{self.log_prefix} 删除后保存失败: {save_e}")

            return ok, msg, 0

        except Exception as e:  # noqa: BLE001
            logger.error(f"{self.log_prefix} 删除命令执行异常: {e}")
            return False, f"❌ 删除失败: {e}", 0

    @staticmethod
    def _looks_like_hash(text: str) -> bool:
        return bool(re.fullmatch(r"[0-9a-fA-F]{64}", text.strip()))

    async def _delete_paragraph(self, paragraph_spec: str) -> Tuple[bool, str]:
        """删除段落（优先 hash，回退按内容匹配）"""
        start_time = time.time()

        query = paragraph_spec.strip()
        if not query:
            return False, "❌ 段落内容不能为空"

        target: Optional[Dict[str, Any]] = None

        if self._looks_like_hash(query):
            target = self.metadata_store.get_paragraph(query)
            if not target:
                return False, f"❌ 未找到段落: {query[:16]}..."
        else:
            matches = self.metadata_store.search_paragraphs_by_content(query)
            if not matches:
                return False, "❌ 未找到匹配的段落"

            if len(matches) > 1:
                query_norm = normalize_text(query)
                exact = [
                    p for p in matches
                    if normalize_text(str(p.get("content", ""))) == query_norm
                ]
                if len(exact) == 1:
                    target = exact[0]
                else:
                    previews: List[str] = []
                    for p in matches[:5]:
                        content_preview = str(p.get("content", "")).replace("\n", " ")
                        if len(content_preview) > 40:
                            content_preview = content_preview[:40] + "..."
                        previews.append(f"- {p['hash'][:16]}... {content_preview}")
                    return False, "⚠️ 匹配到多个段落，请使用 hash 精确删除:\n" + "\n".join(previews)
            else:
                target = matches[0]

        paragraph_hash = str(target["hash"])

        cleanup_plan = self.metadata_store.delete_paragraph_atomic(paragraph_hash)

        relation_prune_ops = cleanup_plan.get("relation_prune_ops", []) or []

        # 优先按 relation hash 精确修剪边映射
        if relation_prune_ops and hasattr(self.graph_store, "prune_relation_hashes"):
            self.graph_store.prune_relation_hashes(relation_prune_ops)
        elif cleanup_plan.get("edges_to_remove"):
            return (
                False,
                "❌ 删除中止：检测到旧图边清理回退需求（缺少 edge-hash-map），请先执行 "
                "`python plugins/A_memorix/scripts/release_vnext_migrate.py migrate`",
            )

        # 向量删除：段落向量 + 被剪掉的关系向量
        vector_ids: List[str] = []
        vector_id_to_remove = cleanup_plan.get("vector_id_to_remove")
        if vector_id_to_remove:
            vector_ids.append(str(vector_id_to_remove))
        for op in relation_prune_ops:
            if len(op) >= 3 and op[2]:
                vector_ids.append(str(op[2]))

        deleted_vectors = 0
        if vector_ids:
            dedup_ids = list(dict.fromkeys(vector_ids))
            deleted_vectors = self.vector_store.delete(dedup_ids)

        elapsed = time.time() - start_time
        result_lines = [
            "✅ 段落删除完成",
            f"📄 Hash: {paragraph_hash[:16]}...",
            f"🔗 清理关系: {len(relation_prune_ops)}",
            f"🧹 清理向量: {deleted_vectors}",
            f"⏱️ 耗时: {elapsed*1000:.1f}ms",
        ]
        return True, "\n".join(result_lines)

    async def _delete_entity(self, entity_name: str) -> Tuple[bool, str]:
        """删除实体"""
        start_time = time.time()

        entity_name = entity_name.strip()
        if not entity_name:
            return False, "❌ 实体名称不能为空"

        canonical_name = entity_name.lower()

        if not self.graph_store.has_node(canonical_name):
            return False, f"❌ 实体不存在: {canonical_name}"

        neighbors = self.graph_store.get_neighbors(canonical_name)
        edge_count = len(neighbors)
        related_paragraphs = self.metadata_store.get_paragraphs_by_entity(canonical_name)

        # 预先记录相关关系 hash，便于删除对应向量
        rel_as_subject = self.metadata_store.get_relations(subject=canonical_name)
        rel_as_object = self.metadata_store.get_relations(object=canonical_name)
        relation_hashes = {
            str(r["hash"]) for r in (rel_as_subject + rel_as_object) if r.get("hash")
        }

        deleted_nodes = self.graph_store.delete_nodes([canonical_name])
        meta_deleted = self.metadata_store.delete_entity(canonical_name)

        vector_ids = [compute_hash(canonical_name)] + list(relation_hashes)
        deleted_vectors = 0
        try:
            deleted_vectors = self.vector_store.delete(vector_ids)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"{self.log_prefix} 删除实体向量失败 {canonical_name}: {e}")

        if deleted_nodes <= 0 and not meta_deleted:
            return False, f"❌ 实体删除失败: {canonical_name}"

        elapsed = time.time() - start_time
        result_lines = [
            "✅ 实体删除完成",
            f"🏷️ 实体名称: {canonical_name}",
            f"🔗 关联边数: {edge_count}",
            f"📄 相关段落: {len(related_paragraphs)}",
            f"🧹 清理向量: {deleted_vectors}",
            f"⏱️ 耗时: {elapsed*1000:.1f}ms",
            "",
            "⚠️ 注意: 相关段落未删除，如需删除请使用 /delete paragraph",
        ]

        return True, "\n".join(result_lines)

    async def _delete_relation(self, relation_spec: str) -> Tuple[bool, str]:
        """删除关系"""
        start_time = time.time()

        relation_spec = relation_spec.strip()
        if not relation_spec:
            return False, "❌ 关系规格不能为空"

        relation: Optional[Dict[str, Any]] = None
        hash_value = ""

        if self._looks_like_hash(relation_spec):
            hash_value = relation_spec.lower()
            relation = self.metadata_store.get_relation(hash_value)
            if not relation:
                return False, f"❌ 未找到关系: {hash_value[:16]}..."
        else:
            if "|" in relation_spec:
                parts = [p.strip() for p in relation_spec.split("|")]
                if len(parts) != 3:
                    return False, "❌ 关系格式错误，应使用: subject|predicate|object"
                subject, predicate, obj = parts
            else:
                parts = relation_spec.split(maxsplit=2)
                if len(parts) != 3:
                    return False, "❌ 关系格式错误，应使用: subject predicate object"
                subject, predicate, obj = parts

            s_canon = subject.strip().lower()
            p_canon = predicate.strip().lower()
            o_canon = obj.strip().lower()

            relation_key = f"{s_canon}|{p_canon}|{o_canon}"
            hash_value = compute_hash(relation_key)

            relation = self.metadata_store.get_relation(hash_value)
            if not relation:
                return False, f"❌ 未找到关系 (hash不匹配): {subject} {predicate} {obj}"

        success = self.metadata_store.delete_relation(hash_value)
        if not success:
            return False, f"❌ 关系删除失败: {hash_value[:16]}..."

        subject = str(relation.get("subject", ""))
        obj = str(relation.get("object", ""))

        has_hash_map = self.graph_store.edge_contains_relation_hash(subject, obj, hash_value)
        if not has_hash_map:
            return False, (
                "❌ 检测到关系映射缺失，已拒绝执行粗粒度删边。"
                " 请先执行 scripts/release_vnext_migrate.py migrate。"
            )
        self.graph_store.prune_relation_hashes([(subject, obj, hash_value)])

        deleted_vectors = 0
        try:
            deleted_vectors = self.vector_store.delete([hash_value])
        except Exception as e:  # noqa: BLE001
            logger.warning(f"{self.log_prefix} 删除关系向量失败 {hash_value[:16]}...: {e}")

        elapsed = time.time() - start_time
        result_lines = [
            "✅ 关系删除完成",
            f"🔗 Hash: {hash_value[:16]}...",
            f"📌 {subject} {relation.get('predicate', '')} {obj}",
            f"🧹 清理向量: {deleted_vectors}",
            f"⏱️ 耗时: {elapsed*1000:.1f}ms",
        ]

        return True, "\n".join(result_lines)

    async def _clear_knowledge_base(self) -> Tuple[bool, str]:
        """清空知识库"""
        start_time = time.time()

        try:
            num_paragraphs = self.metadata_store.count_paragraphs()
            num_relations = self.metadata_store.count_relations()
            num_entities = self.metadata_store.count_entities()
            num_vectors = self.vector_store.num_vectors

            self.vector_store.clear()
            self.graph_store.clear()
            self.metadata_store.clear_all()

            elapsed = time.time() - start_time

            result_lines = [
                "⚠️ 知识库已清空",
                "",
                "📊 已删除内容:",
                f"  - 段落: {num_paragraphs}",
                f"  - 关系: {num_relations}",
                f"  - 实体: {num_entities}",
                f"  - 向量: {num_vectors}",
                "",
                f"⏱️ 耗时: {elapsed*1000:.1f}ms",
                "",
                "⚠️ 此操作不可撤销！",
            ]

            return True, "\n".join(result_lines)

        except Exception as e:  # noqa: BLE001
            return False, f"❌ 清空知识库失败: {str(e)}"

    def _get_deletion_stats(self) -> Tuple[bool, str]:
        """获取删除统计信息"""
        deleted_paragraphs = self.metadata_store.count_paragraphs(include_deleted=True, only_deleted=True)
        deleted_relations = self.metadata_store.count_relations(include_deleted=True, only_deleted=True)

        current_paragraphs = self.metadata_store.count_paragraphs()
        current_relations = self.metadata_store.count_relations()
        current_entities = self.metadata_store.count_entities()

        lines = [
            "📊 删除统计信息",
            "",
            "🗑️ 已删除（软删除）:",
            f"  - 段落: {deleted_paragraphs}",
            f"  - 关系: {deleted_relations}",
            "",
            "📦 当前内容:",
            f"  - 段落: {current_paragraphs}",
            f"  - 关系: {current_relations}",
            f"  - 实体: {current_entities}",
            "",
            "💡 提示:",
            "  - 段落和关系使用软删除，可通过重建索引彻底清除",
            "  - 使用 /delete clear 清空整个知识库",
        ]

        return True, "\n".join(lines)

    def _get_help_message(self) -> str:
        """获取帮助消息"""
        return """📖 删除命令帮助

用法:
  /delete paragraph <hash或内容>  - 删除段落（软删除）
  /delete entity <实体名称>       - 删除实体
  /delete relation <关系规格>     - 删除关系
  /delete clear                  - 清空知识库（危险操作！）
  /delete stats                  - 显示删除统计
  /delete help                   - 显示此帮助

快捷模式:
  /delete p <hash或内容>         - 删除段落（paragraph的简写）
  /delete e <实体名称>           - 删除实体（entity的简写）
  /delete r <关系规格>           - 删除关系（relation的简写）

示例:
  /delete paragraph a1b2c3d4...
  /delete paragraph 人工智能的定义
  /delete entity Apple
  /delete relation Apple|founded|Steve Jobs
  /delete relation founded by Steve Jobs
  /delete stats

关系格式:
  - subject|predicate|object（使用|分隔）
  - subject predicate object（使用空格分隔）
  - 完整的64位hash值（精确删除）

注意事项:
  - 段落删除采用软删除，不会立即物理删除
  - 删除实体不会删除相关段落，仅删除实体节点
  - 删除关系会同时删除图中的边
  - clear操作不可撤销，请谨慎使用
"""
