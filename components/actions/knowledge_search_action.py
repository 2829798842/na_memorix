"""
知识检索Action组件

提供基于双路检索的知识搜索功能。
"""

from typing import Tuple, Optional, List, Dict, Any

from src.common.logger import get_logger
from src.plugin_system.base.base_action import BaseAction
from src.plugin_system.base.component_types import ActionActivationType
from src.chat.message_receive.chat_stream import ChatStream

# 导入核心模块
from ...core import (
    DualPathRetriever,
    DynamicThresholdFilter,
)
from ...core.runtime import build_search_runtime
from ...core.utils.search_execution_service import (
    SearchExecutionRequest,
    SearchExecutionService,
)

logger = get_logger("A_Memorix.KnowledgeSearchAction")


class KnowledgeSearchAction(BaseAction):
    """知识检索Action

    功能：
    - 双路检索（段落+关系）
    - 智能结果融合
    - PPR重排序
    - 动态阈值过滤
    """

    # Action基本信息
    action_name = "knowledge_search"
    action_description = "在知识库中搜索相关内容，支持段落和关系的双路检索"

    # 激活配置
    activation_type = ActionActivationType.ALWAYS
    parallel_action = True

    # Action参数
    action_parameters = {
        "query_type": {
            "type": "string",
            "description": "查询模式: semantic(语义)、time(时间)、hybrid(语义+时间)",
            "required": False,
            "enum": ["semantic", "time", "hybrid"],
            "default": "semantic",
        },
        "query": {
            "type": "string",
            "description": "搜索查询文本（semantic/hybrid必填，time可选）",
            "required": False,
        },
        "time_from": {
            "type": "string",
            "description": "开始时间，仅支持 YYYY/MM/DD 或 YYYY/MM/DD HH:mm（日期自动按 00:00 展开）",
            "required": False,
        },
        "time_to": {
            "type": "string",
            "description": "结束时间，仅支持 YYYY/MM/DD 或 YYYY/MM/DD HH:mm（日期自动按 23:59 展开）",
            "required": False,
        },
        "person": {
            "type": "string",
            "description": "按人物过滤（可选）",
            "required": False,
        },
        "source": {
            "type": "string",
            "description": "按来源过滤（可选）",
            "required": False,
        },
        "top_k": {
            "type": "integer",
            "description": "返回结果数量",
            "default": 10,
            "min": 1,
            "max": 50,
        },
        "use_threshold": {
            "type": "boolean",
            "description": "是否使用动态阈值过滤",
            "default": True,
        },
        "enable_ppr": {
            "type": "boolean",
            "description": "是否启用PPR重排序",
            "default": True,
        },
    }

    # Action依赖
    action_require = ["vector_store", "graph_store", "metadata_store", "embedding_manager"]

    def __init__(self, *args, **kwargs):
        """初始化知识检索Action"""
        super().__init__(*args, **kwargs)

        # 初始化检索器
        self.retriever: Optional[DualPathRetriever] = None
        self.threshold_filter: Optional[DynamicThresholdFilter] = None
        self._initialize_retriever()
 
    @property
    def debug_enabled(self) -> bool:
        """检查是否启用了调试模式"""
        advanced = self.plugin_config.get("advanced", {})
        if isinstance(advanced, dict):
            return advanced.get("debug", False)
        return self.plugin_config.get("debug", False)
 
    def _initialize_retriever(self) -> None:
        """初始化检索器"""
        runtime = build_search_runtime(
            plugin_config=self.plugin_config,
            logger_obj=logger,
            owner_tag="action",
            log_prefix=self.log_prefix,
        )
        self.retriever = runtime.retriever
        self.threshold_filter = runtime.threshold_filter

    async def execute(self) -> Tuple[bool, str]:
        """执行知识检索

        Returns:
            Tuple[bool, str]: (是否成功, 结果文本)
        """
        # 检查检索器是否可用
        if not self.retriever:
            return False, "知识检索器未初始化"

        # 获取查询参数
        query = str(self.action_data.get("query", "") or "").strip()
        query_type = str(self.action_data.get("query_type", "") or "").strip().lower()
        time_from_raw = self.action_data.get("time_from")
        time_to_raw = self.action_data.get("time_to")
        person = self.action_data.get("person")
        source = self.action_data.get("source")
        top_k_raw = self.action_data.get("top_k")
        use_threshold = self.action_data.get("use_threshold", True)
        enable_ppr = self.action_data.get("enable_ppr", True)
        if not query_type:
            if time_from_raw or time_to_raw:
                query_type = "hybrid" if query else "time"
            else:
                query_type = "semantic"
        search_owner = str(self.get_config("routing.search_owner", "action") or "action").strip().lower()
        if search_owner == "tool":
            logger.info(f"{self.log_prefix} routing.search_owner=tool，Action检索链路跳过")
            return True, ""

        request = SearchExecutionRequest(
            caller="action",
            stream_id=self.chat_id,
            group_id=self.group_id,
            user_id=self.user_id,
            query_type=query_type,
            query=query,
            top_k=top_k_raw,
            time_from=str(time_from_raw) if time_from_raw is not None else None,
            time_to=str(time_to_raw) if time_to_raw is not None else None,
            person=str(person).strip() if person else None,
            source=str(source).strip() if source else None,
            use_threshold=bool(use_threshold),
            enable_ppr=bool(enable_ppr),
        )

        execution = await SearchExecutionService.execute(
            retriever=self.retriever,
            threshold_filter=self.threshold_filter,
            plugin_config=self.plugin_config,
            request=request,
            enforce_chat_filter=True,
            reinforce_access=True,
        )
        if not execution.success:
            return False, execution.error

        if execution.chat_filtered:
            return True, ""

        results = execution.results
        elapsed_ms = execution.elapsed_ms
        if not results:
            response = f"未找到相关内容（检索耗时: {elapsed_ms:.1f}ms）"
            logger.info(f"{self.log_prefix} {response}")
            return True, response

        query_display = query if query else "N/A"
        response = self._format_results(results, query_display, elapsed_ms)
        logger.info(
            f"{self.log_prefix} 检索完成: 返回{len(results)}条结果, 耗时{elapsed_ms:.1f}ms"
        )
        return True, response

    def _format_results(self, results: List[Any], query: str, elapsed_ms: float) -> str:
        """格式化检索结果

        Args:
            results: 检索结果列表
            query: 原始查询
            elapsed_ms: 检索耗时

        Returns:
            格式化的结果文本
        """
        lines = []
        lines.append(f"🔍 知识检索结果（查询: '{query}'，耗时: {elapsed_ms:.1f}ms）")
        lines.append("")

        # 按类型分组
        paragraphs = []
        relations = []

        for result in results:
            if result.result_type == "paragraph":
                paragraphs.append(result)
            elif result.result_type == "relation":
                relations.append(result)

        # 添加段落结果
        if paragraphs:
            lines.append("📄 匹配的段落：")
            for i, result in enumerate(paragraphs, 1):
                score_pct = result.score * 100
                summary = result.content[:100] + ("..." if len(result.content) > 100 else "")
                lines.append(f"  {i}. [{score_pct:.1f}%] {summary}")
                time_meta = result.metadata.get("time_meta", {})
                if time_meta:
                    basis = time_meta.get("match_basis", "none")
                    s_text = time_meta.get("effective_start_text") or "N/A"
                    e_text = time_meta.get("effective_end_text") or "N/A"
                    lines.append(f"     ⏱️ {s_text} ~ {e_text} ({basis})")
            lines.append("")

        # 添加关系结果
        if relations:
            lines.append("🔗 匹配的关系：")
            for i, result in enumerate(relations, 1):
                score_pct = result.score * 100
                subject = result.metadata.get("subject", "")
                predicate = result.metadata.get("predicate", "")
                obj = result.metadata.get("object", "")
                lines.append(f"  {i}. [{score_pct:.1f}%] {subject} {predicate} {obj}")
                time_meta = result.metadata.get("time_meta", {})
                if time_meta:
                    basis = time_meta.get("match_basis", "none")
                    s_text = time_meta.get("effective_start_text") or "N/A"
                    e_text = time_meta.get("effective_end_text") or "N/A"
                    lines.append(f"     ⏱️ {s_text} ~ {e_text} ({basis})")
            lines.append("")

        # 添加统计信息
        lines.append(f"📊 统计: 共{len(results)}条结果（段落: {len(paragraphs)}, 关系: {len(relations)}）")

        return "\n".join(lines)

    async def search_batch(
        self,
        queries: List[str],
        top_k: int = 10,
    ) -> Dict[str, List[Any]]:
        """批量检索知识

        Args:
            queries: 查询列表
            top_k: 每个查询返回的结果数

        Returns:
            查询到结果的映射 {query: results}
        """
        if not self.retriever:
            logger.error(f"{self.log_prefix} 检索器未初始化")
            return {}

        results_map = {}

        for query in queries:
            try:
                results = self.retriever.retrieve(query, top_k=top_k)
                results_map[query] = results
                logger.info(
                    f"{self.log_prefix} 批量检索: '{query}' -> {len(results)}条结果"
                )
            except Exception as e:
                logger.error(f"{self.log_prefix} 批量检索失败 '{query}': {e}")
                results_map[query] = []

        return results_map

    def get_statistics(self) -> Dict[str, Any]:
        """获取检索统计信息

        Returns:
            统计信息字典
        """
        if not self.retriever:
            return {
                "status": "not_initialized",
            }

        stats = {
            "status": "active",
            "config": {
                "top_k_paragraphs": self.retriever.config.top_k_paragraphs,
                "top_k_relations": self.retriever.config.top_k_relations,
                "alpha": self.retriever.config.alpha,
                "enable_ppr": self.retriever.config.enable_ppr,
                "enable_parallel": self.retriever.config.enable_parallel,
                "retrieval_strategy": self.retriever.config.retrieval_strategy.value,
            },
        }

        # 添加存储统计
        if hasattr(self.retriever, "vector_store"):
            stats["vector_store"] = {
                "num_vectors": self.retriever.vector_store.num_vectors,
                "dimension": self.retriever.vector_store.dimension,
            }

        if hasattr(self.retriever, "graph_store"):
            stats["graph_store"] = {
                "num_nodes": self.retriever.graph_store.num_nodes,
                "num_edges": self.retriever.graph_store.num_edges,
            }

        return stats

    def __repr__(self) -> str:
        return (
            f"KnowledgeSearchAction("
            f"retriever_initialized={self.retriever is not None})"
        )
