"""
知识查询Tool组件

提供LLM可调用的知识查询工具。
"""

from typing import Any, List, Tuple, Optional, Dict

from src.common.logger import get_logger
from src.plugin_system.base.base_tool import BaseTool
from src.plugin_system.base.component_types import ToolParamType
from src.chat.message_receive.chat_stream import ChatStream

# 导入核心模块
from ...core import (
    DualPathRetriever,
    DynamicThresholdFilter,
)
from ...core.runtime import build_search_runtime
from .query_modes_entity import query_entity as query_entity_mode
from .query_modes_person import (
    is_person_profile_injection_enabled as is_person_profile_injection_enabled_mode,
    query_person as query_person_mode,
)
from .query_modes_relation import (
    extract_entities_from_query as extract_entities_from_query_mode,
    path_search as path_search_mode,
    query_relation as query_relation_mode,
    semantic_search_relation as semantic_search_relation_mode,
)
from .query_tool_orchestrator import (
    build_forward_search_content as build_forward_search_content_orchestrator,
    build_forward_time_content as build_forward_time_content_orchestrator,
    direct_execute_tool as direct_execute_tool_orchestrator,
    execute_forward_search_or_time as execute_forward_search_or_time_orchestrator,
    execute_tool as execute_tool_orchestrator,
    get_search_owner as get_search_owner_orchestrator,
    get_tool_search_mode as get_tool_search_mode_orchestrator,
    resolve_search_context as resolve_search_context_orchestrator,
)

logger = get_logger("A_Memorix.KnowledgeQueryTool")


class KnowledgeQueryTool(BaseTool):
    """知识查询Tool

    功能：
    - search/time 检索（统一 forward 链路）
    - 实体查询
    - 关系查询
    - 统计信息
    - LLM可直接调用
    """

    # Tool基本信息
    name = "knowledge_query"
    description = "查询A_Memorix知识库，支持检索、实体查询、关系查询和统计信息"

    # Tool参数定义
    parameters: List[Tuple[str, ToolParamType, str, bool, List[str] | None]] = [
        (
            "query_type",
            ToolParamType.STRING,
            "查询类型：search(检索)、time(时序检索)、episode(情景记忆)、aggregate(聚合)、entity(实体)、relation(关系)、person(人物画像)、stats(统计)",
            True,
            ["search", "time", "episode", "aggregate", "entity", "relation", "person", "stats"],
        ),
        (
            "query",
            ToolParamType.STRING,
            "查询内容（检索文本/实体名称/关系规格），stats模式不需要",
            False,
            None,
        ),
        (
            "person_id",
            ToolParamType.STRING,
            "人物ID（person模式可选；为空时会尝试通过query或会话上下文解析）",
            False,
            None,
        ),
        (
            "top_k",
            ToolParamType.INTEGER,
            "返回结果数量（search/time/episode模式）",
            False,
            None,
        ),
        (
            "use_threshold",
            ToolParamType.BOOLEAN,
            "是否使用动态阈值过滤（search/time模式）",
            False,
            None,
        ),
        (
            "mix",
            ToolParamType.BOOLEAN,
            "aggregate 模式是否输出混合融合结果（Weighted RRF）",
            False,
            None,
        ),
        (
            "mix_top_k",
            ToolParamType.INTEGER,
            "aggregate 模式混合结果返回数量（默认回落 top_k）",
            False,
            None,
        ),
        (
            "time_from",
            ToolParamType.STRING,
            "开始时间（time模式，仅支持 YYYY/MM/DD 或 YYYY/MM/DD HH:mm；日期按 00:00 展开）",
            False,
            None,
        ),
        (
            "time_to",
            ToolParamType.STRING,
            "结束时间（time模式，仅支持 YYYY/MM/DD 或 YYYY/MM/DD HH:mm；日期按 23:59 展开）",
            False,
            None,
        ),
        (
            "person",
            ToolParamType.STRING,
            "人物过滤（time模式可选）",
            False,
            None,
        ),
        (
            "source",
            ToolParamType.STRING,
            "来源过滤（time/episode模式可选）",
            False,
            None,
        ),
        (
            "include_paragraphs",
            ToolParamType.BOOLEAN,
            "episode 模式是否附带关联段落明细",
            False,
            None,
        ),
    ]

    # LLM可用
    available_for_llm = True

    def __init__(self, plugin_config: Optional[dict] = None, chat_stream: Optional["ChatStream"] = None):
        """初始化知识查询Tool"""
        super().__init__(plugin_config, chat_stream)

        # 获取存储实例
        self.vector_store = self.plugin_config.get("vector_store")
        self.graph_store = self.plugin_config.get("graph_store")
        self.metadata_store = self.plugin_config.get("metadata_store")
        self.embedding_manager = self.plugin_config.get("embedding_manager")
        self.sparse_index = self.plugin_config.get("sparse_index")

        # 初始化检索器
        self.retriever: Optional[DualPathRetriever] = None
        self.threshold_filter: Optional[DynamicThresholdFilter] = None

        # 设置日志前缀
        chat_id = self.chat_id if self.chat_id else "unknown"
        self.log_prefix = f"[KnowledgeQueryTool-{chat_id}]"

        # 初始化组件
        self._initialize_components()

    @property
    def debug_enabled(self) -> bool:
        """检查是否启用了调试模式"""
        advanced = self.plugin_config.get("advanced", {})
        if isinstance(advanced, dict):
            return advanced.get("debug", False)
        return self.plugin_config.get("debug", False)

    def _initialize_components(self) -> None:
        """初始化检索和过滤组件"""
        runtime = build_search_runtime(
            plugin_config=self.plugin_config,
            logger_obj=logger,
            owner_tag="tool",
            log_prefix=self.log_prefix,
        )
        self.vector_store = runtime.vector_store
        self.graph_store = runtime.graph_store
        self.metadata_store = runtime.metadata_store
        self.embedding_manager = runtime.embedding_manager
        self.sparse_index = runtime.sparse_index
        self.retriever = runtime.retriever
        self.threshold_filter = runtime.threshold_filter

    def _get_search_owner(self) -> str:
        return get_search_owner_orchestrator(self)

    def _get_tool_search_mode(self) -> str:
        return get_tool_search_mode_orchestrator(self)

    def _resolve_search_context(
        self,
        function_args: Dict[str, Any],
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        return resolve_search_context_orchestrator(self, function_args)

    def _build_forward_search_content(self, results: List[Dict[str, Any]]) -> str:
        return build_forward_search_content_orchestrator(self, results)

    def _build_forward_time_content(self, results: List[Dict[str, Any]]) -> str:
        return build_forward_time_content_orchestrator(self, results)

    async def _execute_forward_search_or_time(
        self,
        *,
        query_type: str,
        query: str,
        top_k: int,
        use_threshold: bool,
        time_from: Optional[str],
        time_to: Optional[str],
        person: Optional[str],
        source: Optional[str],
        function_args: Dict[str, Any],
    ) -> Dict[str, Any]:
        return await execute_forward_search_or_time_orchestrator(
            self,
            query_type=query_type,
            query=query,
            top_k=top_k,
            use_threshold=use_threshold,
            time_from=time_from,
            time_to=time_to,
            person=person,
            source=source,
            function_args=function_args,
        )

    async def execute(self, function_args: dict[str, Any]) -> dict[str, Any]:
        return await execute_tool_orchestrator(self, function_args)

    async def direct_execute(
        self,
        query_type: str = "search",
        query: str = "",
        top_k: int = 10,
        use_threshold: bool = True,
        mix: bool = False,
        mix_top_k: Optional[int] = None,
        time_from: Optional[str] = None,
        time_to: Optional[str] = None,
        person: Optional[str] = None,
        source: Optional[str] = None,
        person_id: Optional[str] = None,
        include_paragraphs: bool = False,
        for_injection: bool = False,
        force_refresh: bool = False,
        stream_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await direct_execute_tool_orchestrator(
            self,
            query_type=query_type,
            query=query,
            top_k=top_k,
            use_threshold=use_threshold,
            mix=mix,
            mix_top_k=mix_top_k,
            time_from=time_from,
            time_to=time_to,
            person=person,
            source=source,
            person_id=person_id,
            include_paragraphs=include_paragraphs,
            for_injection=for_injection,
            force_refresh=force_refresh,
            stream_id=stream_id,
            user_id=user_id,
        )

    def _is_person_profile_injection_enabled(self, stream_id: Optional[str], user_id: Optional[str]) -> bool:
        return is_person_profile_injection_enabled_mode(self, stream_id, user_id)

    async def _query_person(
        self,
        query: str,
        person_id: Optional[str],
        top_k: int,
        for_injection: bool = False,
        force_refresh: bool = False,
        stream_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await query_person_mode(
            self,
            query=query,
            person_id=person_id,
            top_k=top_k,
            for_injection=for_injection,
            force_refresh=force_refresh,
            stream_id=stream_id,
            user_id=user_id,
        )

    async def _query_entity(self, entity_name: str) -> Dict[str, Any]:
        return await query_entity_mode(self, entity_name)

    async def _query_relation(self, relation_spec: str) -> Dict[str, Any]:
        return await query_relation_mode(self, relation_spec)

    async def _semantic_search_relation(
        self,
        query: str,
        min_score: float,
    ) -> Dict[str, Any]:
        return await semantic_search_relation_mode(self, query, min_score)

    def _path_search(self, query: str) -> Optional[Dict[str, Any]]:
        return path_search_mode(self, query)

    def _extract_entities_from_query(self, query: str) -> List[str]:
        return extract_entities_from_query_mode(self, query)

    def _get_stats(self) -> Dict[str, Any]:
        """获取统计信息

        Returns:
            统计信息字典
        """
        stats = {
            "vector_store": {
                "num_vectors": self.vector_store.num_vectors if self.vector_store else 0,
                "dimension": self.vector_store.dimension if self.vector_store else 0,
            },
            "graph_store": {
                "num_nodes": self.graph_store.num_nodes if self.graph_store else 0,
                "num_edges": self.graph_store.num_edges if self.graph_store else 0,
            },
            "metadata_store": {
                "num_paragraphs": self.metadata_store.count_paragraphs() if self.metadata_store else 0,
                "num_relations": self.metadata_store.count_relations() if self.metadata_store else 0,
                "num_entities": self.metadata_store.count_entities() if self.metadata_store else 0,
            },
            "sparse": self.sparse_index.stats() if self.sparse_index else None,
            "relation_vectorization": {},
            "runtime_self_check": None,
        }
        plugin_instance = self.plugin_config.get("plugin_instance")
        if plugin_instance is not None and hasattr(plugin_instance, "get_relation_vector_stats"):
            try:
                stats["relation_vectorization"] = plugin_instance.get_relation_vector_stats()
            except Exception as e:
                logger.warning(f"{self.log_prefix} 获取关系向量统计失败: {e}")
        if plugin_instance is not None:
            report = getattr(plugin_instance, "_runtime_self_check_report", None)
            if isinstance(report, dict) and report:
                stats["runtime_self_check"] = dict(report)

        # Format a human-readable summary
        content = (
            f"📊 知识库统计信息\n\n"
            f"📦 向量存储:\n"
            f"  - 向量数量: {stats['vector_store']['num_vectors']}\n"
            f"  - 维度: {stats['vector_store']['dimension']}\n\n"
            f"🕸️ 图存储:\n"
            f"  - 节点数: {stats['graph_store']['num_nodes']}\n"
            f"  - 边数: {stats['graph_store']['num_edges']}\n\n"
            f"📝 元数据存储:\n"
            f"  - 段落数: {stats['metadata_store']['num_paragraphs']}\n"
            f"  - 关系数: {stats['metadata_store']['num_relations']}\n"
            f"  - 实体数: {stats['metadata_store']['num_entities']}"
        )
        sparse_stats = stats.get("sparse")
        if sparse_stats:
            content += (
                f"\n\n🧩 稀疏检索:\n"
                f"  - 启用: {'是' if sparse_stats.get('enabled') else '否'}\n"
                f"  - 已加载: {'是' if sparse_stats.get('loaded') else '否'}\n"
                f"  - Tokenizer: {sparse_stats.get('tokenizer_mode', 'N/A')}\n"
                f"  - FTS文档数: {sparse_stats.get('doc_count', 0)}"
            )

        rel_stats = stats.get("relation_vectorization") or {}
        rel_states = rel_stats.get("states") if isinstance(rel_stats, dict) else None
        if rel_states:
            ready_cov = float(rel_stats.get("relation_ready_coverage", 0.0) or 0.0) * 100
            vector_cov = float(rel_stats.get("relation_vector_coverage", 0.0) or 0.0) * 100
            content += (
                f"\n\n🧠 关系向量化:\n"
                f"  - total: {rel_states.get('total', 0)}\n"
                f"  - ready: {rel_states.get('ready', 0)}\n"
                f"  - pending: {rel_states.get('pending', 0)}\n"
                f"  - failed: {rel_states.get('failed', 0)}\n"
                f"  - none: {rel_states.get('none', 0)}\n"
                f"  - orphan_vectors: {rel_stats.get('orphan_vectors', 0)}\n"
                f"  - ready_coverage: {ready_cov:.1f}%\n"
                f"  - vector_coverage: {vector_cov:.1f}%\n"
                f"  - ready_but_missing_vector: {rel_stats.get('ready_but_missing_vector', 0)}"
            )

        runtime_self_check = stats.get("runtime_self_check")
        if isinstance(runtime_self_check, dict) and runtime_self_check:
            content += (
                f"\n\n🩺 Runtime 自检:\n"
                f"  - ok: {'是' if runtime_self_check.get('ok') else '否'}\n"
                f"  - code: {runtime_self_check.get('code', 'unknown')}\n"
                f"  - configured_dimension: {runtime_self_check.get('configured_dimension', 0)}\n"
                f"  - vector_store_dimension: {runtime_self_check.get('vector_store_dimension', 0)}\n"
                f"  - detected_dimension: {runtime_self_check.get('detected_dimension', 0)}\n"
                f"  - encoded_dimension: {runtime_self_check.get('encoded_dimension', 0)}"
            )

        return {
            "success": True,
            "query_type": "stats",
            "content": content,
            "statistics": stats,
        }

    def get_tool_info_summary(self) -> str:
        """获取工具信息摘要

        Returns:
            工具信息摘要文本
        """
        if not self.retriever:
            return "❌ 知识查询Tool未初始化"

        lines = [
            "🔧 知识查询Tool信息",
            "",
            "📋 基本信息:",
            f"  - 名称: {self.name}",
            f"  - 描述: {self.description}",
            f"  - LLM可用: {'是' if self.available_for_llm else '否'}",
            "",
            "⚙️ 检索配置:",
            f"  - Top-K段落: {self.retriever.config.top_k_paragraphs}",
            f"  - Top-K关系: {self.retriever.config.top_k_relations}",
            f"  - 融合系数(alpha): {self.retriever.config.alpha}",
            f"  - 融合方法: {self.retriever.config.fusion.method}",
            f"  - PPR启用: {'是' if self.retriever.config.enable_ppr else '否'}",
            f"  - 并行检索: {'是' if self.retriever.config.enable_parallel else '否'}",
            "",
            "📊 存储统计:",
            f"  - 向量数量: {self.vector_store.num_vectors if self.vector_store else 0}",
            f"  - 节点数量: {self.graph_store.num_nodes if self.graph_store else 0}",
            f"  - 边数量: {self.graph_store.num_edges if self.graph_store else 0}",
            f"  - 段落数量: {self.metadata_store.count_paragraphs() if self.metadata_store else 0}",
        ]
        if self.sparse_index:
            sparse_stats = self.sparse_index.stats()
            lines.extend([
                "",
                "🧩 稀疏检索:",
                f"  - 启用: {'是' if sparse_stats.get('enabled') else '否'}",
                f"  - 已加载: {'是' if sparse_stats.get('loaded') else '否'}",
                f"  - Tokenizer: {sparse_stats.get('tokenizer_mode', 'N/A')}",
            ])

        return "\n".join(lines)
