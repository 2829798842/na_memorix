"""构建 na_memorix 运行时上下文并衔接外部基础设施。"""

import asyncio
from pathlib import Path
from typing import Any, Dict

from core.embedding.api_adapter import create_embedding_api_adapter
from core.retrieval.dual_path import DualPathRetriever, DualPathRetrieverConfig, FusionConfig
from core.retrieval.sparse_bm25 import SparseBM25Config, SparseBM25Index
from core.retrieval.threshold import DynamicThresholdFilter, ThresholdConfig, ThresholdMethod
from core.storage import GraphStore, MetadataStore, QuantizationType, SparseMatrixFormat, VectorStore
from core.utils.person_profile_service import PersonProfileService
from qdrant_client import QdrantClient

from nekro_agent.core.vector_db import get_qdrant_config

from .common.logging import get_logger
from .context import AppContext
from .settings import AppSettings, resolve_openapi_endpoint_config

logger = get_logger("A_Memorix.Bootstrap")


def _safe_int(value: Any, default: int) -> int:
    """将任意值安全转换为整数。

    Args:
        value: 待转换的原始值。
        default: 转换失败时回退的默认值。

    Returns:
        int: 转换结果或默认值。
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    """将任意值安全转换为浮点数。

    Args:
        value: 待转换的原始值。
        default: 转换失败时回退的默认值。

    Returns:
        float: 转换结果或默认值。
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolve_existing_qdrant_dimension(*collection_names: str) -> int:
    """从现有 Qdrant 集合中探测已落地的向量维度。

    Args:
        *collection_names: 按优先级检查的集合名称。

    Returns:
        int: 找到的维度值；若未找到则返回 ``0``。
    """
    cfg = get_qdrant_config()
    client = QdrantClient(url=cfg.url, api_key=cfg.api_key, timeout=15.0)
    try:
        for collection_name in collection_names:
            if not str(collection_name or "").strip():
                continue
            try:
                collection = client.get_collection(collection_name)
            except Exception:
                continue
            params = getattr(collection.config, "params", None)
            vectors = getattr(params, "vectors", None)
            size = getattr(vectors, "size", None)
            if size:
                return int(size)
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    return 0


def _probe_embedding_dimension(adapter: Any, fallback_dim: int) -> int:
    """向远端嵌入接口探测真实维度。

    Args:
        adapter: 负责发起嵌入请求的适配器对象。
        fallback_dim: 探测失败时使用的默认维度。

    Returns:
        int: 探测成功后的真实维度，或回退维度。
    """
    try:
        detected = int(asyncio.run(adapter._detect_dimension()))  # noqa: SLF001
        if detected > 0:
            return detected
    except Exception as exc:
        logger.warning("Embedding dimension probe failed, fallback to configured dimension: %s", exc)
    return int(fallback_dim)


def _rebuild_graph_from_relations(graph_store: GraphStore, metadata_store: MetadataStore) -> int:
    """根据关系表内容重建图缓存。

    Args:
        graph_store: 需要重建的图存储对象。
        metadata_store: 提供三元组数据的元数据存储对象。

    Returns:
        int: 成功回灌到图中的关系数量。
    """
    triples = metadata_store.get_all_triples()
    if not triples:
        return 0

    graph_store.clear()
    nodes = sorted({str(subject) for subject, _, _, _ in triples} | {str(obj) for _, _, obj, _ in triples})
    if nodes:
        graph_store.add_nodes(nodes)
    graph_store.add_edges(
        [(str(subject), str(obj)) for subject, _, obj, _ in triples],
        weights=[1.0] * len(triples),
        relation_hashes=[str(hash_value) for _, _, _, hash_value in triples],
    )
    graph_store.save()
    return len(triples)


def build_context(settings: AppSettings) -> AppContext:
    """根据应用配置组装完整运行时上下文。

    Args:
        settings: 应用级配置对象。

    Returns:
        AppContext: 包含存储、检索、嵌入与服务对象的运行时上下文。
    """
    data_dir = settings.data_dir
    vectors_dir = data_dir / "vectors"
    graph_dir = data_dir / "graph"
    metadata_dir = data_dir / "metadata"
    data_dir.mkdir(parents=True, exist_ok=True)
    vectors_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    endpoint_cfg = resolve_openapi_endpoint_config(settings.config, section="embedding")
    retry_cfg = settings.get("embedding.retry", {}) or {}
    chunk_collection = str(settings.get("qdrant.chunk_collection", "na_memorix_chunks") or "na_memorix_chunks")
    relation_collection = str(settings.get("qdrant.relation_collection", "na_memorix_relations") or "na_memorix_relations")
    configured_dim = _safe_int(settings.get("embedding.dimension", 1024), 1024)
    existing_dim = _resolve_existing_qdrant_dimension(chunk_collection, relation_collection)
    initial_dim = existing_dim or configured_dim
    adapter = create_embedding_api_adapter(
        batch_size=_safe_int(settings.get("embedding.batch_size", 32), 32),
        max_concurrent=_safe_int(settings.get("embedding.max_concurrent", 5), 5),
        default_dimension=initial_dim,
        model_name=str(settings.get("embedding.model_name", "auto")),
        retry_config=retry_cfg,
        base_url=str(endpoint_cfg.get("base_url", "")),
        api_key=str(endpoint_cfg.get("api_key", "")),
        openai_model=str(endpoint_cfg.get("model", "")),
        timeout_seconds=_safe_float(endpoint_cfg.get("timeout_seconds", 30), 30.0),
        max_retries=_safe_int(endpoint_cfg.get("max_retries", 3), 3),
    )

    vector_dim = initial_dim
    auto_detect = bool(settings.get("embedding.auto_detect_dimension", True))
    if auto_detect and existing_dim <= 0:
        # 新库优先向远端探测维度，避免配置值与服务端真实输出不一致。
        probed_dim = _probe_embedding_dimension(adapter, configured_dim)
        if probed_dim != configured_dim:
            logger.info(
                "Vector dimension auto-aligned for fresh store: configured=%s, detected=%s",
                configured_dim,
                probed_dim,
            )
        vector_dim = probed_dim

    quantization_map = {
        "float32": QuantizationType.FLOAT32,
        "int8": QuantizationType.INT8,
        "pq": QuantizationType.PQ,
    }
    quantization_type = quantization_map.get(
        str(settings.get("embedding.quantization_type", "int8")).strip().lower(),
        QuantizationType.INT8,
    )

    matrix_format_map = {
        "csr": SparseMatrixFormat.CSR,
        "csc": SparseMatrixFormat.CSC,
    }
    matrix_format = matrix_format_map.get(
        str(settings.get("graph.sparse_matrix_format", "csr")).strip().lower(),
        SparseMatrixFormat.CSR,
    )
    graph_store = GraphStore(matrix_format=matrix_format, data_dir=graph_dir)
    table_prefix = str(settings.get("storage.table_prefix", "na_memorix") or "na_memorix").strip() or "na_memorix"
    metadata_store = MetadataStore(data_dir=metadata_dir, table_prefix=table_prefix)
    metadata_store.connect()
    graph_store.bind_metadata_store(metadata_store)
    vector_store = VectorStore(
        dimension=vector_dim,
        quantization_type=quantization_type,
        data_dir=vectors_dir,
        metadata_store=metadata_store,
        chunk_collection=chunk_collection,
        relation_collection=relation_collection,
    )
    vector_store.min_train_threshold = _safe_int(settings.get("embedding.min_train_threshold", 40), 40)

    sparse_index = None
    sparse_raw = settings.get("retrieval.sparse", {}) or {}
    try:
        sparse_cfg = SparseBM25Config(**(sparse_raw if isinstance(sparse_raw, dict) else {}))
        sparse_index = SparseBM25Index(metadata_store=metadata_store, config=sparse_cfg)
        if sparse_cfg.enabled and not sparse_cfg.lazy_load:
            sparse_index.ensure_loaded()
    except Exception as exc:
        logger.warning("Sparse index init failed, disabled: %s", exc)

    if vector_store.has_data():
        try:
            vector_store.load()
            logger.info("Loaded vector store with %s vectors", vector_store.num_vectors)
        except Exception as exc:
            logger.warning("Vector load failed: %s", exc)

    graph_pg_exists = metadata_store.graph_has_data()
    legacy_graph_exists = (graph_dir / "graph_metadata.pkl").exists()
    if graph_pg_exists or legacy_graph_exists:
        try:
            graph_store.load()
            if not graph_pg_exists and legacy_graph_exists:
                # 发现旧版本地图库时，首次加载后立即回写到 PostgreSQL 快照表。
                graph_store.save()
                logger.info("Migrated legacy graph cache into PostgreSQL graph tables")
            logger.info("Loaded graph store with %s nodes", graph_store.num_nodes)
        except Exception as exc:
            logger.warning("Graph load failed: %s", exc)
    else:
        try:
            # 没有图快照时，直接从关系表回放，保证新部署可立即查询图结构。
            rebuilt = _rebuild_graph_from_relations(graph_store, metadata_store)
            if rebuilt:
                logger.info("Rebuilt graph store from relation metadata: %s edges", rebuilt)
        except Exception as exc:
            logger.warning("Graph rebuild from relations skipped: %s", exc)

    try:
        if not getattr(graph_store, "_edge_hash_map", {}):
            triples = metadata_store.get_all_triples()
            if triples:
                rebuilt = graph_store.rebuild_edge_hash_map(triples)
                logger.info("Rebuilt edge hash map entries: %s", rebuilt)
                graph_store.save()
    except Exception as exc:
        logger.warning("Edge hash compatibility rebuild skipped: %s", exc)

    retrieval_raw = settings.get("retrieval", {}) or {}
    sparse_for_retriever = retrieval_raw.get("sparse", {}) if isinstance(retrieval_raw, dict) else {}
    fusion_for_retriever = retrieval_raw.get("fusion", {}) if isinstance(retrieval_raw, dict) else {}
    retriever_config = DualPathRetrieverConfig(
        top_k_paragraphs=_safe_int(settings.get("retrieval.top_k_paragraphs", 20), 20),
        top_k_relations=_safe_int(settings.get("retrieval.top_k_relations", 10), 10),
        top_k_final=_safe_int(settings.get("retrieval.top_k_final", 10), 10),
        alpha=_safe_float(settings.get("retrieval.alpha", 0.5), 0.5),
        enable_ppr=bool(settings.get("retrieval.enable_ppr", True)),
        ppr_alpha=_safe_float(settings.get("retrieval.ppr_alpha", 0.85), 0.85),
        ppr_concurrency_limit=_safe_int(settings.get("retrieval.ppr_concurrency_limit", 4), 4),
        enable_parallel=bool(settings.get("retrieval.enable_parallel", True)),
        debug=bool(settings.get("advanced.debug", False)),
        sparse=SparseBM25Config(**(sparse_for_retriever if isinstance(sparse_for_retriever, dict) else {})),
        fusion=FusionConfig(**(fusion_for_retriever if isinstance(fusion_for_retriever, dict) else {})),
    )

    retriever = DualPathRetriever(
        vector_store=vector_store,
        graph_store=graph_store,
        metadata_store=metadata_store,
        embedding_manager=adapter,
        sparse_index=sparse_index,
        config=retriever_config,
    )

    threshold_filter = DynamicThresholdFilter(
        ThresholdConfig(
            method=ThresholdMethod.ADAPTIVE,
            min_threshold=_safe_float(settings.get("threshold.min_threshold", 0.3), 0.3),
            max_threshold=_safe_float(settings.get("threshold.max_threshold", 0.95), 0.95),
            percentile=_safe_float(settings.get("threshold.percentile", 75.0), 75.0),
            std_multiplier=_safe_float(settings.get("threshold.std_multiplier", 1.5), 1.5),
            min_results=_safe_int(settings.get("threshold.min_results", 3), 3),
            enable_auto_adjust=bool(settings.get("threshold.enable_auto_adjust", True)),
        )
    )

    person_profile_service = PersonProfileService(
        metadata_store=metadata_store,
        graph_store=graph_store,
        vector_store=vector_store,
        embedding_manager=adapter,
        sparse_index=sparse_index,
        plugin_config=settings.config,
        retriever=retriever,
    )

    return AppContext(
        settings=settings,
        vector_store=vector_store,
        graph_store=graph_store,
        metadata_store=metadata_store,
        embedding_manager=adapter,
        sparse_index=sparse_index,
        retriever=retriever,
        threshold_filter=threshold_filter,
        person_profile_service=person_profile_service,
        data_dir=data_dir,
        config=settings.config,
    )
