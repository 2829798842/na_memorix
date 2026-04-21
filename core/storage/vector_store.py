"""封装基于 Qdrant 的向量存储与 A_memorix 兼容接口。"""

import asyncio
import threading
import uuid
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from amemorix.common.logging import get_logger
from qdrant_client import QdrantClient
from qdrant_client.http import models

from nekro_agent.core.vector_db import get_qdrant_config

logger = get_logger("A_Memorix.VectorStore")
_POINT_ID_NAMESPACE = uuid.UUID("4b5d8d8f-7e55-4ec0-a9f7-2f42827ed9dc")


class QuantizationType(Enum):
    """向量量化类型枚举。"""

    FLOAT32 = "float32"
    INT8 = "int8"
    PQ = "pq"


class VectorStore:
    """管理段落与关系向量的 Qdrant 存储。

    Attributes:
        dimension (int): 向量维度。
        quantization_type (QuantizationType): 量化类型。
        data_dir (Optional[Path]): 数据目录占位。
        metadata_store (Any): 元数据存储对象。
        chunk_collection (str): 段落向量集合名。
        relation_collection (str): 关系向量集合名。
        min_train_threshold (int): 兼容旧接口保留的训练阈值。
        _client (QdrantClient): Qdrant 客户端。
    """

    def __init__(
        self,
        dimension: int,
        quantization_type: QuantizationType = QuantizationType.FLOAT32,
        data_dir: Optional[Path | str] = None,
        *,
        metadata_store=None,
        chunk_collection: str = "na_memorix_chunks",
        relation_collection: str = "na_memorix_relations",
    ):
        """初始化向量存储。

        Args:
            dimension: 向量维度。
            quantization_type: 量化类型。
            data_dir: 数据目录占位。
            metadata_store: 元数据存储对象。
            chunk_collection: 段落向量集合名。
            relation_collection: 关系向量集合名。
        """
        self.dimension = max(1, int(dimension))
        self.quantization_type = quantization_type
        self.data_dir = Path(data_dir) if data_dir else None
        self.metadata_store = metadata_store
        self.chunk_collection = str(chunk_collection or "na_memorix_chunks")
        self.relation_collection = str(relation_collection or "na_memorix_relations")
        self.min_train_threshold = 0
        self._client = self._create_client()
        self._ensure_collection(self.chunk_collection)
        self._ensure_collection(self.relation_collection)

    def bind_metadata_store(self, metadata_store) -> None:
        """绑定元数据存储对象。

        Args:
            metadata_store: 元数据存储对象。
        """
        self.metadata_store = metadata_store

    def _create_client(self) -> QdrantClient:
        """创建 Qdrant 客户端。

        Returns:
            QdrantClient: 可直接操作集合的客户端。
        """
        cfg = get_qdrant_config()
        return QdrantClient(url=cfg.url, api_key=cfg.api_key, timeout=30.0)

    def _ensure_collection(self, name: str) -> None:
        """确保目标集合存在。

        Args:
            name: 集合名称。
        """
        try:
            self._client.get_collection(name)
            return
        except Exception:
            pass

        self._client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=self.dimension,
                distance=models.Distance.COSINE,
            ),
        )
        logger.info("Created Qdrant collection %s (dim=%s)", name, self.dimension)

    def _kind_for_id(self, point_id: str) -> str:
        """根据元数据推断向量所属类型。

        Args:
            point_id: 向量点 ID。

        Returns:
            str: ``relation`` 或 ``paragraph``。
        """
        if self.metadata_store is not None:
            try:
                if self.metadata_store.get_relation(point_id) is not None:
                    return "relation"
                if self.metadata_store.get_paragraph(point_id) is not None:
                    return "paragraph"
            except Exception:
                pass
        return "paragraph"

    def _table_prefix(self) -> str:
        """返回当前元数据表前缀。

        Returns:
            str: 元数据表前缀；若未绑定则返回空字符串。
        """
        if self.metadata_store is None:
            return ""
        return str(getattr(self.metadata_store, "table_prefix", "") or "")

    @staticmethod
    def _is_valid_qdrant_point_id(value: str) -> bool:
        """判断给定字符串是否已经是 Qdrant 可接受的 point id。"""
        normalized = str(value or "").strip()
        if not normalized:
            return False
        if normalized.isdigit():
            return True
        try:
            uuid.UUID(normalized)
        except ValueError:
            return False
        return True

    @classmethod
    def _point_id_for_hash(cls, hash_value: str) -> int | str:
        """将内部 hash 映射为 Qdrant 接受的整数或 UUID point id。"""
        normalized = str(hash_value or "").strip()
        if not normalized:
            raise ValueError("point hash is empty")
        if normalized.isdigit():
            return int(normalized)
        if cls._is_valid_qdrant_point_id(normalized):
            return normalized
        return str(uuid.uuid5(_POINT_ID_NAMESPACE, normalized))

    @classmethod
    def _candidate_point_ids(cls, hash_value: str) -> list[int | str]:
        """返回用于兼容查询/删除的候选 point id 列表。"""
        normalized = str(hash_value or "").strip()
        if not normalized:
            return []
        candidates: list[int | str] = [cls._point_id_for_hash(normalized)]
        if cls._is_valid_qdrant_point_id(normalized):
            raw: int | str = int(normalized) if normalized.isdigit() else normalized
            if raw not in candidates:
                candidates.append(raw)
        return candidates

    @staticmethod
    def _hash_from_hit(hit: object) -> str:
        """优先从 payload 中还原内部 hash，缺失时回退到 point id。"""
        payload = getattr(hit, "payload", None)
        if isinstance(payload, dict):
            hash_value = str(payload.get("hash", "") or "").strip()
            if hash_value:
                return hash_value
        return str(getattr(hit, "id", "") or "").strip()

    @staticmethod
    def _run_async(coro):
        """在同步上下文中执行异步协程。

        Args:
            coro: 待执行协程。

        Returns:
            Any: 协程执行结果。
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result: dict[str, object] = {}
        error: dict[str, BaseException] = {}

        def _worker() -> None:
            try:
                result["value"] = asyncio.run(coro)
            except BaseException as exc:  # pragma: no cover - passthrough helper
                error["exc"] = exc

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        thread.join()
        if "exc" in error:
            raise error["exc"]
        return result.get("value")

    def _normalize_vectors(self, vectors: np.ndarray | Sequence[Sequence[float]] | Sequence[float]) -> np.ndarray:
        """标准化输入向量形状并校验维度。

        Args:
            vectors: 原始向量输入。

        Returns:
            np.ndarray: 规范化后的二维向量矩阵。

        Raises:
            ValueError: 当向量维度与存储配置不一致时抛出。
        """
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.dimension:
            raise ValueError(f"vector dimension mismatch: expected {self.dimension}, got {arr.shape[1]}")
        return arr

    def add(self, vectors, ids: Sequence[str], *, kinds: Optional[Sequence[str]] = None) -> int:
        """写入一批段落或关系向量。

        Args:
            vectors: 向量输入。
            ids: 向量对应的哈希列表。
            kinds: 显式指定的向量类型列表。

        Returns:
            int: 成功写入的向量数量。
        """
        arr = self._normalize_vectors(vectors)
        if len(ids) != arr.shape[0]:
            raise ValueError("ids and vector rows length mismatch")
        if kinds is not None and len(kinds) != len(ids):
            raise ValueError("kinds and ids length mismatch")

        chunk_points: list[models.PointStruct] = []
        relation_points: list[models.PointStruct] = []
        for idx, point_id in enumerate(ids):
            point_hash = str(point_id)
            qdrant_point_id = self._point_id_for_hash(point_hash)
            kind = str(kinds[idx] if kinds is not None else self._kind_for_id(point_hash)).strip().lower() or "paragraph"
            payload = {
                "hash": point_hash,
                "kind": kind,
                "table_prefix": self._table_prefix(),
            }
            point = models.PointStruct(id=qdrant_point_id, vector=arr[idx].tolist(), payload=payload)
            if kind == "relation":
                relation_points.append(point)
            else:
                chunk_points.append(point)

        if chunk_points:
            self._client.upsert(collection_name=self.chunk_collection, points=chunk_points, wait=True)
        if relation_points:
            self._client.upsert(collection_name=self.relation_collection, points=relation_points, wait=True)
        return len(ids)

    def search(self, query_emb, k: int = 10):
        """在两个集合中检索最相近的向量。

        Args:
            query_emb: 查询向量。
            k: 返回结果上限。

        Returns:
            tuple[list[str], list[float]]: 命中 ID 列表与对应分数列表。
        """
        arr = np.asarray(query_emb, dtype=np.float32).reshape(-1)
        limit = max(1, int(k))
        merged: dict[str, float] = {}

        for collection_name in (self.chunk_collection, self.relation_collection):
            try:
                hits = self._client.search(
                    collection_name=collection_name,
                    query_vector=arr.tolist(),
                    limit=limit,
                    with_payload=True,
                )
            except Exception as exc:
                logger.warning("Qdrant search failed on %s: %s", collection_name, exc)
                continue
            for hit in hits:
                point_id = self._hash_from_hit(hit)
                score = float(hit.score)
                if point_id not in merged or score > merged[point_id]:
                    merged[point_id] = score

        ordered = sorted(merged.items(), key=lambda item: item[1], reverse=True)[:limit]
        return [item[0] for item in ordered], [item[1] for item in ordered]

    def delete(self, ids: Iterable[str]) -> int:
        """删除一批向量点。

        Args:
            ids: 待删除点 ID 列表。

        Returns:
            int: 删除的点数量。
        """
        point_ids: list[int | str] = []
        for item in ids:
            point_ids.extend(self._candidate_point_ids(str(item)))
        point_ids = list(dict.fromkeys(point_ids))
        if not point_ids:
            return 0
        selector = models.PointIdsList(points=point_ids)
        deleted = 0
        for collection_name in (self.chunk_collection, self.relation_collection):
            try:
                self._client.delete(collection_name=collection_name, points_selector=selector, wait=True)
                deleted = len(point_ids)
            except Exception as exc:
                logger.warning("Qdrant delete failed on %s: %s", collection_name, exc)
        return deleted

    def clear(self) -> None:
        """清空并重建段落、关系两个集合。"""
        for collection_name in (self.chunk_collection, self.relation_collection):
            try:
                self._client.delete_collection(collection_name)
            except Exception:
                pass
            self._ensure_collection(collection_name)

    def rebuild_index(self, *, embedding_manager=None, batch_size: int = 32) -> None:
        """根据 PostgreSQL 中的内容重建 Qdrant 向量集合。

        Args:
            embedding_manager: 用于重新编码文本的嵌入管理器。
            batch_size: 编码与写入的批大小。
        """
        self._ensure_collection(self.chunk_collection)
        self._ensure_collection(self.relation_collection)
        if embedding_manager is None or self.metadata_store is None:
            logger.info(
                "VectorStore rebuild_index completed via Qdrant collection verification (%s, %s)",
                self.chunk_collection,
                self.relation_collection,
            )
            return

        batch_size = max(1, int(batch_size))
        paragraph_rows = self.metadata_store.query(
            """
            SELECT hash, content
            FROM paragraphs
            WHERE COALESCE(is_deleted, 0) = 0
            ORDER BY created_at ASC
            """
        )
        relation_rows = self.metadata_store.query(
            """
            SELECT hash, subject, predicate, object
            FROM relations
            ORDER BY created_at ASC
            """
        )

        # 先清空再全量回放，确保集合内容与元数据表保持一致。
        self.clear()

        for start in range(0, len(paragraph_rows), batch_size):
            batch = paragraph_rows[start : start + batch_size]
            texts = [str(row.get("content") or "") for row in batch]
            vectors = self._run_async(embedding_manager.encode_batch(texts))
            self.add(vectors, [str(row.get("hash") or "") for row in batch], kinds=["paragraph"] * len(batch))

        for start in range(0, len(relation_rows), batch_size):
            batch = relation_rows[start : start + batch_size]
            texts = [
                f"{row.get('subject', '')} {row.get('predicate', '')} {row.get('object', '')}"
                for row in batch
            ]
            vectors = self._run_async(embedding_manager.encode_batch(texts))
            self.add(vectors, [str(row.get("hash") or "") for row in batch], kinds=["relation"] * len(batch))

        logger.info(
            "VectorStore rebuilt from PostgreSQL into Qdrant (%s paragraphs, %s relations)",
            len(paragraph_rows),
            len(relation_rows),
        )

    def save(self, data_dir: Optional[Path | str] = None) -> None:
        """保留旧接口的持久化钩子。

        Args:
            data_dir: 兼容旧接口保留的目录参数。
        """
        del data_dir
        # Qdrant is persistent; keeping this as a compatibility no-op.
        self._ensure_collection(self.chunk_collection)
        self._ensure_collection(self.relation_collection)

    def load(self, data_dir: Optional[Path | str] = None) -> None:
        """保留旧接口的加载钩子。

        Args:
            data_dir: 兼容旧接口保留的目录参数。
        """
        del data_dir
        self._ensure_collection(self.chunk_collection)
        self._ensure_collection(self.relation_collection)

    def has_data(self) -> bool:
        """判断当前是否存在已存储向量。

        Returns:
            bool: 至少存在一条向量时返回 ``True``。
        """
        return self.num_vectors > 0

    @property
    def num_vectors(self) -> int:
        """统计两个集合中的向量总数。

        Returns:
            int: 向量总数。
        """
        total = 0
        for collection_name in (self.chunk_collection, self.relation_collection):
            try:
                total += int(self._client.count(collection_name=collection_name, exact=False).count)
            except Exception:
                continue
        return total

    def __contains__(self, hash_value: str) -> bool:
        """判断指定哈希是否已存在于任一集合。

        Args:
            hash_value: 待检查的点 ID。

        Returns:
            bool: 命中任一集合返回 ``True``。
        """
        point_id = str(hash_value or "").strip()
        if not point_id:
            return False
        candidate_ids = self._candidate_point_ids(point_id)
        if not candidate_ids:
            return False
        for collection_name in (self.chunk_collection, self.relation_collection):
            try:
                records = self._client.retrieve(collection_name=collection_name, ids=candidate_ids, with_payload=False)
            except Exception:
                continue
            if records:
                return True
        return False
