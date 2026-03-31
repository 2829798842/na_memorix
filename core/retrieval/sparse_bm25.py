"""实现基于 PostgreSQL 的稀疏检索与 BM25 兼容接口。"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from amemorix.common.logging import get_logger

from ..storage import MetadataStore

logger = get_logger("A_Memorix.SparseBM25")

try:
    import jieba  # type: ignore

    HAS_JIEBA = True
except Exception:
    HAS_JIEBA = False
    jieba = None


@dataclass
class SparseBM25Config:
    """稀疏检索配置。

    Attributes:
        enabled (bool): 是否启用稀疏检索。
        backend (str): 稀疏检索后端名称。
        lazy_load (bool): 是否按需加载索引。
        mode (str): 检索模式。
        tokenizer_mode (str): 分词模式。
        jieba_user_dict (str): 结巴用户词典路径。
        char_ngram_n (int): 字符 n-gram 长度。
        candidate_k (int): 段落候选集大小。
        max_doc_len (int): 截断后的最大文档长度。
        enable_ngram_fallback_index (bool): 是否启用 n-gram 回退索引。
        enable_like_fallback (bool): 是否启用 ILIKE 回退。
        enable_relation_sparse_fallback (bool): 是否启用关系稀疏回退。
        relation_candidate_k (int): 关系候选集大小。
        relation_max_doc_len (int): 截断后的最大关系文档长度。
        unload_on_disable (bool): 停用时是否卸载索引。
        shrink_memory_on_unload (bool): 卸载时是否触发内存收缩钩子。
    """

    enabled: bool = True
    backend: str = "postgres"
    lazy_load: bool = True
    mode: str = "auto"
    tokenizer_mode: str = "jieba"
    jieba_user_dict: str = ""
    char_ngram_n: int = 2
    candidate_k: int = 80
    max_doc_len: int = 2000
    enable_ngram_fallback_index: bool = True
    enable_like_fallback: bool = True
    enable_relation_sparse_fallback: bool = True
    relation_candidate_k: int = 60
    relation_max_doc_len: int = 512
    unload_on_disable: bool = True
    shrink_memory_on_unload: bool = True

    def __post_init__(self) -> None:
        """标准化并校验稀疏检索配置。"""
        self.backend = str(self.backend or "postgres").strip().lower()
        self.mode = str(self.mode or "auto").strip().lower()
        self.tokenizer_mode = str(self.tokenizer_mode or "jieba").strip().lower()
        self.char_ngram_n = max(1, int(self.char_ngram_n))
        self.candidate_k = max(1, int(self.candidate_k))
        self.max_doc_len = max(0, int(self.max_doc_len))
        self.relation_candidate_k = max(1, int(self.relation_candidate_k))
        self.relation_max_doc_len = max(0, int(self.relation_max_doc_len))
        if self.backend != "postgres":
            raise ValueError(f"unsupported sparse backend: {self.backend}")
        if self.mode not in {"auto", "fallback_only", "hybrid"}:
            raise ValueError(f"invalid sparse.mode: {self.mode}")
        if self.tokenizer_mode not in {"jieba", "mixed", "char_2gram"}:
            raise ValueError(f"invalid sparse.tokenizer_mode: {self.tokenizer_mode}")


class SparseBM25Index:
    """管理段落与关系的稀疏检索索引。

    Attributes:
        metadata_store (MetadataStore): 元数据存储对象。
        config (SparseBM25Config): 稀疏检索配置。
        _loaded (bool): 当前索引是否已初始化。
        _jieba_dict_loaded (bool): 是否已加载用户词典。
    """

    def __init__(self, metadata_store: MetadataStore, config: Optional[SparseBM25Config] = None):
        """初始化稀疏检索索引。

        Args:
            metadata_store: 元数据存储对象。
            config: 稀疏检索配置；为空时使用默认配置。
        """
        self.metadata_store = metadata_store
        self.config = config or SparseBM25Config()
        self._loaded = False
        self._jieba_dict_loaded = False

    @property
    def loaded(self) -> bool:
        """返回当前索引是否已加载。

        Returns:
            bool: 已加载返回 ``True``。
        """
        return self._loaded

    def ensure_loaded(self) -> bool:
        """确保稀疏检索依赖的数据库结构已准备完成。

        Returns:
            bool: 加载成功返回 ``True``，否则返回 ``False``。
        """
        if not self.config.enabled:
            self._loaded = False
            return False
        if self.loaded:
            return True

        if not self.metadata_store.ensure_fts_schema():
            return False
        if not self.metadata_store.ensure_fts_backfilled():
            return False
        if self.config.enable_relation_sparse_fallback:
            self.metadata_store.ensure_relations_fts_schema()
            self.metadata_store.ensure_relations_fts_backfilled()
        if self.config.enable_ngram_fallback_index:
            self.metadata_store.ensure_paragraph_ngram_schema()
            self.metadata_store.ensure_paragraph_ngram_backfilled(n=self.config.char_ngram_n)

        self._prepare_tokenizer()
        self._loaded = True
        logger.info(
            "SparseBM25Index loaded: backend=%s tokenizer=%s mode=%s",
            self.config.backend,
            self.config.tokenizer_mode,
            self.config.mode,
        )
        return True

    def _prepare_tokenizer(self) -> None:
        """准备分词器及可选用户词典。"""
        if self._jieba_dict_loaded:
            return
        if self.config.tokenizer_mode not in {"jieba", "mixed"}:
            return
        if not HAS_JIEBA:
            logger.warning("jieba unavailable, sparse tokenizer will fall back to char n-grams")
            return
        user_dict = str(self.config.jieba_user_dict or "").strip()
        if user_dict:
            try:
                jieba.load_userdict(user_dict)  # type: ignore[union-attr]
                logger.info("Loaded jieba user dict: %s", user_dict)
            except Exception as exc:
                logger.warning("Failed to load jieba user dict: %s", exc)
        self._jieba_dict_loaded = True

    def unload(self) -> None:
        """将索引状态标记为未加载。"""
        self._loaded = False
        logger.info("SparseBM25Index unloaded")

    def maybe_unload(self) -> None:
        """在配置要求时卸载稀疏索引。"""
        if not self.config.enabled and self.config.unload_on_disable:
            self.unload()

    def _tokenize_jieba(self, text: str) -> List[str]:
        """使用结巴搜索模式切分文本。

        Args:
            text: 待分词文本。

        Returns:
            List[str]: 归一化后的词元列表。
        """
        if not HAS_JIEBA:
            return []
        try:
            tokens = list(jieba.cut_for_search(text))  # type: ignore[union-attr]
        except Exception:
            return []
        return [token.strip().lower() for token in tokens if token and token.strip()]

    def _tokenize_char_ngram(self, text: str, n: int) -> List[str]:
        """按字符 n-gram 切分文本。

        Args:
            text: 待分词文本。
            n: n-gram 长度。

        Returns:
            List[str]: n-gram 结果列表。
        """
        compact = re.sub(r"\s+", "", str(text or "").lower())
        if not compact:
            return []
        if len(compact) < n:
            return [compact]
        return [compact[idx : idx + n] for idx in range(0, len(compact) - n + 1)]

    def _tokenize(self, text: str) -> List[str]:
        """根据配置选择分词策略。

        Args:
            text: 待分词文本。

        Returns:
            List[str]: 去重后的词元列表。
        """
        raw = str(text or "").strip()
        if not raw:
            return []

        if self.config.tokenizer_mode == "jieba":
            tokens = self._tokenize_jieba(raw)
            return list(dict.fromkeys(tokens or self._tokenize_char_ngram(raw, self.config.char_ngram_n)))

        if self.config.tokenizer_mode == "mixed":
            tokens = self._tokenize_jieba(raw)
            tokens.extend(self._tokenize_char_ngram(raw, self.config.char_ngram_n))
            return list(dict.fromkeys([token for token in tokens if token]))

        return list(dict.fromkeys(self._tokenize_char_ngram(raw, self.config.char_ngram_n)))

    def _build_match_query(self, tokens: List[str]) -> str:
        """构造 PostgreSQL FTS 匹配表达式。

        Args:
            tokens: 已分词词元列表。

        Returns:
            str: 可用于全文检索的匹配表达式。
        """
        safe_tokens: List[str] = []
        for token in tokens:
            value = token.replace('"', '""').strip()
            if not value:
                continue
            safe_tokens.append(f'"{value}"')
        return " OR ".join(safe_tokens[:64])

    def _fallback_substring_search(self, tokens: List[str], limit: int) -> List[Dict[str, Any]]:
        """执行 n-gram 与 ILIKE 回退检索。

        Args:
            tokens: 分词结果。
            limit: 返回结果上限。

        Returns:
            List[Dict[str, Any]]: 回退检索结果列表。
        """
        if not tokens:
            return []

        uniq_tokens = [token for token in dict.fromkeys(tokens) if token][:32]
        if not uniq_tokens:
            return []

        if self.config.enable_ngram_fallback_index:
            try:
                self.metadata_store.ensure_paragraph_ngram_schema()
                self.metadata_store.ensure_paragraph_ngram_backfilled(n=self.config.char_ngram_n)
                rows = self.metadata_store.ngram_search_paragraphs(
                    uniq_tokens,
                    limit=limit,
                    max_doc_len=self.config.max_doc_len,
                )
                if rows:
                    return rows
            except Exception as exc:
                logger.warning("ngram fallback failed, will attempt ILIKE fallback: %s", exc)

        if not self.config.enable_like_fallback:
            return []

        conditions = " OR ".join(["content ILIKE %s"] * len(uniq_tokens))
        params: List[Any] = [f"%{token}%" for token in uniq_tokens]
        params.append(max(int(limit) * 8, 200))
        rows = self.metadata_store.query(
            f"""
            SELECT hash, content
            FROM paragraphs
            WHERE COALESCE(is_deleted, 0) = 0
              AND ({conditions})
            ORDER BY updated_at DESC NULLS LAST, created_at DESC
            LIMIT %s
            """,
            tuple(params),
        )
        if not rows:
            return []

        scored: List[Dict[str, Any]] = []
        token_count = max(1, len(uniq_tokens))
        for row in rows:
            content = str(row.get("content") or "")
            content_low = content.lower()
            matched = [token for token in uniq_tokens if token in content_low]
            if not matched:
                continue
            coverage = len(matched) / token_count
            length_bonus = sum(len(token) for token in matched) / max(1, len(content_low))
            fallback_score = coverage * 0.8 + length_bonus * 0.2
            scored.append(
                {
                    "hash": row["hash"],
                    "content": content[: self.config.max_doc_len] if self.config.max_doc_len > 0 else content,
                    "bm25_score": -float(fallback_score),
                    "fallback_score": float(fallback_score),
                }
            )
        scored.sort(key=lambda item: float(item.get("fallback_score", 0.0)), reverse=True)
        return scored[:limit]

    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        """检索段落内容。

        Args:
            query: 查询文本。
            k: 返回结果上限。

        Returns:
            List[Dict[str, Any]]: 排序后的段落检索结果。
        """
        if not self.config.enabled:
            return []
        if self.config.lazy_load and not self.loaded and not self.ensure_loaded():
            return []
        if not self.loaded:
            return []

        tokens = self._tokenize(query)
        match_query = self._build_match_query(tokens)
        if not match_query:
            return []

        limit = max(1, int(k))
        rows = self.metadata_store.fts_search_bm25(
            match_query=match_query,
            limit=limit,
            max_doc_len=self.config.max_doc_len,
        )
        if not rows:
            rows = self._fallback_substring_search(tokens=tokens, limit=limit)

        results: List[Dict[str, Any]] = []
        for rank, row in enumerate(rows, start=1):
            bm25_score = float(row.get("bm25_score", 0.0))
            results.append(
                {
                    "hash": row["hash"],
                    "content": row["content"],
                    "rank": rank,
                    "bm25_score": bm25_score,
                    "score": -bm25_score,
                }
            )
        return results

    def search_relations(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        """检索关系内容。

        Args:
            query: 查询文本。
            k: 返回结果上限。

        Returns:
            List[Dict[str, Any]]: 排序后的关系检索结果。
        """
        if not self.config.enable_relation_sparse_fallback:
            return []
        if self.config.lazy_load and not self.loaded and not self.ensure_loaded():
            return []
        if not self.loaded:
            return []

        tokens = self._tokenize(query)
        match_query = self._build_match_query(tokens)
        if not match_query:
            return []

        rows = self.metadata_store.fts_search_relations_bm25(
            match_query=match_query,
            limit=max(1, int(k)),
            max_doc_len=self.config.relation_max_doc_len,
        )
        out: List[Dict[str, Any]] = []
        for rank, row in enumerate(rows, start=1):
            bm25_score = float(row.get("bm25_score", 0.0))
            out.append(
                {
                    "hash": row["hash"],
                    "subject": row.get("subject", ""),
                    "predicate": row.get("predicate", ""),
                    "object": row.get("object", ""),
                    "content": row["content"],
                    "rank": rank,
                    "bm25_score": bm25_score,
                    "score": -bm25_score,
                }
            )
        return out

    def upsert_paragraph(self, paragraph_hash: str) -> bool:
        """同步单条段落到稀疏索引。

        Args:
            paragraph_hash: 段落哈希值。

        Returns:
            bool: 同步成功返回 ``True``。
        """
        if not self.ensure_loaded():
            return False
        return self.metadata_store.fts_upsert_paragraph(paragraph_hash)

    def delete_paragraph(self, paragraph_hash: str) -> bool:
        """从稀疏索引中删除段落。

        Args:
            paragraph_hash: 段落哈希值。

        Returns:
            bool: 删除成功返回 ``True``。
        """
        if not self.ensure_loaded():
            return False
        return self.metadata_store.fts_delete_paragraph(paragraph_hash)

    def stats(self) -> Dict[str, Any]:
        """返回当前稀疏索引状态。

        Returns:
            Dict[str, Any]: 索引配置与加载状态信息。
        """
        doc_count = self.metadata_store.fts_doc_count() if self.loaded else 0
        return {
            "enabled": bool(self.config.enabled),
            "backend": self.config.backend,
            "mode": self.config.mode,
            "tokenizer_mode": self.config.tokenizer_mode,
            "enable_ngram_fallback_index": self.config.enable_ngram_fallback_index,
            "enable_like_fallback": self.config.enable_like_fallback,
            "enable_relation_sparse_fallback": self.config.enable_relation_sparse_fallback,
            "loaded": bool(self._loaded),
            "has_jieba": HAS_JIEBA,
            "doc_count": doc_count,
            "candidate_k": self.config.candidate_k,
            "relation_candidate_k": self.config.relation_candidate_k,
        }
