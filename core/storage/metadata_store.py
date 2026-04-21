"""实现基于 PostgreSQL 的元数据存储，并兼容 A_memorix 旧接口。"""

import json
import pickle
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import psycopg2
from amemorix.common.logging import get_logger
from psycopg2.extras import RealDictCursor

from nekro_agent.core.tortoise_config import resolve_db_url

from ..utils.hash import compute_hash, normalize_text
from ..utils.runtime_dependencies import load_jieba
from ..utils.time_parser import normalize_time_meta

logger = get_logger("A_Memorix.MetadataStore")

_BUILTIN_SOURCE_PREFIX = "builtin_memory:"
_CHAT_SUMMARY_SOURCE_PREFIX = "chat_summary:"

_TABLE_NAMES = (
    "paragraphs",
    "entities",
    "relations",
    "paragraph_relations",
    "paragraph_entities",
    "paragraph_ngrams",
    "paragraph_ngram_meta",
    "deleted_relations",
    "graph_nodes",
    "graph_edges",
    "person_profile_switches",
    "person_profile_active_persons",
    "person_profile_snapshots",
    "person_profile_overrides",
    "person_registry",
    "transcript_sessions",
    "transcript_messages",
    "async_tasks",
)

def _json_default(value: Any) -> Any:
    """为 JSON 序列化提供扩展类型转换。

    Args:
        value: 待序列化对象。

    Returns:
        Any: 可直接序列化的值。

    Raises:
        TypeError: 当对象类型不支持 JSON 序列化时抛出。
    """
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _encode_json(value: Any) -> Optional[str]:
    """将对象编码为 JSON 字符串。

    Args:
        value: 待编码对象。

    Returns:
        Optional[str]: JSON 字符串；输入为空时返回 ``None``。
    """
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, default=_json_default)


def _decode_json(raw: Any, default: Any) -> Any:
    """兼容解析 JSON、bytes 或 pickle 载荷。

    Args:
        raw: 原始数据库字段值。
        default: 解析失败时回退的默认值。

    Returns:
        Any: 解析后的对象，失败时返回默认值。
    """
    if raw is None:
        return default
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, memoryview):
        raw = raw.tobytes()
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = bytes(raw).decode("utf-8")
        except Exception:
            try:
                return pickle.loads(bytes(raw))
            except Exception:
                return default
    try:
        return json.loads(str(raw))
    except Exception:
        try:
            return pickle.loads(raw if isinstance(raw, (bytes, bytearray)) else str(raw).encode("latin1"))
        except Exception:
            return default


def _parse_builtin_workspace_id(value: Any) -> Optional[int]:
    text = str(value or "").strip()
    if not text.startswith(_BUILTIN_SOURCE_PREFIX):
        return None
    parts = text.split(":")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except (TypeError, ValueError):
        return None


def _parse_chat_summary_chat_key(value: Any) -> str:
    text = str(value or "").strip()
    if not text.startswith(_CHAT_SUMMARY_SOURCE_PREFIX):
        return ""
    return text[len(_CHAT_SUMMARY_SOURCE_PREFIX) :].strip()


def _resolve_scope_namespace(source: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    metadata_dict = dict(metadata or {})
    normalized_source = str(
        source or metadata_dict.get("record_source", "") or metadata_dict.get("source", "") or ""
    ).strip()

    workspace_id: Optional[int] = None
    workspace_raw = metadata_dict.get("builtin_workspace_id")
    if workspace_raw is not None:
        try:
            workspace_id = int(workspace_raw)
        except (TypeError, ValueError):
            workspace_id = None
    if workspace_id is None:
        workspace_id = _parse_builtin_workspace_id(normalized_source)

    chat_key = str(metadata_dict.get("chat_summary_chat_key", "") or "").strip()
    if not chat_key:
        chat_key = _parse_chat_summary_chat_key(normalized_source)

    scope_parts: List[str] = []
    if workspace_id is not None:
        scope_parts.append(f"workspace:{workspace_id}")
    if chat_key:
        scope_parts.append(f"chat:{chat_key}")
    return "|".join(scope_parts)


class _CompatCursor:
    """为旧版调用点提供近似 sqlite 行为的游标包装器。"""

    def __init__(self, store: "MetadataStore"):
        """初始化兼容游标。

        Args:
            store: 元数据存储对象。
        """
        self._store = store
        self._cursor = store._pg_conn.cursor()  # type: ignore[union-attr]

    def execute(self, sql: str, params: Optional[Sequence[Any]] = None) -> "_CompatCursor":
        """执行单条 SQL 并返回当前游标。

        Args:
            sql: 待执行 SQL。
            params: SQL 参数。

        Returns:
            _CompatCursor: 当前游标对象。
        """
        translated = self._store._translate_sql(sql, for_compat=True)
        self._cursor.execute(translated, tuple(params or ()))
        return self

    def executemany(self, sql: str, seq_of_params: Iterable[Sequence[Any]]) -> "_CompatCursor":
        """批量执行 SQL。

        Args:
            sql: 待执行 SQL。
            seq_of_params: 参数列表。

        Returns:
            _CompatCursor: 当前游标对象。
        """
        translated = self._store._translate_sql(sql, for_compat=True)
        self._cursor.executemany(translated, list(seq_of_params))
        return self

    def fetchone(self) -> Any:
        """获取一行查询结果。"""
        return self._cursor.fetchone()

    def fetchall(self) -> List[Any]:
        """获取全部查询结果。"""
        return self._cursor.fetchall()

    def fetchmany(self, size: Optional[int] = None) -> List[Any]:
        """按指定大小分批获取查询结果。"""
        return self._cursor.fetchmany(size)

    @property
    def rowcount(self) -> int:
        """返回最近一次操作影响的行数。"""
        return int(self._cursor.rowcount or 0)

    def close(self) -> None:
        """关闭底层游标。"""
        self._cursor.close()

    def __enter__(self) -> "_CompatCursor":
        """进入上下文管理器。"""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """退出上下文管理器并关闭游标。"""
        self.close()


class _CompatConnection:
    """暴露最小 sqlite 风格接口的连接包装器。"""

    def __init__(self, store: "MetadataStore"):
        """初始化兼容连接。

        Args:
            store: 元数据存储对象。
        """
        self._store = store

    def cursor(self) -> _CompatCursor:
        """创建兼容游标。

        Returns:
            _CompatCursor: 新建的兼容游标。
        """
        return _CompatCursor(self._store)

    def commit(self) -> None:
        """提交当前事务。"""
        self._store._pg_conn.commit()  # type: ignore[union-attr]

    def rollback(self) -> None:
        """回滚当前事务。"""
        self._store._pg_conn.rollback()  # type: ignore[union-attr]

    def close(self) -> None:
        """关闭兼容连接。"""
        self._store.close()

    @property
    def closed(self) -> bool:
        """返回底层连接是否已关闭。"""
        return bool(self._store._pg_conn is None or self._store._pg_conn.closed)

    @property
    def in_transaction(self) -> bool:
        """返回底层连接是否处于事务中。"""
        return bool(self._store._pg_conn is not None and self._store._pg_conn.status != psycopg2.extensions.STATUS_READY)


class MetadataStore:
    """管理段落、关系、人物画像与异步任务等元数据。

    Attributes:
        data_dir (Optional[Path]): 数据目录占位。
        db_name (str): 兼容旧接口保留的数据库文件名。
        table_prefix (str): PostgreSQL 表名前缀。
        _pg_conn (Any): psycopg2 PostgreSQL 连接对象。
        _conn (Optional[_CompatConnection]): sqlite 风格兼容连接。
        _is_initialized (bool): 是否已完成表结构初始化。
        _db_path (Optional[Path]): 兼容旧接口保留的数据库路径。
        _db_url (str): PostgreSQL 连接地址。
    """

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        db_name: str = "metadata.db",
        table_prefix: str = "na_memorix",
    ):
        """初始化元数据存储配置。

        Args:
            data_dir: 数据目录占位。
            db_name: 兼容旧接口保留的数据库文件名。
            table_prefix: PostgreSQL 表名前缀。
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self.db_name = db_name
        self.table_prefix = str(table_prefix or "na_memorix").strip() or "na_memorix"
        self._pg_conn = None
        self._conn: Optional[_CompatConnection] = None
        self._is_initialized = False
        self._db_path: Optional[Path] = None
        self._db_url = resolve_db_url()

    def __repr__(self) -> str:
        """返回元数据存储的调试摘要字符串。"""
        return f"MetadataStore(prefix={self.table_prefix!r}, connected={self.is_connected()})"

    def __enter__(self) -> "MetadataStore":
        """进入上下文并确保数据库已连接。"""
        if not self.is_connected():
            self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """退出上下文并关闭数据库连接。"""
        self.close()

    def _table(self, name: str) -> str:
        """拼接带前缀的实际表名。

        Args:
            name: 逻辑表名。

        Returns:
            str: 带前缀的实际表名。
        """
        return f"{self.table_prefix}_{name}"

    def _translate_sql(self, sql: str, *, for_compat: bool = False) -> str:
        """将旧版 SQL 翻译为 PostgreSQL 可执行语句。

        Args:
            sql: 原始 SQL。
            for_compat: 是否启用兼容替换规则。

        Returns:
            str: 翻译后的 SQL。
        """
        translated = str(sql)
        for table_name in _TABLE_NAMES:
            translated = re.sub(rf"\b{table_name}\b", self._table(table_name), translated)
        translated = translated.replace("?", "%s")
        if for_compat:
            translated = re.sub(r"\bLIKE\b", "ILIKE", translated, flags=re.IGNORECASE)
        return translated

    def _ensure_connected(self) -> None:
        """确保 PostgreSQL 连接可用。

        Raises:
            RuntimeError: 当数据库尚未连接时抛出。
        """
        if self._pg_conn is None or self._pg_conn.closed:
            raise RuntimeError("MetadataStore is not connected")

    def _cursor(self):
        """创建普通 PostgreSQL 游标。"""
        self._ensure_connected()
        return self._pg_conn.cursor()  # type: ignore[union-attr]

    def _dict_cursor(self):
        """创建返回字典行的 PostgreSQL 游标。"""
        self._ensure_connected()
        return self._pg_conn.cursor(cursor_factory=RealDictCursor)  # type: ignore[union-attr]

    @staticmethod
    def _canonicalize_name(name: str) -> str:
        return str(name or "").strip().lower()

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _unsupported_backend(method_name: str) -> None:
        raise NotImplementedError(f"{method_name} is unsupported in na_memorix PG/Qdrant backend")

    @staticmethod
    def _char_ngrams(text: str, n: int) -> List[str]:
        compact = "".join(str(text or "").lower().split())
        if not compact:
            return []
        if len(compact) < n:
            return [compact]
        return [compact[idx : idx + n] for idx in range(0, len(compact) - n + 1)]

    def _tokenize_for_search(self, text: str, tokenizer_mode: str = "mixed", char_ngram_n: int = 2) -> List[str]:
        raw = str(text or "").strip()
        if not raw:
            return []

        mode = str(tokenizer_mode or "mixed").strip().lower()
        tokens: List[str] = []
        jieba_module = load_jieba(install_if_missing=mode in {"jieba", "mixed"})
        if mode in {"jieba", "mixed"} and jieba_module is not None:
            try:
                tokens.extend(
                    [
                        token.strip().lower()
                        for token in jieba_module.cut_for_search(raw)
                        if token and token.strip()
                    ]
                )
            except Exception:
                pass

        if mode in {"mixed", "char_2gram"} or not tokens:
            tokens.extend(self._char_ngrams(raw, max(1, int(char_ngram_n))))

        return [token for token in dict.fromkeys(tokens) if token]

    def _build_search_lexemes(self, text: str, tokenizer_mode: str = "mixed", char_ngram_n: int = 2) -> str:
        return " ".join(self._tokenize_for_search(text, tokenizer_mode=tokenizer_mode, char_ngram_n=char_ngram_n))

    @staticmethod
    def _parse_match_query(match_query: str) -> List[str]:
        raw = str(match_query or "").strip()
        if not raw:
            return []
        quoted = re.findall(r'"([^"]+)"', raw)
        if quoted:
            return [token.strip().lower() for token in quoted if token and token.strip()]
        parts = re.split(r"\s+OR\s+|\|", raw, flags=re.IGNORECASE)
        return [token.strip().strip('"').lower() for token in parts if token and token.strip().strip('"')]

    @staticmethod
    def _build_tsquery(tokens: Sequence[str]) -> str:
        cleaned: List[str] = []
        for token in tokens:
            text = re.sub(r"[&|!:()<>]", " ", str(token or "").strip().lower()).replace("'", " ")
            cleaned.extend([piece for piece in text.split() if piece])
        uniq = [token for token in dict.fromkeys(cleaned) if token]
        return " | ".join(uniq[:64])

    def connect(self, data_dir: Optional[Union[str, Path]] = None) -> None:
        """连接 PostgreSQL 并按需初始化表结构。

        Args:
            data_dir: 兼容旧接口保留的数据目录参数。
        """
        if data_dir is not None:
            self.data_dir = Path(data_dir)
        if self.data_dir is not None:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self._db_path = self.data_dir / self.db_name
        elif self._db_path is None:
            self._db_path = Path(self.db_name)

        if self._pg_conn is not None and not self._pg_conn.closed:
            return

        self._pg_conn = psycopg2.connect(self._db_url)
        self._pg_conn.autocommit = False
        self._conn = _CompatConnection(self)
        self._ensure_schema()
        self._is_initialized = True
        logger.info("Connected metadata store to PostgreSQL with prefix %s", self.table_prefix)

    def close(self) -> None:
        """关闭 PostgreSQL 连接并清理兼容连接对象。"""
        if self._pg_conn is not None and not self._pg_conn.closed:
            self._pg_conn.close()
        self._pg_conn = None
        self._conn = None

    def is_connected(self) -> bool:
        """判断数据库连接是否可用。

        Returns:
            bool: 已连接返回 ``True``。
        """
        return bool(self._pg_conn is not None and not self._pg_conn.closed)

    def get_db_path(self) -> Path:
        """返回兼容旧接口暴露的数据库路径。

        Returns:
            Path: 逻辑数据库路径。
        """
        if self._db_path is not None:
            return self._db_path
        if self.data_dir is not None:
            return self.data_dir / self.db_name
        return Path(self.db_name)

    def _fetchone_dict(self, sql: str, params: Sequence[Any] = ()) -> Optional[Dict[str, Any]]:
        with self._dict_cursor() as cursor:
            cursor.execute(sql, tuple(params))
            row = cursor.fetchone()
            return dict(row) if row is not None else None

    def _fetchall_dict(self, sql: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        with self._dict_cursor() as cursor:
            cursor.execute(sql, tuple(params))
            return [dict(row) for row in cursor.fetchall()]

    def _execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        with self._cursor() as cursor:
            cursor.execute(sql, tuple(params))
            return int(cursor.rowcount or 0)

    def _maybe_commit(self) -> None:
        self._pg_conn.commit()  # type: ignore[union-attr]

    def rollback(self) -> None:
        if self._pg_conn is not None and not self._pg_conn.closed:
            self._pg_conn.rollback()

    def _decode_paragraph(self, row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        data = dict(row)
        data.pop("search_lexemes", None)
        data.pop("search_document", None)
        metadata = _decode_json(data.pop("metadata_json", None), {})
        data["metadata"] = metadata if isinstance(metadata, dict) else {}
        data["is_deleted"] = bool(data.get("is_deleted", 0))
        data["is_permanent"] = bool(data.get("is_permanent", 0))
        return data

    def _decode_entity(self, row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        data = dict(row)
        data.pop("canonical_name", None)
        metadata = _decode_json(data.pop("metadata_json", None), {})
        data["metadata"] = metadata if isinstance(metadata, dict) else {}
        data["is_deleted"] = bool(data.get("is_deleted", 0))
        return data

    def _decode_relation(self, row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        data = dict(row)
        data.pop("subject_canonical", None)
        data.pop("predicate_canonical", None)
        data.pop("object_canonical", None)
        data.pop("search_lexemes", None)
        data.pop("search_document", None)
        metadata = _decode_json(data.pop("metadata_json", None), {})
        data["metadata"] = metadata if isinstance(metadata, dict) else {}
        data["is_inactive"] = bool(data.get("is_inactive", 0))
        data["is_pinned"] = bool(data.get("is_pinned", 0))
        data["is_permanent"] = bool(data.get("is_permanent", 0))
        return data

    def _decode_deleted_relation(self, row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        decoded = self._decode_relation(row)
        if decoded is None:
            return None
        decoded["support_paragraphs"] = _decode_json(decoded.pop("support_paragraphs_json", None), [])
        return decoded

    def _decode_snapshot(self, row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        data = dict(row)
        data["aliases"] = _decode_json(data.pop("aliases_json", None), [])
        data["relation_edges"] = _decode_json(data.pop("relation_edges_json", None), [])
        data["vector_evidence"] = _decode_json(data.pop("vector_evidence_json", None), [])
        data["evidence_ids"] = _decode_json(data.pop("evidence_ids_json", None), [])
        return data

    def _decode_override(self, row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        return dict(row) if row is not None else None

    def _decode_registry(self, row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        data = dict(row)
        data["metadata"] = _decode_json(data.pop("metadata_json", None), {})
        data["group_nick_name"] = _decode_json(data.get("group_nick_name"), data.get("group_nick_name"))
        data["memory_points"] = _decode_json(data.get("memory_points"), data.get("memory_points"))
        return data

    def _decode_task(self, row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        data = dict(row)
        data["payload"] = _decode_json(data.pop("payload_json", None), {})
        data["result"] = _decode_json(data.pop("result_json", None), {})
        data["cancel_requested"] = bool(data.get("cancel_requested", 0))
        return data

    def _decode_transcript_message(self, row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        data = dict(row)
        data["metadata"] = _decode_json(data.pop("metadata_json", None), {})
        return data

    def _ensure_schema(self) -> None:
        with self._cursor() as cursor:
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table('paragraphs')} (
                    hash TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    vector_index INTEGER,
                    created_at DOUBLE PRECISION,
                    updated_at DOUBLE PRECISION,
                    metadata_json TEXT,
                    source TEXT,
                    word_count INTEGER,
                    event_time DOUBLE PRECISION,
                    event_time_start DOUBLE PRECISION,
                    event_time_end DOUBLE PRECISION,
                    time_granularity TEXT,
                    time_confidence DOUBLE PRECISION DEFAULT 1.0,
                    knowledge_type TEXT DEFAULT 'mixed',
                    is_deleted INTEGER DEFAULT 0,
                    is_permanent INTEGER DEFAULT 0,
                    deleted_at DOUBLE PRECISION,
                    last_accessed DOUBLE PRECISION,
                    access_count INTEGER DEFAULT 0
                )
                """
            )
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table('entities')} (
                    hash TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    canonical_name TEXT NOT NULL UNIQUE,
                    vector_index INTEGER,
                    appearance_count INTEGER DEFAULT 1,
                    created_at DOUBLE PRECISION,
                    metadata_json TEXT,
                    is_deleted INTEGER DEFAULT 0,
                    deleted_at DOUBLE PRECISION
                )
                """
            )
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table('relations')} (
                    hash TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    subject_canonical TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    predicate_canonical TEXT NOT NULL,
                    object TEXT NOT NULL,
                    object_canonical TEXT NOT NULL,
                    vector_index INTEGER,
                    confidence DOUBLE PRECISION DEFAULT 1.0,
                    created_at DOUBLE PRECISION,
                    source_paragraph TEXT,
                    metadata_json TEXT,
                    last_accessed DOUBLE PRECISION,
                    access_count INTEGER DEFAULT 0,
                    is_inactive INTEGER DEFAULT 0,
                    inactive_since DOUBLE PRECISION,
                    is_permanent INTEGER DEFAULT 0,
                    is_pinned INTEGER DEFAULT 0,
                    protected_until DOUBLE PRECISION,
                    last_reinforced DOUBLE PRECISION
                )
                """
            )
            cursor.execute(
                f"""
                ALTER TABLE {self._table('relations')}
                DROP CONSTRAINT IF EXISTS {self.table_prefix}_relations_unique
                """
            )
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table('paragraph_relations')} (
                    paragraph_hash TEXT NOT NULL,
                    relation_hash TEXT NOT NULL,
                    PRIMARY KEY (paragraph_hash, relation_hash),
                    FOREIGN KEY (paragraph_hash) REFERENCES {self._table('paragraphs')}(hash) ON DELETE CASCADE,
                    FOREIGN KEY (relation_hash) REFERENCES {self._table('relations')}(hash) ON DELETE CASCADE
                )
                """
            )
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table('paragraph_entities')} (
                    paragraph_hash TEXT NOT NULL,
                    entity_hash TEXT NOT NULL,
                    mention_count INTEGER DEFAULT 1,
                    PRIMARY KEY (paragraph_hash, entity_hash),
                    FOREIGN KEY (paragraph_hash) REFERENCES {self._table('paragraphs')}(hash) ON DELETE CASCADE,
                    FOREIGN KEY (entity_hash) REFERENCES {self._table('entities')}(hash) ON DELETE CASCADE
                )
                """
            )
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table('deleted_relations')} (
                    hash TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    subject_canonical TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    predicate_canonical TEXT NOT NULL,
                    object TEXT NOT NULL,
                    object_canonical TEXT NOT NULL,
                    vector_index INTEGER,
                    confidence DOUBLE PRECISION DEFAULT 1.0,
                    created_at DOUBLE PRECISION,
                    source_paragraph TEXT,
                    metadata_json TEXT,
                    last_accessed DOUBLE PRECISION,
                    access_count INTEGER DEFAULT 0,
                    is_inactive INTEGER DEFAULT 0,
                    inactive_since DOUBLE PRECISION,
                    is_pinned INTEGER DEFAULT 0,
                    protected_until DOUBLE PRECISION,
                    last_reinforced DOUBLE PRECISION,
                    deleted_at DOUBLE PRECISION,
                    support_paragraphs_json TEXT
                )
                """
            )
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table('person_profile_switches')} (
                    stream_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 0,
                    updated_at DOUBLE PRECISION NOT NULL,
                    PRIMARY KEY (stream_id, user_id)
                )
                """
            )
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table('person_profile_active_persons')} (
                    stream_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    person_id TEXT NOT NULL,
                    last_seen_at DOUBLE PRECISION NOT NULL,
                    PRIMARY KEY (stream_id, user_id, person_id)
                )
                """
            )
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table('person_profile_snapshots')} (
                    snapshot_id BIGSERIAL PRIMARY KEY,
                    person_id TEXT NOT NULL,
                    profile_version INTEGER NOT NULL,
                    profile_text TEXT NOT NULL,
                    aliases_json TEXT,
                    relation_edges_json TEXT,
                    vector_evidence_json TEXT,
                    evidence_ids_json TEXT,
                    updated_at DOUBLE PRECISION NOT NULL,
                    expires_at DOUBLE PRECISION,
                    source_note TEXT,
                    UNIQUE (person_id, profile_version)
                )
                """
            )
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table('person_profile_overrides')} (
                    person_id TEXT PRIMARY KEY,
                    override_text TEXT NOT NULL,
                    updated_at DOUBLE PRECISION NOT NULL,
                    updated_by TEXT,
                    source TEXT
                )
                """
            )
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table('person_registry')} (
                    person_id TEXT PRIMARY KEY,
                    person_name TEXT,
                    nickname TEXT,
                    user_id TEXT,
                    platform TEXT,
                    group_nick_name TEXT,
                    memory_points TEXT,
                    last_know DOUBLE PRECISION,
                    metadata_json TEXT,
                    created_at DOUBLE PRECISION NOT NULL,
                    updated_at DOUBLE PRECISION NOT NULL
                )
                """
            )
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table('transcript_sessions')} (
                    session_id TEXT PRIMARY KEY,
                    source TEXT,
                    metadata_json TEXT,
                    created_at DOUBLE PRECISION NOT NULL,
                    updated_at DOUBLE PRECISION NOT NULL
                )
                """
            )
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table('transcript_messages')} (
                    message_id BIGSERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    ts DOUBLE PRECISION,
                    metadata_json TEXT,
                    created_at DOUBLE PRECISION NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES {self._table('transcript_sessions')}(session_id) ON DELETE CASCADE
                )
                """
            )
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table('async_tasks')} (
                    task_id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT,
                    result_json TEXT,
                    error_message TEXT,
                    created_at DOUBLE PRECISION NOT NULL,
                    updated_at DOUBLE PRECISION NOT NULL,
                    started_at DOUBLE PRECISION,
                    finished_at DOUBLE PRECISION,
                    cancel_requested INTEGER DEFAULT 0
                )
                """
            )
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table('paragraph_ngrams')} (
                    term TEXT NOT NULL,
                    paragraph_hash TEXT NOT NULL,
                    PRIMARY KEY (term, paragraph_hash),
                    FOREIGN KEY (paragraph_hash) REFERENCES {self._table('paragraphs')}(hash) ON DELETE CASCADE
                )
                """
            )
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table('paragraph_ngram_meta')} (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table('graph_nodes')} (
                    canonical_id TEXT PRIMARY KEY,
                    display_name TEXT,
                    attrs_json TEXT,
                    updated_at DOUBLE PRECISION
                )
                """
            )
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table('graph_edges')} (
                    source_canonical TEXT NOT NULL,
                    target_canonical TEXT NOT NULL,
                    source_display TEXT,
                    target_display TEXT,
                    weight DOUBLE PRECISION,
                    relation_hashes_json TEXT,
                    updated_at DOUBLE PRECISION,
                    PRIMARY KEY (source_canonical, target_canonical)
                )
                """
            )

            for table_name, column_name, column_type in [
                ("paragraphs", "search_lexemes", "TEXT"),
                ("paragraphs", "search_document", "TSVECTOR"),
                ("paragraphs", "is_permanent", "INTEGER DEFAULT 0"),
                ("relations", "search_lexemes", "TEXT"),
                ("relations", "search_document", "TSVECTOR"),
                ("relations", "is_permanent", "INTEGER DEFAULT 0"),
            ]:
                cursor.execute(
                    f"""
                    ALTER TABLE {self._table(table_name)}
                    ADD COLUMN IF NOT EXISTS {column_name} {column_type}
                    """
                )

            index_sql = [
                f"CREATE INDEX IF NOT EXISTS {self.table_prefix}_paragraphs_source_idx ON {self._table('paragraphs')}(source)",
                f"CREATE INDEX IF NOT EXISTS {self.table_prefix}_paragraphs_time_idx ON {self._table('paragraphs')}(event_time DESC)",
                f"CREATE INDEX IF NOT EXISTS {self.table_prefix}_paragraphs_time_start_idx ON {self._table('paragraphs')}(event_time_start DESC)",
                f"CREATE INDEX IF NOT EXISTS {self.table_prefix}_paragraphs_time_end_idx ON {self._table('paragraphs')}(event_time_end DESC)",
                f"CREATE INDEX IF NOT EXISTS {self.table_prefix}_entities_canonical_idx ON {self._table('entities')}(canonical_name)",
                f"CREATE INDEX IF NOT EXISTS {self.table_prefix}_relations_subject_idx ON {self._table('relations')}(subject_canonical)",
                f"CREATE INDEX IF NOT EXISTS {self.table_prefix}_relations_object_idx ON {self._table('relations')}(object_canonical)",
                f"CREATE INDEX IF NOT EXISTS {self.table_prefix}_relations_active_idx ON {self._table('relations')}(is_inactive, protected_until, is_pinned)",
                f"CREATE INDEX IF NOT EXISTS {self.table_prefix}_deleted_relations_deleted_at_idx ON {self._table('deleted_relations')}(deleted_at DESC)",
                f"CREATE INDEX IF NOT EXISTS {self.table_prefix}_person_active_seen_idx ON {self._table('person_profile_active_persons')}(last_seen_at DESC)",
                f"CREATE INDEX IF NOT EXISTS {self.table_prefix}_person_snapshot_person_idx ON {self._table('person_profile_snapshots')}(person_id, updated_at DESC)",
                f"CREATE INDEX IF NOT EXISTS {self.table_prefix}_person_registry_lastknow_idx ON {self._table('person_registry')}(last_know DESC)",
                f"CREATE INDEX IF NOT EXISTS {self.table_prefix}_transcript_messages_session_idx ON {self._table('transcript_messages')}(session_id, created_at DESC)",
                f"CREATE INDEX IF NOT EXISTS {self.table_prefix}_async_tasks_status_idx ON {self._table('async_tasks')}(status, updated_at DESC)",
                f"CREATE INDEX IF NOT EXISTS {self.table_prefix}_paragraphs_search_document_idx ON {self._table('paragraphs')} USING GIN(search_document)",
                f"CREATE INDEX IF NOT EXISTS {self.table_prefix}_relations_search_document_idx ON {self._table('relations')} USING GIN(search_document)",
                f"CREATE INDEX IF NOT EXISTS {self.table_prefix}_paragraph_ngrams_hash_idx ON {self._table('paragraph_ngrams')}(paragraph_hash)",
                f"CREATE INDEX IF NOT EXISTS {self.table_prefix}_paragraph_ngrams_term_idx ON {self._table('paragraph_ngrams')}(term)",
                f"CREATE INDEX IF NOT EXISTS {self.table_prefix}_graph_edges_source_idx ON {self._table('graph_edges')}(source_canonical)",
                f"CREATE INDEX IF NOT EXISTS {self.table_prefix}_graph_edges_target_idx ON {self._table('graph_edges')}(target_canonical)",
            ]
            for sql in index_sql:
                cursor.execute(sql)
        self._pg_conn.commit()  # type: ignore[union-attr]

    def query(self, sql: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        """执行查询 SQL 并返回字典结果列表。

        Args:
            sql: 待执行 SQL。
            params: SQL 参数。

        Returns:
            List[Dict[str, Any]]: 查询结果列表。
        """
        translated = self._translate_sql(sql, for_compat=False)
        return self._fetchall_dict(translated, tuple(params))

    def add_paragraph(
        self,
        content: str,
        vector_index: Optional[int] = None,
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        knowledge_type: str = "mixed",
        time_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        text = normalize_text(content)
        if not text:
            raise ValueError("Paragraph content cannot be empty")
        scope_namespace = _resolve_scope_namespace(str(source or ""), metadata)
        hash_basis = f"{scope_namespace}\n{text}" if scope_namespace else text
        hash_value = compute_hash(hash_basis)
        now = time.time()
        normalized_time = normalize_time_meta(time_meta or {})
        search_lexemes = self._build_search_lexemes(text)
        payload = (
            hash_value,
            content,
            vector_index,
            now,
            now,
            _encode_json(metadata or {}),
            str(source or ""),
            len(text.split()),
            normalized_time.get("event_time"),
            normalized_time.get("event_time_start"),
            normalized_time.get("event_time_end"),
            normalized_time.get("time_granularity"),
            normalized_time.get("time_confidence", 1.0),
            str(knowledge_type or "mixed"),
            search_lexemes,
            search_lexemes,
        )
        with self._cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {self._table('paragraphs')} (
                    hash, content, vector_index, created_at, updated_at, metadata_json, source, word_count,
                    event_time, event_time_start, event_time_end, time_granularity, time_confidence,
                    knowledge_type, search_lexemes, search_document
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, to_tsvector('simple', %s))
                ON CONFLICT (hash) DO UPDATE SET
                    updated_at = EXCLUDED.updated_at,
                    vector_index = COALESCE(EXCLUDED.vector_index, {self._table('paragraphs')}.vector_index),
                    metadata_json = COALESCE(NULLIF(EXCLUDED.metadata_json, ''), {self._table('paragraphs')}.metadata_json),
                    source = COALESCE(NULLIF(EXCLUDED.source, ''), {self._table('paragraphs')}.source),
                    word_count = EXCLUDED.word_count,
                    event_time = COALESCE(EXCLUDED.event_time, {self._table('paragraphs')}.event_time),
                    event_time_start = COALESCE(EXCLUDED.event_time_start, {self._table('paragraphs')}.event_time_start),
                    event_time_end = COALESCE(EXCLUDED.event_time_end, {self._table('paragraphs')}.event_time_end),
                    time_granularity = COALESCE(EXCLUDED.time_granularity, {self._table('paragraphs')}.time_granularity),
                    time_confidence = COALESCE(EXCLUDED.time_confidence, {self._table('paragraphs')}.time_confidence),
                    knowledge_type = COALESCE(NULLIF(EXCLUDED.knowledge_type, ''), {self._table('paragraphs')}.knowledge_type),
                    search_lexemes = EXCLUDED.search_lexemes,
                    search_document = EXCLUDED.search_document,
                    is_deleted = 0,
                    deleted_at = NULL
                """,
                payload,
            )
        self._maybe_commit()
        return hash_value

    def add_entity(
        self,
        name: str,
        vector_index: Optional[int] = None,
        source_paragraph: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        canonical = self._canonicalize_name(name)
        if not canonical:
            raise ValueError("Entity name cannot be empty")
        hash_value = compute_hash(canonical)
        now = time.time()
        with self._cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {self._table('entities')} (
                    hash, name, canonical_name, vector_index, appearance_count, created_at, metadata_json, is_deleted, deleted_at
                )
                VALUES (%s, %s, %s, %s, 1, %s, %s, 0, NULL)
                ON CONFLICT (hash) DO UPDATE SET
                    appearance_count = {self._table('entities')}.appearance_count + 1,
                    vector_index = COALESCE(EXCLUDED.vector_index, {self._table('entities')}.vector_index),
                    metadata_json = COALESCE(NULLIF(EXCLUDED.metadata_json, ''), {self._table('entities')}.metadata_json),
                    is_deleted = 0,
                    deleted_at = NULL
                """,
                (hash_value, name, canonical, vector_index, now, _encode_json(metadata or {})),
            )
        if source_paragraph:
            self.link_paragraph_entity(source_paragraph, hash_value)
        self._maybe_commit()
        return hash_value

    def add_relation(
        self,
        subject: str,
        predicate: str,
        obj: str,
        vector_index: Optional[int] = None,
        confidence: float = 1.0,
        source_paragraph: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        s_canon = self._canonicalize_name(subject)
        p_canon = self._canonicalize_name(predicate)
        o_canon = self._canonicalize_name(obj)
        if not all((s_canon, p_canon, o_canon)):
            raise ValueError("Relation components cannot be empty")
        relation_key = f"{s_canon}|{p_canon}|{o_canon}"
        scope_namespace = _resolve_scope_namespace("", metadata)
        if not scope_namespace and source_paragraph:
            paragraph = self.get_paragraph(str(source_paragraph or "").strip())
            if paragraph is not None:
                scope_namespace = _resolve_scope_namespace(
                    str(paragraph.get("source", "") or ""),
                    dict(paragraph.get("metadata", {}) or {}),
                )
        hash_basis = f"{scope_namespace}|{relation_key}" if scope_namespace else relation_key
        hash_value = compute_hash(hash_basis)
        now = time.time()
        relation_text = f"{subject} {predicate} {obj}"
        search_lexemes = self._build_search_lexemes(relation_text)
        with self._cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {self._table('relations')} (
                    hash, subject, subject_canonical, predicate, predicate_canonical, object, object_canonical,
                    vector_index, confidence, created_at, source_paragraph, metadata_json, last_accessed,
                    access_count, is_inactive, inactive_since, is_pinned, protected_until, last_reinforced,
                    search_lexemes, search_document
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 0, 0, NULL, 0, NULL, NULL, %s, to_tsvector('simple', %s))
                ON CONFLICT (hash) DO UPDATE SET
                    vector_index = COALESCE(EXCLUDED.vector_index, {self._table('relations')}.vector_index),
                    confidence = GREATEST(COALESCE({self._table('relations')}.confidence, 0), COALESCE(EXCLUDED.confidence, 0)),
                    source_paragraph = COALESCE(NULLIF(EXCLUDED.source_paragraph, ''), {self._table('relations')}.source_paragraph),
                    metadata_json = COALESCE(NULLIF(EXCLUDED.metadata_json, ''), {self._table('relations')}.metadata_json),
                    search_lexemes = EXCLUDED.search_lexemes,
                    search_document = EXCLUDED.search_document,
                    is_inactive = 0,
                    inactive_since = NULL
                """,
                (
                    hash_value,
                    subject,
                    s_canon,
                    predicate,
                    p_canon,
                    obj,
                    o_canon,
                    vector_index,
                    float(confidence),
                    now,
                    source_paragraph,
                    _encode_json(metadata or {}),
                    now,
                    search_lexemes,
                    search_lexemes,
                ),
            )
        if source_paragraph:
            self.link_paragraph_relation(source_paragraph, hash_value)
        self._maybe_commit()
        return hash_value

    def link_paragraph_entity(self, paragraph_hash: str, entity_hash: str, mention_count: int = 1) -> bool:
        with self._cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {self._table('paragraph_entities')} (paragraph_hash, entity_hash, mention_count)
                VALUES (%s, %s, %s)
                ON CONFLICT (paragraph_hash, entity_hash) DO UPDATE SET
                    mention_count = {self._table('paragraph_entities')}.mention_count + EXCLUDED.mention_count
                """,
                (paragraph_hash, entity_hash, max(1, int(mention_count))),
            )
        self._maybe_commit()
        return True

    def link_paragraph_relation(self, paragraph_hash: str, relation_hash: str) -> bool:
        with self._cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {self._table('paragraph_relations')} (paragraph_hash, relation_hash)
                VALUES (%s, %s)
                ON CONFLICT (paragraph_hash, relation_hash) DO NOTHING
                """,
                (paragraph_hash, relation_hash),
            )
        self._maybe_commit()
        return True

    def get_paragraph(self, hash_value: str) -> Optional[Dict[str, Any]]:
        row = self._fetchone_dict(
            f"SELECT * FROM {self._table('paragraphs')} WHERE hash = %s",
            (str(hash_value or "").strip(),),
        )
        return self._decode_paragraph(row)

    def update_paragraph_time_meta(
        self,
        paragraph_hash: str,
        time_meta: Dict[str, Any],
    ) -> bool:
        normalized = normalize_time_meta(time_meta or {})
        if not normalized:
            return False

        assignments: List[str] = []
        params: List[Any] = []
        for key in [
            "event_time",
            "event_time_start",
            "event_time_end",
            "time_granularity",
            "time_confidence",
        ]:
            if key in normalized:
                assignments.append(f"{key} = %s")
                params.append(normalized[key])

        if not assignments:
            return False

        assignments.append("updated_at = %s")
        params.append(time.time())
        params.append(str(paragraph_hash or "").strip())
        with self._cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE {self._table('paragraphs')}
                SET {', '.join(assignments)}
                WHERE hash = %s
                """,
                tuple(params),
            )
            affected = int(cursor.rowcount or 0)
        self._maybe_commit()
        return affected > 0

    def get_entity(self, hash_or_name: str) -> Optional[Dict[str, Any]]:
        value = str(hash_or_name or "").strip()
        if not value:
            return None
        canonical = self._canonicalize_name(value)
        row = self._fetchone_dict(
            f"""
            SELECT * FROM {self._table('entities')}
            WHERE hash = %s OR canonical_name = %s
            LIMIT 1
            """,
            (value.lower(), canonical),
        )
        return self._decode_entity(row)

    def get_relation(self, hash_value: str) -> Optional[Dict[str, Any]]:
        row = self._fetchone_dict(
            f"SELECT * FROM {self._table('relations')} WHERE hash = %s",
            (str(hash_value or "").strip().lower(),),
        )
        return self._decode_relation(row)

    def get_paragraph_entities(self, paragraph_hash: str) -> List[Dict[str, Any]]:
        rows = self._fetchall_dict(
            f"""
            SELECT e.*, pe.mention_count
            FROM {self._table('paragraph_entities')} pe
            JOIN {self._table('entities')} e ON e.hash = pe.entity_hash
            WHERE pe.paragraph_hash = %s
            ORDER BY pe.mention_count DESC, e.created_at ASC
            """,
            (paragraph_hash,),
        )
        results: List[Dict[str, Any]] = []
        for row in rows:
            decoded = self._decode_entity(row)
            if decoded is not None:
                results.append(decoded)
        return results

    def get_paragraph_relations(self, paragraph_hash: str) -> List[Dict[str, Any]]:
        rows = self._fetchall_dict(
            f"""
            SELECT r.*
            FROM {self._table('paragraph_relations')} pr
            JOIN {self._table('relations')} r ON r.hash = pr.relation_hash
            WHERE pr.paragraph_hash = %s
            ORDER BY r.created_at DESC
            """,
            (paragraph_hash,),
        )
        results: List[Dict[str, Any]] = []
        for row in rows:
            decoded = self._decode_relation(row)
            if decoded is not None:
                results.append(decoded)
        return results

    def get_paragraphs_by_entity(self, entity_name: str) -> List[Dict[str, Any]]:
        canonical = self._canonicalize_name(entity_name)
        if not canonical:
            return []
        rows = self._fetchall_dict(
            f"""
            SELECT p.*
            FROM {self._table('paragraph_entities')} pe
            JOIN {self._table('entities')} e ON e.hash = pe.entity_hash
            JOIN {self._table('paragraphs')} p ON p.hash = pe.paragraph_hash
            WHERE e.canonical_name = %s AND COALESCE(p.is_deleted, 0) = 0
            ORDER BY p.updated_at DESC NULLS LAST, p.created_at DESC
            """,
            (canonical,),
        )
        results: List[Dict[str, Any]] = []
        for row in rows:
            decoded = self._decode_paragraph(row)
            if decoded is not None:
                results.append(decoded)
        return results

    def get_paragraphs_by_relation(self, relation_hash: str) -> List[Dict[str, Any]]:
        rows = self._fetchall_dict(
            f"""
            SELECT p.*
            FROM {self._table('paragraph_relations')} pr
            JOIN {self._table('paragraphs')} p ON p.hash = pr.paragraph_hash
            WHERE pr.relation_hash = %s AND COALESCE(p.is_deleted, 0) = 0
            ORDER BY p.updated_at DESC NULLS LAST, p.created_at DESC
            """,
            (relation_hash,),
        )
        results: List[Dict[str, Any]] = []
        for row in rows:
            decoded = self._decode_paragraph(row)
            if decoded is not None:
                results.append(decoded)
        return results

    def get_paragraphs_by_source(self, source: str) -> List[Dict[str, Any]]:
        rows = self._fetchall_dict(
            f"""
            SELECT * FROM {self._table('paragraphs')}
            WHERE source = %s AND COALESCE(is_deleted, 0) = 0
            ORDER BY updated_at DESC NULLS LAST, created_at DESC
            """,
            (str(source or ""),),
        )
        results: List[Dict[str, Any]] = []
        for row in rows:
            decoded = self._decode_paragraph(row)
            if decoded is not None:
                results.append(decoded)
        return results

    def get_relations(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        conditions: List[str] = []
        params: List[Any] = []
        if subject:
            conditions.append("subject_canonical = %s")
            params.append(self._canonicalize_name(subject))
        if predicate:
            conditions.append("predicate_canonical = %s")
            params.append(self._canonicalize_name(predicate))
        if object:
            conditions.append("object_canonical = %s")
            params.append(self._canonicalize_name(object))
        where_sql = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self._fetchall_dict(
            f"""
            SELECT * FROM {self._table('relations')}
            {where_sql}
            ORDER BY last_accessed DESC NULLS LAST, created_at DESC
            """,
            tuple(params),
        )
        results: List[Dict[str, Any]] = []
        for row in rows:
            decoded = self._decode_relation(row)
            if decoded is not None:
                results.append(decoded)
        return results

    def get_all_triples(self) -> List[Tuple[str, str, str, str]]:
        with self._cursor() as cursor:
            cursor.execute(
                f"SELECT subject, predicate, object, hash FROM {self._table('relations')} ORDER BY created_at ASC"
            )
            return [(str(row[0]), str(row[1]), str(row[2]), str(row[3])) for row in cursor.fetchall()]

    def get_all_sources(self) -> List[str]:
        with self._cursor() as cursor:
            cursor.execute(
                f"""
                SELECT DISTINCT source
                FROM {self._table('paragraphs')}
                WHERE source IS NOT NULL AND source <> '' AND COALESCE(is_deleted, 0) = 0
                ORDER BY source ASC
                """
            )
            return [str(row[0]) for row in cursor.fetchall()]

    def search_paragraphs_by_content(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        keyword = str(query or "").strip()
        if not keyword:
            return []
        rows = self._fetchall_dict(
            f"""
            SELECT *
            FROM {self._table('paragraphs')}
            WHERE COALESCE(is_deleted, 0) = 0 AND content ILIKE %s
            ORDER BY updated_at DESC NULLS LAST, created_at DESC
            LIMIT %s
            """,
            (f"%{keyword}%", max(1, int(limit))),
        )
        results: List[Dict[str, Any]] = []
        for row in rows:
            decoded = self._decode_paragraph(row)
            if decoded is not None:
                results.append(decoded)
        return results

    def count_paragraphs(self, include_deleted: bool = False, only_deleted: bool = False) -> int:
        where_sql = "WHERE COALESCE(is_deleted, 0) = 1" if only_deleted else ""
        if not only_deleted and not include_deleted:
            where_sql = "WHERE COALESCE(is_deleted, 0) = 0"
        with self._cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM {self._table('paragraphs')} {where_sql}")
            return int(cursor.fetchone()[0] or 0)

    def count_relations(self, include_deleted: bool = False, only_deleted: bool = False) -> int:
        if only_deleted:
            with self._cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) FROM {self._table('deleted_relations')}")
                return int(cursor.fetchone()[0] or 0)
        with self._cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM {self._table('relations')}")
            live_count = int(cursor.fetchone()[0] or 0)
            if include_deleted:
                cursor.execute(f"SELECT COUNT(*) FROM {self._table('deleted_relations')}")
                return live_count + int(cursor.fetchone()[0] or 0)
            return live_count

    def count_entities(self) -> int:
        with self._cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM {self._table('entities')} WHERE COALESCE(is_deleted, 0) = 0")
            return int(cursor.fetchone()[0] or 0)

    def delete_paragraph(self, hash_value: str) -> bool:
        plan = self.delete_paragraph_atomic(str(hash_value or "").strip())
        return bool(plan.get("vector_id_to_remove"))

    def delete_entity(self, hash_or_name: str) -> int:
        target = self.get_entity(hash_or_name)
        if not target:
            return 0
        with self._cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE {self._table('entities')}
                SET is_deleted = 1, deleted_at = %s
                WHERE hash = %s
                """,
                (time.time(), target["hash"]),
            )
            affected = int(cursor.rowcount or 0)
        self._maybe_commit()
        return affected

    def delete_relation(self, hash_value: str) -> bool:
        deleted = self.backup_and_delete_relations([str(hash_value or "").strip().lower()])
        return deleted > 0

    def update_vector_index(
        self,
        item_type: str,
        hash_value: str,
        vector_index: int,
    ) -> bool:
        table_map = {
            "paragraph": "paragraphs",
            "entity": "entities",
            "relation": "relations",
        }
        table_name = table_map.get(str(item_type or "").strip())
        if not table_name:
            raise ValueError(f"invalid item_type: {item_type}")
        with self._cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE {self._table(table_name)}
                SET vector_index = %s
                WHERE hash = %s
                """,
                (int(vector_index), str(hash_value or "").strip()),
            )
            affected = int(cursor.rowcount or 0)
        self._maybe_commit()
        return affected > 0

    def record_access(self, hash_value: str, item_type: str) -> bool:
        table_map = {
            "paragraph": "paragraphs",
            "relation": "relations",
        }
        table_name = table_map.get(str(item_type or "").strip())
        if not table_name:
            return False

        now = time.time()
        with self._cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE {self._table(table_name)}
                SET last_accessed = %s,
                    access_count = COALESCE(access_count, 0) + 1
                WHERE hash = %s
                """,
                (now, str(hash_value or "").strip()),
            )
            affected = int(cursor.rowcount or 0)
        self._maybe_commit()
        return affected > 0

    def revive_if_deleted(
        self,
        entity_hashes: Optional[List[str]] = None,
        paragraph_hashes: Optional[List[str]] = None,
    ) -> int:
        affected = 0
        with self._cursor() as cursor:
            if entity_hashes:
                cursor.execute(
                    f"""
                    UPDATE {self._table('entities')}
                    SET is_deleted = 0, deleted_at = NULL
                    WHERE hash = ANY(%s) AND COALESCE(is_deleted, 0) = 1
                    """,
                    (entity_hashes,),
                )
                affected += int(cursor.rowcount or 0)
            if paragraph_hashes:
                cursor.execute(
                    f"""
                    UPDATE {self._table('paragraphs')}
                    SET is_deleted = 0, deleted_at = NULL
                    WHERE hash = ANY(%s) AND COALESCE(is_deleted, 0) = 1
                    """,
                    (paragraph_hashes,),
                )
                affected += int(cursor.rowcount or 0)
        if affected:
            self._maybe_commit()
        return affected

    def revive_entities_by_names(self, names: List[str]) -> int:
        hashes = [compute_hash(self._canonicalize_name(name)) for name in names if self._canonicalize_name(name)]
        if not hashes:
            return 0
        return self.revive_if_deleted(entity_hashes=hashes)

    def delete_paragraph_atomic(self, paragraph_hash: str) -> Dict[str, Any]:
        cleanup_plan: Dict[str, Any] = {
            "paragraph_hash": paragraph_hash,
            "vector_id_to_remove": None,
            "edges_to_remove": [],
            "relation_prune_ops": [],
        }
        with self._cursor() as cursor:
            cursor.execute(f"SELECT hash FROM {self._table('paragraphs')} WHERE hash = %s", (paragraph_hash,))
            if cursor.fetchone() is None:
                return cleanup_plan

            cleanup_plan["vector_id_to_remove"] = paragraph_hash
            cursor.execute(
                f"SELECT relation_hash FROM {self._table('paragraph_relations')} WHERE paragraph_hash = %s",
                (paragraph_hash,),
            )
            candidate_relations = [str(row[0]) for row in cursor.fetchall()]

            cursor.execute(
                f"DELETE FROM {self._table('paragraph_entities')} WHERE paragraph_hash = %s",
                (paragraph_hash,),
            )
            cursor.execute(
                f"DELETE FROM {self._table('paragraph_relations')} WHERE paragraph_hash = %s",
                (paragraph_hash,),
            )
            cursor.execute(
                f"DELETE FROM {self._table('paragraphs')} WHERE hash = %s",
                (paragraph_hash,),
            )

            orphaned_hashes: List[str] = []
            for rel_hash in candidate_relations:
                cursor.execute(
                    f"SELECT COUNT(*) FROM {self._table('paragraph_relations')} WHERE relation_hash = %s",
                    (rel_hash,),
                )
                support_count = int(cursor.fetchone()[0] or 0)
                if support_count > 0:
                    continue
                relation = self.get_relation(rel_hash)
                if relation is None:
                    continue
                cleanup_plan["relation_prune_ops"].append(
                    (str(relation["subject"]), str(relation["object"]), rel_hash)
                )
                cursor.execute(
                    f"""
                    SELECT COUNT(*) FROM {self._table('relations')}
                    WHERE subject_canonical = %s AND object_canonical = %s AND hash <> %s
                    """,
                    (
                        self._canonicalize_name(str(relation["subject"])),
                        self._canonicalize_name(str(relation["object"])),
                        rel_hash,
                    ),
                )
                sibling_count = int(cursor.fetchone()[0] or 0)
                if sibling_count == 0:
                    cleanup_plan["edges_to_remove"].append((str(relation["subject"]), str(relation["object"])))
                orphaned_hashes.append(rel_hash)

            if orphaned_hashes:
                self._backup_and_delete_relations_tx(cursor, orphaned_hashes)
        self._maybe_commit()
        return cleanup_plan

    def _backup_and_delete_relations_tx(self, cursor, hashes: List[str]) -> int:
        if not hashes:
            return 0
        cursor.execute(
            f"""
            SELECT relation_hash, array_agg(paragraph_hash ORDER BY paragraph_hash) AS paragraphs
            FROM {self._table('paragraph_relations')}
            WHERE relation_hash = ANY(%s)
            GROUP BY relation_hash
            """,
            (hashes,),
        )
        support_map = {str(row[0]): list(row[1] or []) for row in cursor.fetchall()}
        cursor.execute(
            f"SELECT * FROM {self._table('relations')} WHERE hash = ANY(%s)",
            (hashes,),
        )
        rows = cursor.fetchall()
        if not rows:
            return 0
        deleted_at = time.time()
        for row in rows:
            relation_hash = str(row[0])
            cursor.execute(
                f"""
                INSERT INTO {self._table('deleted_relations')} (
                    hash, subject, subject_canonical, predicate, predicate_canonical, object, object_canonical,
                    vector_index, confidence, created_at, source_paragraph, metadata_json, last_accessed,
                    access_count, is_inactive, inactive_since, is_pinned, protected_until, last_reinforced,
                    deleted_at, support_paragraphs_json
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (hash) DO UPDATE SET
                    subject = EXCLUDED.subject,
                    subject_canonical = EXCLUDED.subject_canonical,
                    predicate = EXCLUDED.predicate,
                    predicate_canonical = EXCLUDED.predicate_canonical,
                    object = EXCLUDED.object,
                    object_canonical = EXCLUDED.object_canonical,
                    vector_index = EXCLUDED.vector_index,
                    confidence = EXCLUDED.confidence,
                    created_at = EXCLUDED.created_at,
                    source_paragraph = EXCLUDED.source_paragraph,
                    metadata_json = EXCLUDED.metadata_json,
                    last_accessed = EXCLUDED.last_accessed,
                    access_count = EXCLUDED.access_count,
                    is_inactive = EXCLUDED.is_inactive,
                    inactive_since = EXCLUDED.inactive_since,
                    is_pinned = EXCLUDED.is_pinned,
                    protected_until = EXCLUDED.protected_until,
                    last_reinforced = EXCLUDED.last_reinforced,
                    deleted_at = EXCLUDED.deleted_at,
                    support_paragraphs_json = EXCLUDED.support_paragraphs_json
                """,
                (
                    row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9],
                    row[10], row[11], row[12], row[13], row[14], row[15], row[16], row[17], row[18],
                    deleted_at, _encode_json(support_map.get(relation_hash, [])),
                ),
            )

        cursor.execute(
            f"DELETE FROM {self._table('paragraph_relations')} WHERE relation_hash = ANY(%s)",
            (hashes,),
        )
        cursor.execute(
            f"DELETE FROM {self._table('relations')} WHERE hash = ANY(%s)",
            (hashes,),
        )
        return int(cursor.rowcount or 0)

    def backup_and_delete_relations(self, hashes: List[str]) -> int:
        with self._cursor() as cursor:
            deleted = self._backup_and_delete_relations_tx(cursor, hashes)
        self._maybe_commit()
        return deleted

    def restore_relation(self, hash_value: str) -> Optional[Dict[str, Any]]:
        row = self._fetchone_dict(
            f"SELECT * FROM {self._table('deleted_relations')} WHERE hash = %s",
            (str(hash_value or "").strip().lower(),),
        )
        decoded = self._decode_deleted_relation(row)
        if decoded is None:
            return None
        support_paragraphs = decoded.get("support_paragraphs", []) or []
        with self._cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {self._table('relations')} (
                    hash, subject, subject_canonical, predicate, predicate_canonical, object, object_canonical,
                    vector_index, confidence, created_at, source_paragraph, metadata_json, last_accessed,
                    access_count, is_inactive, inactive_since, is_pinned, protected_until, last_reinforced
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (hash) DO UPDATE SET
                    subject = EXCLUDED.subject,
                    subject_canonical = EXCLUDED.subject_canonical,
                    predicate = EXCLUDED.predicate,
                    predicate_canonical = EXCLUDED.predicate_canonical,
                    object = EXCLUDED.object,
                    object_canonical = EXCLUDED.object_canonical,
                    vector_index = EXCLUDED.vector_index,
                    confidence = EXCLUDED.confidence,
                    source_paragraph = EXCLUDED.source_paragraph,
                    metadata_json = EXCLUDED.metadata_json,
                    last_accessed = EXCLUDED.last_accessed,
                    access_count = EXCLUDED.access_count,
                    is_inactive = 0,
                    inactive_since = NULL,
                    is_pinned = EXCLUDED.is_pinned,
                    protected_until = EXCLUDED.protected_until,
                    last_reinforced = EXCLUDED.last_reinforced
                """,
                (
                    decoded["hash"],
                    decoded["subject"],
                    self._canonicalize_name(decoded["subject"]),
                    decoded["predicate"],
                    self._canonicalize_name(decoded["predicate"]),
                    decoded["object"],
                    self._canonicalize_name(decoded["object"]),
                    decoded.get("vector_index"),
                    decoded.get("confidence", 1.0),
                    decoded.get("created_at"),
                    decoded.get("source_paragraph"),
                    _encode_json(decoded.get("metadata", {})),
                    decoded.get("last_accessed"),
                    decoded.get("access_count", 0),
                    0,
                    None,
                    1 if decoded.get("is_pinned") else 0,
                    decoded.get("protected_until"),
                    decoded.get("last_reinforced"),
                ),
            )
            for paragraph_hash in support_paragraphs:
                cursor.execute(
                    f"""
                    INSERT INTO {self._table('paragraph_relations')} (paragraph_hash, relation_hash)
                    VALUES (%s, %s)
                    ON CONFLICT (paragraph_hash, relation_hash) DO NOTHING
                    """,
                    (paragraph_hash, decoded["hash"]),
                )
            cursor.execute(
                f"DELETE FROM {self._table('deleted_relations')} WHERE hash = %s",
                (decoded["hash"],),
            )
        self._maybe_commit()
        return self.get_relation(decoded["hash"])

    def restore_relation_metadata(self, hash_value: str) -> Optional[Dict[str, Any]]:
        return self.restore_relation(hash_value)

    def get_deleted_relations(self, limit: int = 50) -> List[Dict[str, Any]]:
        rows = self._fetchall_dict(
            f"""
            SELECT * FROM {self._table('deleted_relations')}
            ORDER BY deleted_at DESC NULLS LAST
            LIMIT %s
            """,
            (max(1, int(limit)),),
        )
        results: List[Dict[str, Any]] = []
        for row in rows:
            decoded = self._decode_deleted_relation(row)
            if decoded is not None:
                results.append(decoded)
        return results

    def get_deleted_relation(self, hash_value: str) -> Optional[Dict[str, Any]]:
        row = self._fetchone_dict(
            f"SELECT * FROM {self._table('deleted_relations')} WHERE hash = %s",
            (str(hash_value or "").strip().lower(),),
        )
        return self._decode_deleted_relation(row)

    def get_deleted_entities(self, limit: int = 50) -> List[Dict[str, Any]]:
        rows = self._fetchall_dict(
            f"""
            SELECT * FROM {self._table('entities')}
            WHERE COALESCE(is_deleted, 0) = 1
            ORDER BY deleted_at DESC NULLS LAST
            LIMIT %s
            """,
            (max(1, int(limit)),),
        )
        results: List[Dict[str, Any]] = []
        for row in rows:
            decoded = self._decode_entity(row)
            if decoded is not None:
                decoded["type"] = "entity"
                results.append(decoded)
        return results

    def get_relation_status_batch(self, hashes: List[str]) -> Dict[str, Dict[str, Any]]:
        if not hashes:
            return {}
        rows = self._fetchall_dict(
            f"""
            SELECT hash, is_inactive, inactive_since, is_pinned, protected_until, last_reinforced
            FROM {self._table('relations')}
            WHERE hash = ANY(%s)
            """,
            (hashes,),
        )
        return {
            str(row["hash"]): {
                "is_inactive": bool(row.get("is_inactive", 0)),
                "inactive_since": row.get("inactive_since"),
                "is_pinned": bool(row.get("is_pinned", 0)),
                "protected_until": row.get("protected_until"),
                "last_reinforced": row.get("last_reinforced"),
            }
            for row in rows
        }

    def get_entity_status_batch(self, hashes: List[str]) -> Dict[str, Dict[str, Any]]:
        if not hashes:
            return {}
        rows = self._fetchall_dict(
            f"""
            SELECT hash, is_deleted, deleted_at
            FROM {self._table('entities')}
            WHERE hash = ANY(%s)
            """,
            (hashes,),
        )
        return {
            str(row["hash"]): {
                "is_deleted": bool(row.get("is_deleted", 0)),
                "deleted_at": row.get("deleted_at"),
            }
            for row in rows
        }

    def reinforce_relations(self, hashes: List[str]) -> int:
        if not hashes:
            return 0
        now = time.time()
        with self._cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE {self._table('relations')}
                SET last_accessed = %s,
                    access_count = COALESCE(access_count, 0) + 1,
                    is_inactive = 0,
                    inactive_since = NULL,
                    last_reinforced = %s
                WHERE hash = ANY(%s)
                """,
                (now, now, hashes),
            )
            affected = int(cursor.rowcount or 0)
        self._maybe_commit()
        return affected

    def update_relation_timestamp(self, hash_value: str, access_count_delta: int = 1) -> None:
        now = time.time()
        with self._cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE {self._table('relations')}
                SET last_accessed = %s,
                    access_count = COALESCE(access_count, 0) + %s
                WHERE hash = %s
                """,
                (now, int(access_count_delta), str(hash_value or "").strip().lower()),
            )
        self._maybe_commit()

    def mark_relations_active(self, hashes: List[str], boost_weight: Optional[float] = None) -> int:
        if not hashes:
            return 0
        assignments = [
            "is_inactive = 0",
            "inactive_since = NULL",
        ]
        params: List[Any] = []
        if boost_weight is not None:
            assignments.append("confidence = GREATEST(confidence, %s)")
            params.append(float(boost_weight))
        with self._cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE {self._table('relations')}
                SET {', '.join(assignments)}
                WHERE hash = ANY(%s)
                """,
                tuple(params + [hashes]),
            )
            affected = int(cursor.rowcount or 0)
        self._maybe_commit()
        return affected

    def mark_relations_inactive(self, hashes: List[str], inactive_since: Optional[float] = None) -> int:
        if not hashes:
            return 0
        now = float(inactive_since) if inactive_since is not None else time.time()
        with self._cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE {self._table('relations')}
                SET is_inactive = 1,
                    inactive_since = COALESCE(inactive_since, %s)
                WHERE hash = ANY(%s)
                """,
                (now, hashes),
            )
            affected = int(cursor.rowcount or 0)
        self._maybe_commit()
        return affected

    def update_relations_protection(
        self,
        hashes: List[str],
        *,
        is_pinned: Optional[bool] = None,
        protected_until: Optional[float] = None,
        last_reinforced: Optional[float] = None,
    ) -> int:
        if not hashes:
            return 0
        assignments: List[str] = []
        params: List[Any] = []
        if is_pinned is not None:
            assignments.append("is_pinned = %s")
            params.append(1 if is_pinned else 0)
        if protected_until is not None:
            assignments.append("protected_until = %s")
            params.append(protected_until)
        if last_reinforced is not None:
            assignments.append("last_reinforced = %s")
            params.append(last_reinforced)
        if not assignments:
            return 0
        with self._cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE {self._table('relations')}
                SET {', '.join(assignments)}
                WHERE hash = ANY(%s)
                """,
                tuple(params + [hashes]),
            )
            affected = int(cursor.rowcount or 0)
        self._maybe_commit()
        return affected

    def protect_relations(self, hashes: List[str], is_pinned: bool = False, ttl_seconds: float = 0.0) -> int:
        now = time.time()
        protected_until = None if is_pinned else now + max(0.0, float(ttl_seconds or 0.0))
        return self.update_relations_protection(
            hashes,
            is_pinned=is_pinned,
            protected_until=protected_until,
            last_reinforced=now,
        )

    def get_protected_relations_hashes(self) -> List[str]:
        now = time.time()
        with self._cursor() as cursor:
            cursor.execute(
                f"""
                SELECT hash
                FROM {self._table('relations')}
                WHERE COALESCE(is_pinned, 0) = 1
                   OR COALESCE(protected_until, 0) > %s
                """,
                (now,),
            )
            return [str(row[0]) for row in cursor.fetchall()]

    def get_prune_candidates(self, cutoff: float) -> List[str]:
        now = time.time()
        with self._cursor() as cursor:
            cursor.execute(
                f"""
                SELECT hash
                FROM {self._table('relations')}
                WHERE COALESCE(is_inactive, 0) = 1
                  AND COALESCE(inactive_since, 0) <= %s
                  AND COALESCE(is_pinned, 0) = 0
                  AND COALESCE(protected_until, 0) <= %s
                """,
                (float(cutoff), now),
            )
            return [str(row[0]) for row in cursor.fetchall()]

    def set_person_profile_switch(self, stream_id: str, user_id: str, enabled: bool) -> Dict[str, Any]:
        now = time.time()
        with self._cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {self._table('person_profile_switches')} (stream_id, user_id, enabled, updated_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (stream_id, user_id) DO UPDATE SET
                    enabled = EXCLUDED.enabled,
                    updated_at = EXCLUDED.updated_at
                """,
                (stream_id, user_id, 1 if enabled else 0, now),
            )
        self._maybe_commit()
        return {"stream_id": stream_id, "user_id": user_id, "enabled": bool(enabled), "updated_at": now}

    def get_person_profile_switch(self, stream_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        row = self._fetchone_dict(
            f"""
            SELECT stream_id, user_id, enabled, updated_at
            FROM {self._table('person_profile_switches')}
            WHERE stream_id = %s AND user_id = %s
            """,
            (stream_id, user_id),
        )
        if row is None:
            return None
        row["enabled"] = bool(row.get("enabled", 0))
        return row

    def get_enabled_person_profile_switches(self) -> List[Dict[str, Any]]:
        rows = self._fetchall_dict(
            f"""
            SELECT stream_id, user_id, enabled, updated_at
            FROM {self._table('person_profile_switches')}
            WHERE enabled = 1
            ORDER BY updated_at DESC
            """
        )
        for row in rows:
            row["enabled"] = bool(row.get("enabled", 0))
        return rows

    def touch_active_person(
        self,
        stream_id: str,
        user_id: str,
        person_id: str,
        last_seen_at: Optional[float] = None,
    ) -> Dict[str, Any]:
        seen_at = float(last_seen_at if last_seen_at is not None else time.time())
        with self._cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {self._table('person_profile_active_persons')} (stream_id, user_id, person_id, last_seen_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (stream_id, user_id, person_id) DO UPDATE SET
                    last_seen_at = EXCLUDED.last_seen_at
                """,
                (stream_id, user_id, person_id, seen_at),
            )
        self._maybe_commit()
        return {"stream_id": stream_id, "user_id": user_id, "person_id": person_id, "last_seen_at": seen_at}

    def mark_person_profile_active(
        self,
        stream_id: str,
        user_id: str,
        person_id: str,
        seen_at: Optional[float] = None,
    ) -> None:
        if not stream_id or not user_id or not person_id:
            return
        self.touch_active_person(stream_id=stream_id, user_id=user_id, person_id=person_id, last_seen_at=seen_at)

    def get_active_person_ids_for_enabled_switches(self, active_after: Optional[float] = None, limit: int = 50) -> List[str]:
        with self._cursor() as cursor:
            sql = f"""
                SELECT ap.person_id, MAX(ap.last_seen_at) AS last_seen
                FROM {self._table('person_profile_active_persons')} ap
                JOIN {self._table('person_profile_switches')} sw
                  ON sw.stream_id = ap.stream_id
                 AND sw.user_id = ap.user_id
                WHERE sw.enabled = 1
            """
            params: List[Any] = []
            if active_after is not None:
                sql += " AND ap.last_seen_at >= %s"
                params.append(float(active_after))
            sql += """
                GROUP BY ap.person_id
                ORDER BY last_seen DESC
                LIMIT %s
            """
            params.append(max(1, int(limit)))
            cursor.execute(sql, tuple(params))
            return [str(row[0]) for row in cursor.fetchall() if row and row[0]]

    def upsert_person_profile_snapshot(
        self,
        person_id: str,
        profile_text: str,
        aliases: Optional[List[str]] = None,
        relation_edges: Optional[List[Dict[str, Any]]] = None,
        vector_evidence: Optional[List[Dict[str, Any]]] = None,
        evidence_ids: Optional[List[str]] = None,
        expires_at: Optional[float] = None,
        source_note: Optional[str] = None,
    ) -> Dict[str, Any]:
        pid = str(person_id or "").strip()
        if not pid:
            raise ValueError("person_id is empty")
        now = time.time()
        with self._cursor() as cursor:
            cursor.execute(
                f"SELECT COALESCE(MAX(profile_version), 0) + 1 FROM {self._table('person_profile_snapshots')} WHERE person_id = %s",
                (pid,),
            )
            next_version = int(cursor.fetchone()[0] or 1)
            cursor.execute(
                f"""
                INSERT INTO {self._table('person_profile_snapshots')} (
                    person_id, profile_version, profile_text, aliases_json, relation_edges_json,
                    vector_evidence_json, evidence_ids_json, updated_at, expires_at, source_note
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    pid,
                    next_version,
                    str(profile_text or ""),
                    _encode_json(aliases or []),
                    _encode_json(relation_edges or []),
                    _encode_json(vector_evidence or []),
                    _encode_json(evidence_ids or []),
                    now,
                    expires_at,
                    source_note,
                ),
            )
            row = dict(cursor.fetchone())
        self._maybe_commit()
        return self._decode_snapshot(row) or {}

    def get_latest_person_profile_snapshot(self, person_id: str) -> Optional[Dict[str, Any]]:
        row = self._fetchone_dict(
            f"""
            SELECT *
            FROM {self._table('person_profile_snapshots')}
            WHERE person_id = %s
            ORDER BY updated_at DESC, profile_version DESC
            LIMIT 1
            """,
            (str(person_id or "").strip(),),
        )
        return self._decode_snapshot(row)

    def set_person_profile_override(
        self,
        person_id: str,
        override_text: str,
        updated_by: str = "",
        source: str = "",
    ) -> Dict[str, Any]:
        pid = str(person_id or "").strip()
        if not pid:
            raise ValueError("person_id is empty")
        now = time.time()
        with self._cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {self._table('person_profile_overrides')} (person_id, override_text, updated_at, updated_by, source)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (person_id) DO UPDATE SET
                    override_text = EXCLUDED.override_text,
                    updated_at = EXCLUDED.updated_at,
                    updated_by = EXCLUDED.updated_by,
                    source = EXCLUDED.source
                RETURNING *
                """,
                (pid, str(override_text or ""), now, str(updated_by or ""), str(source or "")),
            )
            row = dict(cursor.fetchone())
        self._maybe_commit()
        return self._decode_override(row) or {}

    def get_person_profile_override(self, person_id: str) -> Optional[Dict[str, Any]]:
        row = self._fetchone_dict(
            f"SELECT * FROM {self._table('person_profile_overrides')} WHERE person_id = %s",
            (str(person_id or "").strip(),),
        )
        return self._decode_override(row)

    def delete_person_profile_override(self, person_id: str) -> int:
        affected = self._execute(
            f"DELETE FROM {self._table('person_profile_overrides')} WHERE person_id = %s",
            (str(person_id or "").strip(),),
        )
        self._maybe_commit()
        return affected

    def upsert_person_registry(
        self,
        *,
        person_id: str,
        person_name: str = "",
        nickname: str = "",
        user_id: str = "",
        platform: str = "",
        group_nick_name: Any = None,
        memory_points: Any = None,
        last_know: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        pid = str(person_id or "").strip()
        if not pid:
            raise ValueError("person_id is empty")
        now = time.time()
        payload = (
            pid,
            str(person_name or "").strip(),
            str(nickname or "").strip(),
            str(user_id or "").strip(),
            str(platform or "").strip(),
            _encode_json(group_nick_name),
            _encode_json(memory_points),
            self._safe_float(last_know),
            _encode_json(metadata or {}),
            now,
            now,
        )
        with self._cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {self._table('person_registry')} (
                    person_id, person_name, nickname, user_id, platform, group_nick_name,
                    memory_points, last_know, metadata_json, created_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (person_id) DO UPDATE SET
                    person_name = COALESCE(NULLIF(EXCLUDED.person_name, ''), {self._table('person_registry')}.person_name),
                    nickname = COALESCE(NULLIF(EXCLUDED.nickname, ''), {self._table('person_registry')}.nickname),
                    user_id = COALESCE(NULLIF(EXCLUDED.user_id, ''), {self._table('person_registry')}.user_id),
                    platform = COALESCE(NULLIF(EXCLUDED.platform, ''), {self._table('person_registry')}.platform),
                    group_nick_name = COALESCE(NULLIF(EXCLUDED.group_nick_name, ''), {self._table('person_registry')}.group_nick_name),
                    memory_points = COALESCE(NULLIF(EXCLUDED.memory_points, ''), {self._table('person_registry')}.memory_points),
                    last_know = COALESCE(EXCLUDED.last_know, {self._table('person_registry')}.last_know),
                    metadata_json = COALESCE(NULLIF(EXCLUDED.metadata_json, ''), {self._table('person_registry')}.metadata_json),
                    updated_at = EXCLUDED.updated_at
                RETURNING *
                """,
                payload,
            )
            row = dict(cursor.fetchone())
        self._maybe_commit()
        return self._decode_registry(row) or {}

    def get_person_registry(self, person_id: str) -> Optional[Dict[str, Any]]:
        row = self._fetchone_dict(
            f"SELECT * FROM {self._table('person_registry')} WHERE person_id = %s",
            (str(person_id or "").strip(),),
        )
        return self._decode_registry(row)

    def resolve_person_registry(self, value: str) -> Optional[str]:
        needle = str(value or "").strip()
        if not needle:
            return None
        row = self._fetchone_dict(
            f"""
            SELECT person_id
            FROM {self._table('person_registry')}
            WHERE person_id = %s
               OR user_id = %s
               OR person_name ILIKE %s
               OR nickname ILIKE %s
               OR group_nick_name ILIKE %s
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (needle, needle, needle, needle, f"%{needle}%"),
        )
        return str(row["person_id"]) if row else None

    def list_person_registry(self, keyword: str = "", page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        kw = str(keyword or "").strip()
        page = max(1, int(page))
        page_size = max(1, int(page_size))
        offset = (page - 1) * page_size
        params: List[Any] = []
        where_sql = ""
        if kw:
            where_sql = (
                "WHERE person_name ILIKE %s OR nickname ILIKE %s OR user_id ILIKE %s "
                "OR person_id ILIKE %s OR group_nick_name ILIKE %s"
            )
            like_kw = f"%{kw}%"
            params.extend([like_kw, like_kw, like_kw, like_kw, like_kw])

        with self._cursor() as cursor:
            cursor.execute(
                f"SELECT COUNT(*) FROM {self._table('person_registry')} {where_sql}",
                tuple(params),
            )
            total = int(cursor.fetchone()[0] or 0)

        rows = self._fetchall_dict(
            f"""
            SELECT *
            FROM {self._table('person_registry')}
            {where_sql}
            ORDER BY last_know DESC NULLS LAST, updated_at DESC
            LIMIT %s OFFSET %s
            """,
            tuple(params + [page_size, offset]),
        )
        items: List[Dict[str, Any]] = []
        for row in rows:
            decoded = self._decode_registry(row)
            if decoded is not None:
                items.append(decoded)
        return {
            "success": True,
            "keyword": kw,
            "page": page,
            "page_size": page_size,
            "total": total,
            "items": items,
        }

    def upsert_transcript_session(
        self,
        *,
        session_id: Optional[str] = None,
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        sid = str(session_id or uuid.uuid4().hex).strip() or uuid.uuid4().hex
        now = time.time()
        with self._cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {self._table('transcript_sessions')} (session_id, source, metadata_json, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (session_id) DO UPDATE SET
                    source = COALESCE(NULLIF(EXCLUDED.source, ''), {self._table('transcript_sessions')}.source),
                    metadata_json = COALESCE(NULLIF(EXCLUDED.metadata_json, ''), {self._table('transcript_sessions')}.metadata_json),
                    updated_at = EXCLUDED.updated_at
                RETURNING *
                """,
                (sid, str(source or ""), _encode_json(metadata or {}), now, now),
            )
            row = dict(cursor.fetchone())
        self._maybe_commit()
        row["metadata"] = _decode_json(row.pop("metadata_json", None), {})
        return row

    def append_transcript_messages(self, session_id: str, messages: List[Dict[str, Any]]) -> int:
        if not messages:
            return 0
        now = time.time()
        with self._cursor() as cursor:
            for message in messages:
                cursor.execute(
                    f"""
                    INSERT INTO {self._table('transcript_messages')} (
                        session_id, role, content, ts, metadata_json, created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(session_id or "").strip(),
                        str(message.get("role", "user") or "user"),
                        str(message.get("content", "") or ""),
                        self._safe_float(message.get("ts") or message.get("timestamp")),
                        _encode_json(message.get("metadata") if isinstance(message.get("metadata"), dict) else {}),
                        now,
                    ),
                )
            count = len(messages)
            cursor.execute(
                f"""
                UPDATE {self._table('transcript_sessions')}
                SET updated_at = %s
                WHERE session_id = %s
                """,
                (now, str(session_id or "").strip()),
            )
        self._maybe_commit()
        return count

    def get_transcript_messages(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        rows = self._fetchall_dict(
            f"""
            SELECT *
            FROM {self._table('transcript_messages')}
            WHERE session_id = %s
            ORDER BY created_at DESC, message_id DESC
            LIMIT %s
            """,
            (str(session_id or "").strip(), max(1, int(limit))),
        )
        items: List[Dict[str, Any]] = []
        for row in rows:
            decoded = self._decode_transcript_message(row)
            if decoded is not None:
                items.append(decoded)
        items.reverse()
        return items

    # A_memorix 原本缺少统一任务表，这里新增任务元数据存储以支撑插件侧异步导入、摘要和重建索引。
    def create_async_task(self, task_id: str, task_type: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """创建异步任务记录。

        Args:
            task_id: 任务唯一标识。
            task_type: 任务类型。
            payload: 任务参数。

        Returns:
            Dict[str, Any]: 创建后的任务记录。
        """
        now = time.time()
        with self._dict_cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {self._table('async_tasks')} (
                    task_id, task_type, status, payload_json, result_json, error_message,
                    created_at, updated_at, started_at, finished_at, cancel_requested
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NULL, NULL, 0)
                ON CONFLICT (task_id) DO UPDATE SET
                    task_type = EXCLUDED.task_type,
                    payload_json = EXCLUDED.payload_json,
                    updated_at = EXCLUDED.updated_at
                RETURNING *
                """,
                (
                    str(task_id or "").strip(),
                    str(task_type or "").strip(),
                    "queued",
                    _encode_json(payload or {}),
                    _encode_json({}),
                    "",
                    now,
                    now,
                ),
            )
            row = cursor.fetchone()
        self._maybe_commit()
        return self._decode_task(row) or {}

    def get_async_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """按任务 ID 查询任务记录。

        Args:
            task_id: 任务唯一标识。

        Returns:
            Optional[Dict[str, Any]]: 命中的任务记录；不存在时返回 ``None``。
        """
        row = self._fetchone_dict(
            f"SELECT * FROM {self._table('async_tasks')} WHERE task_id = %s",
            (str(task_id or "").strip(),),
        )
        return self._decode_task(row)

    def list_async_tasks(self, task_type: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """列出异步任务记录。

        Args:
            task_type: 可选任务类型过滤条件。
            limit: 返回记录上限。

        Returns:
            List[Dict[str, Any]]: 已解码的任务记录列表。
        """
        where_sql = ""
        params: List[Any] = []
        if task_type:
            where_sql = "WHERE task_type = %s"
            params.append(str(task_type))
        rows = self._fetchall_dict(
            f"""
            SELECT *
            FROM {self._table('async_tasks')}
            {where_sql}
            ORDER BY updated_at DESC NULLS LAST, created_at DESC NULLS LAST
            LIMIT %s
            """,
            tuple(params + [max(1, int(limit))]),
        )
        items: List[Dict[str, Any]] = []
        for row in rows:
            decoded = self._decode_task(row)
            if decoded is not None:
                items.append(decoded)
        return items

    def update_async_task(
        self,
        *,
        task_id: str,
        status: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        result: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        started_at: Optional[float] = None,
        finished_at: Optional[float] = None,
        cancel_requested: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        """更新异步任务状态与结果。

        Args:
            task_id: 任务唯一标识。
            status: 任务状态。
            payload: 更新后的任务参数。
            result: 任务执行结果。
            error_message: 错误信息。
            started_at: 开始时间戳。
            finished_at: 完成时间戳。
            cancel_requested: 是否请求取消。

        Returns:
            Optional[Dict[str, Any]]: 更新后的任务记录；任务不存在时返回 ``None``。
        """
        assignments = ["updated_at = %s"]
        params: List[Any] = [time.time()]
        if status is not None:
            assignments.append("status = %s")
            params.append(str(status))
        if payload is not None:
            assignments.append("payload_json = %s")
            params.append(_encode_json(payload))
        if result is not None:
            assignments.append("result_json = %s")
            params.append(_encode_json(result))
        if error_message is not None:
            assignments.append("error_message = %s")
            params.append(str(error_message))
        if started_at is not None:
            assignments.append("started_at = %s")
            params.append(started_at)
        if finished_at is not None:
            assignments.append("finished_at = %s")
            params.append(finished_at)
        if cancel_requested is not None:
            assignments.append("cancel_requested = %s")
            params.append(1 if cancel_requested else 0)
        params.append(str(task_id or "").strip())

        with self._cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE {self._table('async_tasks')}
                SET {', '.join(assignments)}
                WHERE task_id = %s
                """,
                tuple(params),
            )
        self._maybe_commit()
        return self.get_async_task(task_id)

    def query_paragraphs_temporal(
        self,
        *,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
        person: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100,
        allow_created_fallback: bool = True,
    ) -> List[Dict[str, Any]]:
        """按时间范围查询段落，并支持人物过滤。

        Args:
            start_ts: 起始时间戳。
            end_ts: 结束时间戳。
            person: 人物过滤条件。
            source: 来源过滤条件。
            limit: 返回记录上限。
            allow_created_fallback: 是否允许回退到创建时间排序。

        Returns:
            List[Dict[str, Any]]: 满足条件的段落列表。
        """
        conditions = ["COALESCE(is_deleted, 0) = 0"]
        params: List[Any] = []
        if source:
            conditions.append("source = %s")
            params.append(str(source))

        effective_expr = (
            "COALESCE(event_time_end, event_time, event_time_start, created_at)"
            if allow_created_fallback
            else "COALESCE(event_time_end, event_time, event_time_start)"
        )
        effective_start_expr = (
            "COALESCE(event_time_start, event_time, event_time_end, created_at)"
            if allow_created_fallback
            else "COALESCE(event_time_start, event_time, event_time_end)"
        )
        effective_end_expr = (
            "COALESCE(event_time_end, event_time, event_time_start, created_at)"
            if allow_created_fallback
            else "COALESCE(event_time_end, event_time, event_time_start)"
        )

        if start_ts is not None and end_ts is not None:
            conditions.append(f"{effective_end_expr} >= %s AND {effective_start_expr} <= %s")
            params.extend([float(start_ts), float(end_ts)])
        elif start_ts is not None:
            conditions.append(f"{effective_end_expr} >= %s")
            params.append(float(start_ts))
        elif end_ts is not None:
            conditions.append(f"{effective_start_expr} <= %s")
            params.append(float(end_ts))

        rows = self._fetchall_dict(
            f"""
            SELECT *
            FROM {self._table('paragraphs')}
            WHERE {' AND '.join(conditions)}
            ORDER BY {effective_expr} DESC NULLS LAST, updated_at DESC NULLS LAST, created_at DESC
            LIMIT %s
            """,
            tuple(params + [max(1, int(limit) * 4)]),
        )
        items: List[Dict[str, Any]] = []
        for row in rows:
            decoded = self._decode_paragraph(row)
            if decoded is not None:
                items.append(decoded)
        if person:
            person_lower = self._canonicalize_name(person)
            filtered: List[Dict[str, Any]] = []
            for paragraph in items:
                entities = self.get_paragraph_entities(str(paragraph["hash"]))
                if any(person_lower in self._canonicalize_name(str(entity.get("name", ""))) for entity in entities):
                    filtered.append(paragraph)
            items = filtered
        return items[: max(1, int(limit))]

    def clear_all(self) -> None:
        tables = [
            "paragraph_relations",
            "paragraph_entities",
            "relations",
            "deleted_relations",
            "entities",
            "paragraphs",
            "person_profile_switches",
            "person_profile_active_persons",
            "person_profile_snapshots",
            "person_profile_overrides",
            "person_registry",
            "transcript_messages",
            "transcript_sessions",
            "async_tasks",
        ]
        with self._cursor() as cursor:
            for table in tables:
                cursor.execute(f"DELETE FROM {self._table(table)}")
        self._maybe_commit()

    def get_statistics(self) -> Dict[str, Any]:
        with self._cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM {self._table('paragraphs')} WHERE COALESCE(is_deleted, 0) = 0")
            paragraphs = int(cursor.fetchone()[0] or 0)
            cursor.execute(f"SELECT COUNT(*) FROM {self._table('paragraphs')} WHERE COALESCE(is_deleted, 0) = 1")
            deleted_paragraphs = int(cursor.fetchone()[0] or 0)
            cursor.execute(f"SELECT COUNT(*) FROM {self._table('entities')} WHERE COALESCE(is_deleted, 0) = 0")
            entities = int(cursor.fetchone()[0] or 0)
            cursor.execute(f"SELECT COUNT(*) FROM {self._table('entities')} WHERE COALESCE(is_deleted, 0) = 1")
            deleted_entities = int(cursor.fetchone()[0] or 0)
            cursor.execute(f"SELECT COUNT(*) FROM {self._table('relations')}")
            relations = int(cursor.fetchone()[0] or 0)
            cursor.execute(f"SELECT COUNT(*) FROM {self._table('deleted_relations')}")
            deleted_relations = int(cursor.fetchone()[0] or 0)
            cursor.execute(f"SELECT COUNT(*) FROM {self._table('transcript_sessions')}")
            transcript_sessions = int(cursor.fetchone()[0] or 0)
            cursor.execute(f"SELECT COUNT(*) FROM {self._table('async_tasks')}")
            async_tasks = int(cursor.fetchone()[0] or 0)
            cursor.execute(f"SELECT COALESCE(SUM(word_count), 0) FROM {self._table('paragraphs')} WHERE COALESCE(is_deleted, 0) = 0")
            total_words = int(cursor.fetchone()[0] or 0)
        return {
            "backend": "postgres",
            "table_prefix": self.table_prefix,
            "paragraph_count": paragraphs,
            "entity_count": entities,
            "relation_count": relations,
            "total_words": total_words,
            "paragraphs": paragraphs,
            "deleted_paragraphs": deleted_paragraphs,
            "entities": entities,
            "deleted_entities": deleted_entities,
            "relations": relations,
            "deleted_relations": deleted_relations,
            "transcript_sessions": transcript_sessions,
            "async_tasks": async_tasks,
        }

    def graph_has_data(self) -> bool:
        if not self.is_connected():
            return False
        with self._cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM {self._table('graph_nodes')}")
            node_count = int(cursor.fetchone()[0] or 0)
            cursor.execute(f"SELECT COUNT(*) FROM {self._table('graph_edges')}")
            edge_count = int(cursor.fetchone()[0] or 0)
        return node_count > 0 or edge_count > 0

    def clear_graph_snapshot(self) -> None:
        with self._cursor() as cursor:
            cursor.execute(f"DELETE FROM {self._table('graph_edges')}")
            cursor.execute(f"DELETE FROM {self._table('graph_nodes')}")
        self._maybe_commit()

    def save_graph_snapshot(
        self,
        *,
        nodes: Sequence[Dict[str, Any]],
        edges: Sequence[Dict[str, Any]],
    ) -> None:
        now = time.time()
        with self._cursor() as cursor:
            cursor.execute(f"DELETE FROM {self._table('graph_edges')}")
            cursor.execute(f"DELETE FROM {self._table('graph_nodes')}")
            if nodes:
                cursor.executemany(
                    f"""
                    INSERT INTO {self._table('graph_nodes')} (
                        canonical_id, display_name, attrs_json, updated_at
                    )
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (canonical_id) DO UPDATE SET
                        display_name = EXCLUDED.display_name,
                        attrs_json = EXCLUDED.attrs_json,
                        updated_at = EXCLUDED.updated_at
                    """,
                    [
                        (
                            str(node.get("canonical_id") or ""),
                            str(node.get("display_name") or node.get("canonical_id") or ""),
                            _encode_json(node.get("attrs") or {}),
                            float(node.get("updated_at") or now),
                        )
                        for node in nodes
                        if str(node.get("canonical_id") or "").strip()
                    ],
                )
            if edges:
                cursor.executemany(
                    f"""
                    INSERT INTO {self._table('graph_edges')} (
                        source_canonical, target_canonical, source_display, target_display,
                        weight, relation_hashes_json, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (source_canonical, target_canonical) DO UPDATE SET
                        source_display = EXCLUDED.source_display,
                        target_display = EXCLUDED.target_display,
                        weight = EXCLUDED.weight,
                        relation_hashes_json = EXCLUDED.relation_hashes_json,
                        updated_at = EXCLUDED.updated_at
                    """,
                    [
                        (
                            str(edge.get("source_canonical") or ""),
                            str(edge.get("target_canonical") or ""),
                            str(edge.get("source_display") or edge.get("source_canonical") or ""),
                            str(edge.get("target_display") or edge.get("target_canonical") or ""),
                            float(edge.get("weight") or 0.0),
                            _encode_json(edge.get("relation_hashes") or []),
                            float(edge.get("updated_at") or now),
                        )
                        for edge in edges
                        if str(edge.get("source_canonical") or "").strip()
                        and str(edge.get("target_canonical") or "").strip()
                    ],
                )
        self._maybe_commit()

    def load_graph_snapshot(self) -> Dict[str, List[Dict[str, Any]]]:
        nodes = self._fetchall_dict(
            f"""
            SELECT canonical_id, display_name, attrs_json, updated_at
            FROM {self._table('graph_nodes')}
            ORDER BY canonical_id ASC
            """
        )
        edges = self._fetchall_dict(
            f"""
            SELECT source_canonical, target_canonical, source_display, target_display,
                   weight, relation_hashes_json, updated_at
            FROM {self._table('graph_edges')}
            ORDER BY source_canonical ASC, target_canonical ASC
            """
        )
        return {
            "nodes": [
                {
                    "canonical_id": str(row.get("canonical_id") or ""),
                    "display_name": str(row.get("display_name") or row.get("canonical_id") or ""),
                    "attrs": _decode_json(row.get("attrs_json"), {}),
                    "updated_at": self._safe_float(row.get("updated_at")) or 0.0,
                }
                for row in nodes
            ],
            "edges": [
                {
                    "source_canonical": str(row.get("source_canonical") or ""),
                    "target_canonical": str(row.get("target_canonical") or ""),
                    "source_display": str(row.get("source_display") or row.get("source_canonical") or ""),
                    "target_display": str(row.get("target_display") or row.get("target_canonical") or ""),
                    "weight": self._safe_float(row.get("weight")) or 0.0,
                    "relation_hashes": _decode_json(row.get("relation_hashes_json"), []),
                    "updated_at": self._safe_float(row.get("updated_at")) or 0.0,
                }
                for row in edges
            ],
        }

    def has_table(self, table_name: str) -> bool:
        if not self.is_connected():
            return False
        target = str(table_name or "").strip()
        if not target:
            return False
        candidates = [target]
        if not target.startswith(f"{self.table_prefix}_"):
            candidates.append(self._table(target))
        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = current_schema()
                  AND table_name = ANY(%s)
                LIMIT 1
                """,
                (candidates,),
            )
            return cursor.fetchone() is not None

    def has_data(self) -> bool:
        if not self.is_connected():
            return False
        return any(
            [
                self.count_paragraphs(include_deleted=True) > 0,
                self.count_relations(include_deleted=True) > 0,
                self.count_entities() > 0,
            ]
        )

    def ensure_fts_schema(self, conn=None) -> bool:
        del conn
        try:
            self._ensure_schema()
            return True
        except Exception as exc:
            logger.warning("ensure_fts_schema failed: %s", exc)
            return False

    def ensure_fts_backfilled(self, conn=None) -> bool:
        del conn
        try:
            rows = self._fetchall_dict(
                f"""
                SELECT hash, content
                FROM {self._table('paragraphs')}
                WHERE search_document IS NULL OR COALESCE(search_lexemes, '') = ''
                """
            )
            if not rows:
                return True
            with self._cursor() as cursor:
                for row in rows:
                    search_lexemes = self._build_search_lexemes(str(row.get("content") or ""))
                    cursor.execute(
                        f"""
                        UPDATE {self._table('paragraphs')}
                        SET search_lexemes = %s,
                            search_document = to_tsvector('simple', %s)
                        WHERE hash = %s
                        """,
                        (search_lexemes, search_lexemes, str(row.get("hash") or "")),
                    )
            self._maybe_commit()
            return True
        except Exception as exc:
            logger.warning("ensure_fts_backfilled failed: %s", exc)
            self.rollback()
            return False

    def ensure_relations_fts_schema(self, conn=None) -> bool:
        del conn
        try:
            self._ensure_schema()
            return True
        except Exception as exc:
            logger.warning("ensure_relations_fts_schema failed: %s", exc)
            return False

    def ensure_relations_fts_backfilled(self, conn=None) -> bool:
        del conn
        try:
            rows = self._fetchall_dict(
                f"""
                SELECT hash, subject, predicate, object
                FROM {self._table('relations')}
                WHERE search_document IS NULL OR COALESCE(search_lexemes, '') = ''
                """
            )
            if not rows:
                return True
            with self._cursor() as cursor:
                for row in rows:
                    relation_text = f"{row.get('subject', '')} {row.get('predicate', '')} {row.get('object', '')}"
                    search_lexemes = self._build_search_lexemes(relation_text)
                    cursor.execute(
                        f"""
                        UPDATE {self._table('relations')}
                        SET search_lexemes = %s,
                            search_document = to_tsvector('simple', %s)
                        WHERE hash = %s
                        """,
                        (search_lexemes, search_lexemes, str(row.get("hash") or "")),
                    )
            self._maybe_commit()
            return True
        except Exception as exc:
            logger.warning("ensure_relations_fts_backfilled failed: %s", exc)
            self.rollback()
            return False

    def ensure_paragraph_ngram_schema(self, conn=None) -> bool:
        del conn
        try:
            self._ensure_schema()
            return True
        except Exception as exc:
            logger.warning("ensure_paragraph_ngram_schema failed: %s", exc)
            return False

    def ensure_paragraph_ngram_backfilled(self, conn=None, batch_size: int = 2000, n: int = 2) -> bool:
        del conn
        n = max(1, int(n))
        batch_size = max(100, int(batch_size))
        try:
            row = self._fetchone_dict(
                f"SELECT value FROM {self._table('paragraph_ngram_meta')} WHERE key = %s",
                ("ngram_n",),
            )
            current_n = self._safe_int(row.get("value")) if row else None
            paragraph_count = self.count_paragraphs()
            indexed_row = self._fetchone_dict(
                f"SELECT COUNT(DISTINCT paragraph_hash) AS count FROM {self._table('paragraph_ngrams')}"
            )
            indexed_docs = int(indexed_row.get("count") or 0) if indexed_row else 0
            if current_n == n and paragraph_count == indexed_docs:
                return True

            rows = self._fetchall_dict(
                f"""
                SELECT hash, content
                FROM {self._table('paragraphs')}
                WHERE COALESCE(is_deleted, 0) = 0
                """
            )
            with self._cursor() as cursor:
                cursor.execute(f"DELETE FROM {self._table('paragraph_ngrams')}")
                batch: List[Tuple[str, str]] = []
                for row in rows:
                    paragraph_hash = str(row.get("hash") or "")
                    terms = list(dict.fromkeys(self._char_ngrams(str(row.get("content") or ""), n)))
                    for term in terms:
                        batch.append((term, paragraph_hash))
                    if len(batch) >= batch_size:
                        cursor.executemany(
                            f"""
                            INSERT INTO {self._table('paragraph_ngrams')} (term, paragraph_hash)
                            VALUES (%s, %s)
                            ON CONFLICT (term, paragraph_hash) DO NOTHING
                            """,
                            batch,
                        )
                        batch.clear()
                if batch:
                    cursor.executemany(
                        f"""
                        INSERT INTO {self._table('paragraph_ngrams')} (term, paragraph_hash)
                        VALUES (%s, %s)
                        ON CONFLICT (term, paragraph_hash) DO NOTHING
                        """,
                        batch,
                    )
                cursor.execute(
                    f"""
                    INSERT INTO {self._table('paragraph_ngram_meta')} (key, value)
                    VALUES (%s, %s)
                    ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                    """,
                    ("ngram_n", str(n)),
                )
                cursor.execute(
                    f"""
                    INSERT INTO {self._table('paragraph_ngram_meta')} (key, value)
                    VALUES (%s, %s)
                    ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                    """,
                    ("paragraph_count", str(paragraph_count)),
                )
            self._maybe_commit()
            return True
        except Exception as exc:
            logger.warning("ensure_paragraph_ngram_backfilled failed: %s", exc)
            self.rollback()
            return False

    def fts_upsert_paragraph(self, paragraph_hash: str, conn=None) -> bool:
        del conn
        paragraph = self.get_paragraph(paragraph_hash)
        if not paragraph:
            return False
        search_lexemes = self._build_search_lexemes(str(paragraph.get("content") or ""))
        with self._cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE {self._table('paragraphs')}
                SET search_lexemes = %s,
                    search_document = to_tsvector('simple', %s)
                WHERE hash = %s
                """,
                (search_lexemes, search_lexemes, str(paragraph_hash or "").strip()),
            )
        self._maybe_commit()
        return True

    def fts_delete_paragraph(self, paragraph_hash: str, conn=None) -> bool:
        del conn
        with self._cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE {self._table('paragraphs')}
                SET search_lexemes = NULL,
                    search_document = NULL
                WHERE hash = %s
                """,
                (str(paragraph_hash or "").strip(),),
            )
            cursor.execute(
                f"DELETE FROM {self._table('paragraph_ngrams')} WHERE paragraph_hash = %s",
                (str(paragraph_hash or "").strip(),),
            )
            affected = int(cursor.rowcount or 0)
        self._maybe_commit()
        return affected >= 0

    def fts_search_bm25(self, match_query: str, limit: int = 20, max_doc_len: int = 2000, conn=None) -> List[Dict[str, Any]]:
        del conn
        tokens = self._parse_match_query(match_query)
        tsquery = self._build_tsquery(tokens)
        if not tsquery:
            return []
        try:
            rows = self._fetchall_dict(
                f"""
                WITH search_query AS (
                    SELECT to_tsquery('simple', %s) AS q
                )
                SELECT p.hash, p.content, ts_rank_cd(p.search_document, sq.q) AS rank_score
                FROM {self._table('paragraphs')} p
                CROSS JOIN search_query sq
                WHERE COALESCE(p.is_deleted, 0) = 0
                  AND p.search_document IS NOT NULL
                  AND p.search_document @@ sq.q
                ORDER BY rank_score DESC, p.updated_at DESC NULLS LAST, p.created_at DESC
                LIMIT %s
                """,
                (tsquery, max(1, int(limit))),
            )
        except Exception as exc:
            logger.warning("fts_search_bm25 failed: %s", exc)
            return []
        results: List[Dict[str, Any]] = []
        for row in rows:
            content = str(row.get("content") or "")
            if max_doc_len > 0:
                content = content[:max_doc_len]
            rank_score = float(row.get("rank_score") or 0.0)
            results.append(
                {
                    "hash": str(row.get("hash") or ""),
                    "content": content,
                    "bm25_score": -rank_score,
                }
            )
        return results

    def fts_search_relations_bm25(
        self,
        match_query: str,
        limit: int = 20,
        max_doc_len: int = 512,
        conn=None,
    ) -> List[Dict[str, Any]]:
        del conn
        tokens = self._parse_match_query(match_query)
        tsquery = self._build_tsquery(tokens)
        if not tsquery:
            return []
        try:
            rows = self._fetchall_dict(
                f"""
                WITH search_query AS (
                    SELECT to_tsquery('simple', %s) AS q
                )
                SELECT r.hash, r.subject, r.predicate, r.object,
                       ts_rank_cd(r.search_document, sq.q) AS rank_score
                FROM {self._table('relations')} r
                CROSS JOIN search_query sq
                WHERE r.search_document IS NOT NULL
                  AND r.search_document @@ sq.q
                ORDER BY rank_score DESC, r.last_accessed DESC NULLS LAST, r.created_at DESC
                LIMIT %s
                """,
                (tsquery, max(1, int(limit))),
            )
        except Exception as exc:
            logger.warning("fts_search_relations_bm25 failed: %s", exc)
            return []
        results: List[Dict[str, Any]] = []
        for row in rows:
            content = f"{row.get('subject', '')} {row.get('predicate', '')} {row.get('object', '')}"
            if max_doc_len > 0:
                content = content[:max_doc_len]
            rank_score = float(row.get("rank_score") or 0.0)
            results.append(
                {
                    "hash": str(row.get("hash") or ""),
                    "subject": str(row.get("subject") or ""),
                    "predicate": str(row.get("predicate") or ""),
                    "object": str(row.get("object") or ""),
                    "content": content,
                    "bm25_score": -rank_score,
                }
            )
        return results

    def ngram_search_paragraphs(
        self,
        tokens: Union[str, Sequence[str]],
        limit: int = 20,
        max_doc_len: int = 2000,
        conn=None,
    ) -> List[Dict[str, Any]]:
        del conn
        if isinstance(tokens, str):
            uniq = [token for token in dict.fromkeys(self._char_ngrams(tokens, 2)) if token]
        else:
            uniq = [token for token in dict.fromkeys([str(item or "").strip().lower() for item in tokens]) if token]
        if not uniq:
            return []
        try:
            rows = self._fetchall_dict(
                f"""
                SELECT p.hash, p.content, COUNT(*) AS hit_terms
                FROM {self._table('paragraph_ngrams')} ng
                JOIN {self._table('paragraphs')} p ON p.hash = ng.paragraph_hash
                WHERE ng.term = ANY(%s)
                  AND COALESCE(p.is_deleted, 0) = 0
                GROUP BY p.hash, p.content
                ORDER BY hit_terms DESC, MAX(p.updated_at) DESC NULLS LAST, MAX(p.created_at) DESC
                LIMIT %s
                """,
                (uniq, max(1, int(limit))),
            )
        except Exception as exc:
            logger.warning("ngram_search_paragraphs failed: %s", exc)
            return []
        token_count = max(1, len(uniq))
        results: List[Dict[str, Any]] = []
        for row in rows:
            content = str(row.get("content") or "")
            if max_doc_len > 0:
                content = content[:max_doc_len]
            score = float(int(row.get("hit_terms") or 0) / token_count)
            results.append(
                {
                    "hash": str(row.get("hash") or ""),
                    "content": content,
                    "bm25_score": -score,
                    "fallback_score": score,
                }
            )
        return results

    def fts_doc_count(self, conn=None) -> int:
        del conn
        with self._cursor() as cursor:
            cursor.execute(
                f"""
                SELECT COUNT(*)
                FROM {self._table('paragraphs')}
                WHERE search_document IS NOT NULL
                """
            )
            return int(cursor.fetchone()[0] or 0)

    def shrink_memory(self, conn=None) -> None:
        del conn
        # PostgreSQL manages memory at the backend level; keep the A-style hook as a no-op.
        return None

    # 与 A_memorix 相比，na_memorix 在 PostgreSQL 中显式持久化永久记忆标记，便于 GC 与冻结流程协同。
    def set_permanence(self, hash_value: str, item_type: str, is_permanent: bool) -> bool:
        """设置段落或关系的永久保留标记。

        Args:
            hash_value: 条目哈希值。
            item_type: 条目类型，仅支持 ``paragraph`` 或 ``relation``。
            is_permanent: 是否设为永久保留。

        Returns:
            bool: 更新成功返回 ``True``。
        """
        table_map = {
            "paragraph": "paragraphs",
            "relation": "relations",
        }
        table_name = table_map.get(str(item_type or "").strip())
        if not table_name:
            raise ValueError(f"类型 {item_type} 不支持设置永久性")

        with self._cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE {self._table(table_name)}
                SET is_permanent = %s
                WHERE hash = %s
                """,
                (1 if is_permanent else 0, str(hash_value or "").strip()),
            )
            affected = int(cursor.rowcount or 0)
        self._maybe_commit()
        if affected > 0:
            logger.debug(
                "设置永久记忆: %s/%s -> %s",
                item_type,
                str(hash_value or "")[:8],
                bool(is_permanent),
            )
            return True
        return False

    def get_entity_gc_candidates(self, isolated_hashes: List[str], retention_seconds: float) -> List[str]:
        """筛选可进入实体回收流程的候选项。

        Args:
            isolated_hashes: 图中孤立实体哈希或名称列表。
            retention_seconds: 最短保留时长。

        Returns:
            List[str]: 满足回收条件的实体哈希列表。
        """
        if not isolated_hashes:
            return []

        normalized_hashes: List[str] = []
        for item in isolated_hashes:
            if not item:
                continue
            value = str(item).strip()
            if len(value) == 64 and all(ch in "0123456789abcdefABCDEF" for ch in value):
                normalized_hashes.append(value.lower())
                continue
            canonical = self._canonicalize_name(value)
            if canonical:
                normalized_hashes.append(compute_hash(canonical))

        normalized_hashes = list(dict.fromkeys(normalized_hashes))
        if not normalized_hashes:
            return []

        cutoff = time.time() - max(0.0, float(retention_seconds or 0.0))
        rows = self._fetchall_dict(
            f"""
            SELECT e.hash
            FROM {self._table('entities')} e
            WHERE e.hash = ANY(%s)
              AND COALESCE(e.is_deleted, 0) = 0
              AND (e.created_at IS NULL OR e.created_at < %s)
              AND NOT EXISTS (
                    SELECT 1
                    FROM {self._table('paragraph_entities')} pe
                    JOIN {self._table('paragraphs')} p ON p.hash = pe.paragraph_hash
                    WHERE pe.entity_hash = e.hash
                      AND COALESCE(p.is_deleted, 0) = 0
              )
            ORDER BY e.created_at ASC NULLS FIRST, e.hash ASC
            """,
            (normalized_hashes, cutoff),
        )
        return [str(row.get("hash") or "") for row in rows if str(row.get("hash") or "").strip()]

    def get_paragraph_gc_candidates(self, retention_seconds: float) -> List[str]:
        """筛选可进入段落回收流程的候选项。

        Args:
            retention_seconds: 最短保留时长。

        Returns:
            List[str]: 满足回收条件的段落哈希列表。
        """
        cutoff = time.time() - max(0.0, float(retention_seconds or 0.0))
        rows = self._fetchall_dict(
            f"""
            SELECT p.hash
            FROM {self._table('paragraphs')} p
            WHERE COALESCE(p.is_deleted, 0) = 0
              AND COALESCE(p.is_permanent, 0) = 0
              AND (p.created_at IS NULL OR p.created_at < %s)
              AND NOT EXISTS (
                    SELECT 1
                    FROM {self._table('paragraph_relations')} pr
                    WHERE pr.paragraph_hash = p.hash
              )
              AND NOT EXISTS (
                    SELECT 1
                    FROM {self._table('paragraph_entities')} pe
                    WHERE pe.paragraph_hash = p.hash
              )
            ORDER BY p.created_at ASC NULLS FIRST, p.hash ASC
            """,
            (cutoff,),
        )
        return [str(row.get("hash") or "") for row in rows if str(row.get("hash") or "").strip()]

    def mark_as_deleted(self, hashes: List[str], type_: str) -> int:
        """将实体或段落标记为软删除。

        Args:
            hashes: 待标记哈希列表。
            type_: 条目类型，支持 ``entity`` 或 ``paragraph``。

        Returns:
            int: 成功标记的条目数量。
        """
        if not hashes:
            return 0

        table_name = "entities" if str(type_ or "").strip() == "entity" else "paragraphs"
        now = time.time()
        affected = 0
        batch_size = 900

        with self._cursor() as cursor:
            for start in range(0, len(hashes), batch_size):
                batch = [str(item or "").strip() for item in hashes[start : start + batch_size] if str(item or "").strip()]
                if not batch:
                    continue
                cursor.execute(
                    f"""
                    UPDATE {self._table(table_name)}
                    SET is_deleted = 1,
                        deleted_at = %s
                    WHERE COALESCE(is_deleted, 0) = 0
                      AND hash = ANY(%s)
                    """,
                    (now, batch),
                )
                affected += int(cursor.rowcount or 0)
        self._maybe_commit()
        if affected > 0:
            logger.info("软删除标记 (%s): %s 项", table_name, affected)
        return affected

    def sweep_deleted_items(self, type_: str, grace_period_seconds: float) -> List[Tuple[str, str]]:
        """列出超过宽限期、可物理删除的软删除条目。

        Args:
            type_: 条目类型，支持 ``entity`` 或 ``paragraph``。
            grace_period_seconds: 软删除宽限期。

        Returns:
            List[Tuple[str, str]]: 待物理删除的 ``(hash, name)`` 列表。
        """
        table_name = "entities" if str(type_ or "").strip() == "entity" else "paragraphs"
        cutoff = time.time() - max(0.0, float(grace_period_seconds or 0.0))
        columns = "hash, name" if table_name == "entities" else "hash, '' AS name"
        rows = self._fetchall_dict(
            f"""
            SELECT {columns}
            FROM {self._table(table_name)}
            WHERE COALESCE(is_deleted, 0) = 1
              AND deleted_at < %s
            ORDER BY deleted_at ASC, hash ASC
            """,
            (cutoff,),
        )
        return [(str(row.get("hash") or ""), str(row.get("name") or "")) for row in rows]

    def physically_delete_entities(self, hashes: List[str]) -> int:
        """物理删除实体记录。

        Args:
            hashes: 待删除实体哈希列表。

        Returns:
            int: 成功删除的实体数量。
        """
        if not hashes:
            return 0

        affected = 0
        batch_size = 900
        with self._cursor() as cursor:
            for start in range(0, len(hashes), batch_size):
                batch = [str(item or "").strip() for item in hashes[start : start + batch_size] if str(item or "").strip()]
                if not batch:
                    continue
                cursor.execute(
                    f"DELETE FROM {self._table('entities')} WHERE hash = ANY(%s)",
                    (batch,),
                )
                affected += int(cursor.rowcount or 0)
        self._maybe_commit()
        return affected

    def physically_delete_paragraphs(self, hashes: List[str]) -> int:
        """物理删除段落记录。

        Args:
            hashes: 待删除段落哈希列表。

        Returns:
            int: 成功删除的段落数量。
        """
        if not hashes:
            return 0

        affected = 0
        batch_size = 900
        with self._cursor() as cursor:
            for start in range(0, len(hashes), batch_size):
                batch = [str(item or "").strip() for item in hashes[start : start + batch_size] if str(item or "").strip()]
                if not batch:
                    continue
                cursor.execute(
                    f"DELETE FROM {self._table('paragraphs')} WHERE hash = ANY(%s)",
                    (batch,),
                )
                affected += int(cursor.rowcount or 0)
        self._maybe_commit()
        return affected

    def vacuum(self) -> None:
        """执行 PostgreSQL `VACUUM ANALYZE` 以整理统计信息。"""
        if self.is_connected():
            self._maybe_commit()

        conn = psycopg2.connect(self._db_url)
        try:
            conn.autocommit = True
            with conn.cursor() as cursor:
                for table_name in _TABLE_NAMES:
                    cursor.execute(f"VACUUM ANALYZE {self._table(table_name)}")
            logger.info("数据库优化完成 (PostgreSQL VACUUM ANALYZE)")
        finally:
            conn.close()
