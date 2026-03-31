"""封装 OpenAI 兼容嵌入接口并保留 A_memorix 适配层能力。"""

import asyncio
import os
import time
from typing import List, Optional, Union

import numpy as np
from openai import AsyncOpenAI

from nekro_agent.core.config import config as app_config

from amemorix.common.logging import get_logger

logger = get_logger("A_Memorix.EmbeddingAPIAdapter")


def _first_env(*keys: str) -> str:
    """按顺序读取首个非空环境变量值。

    Args:
        *keys: 候选环境变量名列表。

    Returns:
        str: 首个命中的非空值；若均为空则返回空字符串。
    """
    for key in keys:
        value = str(os.getenv(key, "") or "").strip()
        if value:
            return value
    return ""


class EmbeddingAPIAdapter:
    """提供带重试与维度探测能力的嵌入适配器。

    Attributes:
        batch_size (int): 默认批大小。
        max_concurrent (int): 并发请求上限。
        default_dimension (int): 默认向量维度。
        enable_cache (bool): 是否启用缓存标记。
        model_name (str): 逻辑模型名称。
        timeout_seconds (float): 单次请求超时时间。
        max_retries (int): 最大重试次数。
        base_url (str): OpenAI 兼容服务地址。
        api_key (str): 接口访问密钥。
        openai_model (str): 实际使用的嵌入模型名称。
        retry_config (dict): 重试策略配置。
        max_attempts (int): 最终请求尝试次数。
        max_wait_seconds (float): 指数退避最大等待秒数。
        min_wait_seconds (float): 指数退避最小等待秒数。
        _dimension (Optional[int]): 已探测到的维度。
        _dimension_detected (bool): 是否已完成维度探测。
        _client (Optional[AsyncOpenAI]): 惰性创建的异步客户端。
    """

    def __init__(
        self,
        batch_size: int = 32,
        max_concurrent: int = 5,
        default_dimension: int = 1024,
        enable_cache: bool = False,
        model_name: str = "auto",
        retry_config: Optional[dict] = None,
        base_url: str = "",
        api_key: str = "",
        openai_model: str = "",
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
    ):
        """初始化嵌入适配器配置。

        Args:
            batch_size: 默认批大小。
            max_concurrent: 并发请求上限。
            default_dimension: 默认向量维度。
            enable_cache: 是否启用缓存标记。
            model_name: 逻辑模型名称。
            retry_config: 重试策略配置。
            base_url: OpenAI 兼容服务地址。
            api_key: 接口访问密钥。
            openai_model: 实际使用的嵌入模型名称。
            timeout_seconds: 单次请求超时时间。
            max_retries: 最大重试次数。
        """
        self.batch_size = max(1, int(batch_size))
        self.max_concurrent = max(1, int(max_concurrent))
        self.default_dimension = max(1, int(default_dimension))
        self.enable_cache = bool(enable_cache)
        self.model_name = str(model_name or "auto")
        self.timeout_seconds = float(timeout_seconds or 30.0)
        self.max_retries = max(1, int(max_retries))

        self.base_url = str(base_url or "").strip()
        self.api_key = str(api_key or "").strip()
        self.openai_model = str(openai_model or "").strip()

        if self.model_name and self.model_name.lower() != "auto":
            try:
                group = app_config.get_model_group_info(self.model_name)
            except Exception:
                group = None
            if group is not None:
                self.base_url = self.base_url or str(group.BASE_URL or "").strip()
                self.api_key = self.api_key or str(group.API_KEY or "").strip()
                self.openai_model = self.openai_model or str(group.CHAT_MODEL or "").strip()

        self.base_url = self.base_url or str(_first_env("OPENAPI_BASE_URL", "OPENAI_BASE_URL")).strip()
        self.api_key = self.api_key or str(_first_env("OPENAPI_API_KEY", "OPENAI_API_KEY")).strip()
        if not self.openai_model:
            if self.model_name and self.model_name.lower() != "auto":
                self.openai_model = self.model_name
            else:
                self.openai_model = str(
                    _first_env(
                        "OPENAPI_EMBEDDING_MODEL",
                        "OPENAI_EMBEDDING_MODEL",
                        "OPENAPI_MODEL",
                        "OPENAI_MODEL",
                    )
                    or "text-embedding-3-large"
                ).strip()

        self.retry_config = retry_config or {}
        self.max_attempts = max(1, int(self.retry_config.get("max_attempts", self.max_retries)))
        self.max_wait_seconds = max(0.1, float(self.retry_config.get("max_wait_seconds", 30)))
        self.min_wait_seconds = max(0.1, float(self.retry_config.get("min_wait_seconds", 1)))

        self._dimension: Optional[int] = None
        self._dimension_detected = False
        self._client: Optional[AsyncOpenAI] = None

        self._total_encoded = 0
        self._total_errors = 0
        self._total_time = 0.0

        logger.info(
            "Embedding adapter initialized: model=%s, default_dim=%s, base_url=%s",
            self.openai_model,
            self.default_dimension,
            self.base_url or "<default>",
        )

    def _get_client(self) -> AsyncOpenAI:
        """惰性创建并复用异步嵌入客户端。

        Returns:
            AsyncOpenAI: 可直接执行嵌入请求的客户端实例。
        """
        if self._client is None:
            kwargs = {
                "api_key": self.api_key or "EMPTY",
                "timeout": self.timeout_seconds,
                "max_retries": 0,  # retries are handled by adapter policy
            }
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def _request_embeddings(
        self,
        inputs: Union[str, List[str]],
        dimensions: Optional[int] = None,
    ) -> List[List[float]]:
        """向远端服务请求嵌入向量。

        Args:
            inputs: 单条或多条输入文本。
            dimensions: 期望的向量维度。

        Returns:
            List[List[float]]: 每条输入对应的向量列表。

        Raises:
            Exception: 当重试耗尽后抛出最后一次异常。
        """
        client = self._get_client()
        payload = {"model": self.openai_model, "input": inputs}
        if dimensions is not None:
            payload["dimensions"] = int(dimensions)

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                resp = await client.embeddings.create(**payload)
                return [list(item.embedding) for item in resp.data]
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_attempts:
                    break
                wait_s = min(
                    self.max_wait_seconds,
                    self.min_wait_seconds * (2 ** (attempt - 1)),
                )
                logger.warning(
                    "Embedding request failed (attempt %s/%s), retry in %.1fs: %s",
                    attempt,
                    self.max_attempts,
                    wait_s,
                    exc,
                )
                await asyncio.sleep(wait_s)

        assert last_error is not None
        raise last_error

    async def _detect_dimension(self) -> int:
        """探测远端模型的真实输出维度。

        Returns:
            int: 已探测或回退后的向量维度。
        """
        if self._dimension_detected and self._dimension is not None:
            return self._dimension

        # Probe with requested dimension first.
        try:
            probed = await self._request_embeddings("dimension_probe", dimensions=self.default_dimension)
            if probed and probed[0]:
                self._dimension = len(probed[0])
                self._dimension_detected = True
                return self._dimension
        except Exception as exc:
            logger.debug("Dimension probe with requested dimension failed: %s", exc)

        # Probe with natural model dimension.
        try:
            probed = await self._request_embeddings("dimension_probe", dimensions=None)
            if probed and probed[0]:
                self._dimension = len(probed[0])
                self._dimension_detected = True
                return self._dimension
        except Exception as exc:
            logger.warning("Dimension detection failed, fallback to default: %s", exc)

        self._dimension = self.default_dimension
        self._dimension_detected = True
        return self.default_dimension

    async def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        normalize: bool = True,
        dimensions: Optional[int] = None,
    ) -> np.ndarray:
        """将文本编码为嵌入向量。

        Args:
            texts: 单条文本或文本列表。
            batch_size: 本次调用的批大小。
            show_progress: 兼容旧接口的进度展示开关。
            normalize: 兼容旧接口的归一化开关。
            dimensions: 指定输出向量维度。

        Returns:
            np.ndarray: 单条输入返回一维向量，多条输入返回二维矩阵。
        """
        del show_progress  # kept for compatibility
        del normalize  # API already returns normalized-ish vectors by model behavior
        start = time.time()

        if isinstance(texts, str):
            input_texts = [texts]
            single = True
        else:
            input_texts = list(texts)
            single = False

        target_dim = dimensions
        if target_dim is None:
            if not self._dimension_detected:
                await self._detect_dimension()
            target_dim = self._dimension or self.default_dimension
        target_dim = int(target_dim)

        if not input_texts:
            empty = np.zeros((0, target_dim), dtype=np.float32)
            return empty[0] if single else empty

        use_batch = max(1, int(batch_size or self.batch_size))
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def _encode_chunk(chunk: List[str]) -> np.ndarray:
            async with semaphore:
                try:
                    # Always send the effective target dimension so providers that
                    # support OpenAI-compatible `dimensions` return stable vector size.
                    vectors = await self._request_embeddings(chunk, dimensions=target_dim)
                    arr = np.asarray(vectors, dtype=np.float32)
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    return arr
                except Exception as exc:
                    self._total_errors += len(chunk)
                    logger.error("Embedding chunk failed: %s", exc)
                    return np.full((len(chunk), target_dim), np.nan, dtype=np.float32)

        tasks = []
        for idx in range(0, len(input_texts), use_batch):
            tasks.append(_encode_chunk(input_texts[idx : idx + use_batch]))
        chunks = await asyncio.gather(*tasks)
        out = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, target_dim), dtype=np.float32)

        self._total_encoded += len(input_texts)
        self._total_time += max(0.0, time.time() - start)
        if out.ndim == 1:
            out = out.reshape(1, -1)
        return out[0] if single else out

    async def encode_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        show_progress: bool = False,
        dimensions: Optional[int] = None,
    ) -> np.ndarray:
        """批量编码文本并临时覆盖并发设置。

        Args:
            texts: 待编码文本列表。
            batch_size: 本次调用的批大小。
            num_workers: 临时并发请求数。
            show_progress: 兼容旧接口的进度展示开关。
            dimensions: 指定输出向量维度。

        Returns:
            np.ndarray: 编码后的二维向量矩阵。
        """
        old = self.max_concurrent
        if num_workers is not None:
            self.max_concurrent = max(1, int(num_workers))
        try:
            return await self.encode(
                texts=texts,
                batch_size=batch_size,
                show_progress=show_progress,
                dimensions=dimensions,
            )
        finally:
            self.max_concurrent = old

    def get_embedding_dimension(self) -> int:
        """返回当前已知的嵌入维度。

        Returns:
            int: 当前维度值。
        """
        if self._dimension is not None:
            return int(self._dimension)
        return int(self.default_dimension)

    def get_model_info(self) -> dict:
        """汇总当前模型与编码统计信息。

        Returns:
            dict: 模型配置与运行统计信息。
        """
        avg_time = self._total_time / self._total_encoded if self._total_encoded > 0 else 0.0
        return {
            "model_name": self.openai_model,
            "dimension": self.get_embedding_dimension(),
            "dimension_detected": self._dimension_detected,
            "batch_size": self.batch_size,
            "max_concurrent": self.max_concurrent,
            "base_url": self.base_url,
            "total_encoded": self._total_encoded,
            "total_errors": self._total_errors,
            "avg_time_per_text": avg_time,
        }

    @property
    def is_model_loaded(self) -> bool:
        """判断模型是否可直接使用。

        Returns:
            bool: 当前实现始终返回 ``True``。
        """
        return True

    def __repr__(self) -> str:
        """返回适配器调试字符串表示。

        Returns:
            str: 包含模型名、维度与编码统计的摘要字符串。
        """
        return (
            "EmbeddingAPIAdapter("
            f"model={self.openai_model}, "
            f"dim={self.get_embedding_dimension()}, "
            f"encoded={self._total_encoded})"
        )


def create_embedding_api_adapter(
    batch_size: int = 32,
    max_concurrent: int = 5,
    default_dimension: int = 1024,
    model_name: str = "auto",
    retry_config: Optional[dict] = None,
    base_url: str = "",
    api_key: str = "",
    openai_model: str = "",
    timeout_seconds: float = 30.0,
    max_retries: int = 3,
) -> EmbeddingAPIAdapter:
    """创建嵌入适配器实例。

    Args:
        batch_size: 默认批大小。
        max_concurrent: 并发请求上限。
        default_dimension: 默认向量维度。
        model_name: 逻辑模型名称。
        retry_config: 重试策略配置。
        base_url: OpenAI 兼容服务地址。
        api_key: 接口访问密钥。
        openai_model: 实际使用的嵌入模型名称。
        timeout_seconds: 单次请求超时时间。
        max_retries: 最大重试次数。

    Returns:
        EmbeddingAPIAdapter: 初始化后的嵌入适配器。
    """
    return EmbeddingAPIAdapter(
        batch_size=batch_size,
        max_concurrent=max_concurrent,
        default_dimension=default_dimension,
        model_name=model_name,
        retry_config=retry_config,
        base_url=base_url,
        api_key=api_key,
        openai_model=openai_model,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )
