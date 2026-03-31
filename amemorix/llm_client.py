"""封装 OpenAI 兼容接口的异步对话补全客户端。"""

import asyncio
import json
import os
from typing import Any, Dict, Optional, Tuple

from openai import AsyncOpenAI

from nekro_agent.core.config import config as app_config

from .common.logging import get_logger

logger = get_logger("A_Memorix.LLMClient")


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


class LLMClient:
    """提供带重试能力的聊天补全客户端。

    Attributes:
        base_url (str): OpenAI 兼容服务地址。
        api_key (str): 调用接口使用的密钥。
        model (str): 最终选用的聊天模型名称。
        timeout_seconds (float): 单次请求超时时间。
        max_retries (int): 请求失败时的最大重试次数。
        _client (Optional[AsyncOpenAI]): 惰性初始化的异步客户端实例。
    """

    def __init__(
        self,
        *,
        base_url: str = "",
        api_key: str = "",
        model: str = "",
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
    ):
        """初始化聊天补全客户端配置。

        Args:
            base_url: OpenAI 兼容服务地址。
            api_key: 接口访问密钥。
            model: 聊天模型名称。
            timeout_seconds: 单次请求超时时间。
            max_retries: 失败后的最大重试次数。
        """
        self.base_url = str(base_url or "").strip()
        self.api_key = str(api_key or "").strip()
        self.model = str(model or "").strip()
        if self.model:
            try:
                group = app_config.get_model_group_info(self.model)
            except Exception:
                group = None
            if group is not None:
                self.base_url = self.base_url or str(group.BASE_URL or "").strip()
                self.api_key = self.api_key or str(group.API_KEY or "").strip()
                self.model = str(group.CHAT_MODEL or self.model).strip()

        self.base_url = self.base_url or str(_first_env("OPENAPI_BASE_URL", "OPENAI_BASE_URL")).strip()
        self.api_key = self.api_key or str(_first_env("OPENAPI_API_KEY", "OPENAI_API_KEY")).strip()
        if not self.model:
            self.model = str(
                _first_env(
                    "OPENAPI_CHAT_MODEL",
                    "OPENAI_CHAT_MODEL",
                    "OPENAPI_MODEL",
                    "OPENAI_MODEL",
                )
                or "gpt-4o-mini"
            ).strip()
        self.timeout_seconds = float(timeout_seconds or 60.0)
        self.max_retries = max(1, int(max_retries))
        self._client: Optional[AsyncOpenAI] = None

    def _get_client(self) -> AsyncOpenAI:
        """惰性创建并复用异步 OpenAI 客户端。

        Returns:
            AsyncOpenAI: 可直接发起聊天请求的客户端实例。
        """
        if self._client is None:
            kwargs = {
                "api_key": self.api_key or "EMPTY",
                "timeout": self.timeout_seconds,
                "max_retries": 0,
            }
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def complete(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 1200) -> str:
        """执行一次聊天补全请求。

        Args:
            prompt: 发送给模型的用户提示词。
            temperature: 采样温度。
            max_tokens: 生成回复的最大 token 数。

        Returns:
            str: 模型返回的文本内容。

        Raises:
            Exception: 当所有重试均失败时抛出最后一次异常。
        """
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                client = self._get_client()
                resp = await client.chat.completions.create(
                    model=self.model,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                    messages=[{"role": "user", "content": prompt}],
                )
                choice = resp.choices[0] if resp.choices else None
                if not choice:
                    return ""
                return str(choice.message.content or "")
            except Exception as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                await asyncio.sleep(min(6.0, 2 ** (attempt - 1)))
        if last_exc is not None:
            raise last_exc
        return ""

    async def complete_json(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> Tuple[bool, Dict[str, Any], str]:
        """请求模型并尽量将结果解析为 JSON。

        Args:
            prompt: 发送给模型的用户提示词。
            temperature: 采样温度。
            max_tokens: 生成回复的最大 token 数。

        Returns:
            Tuple[bool, Dict[str, Any], str]: 依次为解析是否成功、解析后的对象与原始文本。
        """
        text = await self.complete(prompt, temperature=temperature, max_tokens=max_tokens)
        if not text:
            return False, {}, ""
        raw = text.strip()
        try:
            return True, json.loads(raw), raw
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                try:
                    return True, json.loads(raw[start : end + 1]), raw
                except json.JSONDecodeError:
                    pass
        return False, {}, raw
