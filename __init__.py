"""提供 na_memorix 插件包的入口与兼容别名。"""

from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType

__version__ = "0.7.0"
__author__ = "KroMiose"


def _ensure_aliases() -> tuple[ModuleType, ModuleType]:
    """初始化兼容导入别名。

    Returns:
        tuple[ModuleType, ModuleType]: ``amemorix`` 与 ``core`` 两个兼容包对象。
    """
    amemorix_pkg = sys.modules.get("amemorix")
    if amemorix_pkg is None:
        amemorix_pkg = import_module(f"{__name__}.amemorix")
        sys.modules.setdefault("amemorix", amemorix_pkg)

    core_pkg = sys.modules.get("core")
    if core_pkg is None:
        core_pkg = import_module(f"{__name__}.core")
        sys.modules.setdefault("core", core_pkg)

    return amemorix_pkg, core_pkg


def __getattr__(name: str):
    """按需暴露插件入口与兼容子包。

    Args:
        name: 访问的模块属性名。

    Returns:
        Any: 对应的插件对象或兼容包对象。

    Raises:
        AttributeError: 当请求的属性不存在时抛出。
    """
    if name == "plugin":
        _ensure_aliases()
        return import_module(f"{__name__}.plugin").plugin
    if name == "amemorix":
        return _ensure_aliases()[0]
    if name == "core":
        return _ensure_aliases()[1]
    raise AttributeError(name)


__all__ = ["plugin", "__version__", "__author__"]
