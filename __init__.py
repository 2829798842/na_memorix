"""提供 na_memorix 插件包的入口与兼容别名。"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

__version__ = "0.7.0"
__author__ = "KroMiose"
_PACKAGE_DIR = Path(__file__).resolve().parent


def _belongs_to_package(module: ModuleType | None) -> bool:
    """判断模块对象是否来自当前插件包目录。"""

    if module is None:
        return False
    file_path = getattr(module, "__file__", None)
    if not file_path:
        return False
    try:
        return Path(file_path).resolve().is_relative_to(_PACKAGE_DIR)
    except Exception:
        return False


def _purge_stale_modules() -> None:
    """清理当前插件包及其兼容别名的旧子模块缓存。

    Nekro 的插件热重载会卸载包根模块，但子模块往往仍保留在 ``sys.modules`` 中。
    如果这里不主动清理，下一次导入包时会继续复用旧的 ``plugin.py`` / ``server.py``，
    造成“重载后代码没生效”的现象。
    """

    package_prefix = f"{__name__}."
    stale_names = [name for name in list(sys.modules) if name.startswith(package_prefix)]

    for alias in ("amemorix", "core"):
        for name, module in list(sys.modules.items()):
            if (name == alias or name.startswith(f"{alias}.")) and _belongs_to_package(module):
                stale_names.append(name)

    for name in sorted(set(stale_names), reverse=True):
        sys.modules.pop(name, None)


def _ensure_aliases() -> tuple[ModuleType, ModuleType]:
    """初始化兼容导入别名。

    Returns:
        tuple[ModuleType, ModuleType]: ``amemorix`` 与 ``core`` 两个兼容包对象。
    """
    _purge_stale_modules()

    amemorix_pkg = sys.modules.get("amemorix")
    if amemorix_pkg is None:
        amemorix_pkg = import_module(f"{__name__}.amemorix")
        sys.modules.setdefault("amemorix", amemorix_pkg)

    core_pkg = sys.modules.get("core")
    if core_pkg is None:
        core_pkg = import_module(f"{__name__}.core")
        sys.modules.setdefault("core", core_pkg)

    return amemorix_pkg, core_pkg


# 显式绑定插件实例，避免包属性 ``plugin`` 被同名子模块覆盖成 module 对象。
_AMEMORIX_PKG, _CORE_PKG = _ensure_aliases()
plugin = import_module(f"{__name__}.plugin").plugin


def __getattr__(name: str) -> Any:
    """按需暴露插件入口与兼容子包。

    Args:
        name: 访问的模块属性名。

    Returns:
        Any: 对应的插件对象或兼容包对象。

    Raises:
        AttributeError: 当请求的属性不存在时抛出。
    """
    if name == "plugin":
        return plugin
    if name == "amemorix":
        return _AMEMORIX_PKG
    if name == "core":
        return _CORE_PKG
    raise AttributeError(name)


__all__ = ["plugin", "__version__", "__author__", "amemorix", "core"]
