"""运行时依赖检查与动态导入辅助。"""

from __future__ import annotations

import importlib
import threading
from dataclasses import dataclass
from typing import Any

from nekro_agent.api.plugin import dynamic_import_pkg
from amemorix.common.logging import get_logger

logger = get_logger("A_Memorix.RuntimeDeps")

_SCIPY_PACKAGE_SPEC = "scipy"
_SCIPY_IMPORT_NAME = "scipy"


@dataclass(frozen=True)
class RuntimeDependencyStatus:
    """描述单个运行时依赖的可用状态。"""

    name: str
    package_spec: str
    import_name: str
    available: bool
    installed_by_plugin: bool = False
    detail: str = ""

    def to_payload(self) -> dict[str, Any]:
        """转换为适合 API 返回的 payload。"""

        return {
            "name": self.name,
            "package_spec": self.package_spec,
            "import_name": self.import_name,
            "available": self.available,
            "installed_by_plugin": self.installed_by_plugin,
            "detail": self.detail,
        }


_dependency_lock = threading.Lock()
_scipy_status_cache: RuntimeDependencyStatus | None = None


def _import_available(import_name: str) -> bool:
    """检查模块是否已可导入。"""

    try:
        importlib.import_module(import_name)
    except ImportError:
        return False
    return True


def ensure_scipy() -> RuntimeDependencyStatus:
    """确保 SciPy 可用，必要时触发 Nekro 的动态依赖导入。"""

    global _scipy_status_cache

    with _dependency_lock:
        if _scipy_status_cache is not None:
            return _scipy_status_cache

        if _import_available(_SCIPY_IMPORT_NAME):
            _scipy_status_cache = RuntimeDependencyStatus(
                name="SciPy",
                package_spec=_SCIPY_PACKAGE_SPEC,
                import_name=_SCIPY_IMPORT_NAME,
                available=True,
                installed_by_plugin=False,
                detail="SciPy 已就绪。",
            )
            return _scipy_status_cache

        try:
            

            dynamic_import_pkg(_SCIPY_PACKAGE_SPEC, _SCIPY_IMPORT_NAME)
        except Exception as exc:
            detail = (
                "SciPy 运行依赖不可用，插件无法初始化图存储。"
                f" 已尝试通过 dynamic_import_pkg 自动安装，但失败：{exc}"
            )
            logger.warning(detail)
            _scipy_status_cache = RuntimeDependencyStatus(
                name="SciPy",
                package_spec=_SCIPY_PACKAGE_SPEC,
                import_name=_SCIPY_IMPORT_NAME,
                available=False,
                installed_by_plugin=False,
                detail=detail,
            )
            return _scipy_status_cache

        if _import_available(_SCIPY_IMPORT_NAME):
            detail = "SciPy 缺失，已通过 dynamic_import_pkg 自动安装。"
            logger.info(detail)
            _scipy_status_cache = RuntimeDependencyStatus(
                name="SciPy",
                package_spec=_SCIPY_PACKAGE_SPEC,
                import_name=_SCIPY_IMPORT_NAME,
                available=True,
                installed_by_plugin=True,
                detail=detail,
            )
            return _scipy_status_cache

        detail = "SciPy 已尝试动态安装，但仍无法导入。"
        logger.warning(detail)
        _scipy_status_cache = RuntimeDependencyStatus(
            name="SciPy",
            package_spec=_SCIPY_PACKAGE_SPEC,
            import_name=_SCIPY_IMPORT_NAME,
            available=False,
            installed_by_plugin=False,
            detail=detail,
        )
        return _scipy_status_cache


def get_runtime_dependency_report() -> dict[str, Any]:
    """返回运行时依赖整体状态。"""

    items = [ensure_scipy().to_payload()]
    missing = [item["name"] for item in items if not item["available"]]
    detail = "；".join(item["detail"] for item in items if not item["available"])
    return {
        "ready": not missing,
        "items": items,
        "missing": missing,
        "detail": detail,
    }
