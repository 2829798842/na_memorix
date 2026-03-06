"""Runtime coordination utilities for A_Memorix."""

from .search_runtime_initializer import (
    SearchRuntimeBundle,
    SearchRuntimeInitializer,
    build_search_runtime,
)
from .request_router import RequestRouter
from .lifecycle_orchestrator import (
    cancel_background_tasks,
    ensure_initialized,
    initialize_storage_async,
    start_background_tasks,
)

__all__ = [
    "SearchRuntimeBundle",
    "SearchRuntimeInitializer",
    "build_search_runtime",
    "RequestRouter",
    "ensure_initialized",
    "initialize_storage_async",
    "start_background_tasks",
    "cancel_background_tasks",
]
