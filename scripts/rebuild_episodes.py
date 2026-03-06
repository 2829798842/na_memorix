#!/usr/bin/env python3
"""
Episode source 级重建工具。

默认行为：
1. 将目标 source 入队到 episode_rebuild_sources
2. 不直接执行重建

可选行为：
1. `--wait` 时在脚本内串行执行重建
2. 处理完成后按 source 更新重建状态
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys
from typing import Any, Dict, List


CURRENT_DIR = Path(__file__).resolve().parent
PLUGIN_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = PLUGIN_ROOT.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild A_Memorix episodes by source")
    parser.add_argument("--data-dir", default=str(PLUGIN_ROOT / "data"), help="插件数据目录")
    parser.add_argument("--source", type=str, help="指定单个 source 入队/重建")
    parser.add_argument("--all", action="store_true", help="对所有 source 入队/重建")
    parser.add_argument("--wait", action="store_true", help="在脚本内同步执行重建")
    return parser


if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    _build_arg_parser().print_help()
    raise SystemExit(0)


try:
    import tomlkit  # type: ignore
except Exception:
    tomlkit = None

from plugins.A_memorix.core.storage import MetadataStore  # noqa: E402
from plugins.A_memorix.core.utils.episode_service import EpisodeService  # noqa: E402


def _load_plugin_config() -> Dict[str, Any]:
    config_path = PLUGIN_ROOT / "config.toml"
    if tomlkit is None or not config_path.exists():
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            parsed = tomlkit.load(f)
        return dict(parsed) if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _resolve_sources(store: MetadataStore, *, source: str | None, rebuild_all: bool) -> List[str]:
    if rebuild_all:
        return list(store.list_episode_sources_for_rebuild())
    token = str(source or "").strip()
    if not token:
        raise ValueError("必须提供 --source 或 --all")
    return [token]


async def _run_rebuilds(
    store: MetadataStore,
    plugin_config: Dict[str, Any],
    sources: List[str],
) -> int:
    service = EpisodeService(
        metadata_store=store,
        plugin_config=plugin_config,
    )
    status_rows = store.list_episode_source_rebuilds(limit=max(100, len(sources) * 4))
    requested_at_map = {
        str(row.get("source", "") or "").strip(): row.get("requested_at")
        for row in status_rows
    }

    failures: List[str] = []
    for source in sources:
        requested_at = requested_at_map.get(source)
        try:
            requested_at = float(requested_at) if requested_at is not None else None
        except Exception:
            requested_at = None

        started = store.mark_episode_source_running(source, requested_at=requested_at)
        if not started:
            started = store.mark_episode_source_running(source)
        if not started:
            failures.append(f"{source}: unable_to_mark_running")
            continue

        try:
            result = await service.rebuild_source(source)
            store.mark_episode_source_done(source, requested_at=requested_at)
            print(
                "rebuilt"
                f" source={source}"
                f" paragraphs={int(result.get('paragraph_count') or 0)}"
                f" groups={int(result.get('group_count') or 0)}"
                f" episodes={int(result.get('episode_count') or 0)}"
                f" fallback={int(result.get('fallback_count') or 0)}"
            )
        except Exception as e:
            err = str(e)[:500]
            store.mark_episode_source_failed(source, err, requested_at=requested_at)
            failures.append(f"{source}: {err}")
            print(f"failed source={source} error={err}")

    if failures:
        for item in failures:
            print(item)
        return 1
    return 0


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if bool(args.all) == bool(args.source):
        parser.error("必须且只能选择一个：--source 或 --all")

    store = MetadataStore(data_dir=Path(args.data_dir) / "metadata")
    store.connect()
    try:
        sources = _resolve_sources(store, source=args.source, rebuild_all=bool(args.all))
        if not sources:
            print("no sources to rebuild")
            return 0

        enqueued = 0
        reason = "script_rebuild_all" if args.all else "script_rebuild_source"
        for source in sources:
            enqueued += int(store.enqueue_episode_source_rebuild(source, reason=reason))
        print(f"enqueued={enqueued} sources={len(sources)}")

        if not args.wait:
            return 0

        plugin_config = _load_plugin_config()
        return asyncio.run(_run_rebuilds(store, plugin_config, sources))
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
