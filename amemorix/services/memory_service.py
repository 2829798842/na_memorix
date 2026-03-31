"""Memory maintenance API service."""

import datetime
from typing import Any, Dict, List

from core.utils.search_execution_service import SearchExecutionRequest, SearchExecutionService

from amemorix.context import AppContext


class MemoryService:
    """Serve memory maintenance APIs on top of the runtime context."""

    def __init__(self, ctx: AppContext):
        self.ctx = ctx

    async def status(self) -> Dict[str, Any]:
        """Return current memory maintenance counters."""

        def _collect() -> Dict[str, Any]:
            cursor = self.ctx.metadata_store._conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM relations WHERE is_inactive = 0")
            active_count = int(cursor.fetchone()[0] or 0)
            cursor.execute("SELECT COUNT(*) FROM relations WHERE is_inactive = 1")
            inactive_count = int(cursor.fetchone()[0] or 0)
            cursor.execute("SELECT COUNT(*) FROM deleted_relations")
            deleted_count = int(cursor.fetchone()[0] or 0)
            now = datetime.datetime.now().timestamp()
            cursor.execute("SELECT COUNT(*) FROM relations WHERE is_pinned = 1")
            pinned_count = int(cursor.fetchone()[0] or 0)
            cursor.execute("SELECT COUNT(*) FROM relations WHERE protected_until > ?", (now,))
            ttl_count = int(cursor.fetchone()[0] or 0)
            return {
                "active_relations": active_count,
                "inactive_relations": inactive_count,
                "recycle_bin_relations": deleted_count,
                "pinned_relations": pinned_count,
                "ttl_protected_relations": ttl_count,
                "config": self.ctx.get_config("memory", {}),
            }

        return await self.ctx.run_blocking(_collect)

    async def protect(self, query_or_hash: str, hours: float = 24.0) -> Dict[str, Any]:
        """Protect one or more relations by pinning or TTL."""

        hashes = await self._resolve_relations(query_or_hash)
        if not hashes:
            return {"success": False, "message": "relation not found"}

        def _protect() -> Dict[str, Any]:
            now = datetime.datetime.now().timestamp()
            if float(hours) <= 0:
                self.ctx.metadata_store.update_relations_protection(hashes, is_pinned=True)
                mode = "pin"
                until = None
            else:
                until = now + float(hours) * 3600
                self.ctx.metadata_store.update_relations_protection(hashes, protected_until=until, is_pinned=False)
                mode = "ttl"
            return {"success": True, "mode": mode, "count": len(hashes), "protected_until": until}

        return await self.ctx.run_blocking(_protect)

    async def reinforce(self, query_or_hash: str) -> Dict[str, Any]:
        """Reinforce one or more relations and revive inactive edges."""

        hashes = await self._resolve_relations(query_or_hash)
        if not hashes:
            return {"success": False, "message": "relation not found"}

        def _reinforce() -> Dict[str, Any]:
            status_map = self.ctx.metadata_store.get_relation_status_batch(hashes)
            cursor = self.ctx.metadata_store._conn.cursor()
            placeholders = ",".join(["?"] * len(hashes))
            cursor.execute(
                f"SELECT hash, subject, object FROM relations WHERE hash IN ({placeholders})",
                hashes,
            )
            revived = 0
            now = datetime.datetime.now().timestamp()
            for row in cursor.fetchall():
                h, subject, obj = row
                status = status_map.get(h)
                if status and status.get("is_inactive"):
                    revived += 1
                self.ctx.graph_store.update_edge_weight(
                    str(subject),
                    str(obj),
                    1.0,
                    max_weight=float(self.ctx.get_config("memory.max_weight", 10.0)),
                )

            self.ctx.metadata_store.reinforce_relations(hashes)
            self.ctx.metadata_store.update_relations_protection(
                hashes,
                last_reinforced=now,
                protected_until=now + float(self.ctx.get_config("memory.auto_protect_ttl_hours", 24.0)) * 3600,
            )
            self.ctx.graph_store.save()
            return {"success": True, "count": len(hashes), "revived": revived}

        return await self.ctx.run_blocking(_reinforce)

    async def restore(self, hash_value: str, restore_type: str = "relation") -> Dict[str, Any]:
        """Restore a deleted entity or relation."""

        relation_hash = str(hash_value or "").strip().lower()
        if not relation_hash:
            raise ValueError("hash is empty")

        def _restore() -> Dict[str, Any]:
            if restore_type == "entity":
                cursor = self.ctx.metadata_store._conn.cursor()
                cursor.execute(
                    "UPDATE entities SET is_deleted=0, deleted_at=NULL WHERE hash=?",
                    (relation_hash,),
                )
                self.ctx.metadata_store._conn.commit()
                return {"success": True, "type": "entity", "hash": relation_hash}

            record = self.ctx.metadata_store.restore_relation(relation_hash)
            if not record:
                return {"success": False, "message": "relation not found in recycle bin"}

            subject = str(record["subject"])
            obj = str(record["object"])
            self.ctx.metadata_store.revive_entities_by_names([subject, obj])
            self.ctx.graph_store.add_nodes([subject, obj])
            self.ctx.graph_store.add_edges(
                [(subject, obj)],
                weights=[float(record.get("confidence", 1.0) or 1.0)],
                relation_hashes=[relation_hash],
            )
            self.ctx.graph_store.save()
            return {"success": True, "type": "relation", "hash": relation_hash}

        return await self.ctx.run_blocking(_restore)

    async def _resolve_relations(self, query: str) -> List[str]:
        """Resolve a free-form relation query into relation hashes."""

        value = str(query or "").strip()
        if not value:
            return []

        if len(value) in {32, 64} and all(char in "0123456789abcdefABCDEF" for char in value):
            normalized = value.lower()
            if len(normalized) == 64:
                status_map = await self.ctx.run_blocking(self.ctx.metadata_store.get_relation_status_batch, [normalized])
                if status_map:
                    return [normalized]

            def _lookup_prefix() -> List[str]:
                cursor = self.ctx.metadata_store._conn.cursor()
                cursor.execute("SELECT hash FROM relations WHERE hash LIKE ? LIMIT 5", (f"{normalized}%",))
                return [str(row[0]) for row in cursor.fetchall()]

            hits = await self.ctx.run_blocking(_lookup_prefix)
            if hits:
                return hits

        search = await SearchExecutionService.execute(
            retriever=self.ctx.retriever,
            threshold_filter=None,
            plugin_config={
                **self.ctx.config,
                "plugin_instance": self.ctx,
                "graph_store": self.ctx.graph_store,
                "metadata_store": self.ctx.metadata_store,
            },
            request=SearchExecutionRequest(
                caller="v1.memory.resolve",
                query_type="search",
                query=value,
                top_k=10,
                use_threshold=False,
                enable_ppr=True,
            ),
            enforce_chat_filter=False,
            reinforce_access=False,
        )
        if search.success:
            hashes = [item.hash_value for item in search.results if getattr(item, "result_type", "") == "relation"]
            if hashes:
                return hashes[:5]

        def _lookup_by_name() -> List[str]:
            cursor = self.ctx.metadata_store._conn.cursor()
            cursor.execute(
                "SELECT hash FROM relations WHERE subject LIKE ? OR object LIKE ? LIMIT 5",
                (f"%{value}%", f"%{value}%"),
            )
            return [str(row[0]) for row in cursor.fetchall()]

        return await self.ctx.run_blocking(_lookup_by_name)
