"""Entity-mode handlers for KnowledgeQueryTool."""

from __future__ import annotations

from typing import Any, Dict

async def query_entity(tool, entity_name: str) -> Dict[str, Any]:
    """查询实体信息

    Args:
        entity_name: 实体名称

    Returns:
        查询结果字典
    """
    if not entity_name:

        return {
            "success": False,
            "error": "实体名称不能为空",
            "content": "⚠️ 实体名称不能为空",
            "results": [],
        }

    # 检查实体是否存在
    if not tool.graph_store.has_node(entity_name):

        return {
            "success": False,
            "error": f"实体不存在: {entity_name}",
            "content": f"❌ 实体 '{entity_name}' 不存在",
            "results": [],
        }

    # 获取邻居节点
    neighbors = tool.graph_store.get_neighbors(entity_name)

    # 获取相关段落
    paragraphs = tool.metadata_store.get_paragraphs_by_entity(entity_name)

    # 格式化段落
    formatted_paragraphs = [
        {
            "hash": para["hash"],
            "content": para["content"],
            "created_at": para.get("created_at"),
        }
        for para in paragraphs
    ]


    # 生成 content 摘要
    content_lines = [f"实体 '{entity_name}' 信息："]
    content_lines.append(f"- 邻居节点 ({len(neighbors)}): {', '.join(neighbors[:10])}{'...' if len(neighbors)>10 else ''}")
    content_lines.append(f"- 相关段落 ({len(paragraphs)}):")
    for i, para in enumerate(formatted_paragraphs[:3]):
         content_lines.append(f"  {i+1}. {para['content'][:50]}...")
    
    content = "\n".join(content_lines)

    return {
        "success": True,
        "query_type": "entity",
        "entity": entity_name,
        "neighbors": neighbors,
        "related_paragraphs": formatted_paragraphs,
        "neighbor_count": len(neighbors),
        "paragraph_count": len(paragraphs),
        "content": content,
    }

