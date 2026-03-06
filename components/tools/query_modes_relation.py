"""Relation-mode handlers for KnowledgeQueryTool."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.common.logger import get_logger

from ...core import RetrievalStrategy
from ...core.utils.relation_query import parse_relation_query_spec

logger = get_logger("A_Memorix.QueryModesRelation")

_PATH_EDGE_BONUS_UNIT = 0.04
_PATH_EDGE_BONUS_CAP = 0.08


def _build_semantic_relation_content(results: Sequence[Dict[str, Any]], min_score: float) -> str:
    if results:
        lines = [f"找到 {len(results)} 条 [语义候选] 关系："]
        for i, rel in enumerate(results):
            bonus = float(rel.get("path_evidence_bonus", 0.0) or 0.0)
            score_text = f"{float(rel.get('similarity', 0.0) or 0.0):.2f}"
            if bonus > 0.0:
                score_text += f", path+{bonus:.2f}"
            lines.append(
                f"{i+1}. {rel['subject']} {rel['predicate']} {rel['object']} "
                f"(相似度: {score_text})"
            )

        lines.append("")
        lines.append("💡 若需精确过滤，请使用 'Subject|Predicate|Object' 格式")
        return "\n".join(lines)

    return (
        f"未找到相关的关系 (语义相似度均低于 {min_score})。\n"
        "💡 请尝试更具体的关系描述，或使用 'S|P|O' 格式进行精确查询。"
    )


def _extract_semantic_confidence(semantic_result: Dict[str, Any]) -> tuple[int, float]:
    hits_count = int(semantic_result.get("count", 0) or 0)
    max_score = 0.0
    if hits_count > 0 and semantic_result.get("results"):
        try:
            max_score = float(semantic_result["results"][0].get("similarity", 0.0) or 0.0)
        except Exception:
            max_score = 0.0
    return hits_count, max_score


def _build_path_edge_support(path_results: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str], int]:
    support: Dict[Tuple[str, str], int] = {}
    for item in path_results:
        nodes = item.get("nodes", []) if isinstance(item, dict) else []
        if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes)):
            continue
        clean_nodes = [str(node).strip() for node in nodes if str(node).strip()]
        if len(clean_nodes) < 2:
            continue
        for idx in range(len(clean_nodes) - 1):
            key = tuple(sorted((clean_nodes[idx].lower(), clean_nodes[idx + 1].lower())))
            support[key] = support.get(key, 0) + 1
    return support


def _rerank_semantic_results_with_path_evidence(
    semantic_results: Sequence[Dict[str, Any]],
    path_results: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    path_support = _build_path_edge_support(path_results)
    if not path_support:
        return [dict(item) for item in semantic_results]

    reranked: List[Tuple[float, float, int, Dict[str, Any]]] = []
    for index, item in enumerate(semantic_results):
        enriched = dict(item)
        raw_similarity = float(enriched.get("similarity", 0.0) or 0.0)
        subject = str(enriched.get("subject", "") or "").strip().lower()
        obj = str(enriched.get("object", "") or "").strip().lower()
        key = tuple(sorted((subject, obj))) if subject and obj else None
        bonus = 0.0
        if key and key in path_support:
            bonus = min(_PATH_EDGE_BONUS_CAP, _PATH_EDGE_BONUS_UNIT * float(path_support[key]))

        enriched["raw_similarity"] = raw_similarity
        enriched["path_evidence_bonus"] = bonus
        enriched["is_path_supported"] = bonus > 0.0
        enriched["similarity"] = raw_similarity + bonus
        reranked.append((float(enriched["similarity"]), bonus, -index, enriched))

    reranked.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return [item for _, _, _, item in reranked]


def _attach_path_evidence_if_needed(
    tool,
    *,
    query: str,
    semantic_result: Dict[str, Any],
    enable_path_search: bool,
    path_trigger_threshold: float,
) -> Dict[str, Any]:
    if not enable_path_search:
        return semantic_result
    if not semantic_result.get("success", False):
        return semantic_result

    hits_count, max_score = _extract_semantic_confidence(semantic_result)
    if hits_count > 0 and max_score >= path_trigger_threshold:
        return semantic_result

    if tool.debug_enabled:
        logger.info(
            f"{tool.log_prefix} 触发路径证据补充 (Hits={hits_count}, MaxScore={max_score:.2f})"
        )

    path_result = tool._path_search(query)
    if not path_result or not path_result.get("results"):
        return semantic_result

    enriched = dict(semantic_result)
    path_results = list(path_result.get("results", []) or [])
    reranked_results = _rerank_semantic_results_with_path_evidence(
        list(enriched.get("results", []) or []),
        path_results,
    )
    enriched["results"] = reranked_results
    enriched["count"] = len(reranked_results)
    enriched["path_evidence"] = path_results
    enriched["path_evidence_count"] = len(path_results)
    enriched["path_evidence_triggered"] = True

    content = _build_semantic_relation_content(
        reranked_results,
        min_score=float(tool.get_config("retrieval.relation_fallback_min_score", 0.3) or 0.3),
    ).rstrip()
    path_lines = ["", f"🧭 补充路径证据 ({len(path_results)} 条)："]
    for i, item in enumerate(path_results[:3], 1):
        desc = str(item.get("description", "") or "").strip()
        if desc:
            path_lines.append(f"{i}. {desc}")
    enriched["content"] = "\n".join([content] + path_lines).strip()
    return enriched

async def query_relation(tool, relation_spec: str) -> Dict[str, Any]:
    """查询关系信息

    Args:
        relation_spec: 关系规格

    Returns:
        查询结果字典
    """
    # 获取配置
    enable_fallback = tool.get_config("retrieval.relation_semantic_fallback", True)
    fallback_min_score = tool.get_config("retrieval.relation_fallback_min_score", 0.3)
    
    # Path Search 配置
    enable_path_search = tool.get_config("retrieval.relation_enable_path_search", True)
    path_trigger_threshold = tool.get_config("retrieval.relation_path_trigger_threshold", 0.4)

    # 1. 结构化检测
    parsed = parse_relation_query_spec(relation_spec)
    is_structured = parsed.is_structured

    # 2. 自然语言优先处理
    # 如果不是明确的结构化查询，且启用了回退（意味着支持语义模式），则直接使用语义检索
    if not is_structured and enable_fallback:
        semantic_result = await tool._semantic_search_relation(relation_spec, fallback_min_score)
        return _attach_path_evidence_if_needed(
            tool,
            query=relation_spec,
            semantic_result=semantic_result,
            enable_path_search=enable_path_search,
            path_trigger_threshold=path_trigger_threshold,
        )

    # 3. 结构化查询处理 (精确匹配)
    subject, predicate, obj = parsed.subject, parsed.predicate, parsed.object
    if not subject or not predicate:
        # 无法解析为结构化，且没走 NL 路径 (说明 enable_fallback=False)
        return {
            "success": False,
            "error": "关系格式错误 (请使用 S|P|O 或开启语义回退)",
            "content": "❌ 关系格式错误: 请使用 'Subject|Predicate|Object' 格式",
            "results": [],
        }

    # 执行精确查询
    relations = tool.metadata_store.get_relations(
        subject=subject if subject else None,
        predicate=predicate if predicate else None,
        object=obj if obj else None,
    )

    # 4. 结构化查询失败的回退
    # 如果精确匹配无结果，且启用了回退，尝试语义检索
    if not relations and enable_fallback:
         # 使用原始查询字符串进行语义检索
         semantic_result = await tool._semantic_search_relation(relation_spec, fallback_min_score)
         return _attach_path_evidence_if_needed(
             tool,
             query=relation_spec,
             semantic_result=semantic_result,
             enable_path_search=enable_path_search,
             path_trigger_threshold=path_trigger_threshold,
         )

    # 格式化精确匹配结果
    formatted_relations = []
    for rel in relations:
        formatted_relations.append({
            "hash": rel["hash"],
            "subject": rel["subject"],
            "predicate": rel["predicate"],
            "object": rel["object"],
            "confidence": rel.get("confidence", 1.0),
            "is_semantic": False,
        })

    # 生成 content 摘要
    if formatted_relations:
        lines = [f"找到 {len(formatted_relations)} 条精确匹配关系："]
        for i, rel in enumerate(formatted_relations[:10]):
            lines.append(f"{i+1}. {rel['subject']} {rel['predicate']} {rel['object']}")
        content = "\n".join(lines)
    else:
        content = "未找到符合条件的关系。"

    return {
        "success": True,
        "query_type": "relation",
        "spec": {"subject": subject, "predicate": predicate, "object": obj},
        "results": formatted_relations,
        "count": len(formatted_relations),
        "content": content,
    }

async def semantic_search_relation(
    tool,
    query: str,
    min_score: float,
) -> Dict[str, Any]:
    """执行语义关系检索

    Args:
        query: 查询文本
        min_score: 最小相似度阈值

    Returns:
        查询结果
    """
    if not tool.retriever:
         return {
            "success": False,
            "error": "检索器未初始化",
            "content": "❌ 检索器未初始化",
            "results": [],
        }

    # 执行检索 (策略: REL_ONLY, TopK: 5)
    # 护栏 B: TopK 小一点
    results = await tool.retriever.retrieve(
        query,
        top_k=5,
        strategy=RetrievalStrategy.REL_ONLY
    )

    formatted_results = []
    seen_relations = set()

    for res in results:
        # 护栏 B: 阈值过滤
        if res.score < min_score:
            continue
        
        # 护栏 D: 类型过滤 (retrieve REL_ONLY 应该只返回 relation，但防御性检查)
        if res.result_type != "relation":
            continue

        # 获取元数据
        meta = res.metadata
        subj = meta.get("subject", "?")
        pred = meta.get("predicate", "?")
        obj = meta.get("object", "?")
        
        # 护栏 D: 去重
        rel_key = (subj, pred, obj)
        if rel_key in seen_relations:
            continue
        seen_relations.add(rel_key)

        formatted_results.append({
            "hash": res.hash_value,
            "subject": subj,
            "predicate": pred,
            "object": obj,
            "confidence": meta.get("confidence", 1.0),
            "similarity": res.score,
            "is_semantic": True, # 标记为语义结果
        })

    content = _build_semantic_relation_content(formatted_results, min_score)

    return {
        "success": True,
        "query_type": "relation",
        "search_mode": "semantic",
        "query": query,
        "results": formatted_results,
        "count": len(formatted_results),
        "content": content,
    }

def path_search(tool, query: str) -> Optional[Dict[str, Any]]:
    """执行路径搜索 (多跳关系)"""
    # 1. 提取实体
    entities = tool._extract_entities_from_query(query)
    if len(entities) != 2:
        if tool.debug_enabled:
            logger.debug(f"{tool.log_prefix} PathSearch Abort: Requires exactly 2 entities, found {len(entities)}: {entities}")
        return None
        
    start_node, end_node = entities[0], entities[1]
    
    # 2. 查找路径
    paths = tool.graph_store.find_paths(
        start_node, 
        end_node, 
        max_depth=3, # Configurable?
        max_paths=5
    )
    
    if not paths:
        return None
        
    # 3. 丰富路径信息 (查找边上的关系谓语)
    formatted_paths = []
    edge_cache = {} # (u, v) -> relation_str
    
    for path_nodes in paths:
        path_desc = []
        valid_path = True
        
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i+1]
            
            # Check cache
            cache_key = tuple(sorted((u, v))) # Undirected cache key
            if cache_key in edge_cache:
                rel_info = edge_cache[cache_key]
            else:
                # Query metadata for relation u->v or v->u
                # 优先找 u->v
                rels = tool.metadata_store.get_relations(subject=u, object=v)
                direction = "->"
                if not rels:
                    # 尝试 v->u
                    rels = tool.metadata_store.get_relations(subject=v, object=u)
                    direction = "<-"
                
                if rels:
                    # Pick best confidence or first
                    best_rel = max(rels, key=lambda x: x.get("confidence", 1.0))
                    pred = best_rel.get("predicate", "related")
                    rel_info = (pred, direction, u, v) if direction == "->" else (pred, direction, v, u)
                else:
                    rel_info = ("?", "->", u, v) # Should not happen if graph consistent
                    
                edge_cache[cache_key] = rel_info
            
            pred, direction, src, tgt = rel_info
            if direction == "->":
                step_str = f"-[{pred}]->"
            else:
                step_str = f"<-[{pred}]-"
            path_desc.append(step_str)
        
        # Reconstruct full string: Node1 -[pred]-> Node2 ...
        full_path_str = path_nodes[0]
        for i, step in enumerate(path_desc):
            full_path_str += f" {step} {path_nodes[i+1]}"
        
        formatted_paths.append({
            "nodes": path_nodes,
            "description": full_path_str
        })

    # Generate content
    lines = [f"Found {len(formatted_paths)} indirect connection paths between '{start_node}' and '{end_node}':"]
    for i, p in enumerate(formatted_paths):
        lines.append(f"{i+1}. {p['description']}")
        
    content = "\n".join(lines)
    
    return {
        "success": True,
        "query_type": "relation",
        "search_mode": "path",
        "query": query,
        "results": formatted_paths,
        "count": len(formatted_paths),
        "content": content
    }

def extract_entities_from_query(tool, query: str) -> List[str]:
    """从查询中提取已知的图节点实体 (简易启发式)"""
    if not tool.graph_store:
        return []
        
    # 1. 简单的 N-gram 匹配 (N=1..4)
    tokens = query.replace("?", " ").replace("!", " ").replace(".", " ").split()
    found_entities = set()
    
    # 优化: 仅检查 query 中的 potential matches
    # 由于无法遍历所有 node，我们生成 query 的所有子串 check existence
    # O(L^2) where L is query length (small)
    
    n = len(tokens)
    # Max n-gram size: 4 or length of query
    max_n = min(4, n)
    
    # Greedy search: prioritize longer matches
    skip_indices = set()
    
    for size in range(max_n, 0, -1):
        for i in range(n - size + 1):
            # Check if this span is already covered
            if any(idx in skip_indices for idx in range(i, i+size)):
                continue
                
            span = " ".join(tokens[i : i+size])
            # Check original case first, then exact match only (kv store usually case sensitive-ish)
            # But user query might be lower/upper.
            # Use ignore_case=True to support "system" matches "System"
            matched_node = tool.graph_store.find_node(span, ignore_case=True)
            if matched_node:
                found_entities.add(matched_node)
                # Mark indices as covered
                for idx in range(i, i+size):
                    skip_indices.add(idx)
            else:
                pass
                
    return list(found_entities)

