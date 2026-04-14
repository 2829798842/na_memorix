"""宿主检索调优后端的纯计算逻辑。"""

import copy
import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

DEFAULT_OBJECTIVE = "balanced"
DEFAULT_INTENSITY = "standard"
DEFAULT_SAMPLE_SIZE = 24
DEFAULT_TOP_K_EVAL = 20
DEFAULT_POLL_INTERVAL_MS = 1200

OBJECTIVES = {"precision_priority", "balanced", "recall_priority"}
INTENSITIES = {"quick", "standard", "deep"}
SPARSE_MODES = {"auto", "fallback_only", "hybrid"}
TOKENIZER_MODES = {"jieba", "mixed", "char_2gram"}
FUSION_METHODS = {"weighted_rrf", "alpha_legacy"}
NORMALIZE_METHODS = {"minmax"}
RETRIEVAL_PROFILE_KEYS = {
    "top_k_paragraphs",
    "top_k_relations",
    "top_k_final",
    "alpha",
    "enable_ppr",
    "enable_parallel",
    "sparse",
    "fusion",
}
SPARSE_PROFILE_KEYS = {
    "enabled",
    "mode",
    "tokenizer_mode",
    "char_ngram_n",
    "candidate_k",
    "relation_candidate_k",
    "enable_relation_sparse_fallback",
    "enable_ngram_fallback_index",
    "enable_like_fallback",
}
FUSION_PROFILE_KEYS = {
    "method",
    "rrf_k",
    "vector_weight",
    "bm25_weight",
    "normalize_score",
    "normalize_method",
}

DEFAULT_TUNING_PROFILE: dict[str, Any] = {
    "retrieval": {
        "top_k_paragraphs": 20,
        "top_k_relations": 10,
        "top_k_final": 10,
        "alpha": 0.5,
        "enable_ppr": True,
        "enable_parallel": True,
        "sparse": {
            "enabled": True,
            "mode": "auto",
            "tokenizer_mode": "jieba",
            "char_ngram_n": 2,
            "candidate_k": 80,
            "relation_candidate_k": 60,
            "enable_relation_sparse_fallback": True,
            "enable_ngram_fallback_index": True,
            "enable_like_fallback": False,
        },
        "fusion": {
            "method": "weighted_rrf",
            "rrf_k": 60,
            "vector_weight": 0.7,
            "bm25_weight": 0.3,
            "normalize_score": True,
            "normalize_method": "minmax",
        },
    }
}


@dataclass(frozen=True)
class EvaluationCase:
    """一条调优评估样本。"""

    case_id: str
    category: str
    query: str
    expected_hash: str
    expected_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _json_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def deep_merge_dict(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """深度合并字典并返回新对象。"""

    merged = copy.deepcopy(base)
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _expect_dict(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} 必须是对象")
    return value


def _to_int(value: Any, path: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} 必须是整数") from exc
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{path} 必须在 [{minimum}, {maximum}] 之间")
    return parsed


def _to_float(value: Any, path: str, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} 必须是数字") from exc
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{path} 必须在 [{minimum}, {maximum}] 之间")
    return parsed


def _to_bool(value: Any, path: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{path} 必须是布尔值")


def _to_choice(value: Any, path: str, choices: set[str]) -> str:
    parsed = str(value or "").strip().lower()
    if parsed not in choices:
        expected = ", ".join(sorted(choices))
        raise ValueError(f"{path} 必须是以下之一: {expected}")
    return parsed


def _filter_supported_profile_keys(raw: Any, allowed_keys: set[str]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return {key: copy.deepcopy(value) for key, value in raw.items() if key in allowed_keys}


def _normalize_sparse_profile(raw: Any, *, partial: bool) -> dict[str, Any]:
    payload = {} if partial else copy.deepcopy(DEFAULT_TUNING_PROFILE["retrieval"]["sparse"])
    data = _expect_dict(raw, "retrieval.sparse")
    for key, value in data.items():
        if key == "enabled":
            payload[key] = _to_bool(value, "retrieval.sparse.enabled")
        elif key == "mode":
            payload[key] = _to_choice(value, "retrieval.sparse.mode", SPARSE_MODES)
        elif key == "tokenizer_mode":
            payload[key] = _to_choice(value, "retrieval.sparse.tokenizer_mode", TOKENIZER_MODES)
        elif key == "char_ngram_n":
            payload[key] = _to_int(value, "retrieval.sparse.char_ngram_n", 1, 8)
        elif key == "candidate_k":
            payload[key] = _to_int(value, "retrieval.sparse.candidate_k", 1, 400)
        elif key == "relation_candidate_k":
            payload[key] = _to_int(value, "retrieval.sparse.relation_candidate_k", 1, 400)
        elif key == "enable_relation_sparse_fallback":
            payload[key] = _to_bool(value, "retrieval.sparse.enable_relation_sparse_fallback")
        elif key == "enable_ngram_fallback_index":
            payload[key] = _to_bool(value, "retrieval.sparse.enable_ngram_fallback_index")
        elif key == "enable_like_fallback":
            payload[key] = _to_bool(value, "retrieval.sparse.enable_like_fallback")
        else:
            raise ValueError(f"未知调优字段: retrieval.sparse.{key}")
    return payload


def _normalize_fusion_profile(raw: Any, *, partial: bool) -> dict[str, Any]:
    payload = {} if partial else copy.deepcopy(DEFAULT_TUNING_PROFILE["retrieval"]["fusion"])
    data = _expect_dict(raw, "retrieval.fusion")
    for key, value in data.items():
        if key == "method":
            payload[key] = _to_choice(value, "retrieval.fusion.method", FUSION_METHODS)
        elif key == "rrf_k":
            payload[key] = _to_int(value, "retrieval.fusion.rrf_k", 1, 200)
        elif key == "vector_weight":
            payload[key] = _to_float(value, "retrieval.fusion.vector_weight", 0.0, 10.0)
        elif key == "bm25_weight":
            payload[key] = _to_float(value, "retrieval.fusion.bm25_weight", 0.0, 10.0)
        elif key == "normalize_score":
            payload[key] = _to_bool(value, "retrieval.fusion.normalize_score")
        elif key == "normalize_method":
            payload[key] = _to_choice(value, "retrieval.fusion.normalize_method", NORMALIZE_METHODS)
        else:
            raise ValueError(f"未知调优字段: retrieval.fusion.{key}")
    if not partial:
        vector_weight = float(payload.get("vector_weight", 0.0))
        bm25_weight = float(payload.get("bm25_weight", 0.0))
        if vector_weight <= 0 and bm25_weight <= 0:
            payload["vector_weight"] = 0.7
            payload["bm25_weight"] = 0.3
        else:
            total = vector_weight + bm25_weight
            payload["vector_weight"] = round(vector_weight / total, 6)
            payload["bm25_weight"] = round(bm25_weight / total, 6)
    return payload


def _normalize_retrieval_profile(raw: Any, *, partial: bool) -> dict[str, Any]:
    payload = {} if partial else copy.deepcopy(DEFAULT_TUNING_PROFILE["retrieval"])
    data = _expect_dict(raw, "retrieval")
    for key, value in data.items():
        if key == "top_k_paragraphs":
            payload[key] = _to_int(value, "retrieval.top_k_paragraphs", 1, 200)
        elif key == "top_k_relations":
            payload[key] = _to_int(value, "retrieval.top_k_relations", 1, 200)
        elif key == "top_k_final":
            payload[key] = _to_int(value, "retrieval.top_k_final", 1, 100)
        elif key == "alpha":
            payload[key] = round(_to_float(value, "retrieval.alpha", 0.0, 1.0), 6)
        elif key == "enable_ppr":
            payload[key] = _to_bool(value, "retrieval.enable_ppr")
        elif key == "enable_parallel":
            payload[key] = _to_bool(value, "retrieval.enable_parallel")
        elif key == "sparse":
            payload[key] = _normalize_sparse_profile(value, partial=partial)
        elif key == "fusion":
            payload[key] = _normalize_fusion_profile(value, partial=partial)
        else:
            raise ValueError(f"未知调优字段: retrieval.{key}")
    return payload


def normalize_tuning_profile_patch(raw: Any) -> dict[str, Any]:
    """校验并标准化手动应用的局部调优 patch。"""

    data = _expect_dict(raw, "profile")
    if not data:
        return {}

    if "retrieval" in data:
        extra_keys = [key for key in data if key != "retrieval"]
        if extra_keys:
            joined = ", ".join(sorted(extra_keys))
            raise ValueError(f"profile 仅支持 retrieval 顶层对象，发现多余字段: {joined}")
        return {"retrieval": _normalize_retrieval_profile(data.get("retrieval"), partial=True)}

    return {"retrieval": _normalize_retrieval_profile(data, partial=True)}


def extract_tuning_profile(config: dict[str, Any]) -> dict[str, Any]:
    """从完整运行时配置中截出调优页关心的 profile。"""

    retrieval = {}
    if isinstance(config, dict):
        retrieval = config.get("retrieval", {}) or {}
    retrieval_subset = _filter_supported_profile_keys(retrieval, RETRIEVAL_PROFILE_KEYS)
    if "sparse" in retrieval_subset:
        retrieval_subset["sparse"] = _filter_supported_profile_keys(
            retrieval_subset.get("sparse"),
            SPARSE_PROFILE_KEYS,
        )
    if "fusion" in retrieval_subset:
        retrieval_subset["fusion"] = _filter_supported_profile_keys(
            retrieval_subset.get("fusion"),
            FUSION_PROFILE_KEYS,
        )
    return {"retrieval": _normalize_retrieval_profile(retrieval_subset, partial=False)}


def merge_tuning_profile(base_profile: dict[str, Any], patch_profile: dict[str, Any]) -> dict[str, Any]:
    """合并 profile 并重新走一遍标准化，保证结构稳定。"""

    merged = deep_merge_dict(extract_tuning_profile(base_profile), patch_profile or {})
    return extract_tuning_profile(merged)


def resolve_objective(value: Any) -> str:
    raw = str(value or DEFAULT_OBJECTIVE).strip().lower()
    return raw if raw in OBJECTIVES else DEFAULT_OBJECTIVE


def resolve_intensity(value: Any) -> str:
    raw = str(value or DEFAULT_INTENSITY).strip().lower()
    return raw if raw in INTENSITIES else DEFAULT_INTENSITY


def resolve_round_count(intensity: str, requested_rounds: Any = None) -> int:
    if requested_rounds is not None:
        return max(1, min(48, int(requested_rounds)))
    defaults = {
        "quick": 4,
        "standard": 6,
        "deep": 10,
    }
    return defaults.get(resolve_intensity(intensity), defaults[DEFAULT_INTENSITY])


def _clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


def _clamp_float(value: float, minimum: float, maximum: float) -> float:
    return round(max(minimum, min(maximum, float(value))), 6)


def _candidate_label(prefix: str, index: int) -> str:
    return f"{prefix}_{index:02d}"


def generate_candidate_profiles(
    base_profile: dict[str, Any],
    objective: str,
    intensity: str,
    requested_rounds: Any = None,
) -> list[dict[str, Any]]:
    """围绕当前 profile 生成一组候选调优轮次。"""

    base = extract_tuning_profile(base_profile)
    retrieval = base["retrieval"]
    target_rounds = resolve_round_count(intensity, requested_rounds)
    objective_name = resolve_objective(objective)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(label: str, patch: dict[str, Any]) -> None:
        merged = merge_tuning_profile(base, patch)
        item = {
            "label": label,
            "profile": merged,
        }
        encoded = _json_key(item["profile"])
        if encoded in seen:
            return
        seen.add(encoded)
        candidates.append(item)

    add("baseline", {})

    if objective_name == "precision_priority":
        add(
            "precision_anchor",
            {
                "retrieval": {
                    "alpha": _clamp_float(float(retrieval["alpha"]) + 0.12, 0.05, 0.95),
                    "top_k_final": _clamp_int(int(retrieval["top_k_final"]) - 2, 3, 100),
                    "enable_ppr": True,
                    "sparse": {
                        "enabled": False,
                    },
                },
            },
        )
    elif objective_name == "recall_priority":
        add(
            "recall_anchor",
            {
                "retrieval": {
                    "alpha": _clamp_float(float(retrieval["alpha"]) - 0.12, 0.05, 0.95),
                    "top_k_paragraphs": _clamp_int(int(retrieval["top_k_paragraphs"]) + 12, 1, 200),
                    "top_k_relations": _clamp_int(int(retrieval["top_k_relations"]) + 8, 1, 200),
                    "top_k_final": _clamp_int(int(retrieval["top_k_final"]) + 4, 1, 100),
                    "sparse": {
                        "enabled": True,
                        "candidate_k": _clamp_int(
                            int(retrieval["sparse"]["candidate_k"]) + 40,
                            1,
                            400,
                        ),
                        "relation_candidate_k": _clamp_int(
                            int(retrieval["sparse"]["relation_candidate_k"]) + 30,
                            1,
                            400,
                        ),
                    },
                },
            },
        )
    else:
        add(
            "balanced_anchor",
            {
                "retrieval": {
                    "top_k_paragraphs": _clamp_int(int(retrieval["top_k_paragraphs"]) + 6, 1, 200),
                    "top_k_relations": _clamp_int(int(retrieval["top_k_relations"]) + 4, 1, 200),
                    "top_k_final": _clamp_int(int(retrieval["top_k_final"]) + 2, 1, 100),
                    "sparse": {
                        "enabled": True,
                    },
                },
            },
        )

    add(
        "alpha_up",
        {"retrieval": {"alpha": _clamp_float(float(retrieval["alpha"]) + 0.15, 0.05, 0.95)}},
    )
    add(
        "alpha_down",
        {"retrieval": {"alpha": _clamp_float(float(retrieval["alpha"]) - 0.15, 0.05, 0.95)}},
    )
    add(
        "paragraph_bias",
        {"retrieval": {"top_k_paragraphs": _clamp_int(int(retrieval["top_k_paragraphs"]) + 10, 1, 200)}},
    )
    add(
        "relation_bias",
        {"retrieval": {"top_k_relations": _clamp_int(int(retrieval["top_k_relations"]) + 8, 1, 200)}},
    )
    add(
        "final_up",
        {"retrieval": {"top_k_final": _clamp_int(int(retrieval["top_k_final"]) + 4, 1, 100)}},
    )
    add(
        "final_down",
        {"retrieval": {"top_k_final": _clamp_int(int(retrieval["top_k_final"]) - 3, 1, 100)}},
    )
    add(
        "ppr_off",
        {"retrieval": {"enable_ppr": False}},
    )
    add(
        "dense_rrf",
        {
            "retrieval": {
                "fusion": {
                    "method": "weighted_rrf",
                    "vector_weight": 0.82,
                    "bm25_weight": 0.18,
                },
            },
        },
    )
    add(
        "sparse_rrf",
        {
            "retrieval": {
                "sparse": {
                    "enabled": True,
                    "candidate_k": _clamp_int(int(retrieval["sparse"]["candidate_k"]) + 30, 1, 400),
                    "relation_candidate_k": _clamp_int(
                        int(retrieval["sparse"]["relation_candidate_k"]) + 20,
                        1,
                        400,
                    ),
                },
                "fusion": {
                    "method": "weighted_rrf",
                    "vector_weight": 0.58,
                    "bm25_weight": 0.42,
                },
            },
        },
    )
    add(
        "parallel_off",
        {"retrieval": {"enable_parallel": False}},
    )
    add(
        "sparse_off",
        {"retrieval": {"sparse": {"enabled": False}}},
    )

    sweep_seed = random.Random(_json_key(base))
    alpha_offsets = [-0.24, -0.1, 0.1, 0.24]
    para_offsets = [-8, 8, 16]
    rel_offsets = [-6, 6, 12]
    final_offsets = [-2, 2, 6]
    sweep_index = 1

    while len(candidates) < target_rounds:
        alpha_offset = alpha_offsets[(sweep_index - 1) % len(alpha_offsets)]
        para_offset = para_offsets[(sweep_index - 1) % len(para_offsets)]
        rel_offset = rel_offsets[(sweep_index - 1) % len(rel_offsets)]
        final_offset = final_offsets[(sweep_index - 1) % len(final_offsets)]
        candidate_k_delta = sweep_seed.choice([0, 20, 40])
        relation_candidate_delta = sweep_seed.choice([0, 15, 30])
        add(
            _candidate_label("sweep", sweep_index),
            {
                "retrieval": {
                    "alpha": _clamp_float(float(retrieval["alpha"]) + alpha_offset, 0.05, 0.95),
                    "top_k_paragraphs": _clamp_int(int(retrieval["top_k_paragraphs"]) + para_offset, 1, 200),
                    "top_k_relations": _clamp_int(int(retrieval["top_k_relations"]) + rel_offset, 1, 200),
                    "top_k_final": _clamp_int(int(retrieval["top_k_final"]) + final_offset, 1, 100),
                    "sparse": {
                        "enabled": True,
                        "candidate_k": _clamp_int(
                            int(retrieval["sparse"]["candidate_k"]) + candidate_k_delta,
                            1,
                            400,
                        ),
                        "relation_candidate_k": _clamp_int(
                            int(retrieval["sparse"]["relation_candidate_k"]) + relation_candidate_delta,
                            1,
                            400,
                        ),
                    },
                },
            },
        )
        sweep_index += 1

    return candidates[:target_rounds]


def _hit_rank(case: EvaluationCase, ranked_items: Sequence[dict[str, Any]], top_k: int) -> int | None:
    for idx, item in enumerate(list(ranked_items)[: max(1, int(top_k))], start=1):
        if str(item.get("hash", "")) != case.expected_hash:
            continue
        if str(item.get("type", "")) != case.expected_type:
            continue
        return idx
    return None


def calculate_metrics(
    cases: Sequence[EvaluationCase],
    ranked_items_by_case: dict[str, list[dict[str, Any]]],
    *,
    top_k: int,
) -> dict[str, Any]:
    """根据评估样本与检索结果计算调优指标。"""

    if not cases:
        raise ValueError("评估样本为空，无法计算指标")

    total = len(cases)
    precision_at_1 = 0.0
    precision_at_3 = 0.0
    recall_at_k = 0.0
    mrr = 0.0
    empty_rate = 0.0
    relation_total = 0
    relation_hits = 0
    category: dict[str, dict[str, int]] = {}

    for case in cases:
        ranked_items = list(ranked_items_by_case.get(case.case_id, []))
        if not ranked_items:
            empty_rate += 1.0
        rank = _hit_rank(case, ranked_items, max(1, int(top_k)))
        if rank == 1:
            precision_at_1 += 1.0
        if rank is not None and rank <= 3:
            precision_at_3 += 1.0
        if rank is not None:
            recall_at_k += 1.0
            mrr += 1.0 / float(rank)

        bucket = category.setdefault(case.category, {"total": 0, "hit": 0, "hit_at_1": 0})
        bucket["total"] += 1
        if rank is not None:
            bucket["hit"] += 1
        if rank == 1:
            bucket["hit_at_1"] += 1

        if case.expected_type == "relation":
            relation_total += 1
            if rank is not None:
                relation_hits += 1

    metrics = {
        "cases_total": total,
        "precision_at_1": round(precision_at_1 / total, 6),
        "precision_at_3": round(precision_at_3 / total, 6),
        "recall_at_k": round(recall_at_k / total, 6),
        "mrr": round(mrr / total, 6),
        "spo_relation_hit_rate": round(
            relation_hits / relation_total if relation_total > 0 else 0.0,
            6,
        ),
        "empty_rate": round(empty_rate / total, 6),
        "category": category,
    }
    return metrics


def score_round_metrics(metrics: dict[str, Any], objective: str) -> float:
    """按目标函数把各项指标压成单个 round score。"""

    objective_name = resolve_objective(objective)
    if objective_name == "precision_priority":
        weights = {
            "precision_at_1": 0.42,
            "precision_at_3": 0.22,
            "mrr": 0.2,
            "recall_at_k": 0.06,
            "spo_relation_hit_rate": 0.1,
            "empty_rate": -0.08,
        }
    elif objective_name == "recall_priority":
        weights = {
            "precision_at_1": 0.12,
            "precision_at_3": 0.18,
            "mrr": 0.15,
            "recall_at_k": 0.36,
            "spo_relation_hit_rate": 0.12,
            "empty_rate": -0.08,
        }
    else:
        weights = {
            "precision_at_1": 0.24,
            "precision_at_3": 0.18,
            "mrr": 0.18,
            "recall_at_k": 0.22,
            "spo_relation_hit_rate": 0.1,
            "empty_rate": -0.08,
        }

    score = 0.0
    for key, weight in weights.items():
        score += float(metrics.get(key, 0.0)) * weight
    return round(score, 6)


def _fmt_pct(value: Any) -> str:
    return f"{float(value or 0.0) * 100:.2f}%"


def render_markdown_report(task: dict[str, Any]) -> str:
    """渲染调优任务报告。"""

    baseline = dict(task.get("baseline_metrics") or {})
    best = dict(task.get("best_metrics") or {})
    rounds = list(task.get("rounds") or [])
    lines = [
        "# 检索调优报告",
        "",
        f"- 任务 ID: `{task.get('task_id', '')}`",
        f"- 目标函数: `{task.get('objective', DEFAULT_OBJECTIVE)}`",
        f"- 强度: `{task.get('intensity', DEFAULT_INTENSITY)}`",
        f"- 样本数: `{task.get('sample_size', DEFAULT_SAMPLE_SIZE)}`",
        f"- 评估 Top-K: `{task.get('top_k_eval', DEFAULT_TOP_K_EVAL)}`",
        f"- 实际轮次: `{task.get('rounds_done', 0)}/{task.get('rounds_total', 0)}`",
        "",
        "## 基线 vs 最优",
        "",
        "| 指标 | Baseline | Best |",
        "| --- | --- | --- |",
        f"| Precision@1 | {_fmt_pct(baseline.get('precision_at_1'))} | {_fmt_pct(best.get('precision_at_1'))} |",
        f"| Precision@3 | {_fmt_pct(baseline.get('precision_at_3'))} | {_fmt_pct(best.get('precision_at_3'))} |",
        f"| Recall@K | {_fmt_pct(baseline.get('recall_at_k'))} | {_fmt_pct(best.get('recall_at_k'))} |",
        f"| MRR | {float(baseline.get('mrr', 0.0)):.4f} | {float(best.get('mrr', 0.0)):.4f} |",
        f"| SPO Relation Hit | {_fmt_pct(baseline.get('spo_relation_hit_rate'))} | {_fmt_pct(best.get('spo_relation_hit_rate'))} |",
        f"| Empty Rate | {_fmt_pct(baseline.get('empty_rate'))} | {_fmt_pct(best.get('empty_rate'))} |",
        "",
        "## 最优参数",
        "",
        "```json",
        json.dumps(task.get("best_profile") or {}, ensure_ascii=False, indent=2),
        "```",
    ]

    if rounds:
        lines.extend(
            [
                "",
                "## 轮次明细",
                "",
                "| Round | Label | Score | P@1 | Recall@K | MRR | Empty | Latency(ms) |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for item in rounds:
            metrics = dict(item.get("metrics") or {})
            lines.append(
                "| {round_index} | {label} | {score:.4f} | {p1} | {recall} | {mrr:.4f} | {empty} | {latency:.2f} |".format(
                    round_index=int(item.get("round_index", 0)),
                    label=str(item.get("label", "")),
                    score=float(item.get("score", 0.0)),
                    p1=_fmt_pct(metrics.get("precision_at_1")),
                    recall=_fmt_pct(metrics.get("recall_at_k")),
                    mrr=float(metrics.get("mrr", 0.0)),
                    empty=_fmt_pct(metrics.get("empty_rate")),
                    latency=float(item.get("latency_ms", 0.0)),
                )
            )

    return "\n".join(lines).strip() + "\n"


def make_paragraph_query(text: str) -> str:
    """把段落压缩成更适合检索评估的查询串。"""

    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if not compact:
        return ""
    chunks = [
        part.strip()
        for part in re.split(r"[。！？!?；;.\n]+", compact)
        if len(part.strip()) >= 8
    ]
    if chunks:
        chosen = max(chunks, key=len)
    else:
        chosen = compact
    return chosen[:72].strip()


def make_relation_query(subject: str, predicate: str, obj: str, *, seed_text: str = "") -> str:
    """为关系样本构造一个相对自然的检索查询。"""

    parts = [
        f"{subject} {predicate}",
        f"{predicate} {obj}",
        f"{subject} {obj}",
        f"{subject} {predicate} {obj}",
    ]
    filtered = [part.strip() for part in parts if part.strip()]
    if not filtered:
        return ""
    if not seed_text:
        return filtered[-1]
    idx = sum(ord(ch) for ch in seed_text) % len(filtered)
    return filtered[idx]
