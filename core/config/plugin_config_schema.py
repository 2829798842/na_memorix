"""Plugin config schema constants."""

from src.plugin_system import ConfigField

config_section_descriptions = {
    "plugin": "插件基本信息",
    "storage": "存储配置",
    "embedding": "嵌入模型配置",
    "retrieval": "检索配置",
    "threshold": "阈值策略配置",
    "graph": "知识图谱配置",
    "web": "可视化服务器配置",
    "advanced": "高级配置",
    "summarization": "总结与导入配置",
    "schedule": "定时任务配置",
    "filter": "消息过滤配置",
    "routing": "检索路由与兼容开关",
    "person_profile": "人物画像配置",
    "episode": "情景记忆 Episode 配置",
    "memory": "记忆衰减与强化配置",
}

# 配置Schema定义
config_schema: dict = {
    "plugin": {
        "config_version": ConfigField(
            type=str,
            default="4.1.0",
            description="配置文件版本"
        ),
        "enabled": ConfigField(
            type=bool,
            default=False,
            description="是否启用插件"
        ),
    },
    "storage": {
        "data_dir": ConfigField(
            type=str,
            default="./data",  # Changed to relative path default
            description="数据目录（默认为插件目录下的 data）"
        ),
    },
    "embedding": {
        "dimension": ConfigField(
            type=int,
            default=1024,
            description="向量维度 (对于支持动态维度的模型，将尝试请求此维度)"
        ),
        "quantization_type": ConfigField(
            type=str,
            default="int8",
            description="量化类型（vNext 仅支持 int8/SQ8）"
        ),
        "batch_size": ConfigField(
            type=int,
            default=32,
            description="批量生成嵌入的批次大小"
        ),
        "max_concurrent": ConfigField(
            type=int,
            default=5,
            description="嵌入API最大并发请求数"
        ),
        "model_name": ConfigField(
            type=str,
            default="auto",
            description="指定嵌入模型名称 (对应 model_config.toml 中的 name)"
        ),
        "retry": ConfigField(
            type=dict,
            default={
                "max_attempts": 5,
                "max_wait_seconds": 40,
                "min_wait_seconds": 3,
                "backoff_multiplier": 3,
            },
            description="嵌入重试配置: max_attempts, min_wait_seconds, max_wait_seconds, backoff_multiplier"
        ),
    },
    "retrieval": {
        "top_k_relations": ConfigField(
            type=int,
            default=10,
            description="关系检索返回数量"
        ),
        "top_k_paragraphs": ConfigField(
            type=int,
            default=20,
            description="段落检索返回数量"
        ),
        "top_k_final": ConfigField(
            type=int,
            default=10,
            description="最终融合后返回数量"
        ),
        "alpha": ConfigField(
            type=float,
            default=0.5,
            description="双路检索融合权重 (0:仅关系, 1:仅段落)"
        ),
        "enable_ppr": ConfigField(
            type=bool,
            default=True,
            description="是否启用 Personalized PageRank 重排序"
        ),
        "ppr_alpha": ConfigField(
            type=float,
            default=0.85,
            description="PPR的alpha参数"
        ),
        "ppr_concurrency_limit": ConfigField(
            type=int,
            default=4,
            description="PPR计算的最大并发数"
        ),
        "enable_parallel": ConfigField(
            type=bool,
            default=True,
            description="是否启用并行检索"
        ),
        "relation_semantic_fallback": ConfigField(
            type=bool,
            default=True,
            description="是否启用关系查询的语义回退（支持自然语言查询）"
        ),
        "relation_fallback_min_score": ConfigField(
            type=float,
            default=0.3,
            description="关系语义回退的最小相似度阈值"
        ),
        "temporal": ConfigField(
            type=dict,
            default={
                "enabled": True,
                "allow_created_fallback": True,
                "candidate_multiplier": 8,
                "default_top_k": 10,
                "max_scan": 1000,
            },
            description="时序检索配置"
        ),
        "search": ConfigField(
            type=dict,
            default={
                "smart_fallback": {
                    "enabled": True,
                    "threshold": 0.6,
                },
                "safe_content_dedup": {
                    "enabled": True,
                },
                "graph_recall": {
                    "enabled": True,
                    "candidate_k": 24,
                    "max_hop": 1,
                    "allow_two_hop_pair": True,
                    "max_paths": 4,
                },
                "relation_intent": {
                    "enabled": True,
                    "alpha_override": 0.35,
                    "relation_candidate_multiplier": 4,
                    "preserve_top_relations": 3,
                    "force_relation_sparse": True,
                    "pair_predicate_rerank_enabled": True,
                    "pair_predicate_limit": 3,
                },
            },
            description="统一检索后处理配置（smart fallback / safe dedup）"
        ),
        "relation_vectorization": ConfigField(
            type=dict,
            default={
                "enabled": False,
                "write_on_import": True,
                "backfill_enabled": False,
                "backfill_batch_size": 200,
                "backfill_interval_seconds": 5,
                "max_retry": 3,
            },
            description="关系向量化配置（写入与后台回填）"
        ),
        "time": ConfigField(
            type=dict,
            default={
                "skip_threshold_when_query_empty": True,
            },
            description="time 模式行为兼容配置"
        ),
        "aggregate": ConfigField(
            type=dict,
            default={
                "enabled": True,
                "default_top_k": 5,
                "default_mix": False,
                "rrf_k": 60.0,
                "weights": {
                    "search": 1.0,
                    "time": 1.0,
                    "episode": 1.0,
                },
            },
            description="聚合查询配置（search/time/episode 并发 + 可选混合融合）"
        ),
        "sparse": ConfigField(
            type=dict,
            default={
                "enabled": True,
                "backend": "fts5",
                "lazy_load": True,
                "mode": "auto",
                "tokenizer_mode": "jieba",
                "jieba_user_dict": "",
                "char_ngram_n": 2,
                "candidate_k": 80,
                "max_doc_len": 2000,
                "enable_ngram_fallback_index": True,
                "enable_like_fallback": False,
                "enable_relation_sparse_fallback": True,
                "relation_candidate_k": 60,
                "relation_max_doc_len": 512,
                "unload_on_disable": True,
                "shrink_memory_on_unload": True,
            },
            description="稀疏检索配置（FTS5 + BM25）"
        ),
        "fusion": ConfigField(
            type=dict,
            default={
                "method": "weighted_rrf",
                "rrf_k": 60,
                "vector_weight": 0.7,
                "bm25_weight": 0.3,
                "normalize_score": True,
                "normalize_method": "minmax",
            },
            description="检索融合配置"
        ),
    },
    "threshold": {
        "min_threshold": ConfigField(
            type=float,
            default=0.3,
            description="搜索结果的最小阈值"
        ),
        "max_threshold": ConfigField(
            type=float,
            default=0.95,
            description="搜索结果的最大阈值"
        ),
        "percentile": ConfigField(
            type=float,
            default=75.0,
            description="动态阈值的分位数"
        ),
        "std_multiplier": ConfigField(
            type=float,
            default=1.5,
            description="标准差倍数（用于异常值过滤）"
        ),
        "min_results": ConfigField(
            type=int,
            default=3,
            description="即使未达标也强制返回的最小结果数"
        ),
        "enable_auto_adjust": ConfigField(
            type=bool,
            default=True,
            description="是否根据结果分布自动调整阈值"
        ),
    },
    "graph": {
        "sparse_matrix_format": ConfigField(
            type=str,
            default="csr",
            description="稀疏矩阵存储格式: csr, csc"
        ),
    },
    "web": {
        "enabled": ConfigField(
            type=bool,
            default=True,
            description="是否启用可视化编辑服务器"
        ),
        "port": ConfigField(
            type=int,
            default=8082,
            description="服务器端口"
        ),
        "host": ConfigField(
            type=str,
            default="0.0.0.0",
            description="服务器绑定地址"
        ),
        "import": ConfigField(
            type=dict,
            default={
                "enabled": True,
                "max_queue_size": 20,
                "max_files_per_task": 200,
                "max_file_size_mb": 20,
                "max_paste_chars": 200000,
                "default_file_concurrency": 2,
                "default_chunk_concurrency": 4,
                "max_file_concurrency": 6,
                "max_chunk_concurrency": 12,
                "poll_interval_ms": 1000,
                "token": "",
                "path_aliases": {
                    "raw": "./plugins/A_memorix/data/raw",
                    "lpmm": "./data/lpmm_storage",
                    "plugin_data": "./plugins/A_memorix/data",
                },
                "llm_retry": {
                    "max_attempts": 4,
                    "min_wait_seconds": 3,
                    "max_wait_seconds": 40,
                    "backoff_multiplier": 3,
                },
                "convert": {
                    "enable_staging_switch": True,
                    "keep_backup_count": 3,
                },
            },
            description="Web 导入中心配置（队列、并发、大小限制、鉴权）"
        ),
        "tuning": ConfigField(
            type=dict,
            default={
                "enabled": True,
                "poll_interval_ms": 1200,
                "max_queue_size": 8,
                "default_objective": "precision_priority",
                "default_intensity": "standard",
                "default_top_k_eval": 20,
                "default_sample_size": 24,
                "eval_query_timeout_seconds": 10.0,
                "llm_retry": {
                    "max_attempts": 3,
                    "min_wait_seconds": 2,
                    "max_wait_seconds": 20,
                    "backoff_multiplier": 2,
                },
            },
            description="Web 检索调优中心配置（任务轮询、队列与默认调参策略）"
        ),
    },
    "advanced": {
        "enable_auto_save": ConfigField(
            type=bool,
            default=True,
            description="启用自动保存（原子化统一持久化）"
        ),
        "auto_save_interval_minutes": ConfigField(
            type=int,
            default=5,
            description="自动保存间隔（分钟）"
        ),
        "debug": ConfigField(
            type=bool,
            default=False,
            description="启用详细调试日志"
        ),
        "extraction_model": ConfigField(
            type=str,
            default="auto",
            description="指定知识抽取模型名称 (对应 model_config.toml 中的 name)"
        ),
    },
    "summarization": {
        "enabled": ConfigField(
            type=bool,
            default=True,
            description="是否启用总结导入功能"
        ),
        "model_name": ConfigField(
            type=list,
            default=["auto"],
            description="总结模型选择器列表（List[str]）"
        ),
        "context_length": ConfigField(
            type=int,
            default=50,
            description="总结消息的上下文条数"
        ),
        "include_personality": ConfigField(
            type=bool,
            default=True,
            description="总结提示词是否包含机器人人设"
        ),
        "default_knowledge_type": ConfigField(
            type=str,
            default="narrative",
            description="总结导入时的默认知识类型（narrative/factual/quote/structured/mixed）"
        ),
    },
    "schedule": {
        "enabled": ConfigField(
            type=bool,
            default=True,
            description="是否启用定时自动导入"
        ),
        "import_times": ConfigField(
            type=list,
            default=["04:00"],
            description="每日自动导入的时间点列表 (24小时制, 如 ['04:00', '16:00'])"
        ),
    },
    "filter": {
        "enabled": ConfigField(
            type=bool,
            default=True,
            description="是否启用聊天流过滤"
        ),
        "mode": ConfigField(
            type=str,
            default="whitelist",
            description="过滤模式：whitelist(白名单) 或 blacklist(黑名单)"
        ),
        "chats": ConfigField(
            type=list,
            default=[],
            description="聊天流 ID 列表。支持填写: 1. 群号 (group_id, 如: 123456); 2. 私聊用户ID (user_id, 如: 10001); 3. 聊天流唯一标识 (stream_id, MD5格式)。"
        ),
    },
    "routing": {
        "search_owner": ConfigField(
            type=str,
            default="action",
            description="search/time 主责入口：action|tool|dual"
        ),
        "tool_search_mode": ConfigField(
            type=str,
            default="forward",
            description="knowledge_query 的 search/time 模式：forward|disabled"
        ),
        "enable_request_dedup": ConfigField(
            type=bool,
            default=True,
            description="是否启用短时请求去重（抑制 Action+Tool 同轮重复检索）"
        ),
        "request_dedup_ttl_seconds": ConfigField(
            type=int,
            default=2,
            description="请求去重 TTL（秒）"
        ),
    },
    "person_profile": {
        "enabled": ConfigField(
            type=bool,
            default=True,
            description="人物画像模块总开关"
        ),
        "opt_in_required": ConfigField(
            type=bool,
            default=True,
            description="是否要求用户显式开启（默认开启显式开关模式）"
        ),
        "default_injection_enabled": ConfigField(
            type=bool,
            default=False,
            description="当不存在用户开关记录时的默认注入状态"
        ),
        "profile_ttl_minutes": ConfigField(
            type=float,
            default=360.0,
            description="人物画像快照 TTL（分钟）"
        ),
        "refresh_interval_minutes": ConfigField(
            type=int,
            default=30,
            description="定时刷新周期（分钟）"
        ),
        "active_window_hours": ConfigField(
            type=float,
            default=72.0,
            description="活跃人物窗口（小时），仅刷新窗口内人物"
        ),
        "max_refresh_per_cycle": ConfigField(
            type=int,
            default=50,
            description="每轮最多刷新的人物数"
        ),
        "top_k_evidence": ConfigField(
            type=int,
            default=12,
            description="人物画像构建时的向量证据数量上限"
        ),
    },
    "episode": {
        "enabled": ConfigField(
            type=bool,
            default=True,
            description="Episode 模块总开关"
        ),
        "query_enabled": ConfigField(
            type=bool,
            default=True,
            description="是否启用 Episode 查询能力（命令/Tool/API）"
        ),
        "generation_enabled": ConfigField(
            type=bool,
            default=True,
            description="是否启用 Episode 异步生成"
        ),
        "generation_interval_seconds": ConfigField(
            type=int,
            default=30,
            description="Episode 生成循环间隔（秒）"
        ),
        "generation_batch_size": ConfigField(
            type=int,
            default=20,
            description="每批处理的 pending paragraph 数量"
        ),
        "max_retry": ConfigField(
            type=int,
            default=3,
            description="Episode 生成失败最大重试次数"
        ),
        "segmentation_model": ConfigField(
            type=str,
            default="auto",
            description="Episode 语义切分模型选择器（auto/任务名/模型名）"
        ),
        "max_paragraphs_per_call": ConfigField(
            type=int,
            default=20,
            description="单次语义切分最大段落数"
        ),
        "max_chars_per_call": ConfigField(
            type=int,
            default=6000,
            description="单次语义切分最大字符数"
        ),
        "source_time_window_hours": ConfigField(
            type=float,
            default=24.0,
            description="按 source 分组时的时间窗口（小时）"
        ),
        "default_top_k": ConfigField(
            type=int,
            default=5,
            description="Episode 查询默认返回数量"
        ),
    },
    "memory": {
         "half_life_hours": ConfigField(type=float, default=24.0, description="记忆强度半衰期 (小时)"),
         "base_decay_interval_hours": ConfigField(type=float, default=1.0, description="衰减任务执行间隔 (小时)"),
         "prune_threshold": ConfigField(type=float, default=0.1, description="记忆遗忘(冷冻)阈值"),
         "freeze_duration_hours": ConfigField(type=float, default=0.01, description="记忆冷冻保留期 (小时), 过期物理删除"), # 默认可以设小点便于测试，或者保留24h
         "enable_auto_reinforce": ConfigField(type=bool, default=True, description="是否启用自动强化"),
         "reinforce_buffer_max_size": ConfigField(type=int, default=1000, description="强化缓冲区最大大小"),
         "reinforce_cooldown_hours": ConfigField(type=float, default=1.0, description="同一记忆强化的冷却时间"),
         "max_weight": ConfigField(type=float, default=10.0, description="记忆最大权重限制"),
         "revive_boost_weight": ConfigField(type=float, default=0.5, description="复活时的基础增强权重"),
         "auto_protect_ttl_hours": ConfigField(type=float, default=24.0, description="复活/强化后的自动保护时长"),
         "min_active_weight_protected": ConfigField(type=float, default=0.5, description="保护期内记忆的最低权重地板"),
         "enabled": ConfigField(type=bool, default=True, description="V5记忆系统总开关"),
         "orphan": {
             "enable_soft_delete": ConfigField(type=bool, default=True, description="是否启用软删除"),
             "entity_retention_days": ConfigField(type=float, default=7.0, description="实体保留期(天)"),
             "paragraph_retention_days": ConfigField(type=float, default=7.0, description="段落保留期(天)"),
             "sweep_grace_hours": ConfigField(type=float, default=24.0, description="软删除宽限期(小时)"),
         }
    }
}

__all__ = ["config_section_descriptions", "config_schema"]
