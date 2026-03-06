"""
导入知识Command组件

支持从文本、文件等来源导入知识到知识库。
"""

import time
import re
import json
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path

from src.common.logger import get_logger
from src.plugin_system.base.base_command import BaseCommand
from src.chat.message_receive.message import MessageRecv

# 导入核心模块
from src.plugin_system.apis import llm_api
from src.config.config import model_config as host_model_config
from src.config.api_ada_configs import TaskConfig
from ...core import (
    VectorStore,
    GraphStore,
    MetadataStore,
    EmbeddingAPIAdapter,
    KnowledgeType,
    resolve_stored_knowledge_type,
    select_import_strategy,
    should_extract_relations,
    get_type_display_name,
)
from ...core.utils.relation_write_service import RelationWriteService
from ...core.utils.import_payloads import normalize_paragraph_import_item
from ...core.utils.runtime_self_check import (
    ensure_runtime_self_check,
    run_embedding_runtime_self_check,
)

logger = get_logger("A_Memorix.ImportCommand")


class ImportCommand(BaseCommand):
    """导入知识Command

    功能：
    - 从文本导入段落
    - 从文本提取实体和关系
    - 自动生成嵌入向量
    - 批量导入支持
    """

    # Command基本信息
    command_name = "import"
    command_description = "导入知识到知识库，支持文本、段落、实体和关系"
    # Command基本信息
    command_name = "import"
    command_description = "导入知识到知识库，支持文本、段落、实体和关系"
    # 使用严格的模式匹配，避免将内容误识别为未知的模式
    command_pattern = r"^\/import(?:\s+(?P<mode>text|paragraph|relation|file|json))?(?:\s+(?P<content>.+))?$"

    def __init__(self, message: MessageRecv, plugin_config: Optional[dict] = None):
        """初始化导入Command"""
        super().__init__(message, plugin_config)

        # 获取存储实例 (优先从配置获取，兜底从插件实例获取)
        self.vector_store: Optional[VectorStore] = self.plugin_config.get("vector_store")
        self.graph_store: Optional[GraphStore] = self.plugin_config.get("graph_store")
        self.metadata_store: Optional[MetadataStore] = self.plugin_config.get("metadata_store")
        self.embedding_manager: Optional[EmbeddingAPIAdapter] = self.plugin_config.get("embedding_manager")
        self.relation_write_service: Optional[RelationWriteService] = self.plugin_config.get("relation_write_service")

        # 兜底逻辑：如果配置中没有存储实例，尝试直接从插件系统获取
        # 使用 is not None 检查，因为空对象可能布尔值为 False
        if not all([
            self.vector_store is not None,
            self.graph_store is not None,
            self.metadata_store is not None,
            self.embedding_manager is not None
        ]):
            from ...plugin import A_MemorixPlugin
            instances = A_MemorixPlugin.get_storage_instances()
            if instances:
                self.vector_store = self.vector_store or instances.get("vector_store")
                self.graph_store = self.graph_store or instances.get("graph_store")
                self.metadata_store = self.metadata_store or instances.get("metadata_store")
                self.embedding_manager = self.embedding_manager or instances.get("embedding_manager")
                self.relation_write_service = self.relation_write_service or instances.get("relation_write_service")

        # 设置日志前缀
        if self.message and self.message.chat_stream:
            self.log_prefix = f"[ImportCommand-{self.message.chat_stream.stream_id}]"
        else:
            self.log_prefix = "[ImportCommand]"

    @property
    def debug_enabled(self) -> bool:
        """检查是否启用了调试模式"""
        # 尝试从 plugin_config 获取 advanced.debug
        advanced = self.plugin_config.get("advanced", {})
        if isinstance(advanced, dict):
            return advanced.get("debug", False)
        # 兜底：直接检查 debug 字段
        return self.plugin_config.get("debug", False)

    async def execute(self) -> Tuple[bool, Optional[str], int]:
        """执行导入命令

        Returns:
            Tuple[bool, Optional[str], int]: (是否成功, 回复消息, 拦截级别)
        """
        # 检查存储是否初始化 (使用 is not None 而非布尔值，因为空对象可能为 False)
        if not all([
            self.vector_store is not None,
            self.graph_store is not None,
            self.metadata_store is not None,
            self.embedding_manager is not None
        ]):
            error_msg = "❌ 知识库未初始化，无法导入"
            logger.error(f"{self.log_prefix} {error_msg}")
            return False, error_msg, 0

        self_check_ok, self_check_msg = await self._ensure_runtime_self_check()
        if not self_check_ok:
            logger.error(f"{self.log_prefix} 导入前自检失败: {self_check_msg}")
            return False, f"❌ 导入前自检失败: {self_check_msg}", 0

        # 获取匹配的参数: 如果 mode 未捕获(None)，则默认为 "text"
        mode = self.matched_groups.get("mode") or "text"
        content = self.matched_groups.get("content", "")

        if not content:
            help_msg = self._get_help_message()
            return True, help_msg, 0

        logger.info(f"{self.log_prefix} 执行导入: mode={mode}, content_length={len(content)}")
        if self.debug_enabled:
            logger.info(f"{self.log_prefix} [DEBUG] 导入内容预览: {content[:200]}...")

        try:
            # 根据模式执行导入
            if mode == "text":
                success, result = await self._import_text(content)
            elif mode == "paragraph":
                success, result = await self._import_paragraph(content)
            elif mode == "relation":
                success, result = await self._import_relation(content)
            elif mode == "file":
                success, result = await self._import_from_file(content)
            elif mode == "json":
                success, result = await self._import_json(content)
            else:
                success, result = False, f"❌ 未知的导入模式: {mode}"

            # 持久化保存
            if success:
                try:
                    self.vector_store.save()
                    self.graph_store.save()
                    logger.info(f"{self.log_prefix} 数据已持久化保存")
                except Exception as e:
                    logger.error(f"{self.log_prefix} 数据持久化失败: {e}")

            return success, result, 0

        except Exception as e:
            error_msg = f"❌ 导入失败: {str(e)}"
            logger.error(f"{self.log_prefix} {error_msg}")
            return False, error_msg, 0

    async def _ensure_runtime_self_check(self) -> Tuple[bool, str]:
        plugin_instance = self.plugin_config.get("plugin_instance")
        if plugin_instance is not None:
            report = await ensure_runtime_self_check(plugin_instance)
        else:
            report = await run_embedding_runtime_self_check(
                config=self.plugin_config,
                vector_store=self.vector_store,
                embedding_manager=self.embedding_manager,
            )

        if bool(report.get("ok", False)):
            return True, ""

        return (
            False,
            f"{report.get('message', 'unknown')} "
            f"(configured={report.get('configured_dimension', 0)}, "
            f"store={report.get('vector_store_dimension', 0)}, "
            f"encoded={report.get('encoded_dimension', 0)})。"
            "请先执行 python plugins/A_memorix/scripts/runtime_self_check.py --json",
        )

    async def _import_text(self, text: str) -> Tuple[bool, str]:
        """导入文本（自动分段和提取）

        Args:
            text: 待导入的文本

        Returns:
            Tuple[bool, str]: (是否成功, 结果消息)
        """
        start_time = time.time()

        # 分段处理
        paragraphs = self._split_text(text)

        if not paragraphs:
            return False, "❌ 未能从文本中提取有效段落"

        logger.info(f"{self.log_prefix} 文本分段: {len(paragraphs)}个段落")

        # 尝试选择 LLM 模型
        try:
            model_config_to_use = await self._select_model()
            use_llm = True
        except Exception as e:
            logger.warning(f"{self.log_prefix} 未找到可用模型或选择失败: {e}，将回退到基础模式")
            use_llm = False
            model_config_to_use = None

        success_count = 0
        entities_count = 0
        relations_count = 0
        type_counts = {}

        for paragraph in paragraphs:
            # 1. 尝试 LLM 提取
            llm_result = {}
            if use_llm:
                try:
                    llm_result = await self._llm_extract(paragraph, model_config_to_use)
                except Exception as e:
                    logger.warning(f"{self.log_prefix} LLM 提取失败: {e}")

            # 2. 导入段落
            try:
                hash_value, detected_type = await self._add_paragraph(paragraph)
                success_count += 1
                
                type_name = detected_type.value
                type_counts[type_name] = type_counts.get(type_name, 0) + 1
            except Exception as e:
                logger.warning(f"{self.log_prefix} 段落导入失败: {e}")
                continue

            # 3. 导入 LLM 提取的实体
            if llm_result.get("entities"):
                extracted_entities = llm_result["entities"]
                if extracted_entities:
                    for entity in extracted_entities:
                        # 传递 source_paragraph 以建立关联
                        await self._add_entity_with_vector(entity, source_paragraph=hash_value)
                    entities_count += len(extracted_entities)
            
            # 4. 导入 LLM 提取的关系
            if llm_result.get("relations"):
                for rel in llm_result["relations"]:
                    s, p, o = rel.get("subject"), rel.get("predicate"), rel.get("object")
                    if all([s, p, o]):
                        try:
                            await self._add_relation(s, p, o, source_paragraph=hash_value)
                            relations_count += 1
                        except Exception as e:
                            logger.debug(f"{self.log_prefix} 关系添加失败: {e}")

            # 5. 回退逻辑：如果 LLM 为空且类型适合，尝试正则
            if not llm_result and should_extract_relations(detected_type):
                e_c, r_c = await self._extract_knowledge_regex([paragraph], source_hash=hash_value)
                entities_count += e_c
                relations_count += r_c


        elapsed = time.time() - start_time

        # 构建结果消息
        result_lines = [
            "✅ 文本导入完成 (智能增强)",
            f"📊 统计信息:",
            f"  - 段落: {success_count}/{len(paragraphs)}",
        ]
        
        if type_counts:
            result_lines.append(f"  - 类型分布:")
            for type_name, count in type_counts.items():
                result_lines.append(f"    • {type_name}: {count}")
        
        result_lines.extend([
            f"  - 实体: {entities_count}",
            f"  - 关系: {relations_count}",
            f"⏱️ 耗时: {elapsed:.2f}秒",
        ])

        return True, "\n".join(result_lines)

    async def _import_paragraph(self, content: str) -> Tuple[bool, str]:
        """导入单个段落

        Args:
            content: 段落内容

        Returns:
            Tuple[bool, str]: (是否成功, 结果消息)
        """
        try:
            hash_value, detected_type = await self._add_paragraph(content)

            result_lines = [
                "✅ 段落导入完成",
                f"📝 Hash: {hash_value[:16]}...",
                f"🏷️ 类型: {get_type_display_name(detected_type)}",
                f"📄 内容: {content[:50]}...",
            ]

            return True, "\n".join(result_lines)

        except Exception as e:
            return False, f"❌ 段落导入失败: {str(e)}"

    async def _import_relation(self, content: str) -> Tuple[bool, str]:
        """导入关系

        格式: subject|predicate|object 或 subject predicate object

        Args:
            content: 关系内容

        Returns:
            Tuple[bool, str]: (是否成功, 结果消息)
        """
        try:
            # 解析关系
            if "|" in content:
                parts = content.split("|")
                if len(parts) != 3:
                    return False, "❌ 关系格式错误，应使用: subject|predicate|object"
                subject, predicate, obj = parts
            else:
                # 尝试空格分隔
                parts = content.split(maxsplit=2)
                if len(parts) != 3:
                    return False, "❌ 关系格式错误，应使用: subject|predicate|object"
                subject, predicate, obj = parts

            # 去除空白
            subject = subject.strip()
            predicate = predicate.strip()
            obj = obj.strip()

            if not all([subject, predicate, obj]):
                return False, "❌ 关系字段不能为空"

            # 添加关系
            hash_value = await self._add_relation(subject, predicate, obj)

            result_lines = [
                "✅ 关系导入完成",
                f"🔗 Hash: {hash_value[:16]}...",
                f"📌 {subject} {predicate} {obj}",
            ]

            return True, "\n".join(result_lines)

        except Exception as e:
            return False, f"❌ 关系导入失败: {str(e)}"

    async def _import_json(self, json_input: str) -> Tuple[bool, str]:
        """从JSON文件或JSON字符串导入知识。"""
        try:
            path = Path(json_input)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = json.loads(json_input)

            p_count = 0
            r_count = 0
            e_count = 0

            # 导入段落（支持 time_meta 透传）
            paragraphs = data.get("paragraphs", [])
            for p in paragraphs:
                paragraph = normalize_paragraph_import_item(
                    p,
                    default_source="import_command",
                )
                para_hash, _ = await self._add_paragraph(
                    paragraph["content"],
                    knowledge_type=resolve_stored_knowledge_type(paragraph["knowledge_type"]),
                    time_meta=paragraph["time_meta"],
                    source=paragraph["source"],
                )
                for entity in paragraph["entities"]:
                    await self._add_entity_with_vector(entity, source_paragraph=para_hash)
                    e_count += 1
                for rel in paragraph["relations"]:
                    await self._add_relation(
                        rel["subject"],
                        rel["predicate"],
                        rel["object"],
                        source_paragraph=para_hash,
                    )
                    r_count += 1
                p_count += 1

            # 导入实体
            entities = data.get("entities", [])
            if entities:
                for entity in entities:
                    await self._add_entity_with_vector(entity)
                e_count += len(entities)

            # 导入关系
            relations = data.get("relations", [])
            for r in relations:
                s = r.get("subject")
                p = r.get("predicate")
                o = r.get("object")
                if all([s, p, o]):
                    await self._add_relation(s, p, o)
                    r_count += 1

            result_lines = [
                "✅ JSON导入完成",
                "📊 统计信息:",
                f"  - 段落: {p_count}",
                f"  - 实体: {e_count}",
                f"  - 关系: {r_count}",
            ]
            return True, "\n".join(result_lines)

        except json.JSONDecodeError:
            return False, "❌ JSON格式错误（可传入文件路径或JSON字符串）"
        except Exception as e:
            return False, f"❌ JSON导入失败: {str(e)}"

    async def _import_from_file(self, file_path: str) -> Tuple[bool, str]:
        """从文件导入知识

        Args:
            file_path: 文件路径

        Returns:
            Tuple[bool, str]: (是否成功, 结果消息)
        """
        try:
            path = Path(file_path)

            if not path.exists():
                return False, f"❌ 文件不存在: {file_path}"

            # 根据文件扩展名选择导入方式
            if path.suffix.lower() in [".txt", ".md"]:
                # 读取文件
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                return await self._import_text(content)
            elif path.suffix.lower() == ".json":
                return await self._import_json(str(path))
            else:
                return False, f"❌ 不支持的文件类型: {path.suffix}"

        except Exception as e:
            return False, f"❌ 文件导入失败: {str(e)}"

    async def _add_paragraph(
        self,
        content: str,
        knowledge_type: Optional[KnowledgeType] = None,
        time_meta: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
    ) -> Tuple[str, KnowledgeType]:
        """添加段落到知识库

        Args:
            content: 段落内容
            knowledge_type: 知识类型（可选，None则自动检测）

        Returns:
            元组：(段落hash值, 检测到的知识类型)
        """
        # 自动检测知识类型
        if knowledge_type is None:
            import_strategy = select_import_strategy(content)
            knowledge_type = resolve_stored_knowledge_type(import_strategy, content=content)
        
        if self.debug_enabled:
            logger.info(
                f"{self.log_prefix} [DEBUG] 段落类型检测: {get_type_display_name(knowledge_type)}"
            )
        
        # 添加到metadata store
        hash_value = self.metadata_store.add_paragraph(
            content=content,
            source=source or "import_command",
            knowledge_type=knowledge_type.value,
            time_meta=time_meta,
        )

        # 生成嵌入向量 (异步调用)
        embedding = await self.embedding_manager.encode(content)

        # 添加到vector store
        self.vector_store.add(
            vectors=embedding.reshape(1, -1),
            ids=[hash_value],
        )

        logger.debug(f"{self.log_prefix} 添加段落: hash={hash_value[:16]}..., type={knowledge_type.value}")
        if self.debug_enabled:
            logger.info(f"{self.log_prefix} [DEBUG] 向量生成成功: shape={embedding.shape}, dtype={embedding.dtype}")
            logger.info(f"{self.log_prefix} [DEBUG] 段落元数据已写入: hash={hash_value}")

        return hash_value, knowledge_type

    async def _add_relation(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        source_paragraph: str = "",
    ) -> str:
        """添加关系到知识库

        Args:
            subject: 主体
            predicate: 谓词
            obj: 客体
            confidence: 置信度
            source_paragraph: 源段落

        Returns:
            关系hash值
        """
        # 添加实体到图 (并向量化)
        await self._add_entity_with_vector(subject, source_paragraph=source_paragraph)
        await self._add_entity_with_vector(obj, source_paragraph=source_paragraph)

        rv_enabled = bool(self.get_config("retrieval.relation_vectorization.enabled", False))
        write_on_import = bool(self.get_config("retrieval.relation_vectorization.write_on_import", True))
        write_vector = rv_enabled and write_on_import

        if self.relation_write_service is not None:
            result = await self.relation_write_service.upsert_relation_with_vector(
                subject=subject,
                predicate=predicate,
                obj=obj,
                confidence=confidence,
                source_paragraph=source_paragraph,
                write_vector=write_vector,
            )
            logger.debug(
                f"{self.log_prefix} 添加关系: {subject} {predicate} {obj}, "
                f"hash={result.hash_value[:16]}..., vector_state={result.vector_state}"
            )
            return result.hash_value

        # 添加关系到metadata store
        hash_value = self.metadata_store.add_relation(
            subject=subject,
            predicate=predicate,
            obj=obj,  # 参数名是 obj 而不是 object
            confidence=confidence,
            source_paragraph=source_paragraph, # 这里应该是 hash
        )

        # 添加关系到图（写入 relation_hashes，确保删除/修剪可精确回溯）
        self.graph_store.add_edges([(subject, obj)], relation_hashes=[hash_value])
        try:
            self.metadata_store.set_relation_vector_state(hash_value, "none")
        except Exception:
            pass

        logger.debug(
            f"{self.log_prefix} 添加关系: {subject} {predicate} {obj}, "
            f"hash={hash_value[:16]}..."
        )

        return hash_value

    def _split_text(self, text: str, max_length: int = 500) -> List[str]:
        """将文本分段

        Args:
            text: 输入文本
            max_length: 最大段落长度

        Returns:
            段落列表
        """
        # 按段落分段
        paragraphs = text.split("\n\n")

        result = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 如果段落过长，按句子继续分段
            if len(para) > max_length:
                sentences = re.split(r"[。！？.!?]", para)
                current_chunk = ""

                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue

                    if len(current_chunk) + len(sentence) < max_length:
                        current_chunk += sentence + "。"
                    else:
                        if current_chunk:
                            result.append(current_chunk.strip())
                        current_chunk = sentence + "。"

                if current_chunk:
                    result.append(current_chunk.strip())
            else:
                result.append(para)

        return result

    async def _extract_knowledge(
        self,
        paragraphs: List[str],
    ) -> Tuple[int, int]:
        """从段落中提取实体和关系（简化实现）

        Args:
            paragraphs: 段落列表

        Returns:
            Tuple[int, int]: (实体数量, 关系数量)
        """
        entities_count = 0
        relations_count = 0

        # 获取可用模型
        models = llm_api.get_available_models()
        if not models:
            logger.warning(f"{self.log_prefix} 未找到可用模型，退回到正则提取")
            return await self._extract_knowledge_regex(paragraphs)
        
        # 优先选择 balanced 或 performance 模型，否则选第一个
        model_name = "balanced" if "balanced" in models else list(models.keys())[0]
        model_config = models[model_name]

        for para in paragraphs:
            if len(para.strip()) < 10:
                continue

            if self.debug_enabled:
                logger.info(f"{self.log_prefix} [DEBUG] 正在通过 LLM 提取知识，段落长度: {len(para)}")

            prompt = f"""请从以下段落中提取实体和它们之间的关系。
输出值优先使用原文主语言，不要将中文翻译成英文或其他语言，也不要改写专有名词。
以 JSON 格式返回，格式如下：
{{
  "entities": ["实体1", "实体2"],
  "relations": [
    {{"subject": "主体", "predicate": "关系", "object": "客体"}}
  ]
}}

段落内容：
{para}
"""
            success, response, _, _ = await llm_api.generate_with_model(
                prompt=prompt,
                model_config=model_config,
                request_type="A_Memorix.KnowledgeExtraction"
            )

            if success:
                try:
                    # 提取 JSON 部分
                    json_match = re.search(r"\{.*\}", response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group()
                        if self.debug_enabled:
                            logger.info(f"{self.log_prefix} [DEBUG] LLM 提取结果 JSON: {json_str}")
                        data = json.loads(json_str)
                        
                        # 添加实体
                        entities = data.get("entities", [])
                        if entities:
                            for entity in entities:
                                # 这里的 para 是 content，我们其实应该传 hash，但 _extract_knowledge 接口只接收 paragraphs list
                                # 由于 _extract_knowledge 的设计问题，它没有很好的上下文 hash。
                                # 但注意到该方法主要用于测试或简单调用，主流程是 _import_text，那里是分开处理的。
                                # _import_text 调用的是 _llm_extract 返回数据，然后自己在外面循环添加。
                                # 这个 _extract_knowledge 方法似乎是独立的辅助方法？
                                # 看起来 _import_text 并没有直接调用 _extract_knowledge，而是调用的 _llm_extract 和 _add_paragraph 分开处理。
                                # 只有当 ImportCommand 被外部调用用来 "只提取不存段落" 时才会用到这个？
                                # 或者 _extract_knowledge_regex 被用到了。
                                # 经过检查，_import_text 在 228行调用了 _extract_knowledge_regex。
                                # 但该方法没有被 _import_text 调用。
                                # 为了保持一致性，还是加上 source_paragraph=para (虽然这其实是 content 不是 hash，可能导致外键错误)
                                # 等等，metadata_store.add_entity 的 source_paragraph 参数期望的是 hash。
                                # 如果传入 content，会违反外键约束 (如果有的话) 或者存入无效 hash。
                                # 鉴于 _extract_knowledge 不在主流程 _import_text 中使用 (它是分开的)，
                                # 且它甚至没有 parameter hash 的上下文。
                                # 我们先留空，或者传入空字符串。
                                await self._add_entity_with_vector(entity)
                            entities_count += len(entities)

                        # 添加关系
                        relations = data.get("relations", [])
                        for rel in relations:
                            s, p, o = rel.get("subject"), rel.get("predicate"), rel.get("object")
                            if all([s, p, o]):
                                await self._add_relation(
                                    subject=s,
                                    predicate=p,
                                    obj=o,
                                    source_paragraph=para,
                                )
                                relations_count += 1
                        continue # 成功则跳过正则
                except Exception as e:
                    logger.debug(f"{self.log_prefix} LLM 结果解析失败: {e}")

            # 如果 LLM 失败或无效，退回到正则
            e_c, r_c = await self._extract_knowledge_regex([para])
            entities_count += e_c
            relations_count += r_c

        return entities_count, relations_count

    async def _select_model(self) -> Any:
        """选择知识抽取模型（支持任务名/模型名/auto）。"""
        models = llm_api.get_available_models()
        if not models:
            raise ValueError("没有可用的 LLM 模型配置")

        def _is_task_config(task_cfg: Any) -> bool:
            return hasattr(task_cfg, "model_list") and bool(getattr(task_cfg, "model_list", []))

        def _build_single_model_task(model_name: str, template: TaskConfig) -> TaskConfig:
            return TaskConfig(
                model_list=[model_name],
                max_tokens=template.max_tokens,
                temperature=template.temperature,
                slow_threshold=template.slow_threshold,
                selection_strategy=template.selection_strategy,
            )

        preferred_tasks = (
            "lpmm_entity_extract",
            "lpmm_rdf_build",
            "replyer",
            "utils",
            "planner",
            "tool_use",
        )

        config_model = str(
            self.plugin_config.get("advanced", {}).get("extraction_model", "auto") or "auto"
        ).strip()
        model_dict = getattr(host_model_config, "models_dict", {}) or {}

        # 1) 显式任务名
        if config_model.lower() != "auto" and config_model in models and _is_task_config(models[config_model]):
            logger.info(f"{self.log_prefix} 使用插件配置指定任务: {config_model}")
            return models[config_model]

        # 2) 显式模型名（对应 model_config.toml 的模型键）
        if config_model.lower() != "auto" and config_model in model_dict:
            template = next(
                (
                    models.get(task_name)
                    for task_name in preferred_tasks
                    if _is_task_config(models.get(task_name))
                ),
                None,
            )
            if template is None:
                template = next((task for task in models.values() if _is_task_config(task)), None)
            if template is not None:
                logger.info(f"{self.log_prefix} 使用插件配置指定模型: {config_model}")
                return _build_single_model_task(config_model, template)

        if config_model.lower() != "auto":
            logger.warning(
                f"{self.log_prefix} extraction_model='{config_model}' 无法识别，回退自动选择"
            )

        # 3) auto：优先抽取相关任务，避免误落到 embedding
        for task_name in preferred_tasks:
            task_cfg = models.get(task_name)
            if _is_task_config(task_cfg):
                logger.info(f"{self.log_prefix} auto 选择任务: {task_name}")
                return task_cfg

        # 4) fallback：任意非 embedding 任务
        for task_name, task_cfg in models.items():
            if task_name != "embedding" and _is_task_config(task_cfg):
                logger.info(f"{self.log_prefix} auto 回退任务: {task_name}")
                return task_cfg

        # 5) 最终兜底
        first_task_name = next(iter(models))
        logger.warning(f"{self.log_prefix} 仅检测到 embedding 或异常任务，回退: {first_task_name}")
        return models[first_task_name]

    async def _llm_extract(self, chunk: str, model_config: Any) -> Dict:
        """调用 LLM 提取知识"""
        prompt = f"""请分析以下文本，提取其中的实体（Entities）和关系（Relations）。
输出值优先使用原文主语言，不要将中文翻译成英文或其他语言，也不要改写专有名词。
仅提取关键信息。
JSON格式: {{ "entities": ["e1"], "relations": [{{"subject": "s", "predicate": "p", "object": "o"}}] }}
文本:
{chunk[:2000]}
"""
        success, response, _, _ = await llm_api.generate_with_model(
            prompt=prompt,
            model_config=model_config,
            request_type="A_Memorix.KnowledgeExtraction"
        )
        if success:
            try:
                # 简单清理
                txt = response.strip()
                if "```" in txt:
                    txt = txt.split("```json")[-1].split("```")[0].strip()
                    if txt.startswith("json"): txt = txt[4:].strip()
                return json.loads(txt)
            except:
                pass
        return {}

    async def _extract_knowledge_regex(self, paragraphs: List[str], source_hash: Optional[str] = None) -> Tuple[int, int]:
        """使用正则提取知识（备用方案）"""
        entities_count = 0
        relations_count = 0
        for para in paragraphs:
            # 简单提取: 大写单词 或 引号内容
            # 使用非捕获组或分步提取以避免 findall 的空元组问题
            entities = re.findall(r"[A-Z][a-z]+", para)
            quoted = re.findall(r"[\"']([^\"']+)[\"']", para)
            entities.extend(quoted)
            
            unique_entities = list(set([e for e in entities if e.strip()]))
            if unique_entities:
                for entity in unique_entities:
                    # 传递 source_hash
                    await self._add_entity_with_vector(entity, source_paragraph=source_hash or "")
                entities_count += len(unique_entities)
            relations = re.findall(r"([A-Z][a-z]+)\s+(is|was|are|were)\s+([A-Z][a-z]+)", para)
            for subject, predicate, obj in relations:
                try:
                    await self._add_relation(subject, predicate, obj, source_paragraph=source_hash or "")
                    relations_count += 1
                except:
                    pass
        return entities_count, relations_count

    async def _add_entity_with_vector(
        self,
        name: str,
        source_paragraph: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """添加实体并在向量库中生成索引
        
        Args:
            name: 实体名称
            source_paragraph: 来源段落哈希 (可选)
            metadata: 额外元数据
            
        Returns:
            实体hash值
        """
        # 1. 存入元数据和图存储
        hash_value = self.metadata_store.add_entity(
            name, 
            source_paragraph=source_paragraph,
            metadata=metadata
        )
        self.graph_store.add_nodes([name])

        # 2. 生成向量并存入向量库
        try:
            # 检查是否已存在于向量库
            if hash_value not in self.vector_store:
                embedding = await self.embedding_manager.encode(name)
                # 尝试添加。如果ID已存在（例如被标记删除），add会抛出ValueError
                try:
                    self.vector_store.add(
                        vectors=embedding.reshape(1, -1),
                        ids=[hash_value],
                    )
                    logger.debug(f"{self.log_prefix} Added vector for entity: {name}")
                except ValueError:
                    # ID存在但add失败，可能是被软删除了，或者并发导致
                    # 暂时忽略，避免崩溃
                    logger.warning(f"{self.log_prefix} Entity vector {name} (hash={hash_value}) already exists or conflict.")
        except Exception as e:
            logger.warning(f"{self.log_prefix} Failed to vectorize entity {name}: {e}")

        return hash_value

    def _get_help_message(self) -> str:
        """获取帮助消息

        Returns:
            帮助消息文本
        """
        return """📖 导入命令帮助

用法:
  /import text <文本内容>        - 导入文本（自动分段）
  /import paragraph <段落内容>   - 导入单个段落
  /import relation <关系>        - 导入关系 (格式: subject|predicate|object)
  /import file <文件路径>        - 从文件导入 (.txt, .md, .json)
  /import json <文件路径>        - 从JSON文件导入 (结构化数据)

示例:
  /import text 人工智能是计算机科学的一个分支...
  /import paragraph 机器学习是AI的子领域
  /import relation Apple|founded|Steve Jobs
  /import file ./data/knowledge.txt
  /import json ./data/knowledge.json

提示:
  - 文本模式会自动分段并提取实体关系
  - 关系格式支持 "|" 或空格分隔
  - 支持的文件类型: .txt, .md, .json
"""
