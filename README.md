# na_memorix

`na_memorix` 是一个基于 A_memorix 特化为 Nekro Agent 的知识库/记忆插件。它保留了核心检索与图谱处理链路，但将运行方式、存储后端和宿主集成方式改成了更适合 Nekro 插件体系的实现。

## 它能做什么

- 导入文本知识、聊天总结、实体与关系数据。
- 执行语义检索、时间检索、实体检索和关系检索。
- 维护记忆图谱，支持边权调整、冻结、恢复、强化、保护等操作。
- 为 Agent 自动注入记忆上下文。
- 提供人物画像查询、覆盖和注册表管理。
- 提供 Web 可视化界面，用于浏览图谱、查看来源、管理记忆和触发重建索引。
- 暴露兼容的 `/api/*` 与 `/v1/*` 接口，便于前端和旧调用链继续工作。

## Web 界面入口

- [打开统一入口页](/plugins/litroenade.na_memorix/launcher)
- [打开主面板](/plugins/litroenade.na_memorix/)
- [打开导入中心](/plugins/litroenade.na_memorix/import)
- [打开检索调优](/plugins/litroenade.na_memorix/tuning)
- [项目仓库](https://github.com/2829798842/na_memorix)

### 存储后端

- 元数据后端：PostgreSQL
- 向量后端：Qdrant
- 图快照：持久化到 PostgreSQL 图表；本地旧图文件仅作为兼容迁移输入
- 稀疏检索默认后端：PostgreSQL

相关实现见：

- [core/storage/metadata_store.py](./core/storage/metadata_store.py)
- [core/storage/vector_store.py](./core/storage/vector_store.py)
- [amemorix/bootstrap.py](./amemorix/bootstrap.py)

### 外部依赖来源

- PostgreSQL 连接由宿主侧数据库配置解析，`MetadataStore` 通过 `resolve_db_url()` 建立连接。
- Qdrant 连接使用 Nekro 的向量库配置。
- Embedding/OpenAI 兼容接口配置来自插件 runtime 配置与环境变量合并结果，兼容：
  - `[embedding.openapi]`
  - `[embedding.openai]`
  - `OPENAPI_*`
  - `OPENAI_*`

### 插件数据目录

插件运行时目录位于：

```text
data/plugin_data/{plugin.key}/runtime/
```

这里主要保存本地运行时缓存、图文件兼容数据和向量目录等插件专属文件。

## 关键配置项

插件配置面定义在 [plugin.py](./plugin.py) 中，常用项如下：

- `GLOBAL_MEMORY_ENABLED`
  - 全局总开关，影响自动注入、后台维护、人物画像刷新等主流程。
- `AUTO_INJECT_ENABLED`
  - 是否允许自动向 Agent 注入记忆。
- `AUTO_INJECT_TOP_K`
  - 自动注入时最多返回多少条结果。
- `AUTO_INJECT_MIN_SCORE`
  - 自动注入时的最低分数阈值。
- `CHAT_FILTER_ENABLED`
  - 是否启用聊天过滤。
- `CHAT_FILTER_MODE`
  - 聊天过滤模式，支持 `whitelist` 与 `blacklist`。
- `CHAT_FILTER_CHATS`
  - 过滤目标列表，支持 `stream:`、`group:`、`user:` 前缀。
- `EMBEDDING_MODEL_GROUP`
  - Embedding 模型组名。
- `EMBEDDING_DIMENSION`
  - Embedding 维度。
- `SUMMARIZATION_MODEL_GROUP`
  - 聊天总结模型组名。
- `PERSON_PROFILE_ENABLED`
  - 是否启用人物画像。
- `WEB_READ_ONLY`
  - Web UI 是否只读。
- `CHUNK_COLLECTION_NAME`
  - Qdrant 段落集合名。
- `RELATION_COLLECTION_NAME`
  - Qdrant 关系集合名。
- `TABLE_PREFIX`
  - PostgreSQL 表名前缀。

## 前端说明

- 前端静态资源位于 [web](./web) 目录。
- 页面由插件路由挂载，不应假定自己运行在站点根路径。
- 前端请求应始终使用插件相对路径，避免在宿主前缀下出现路径错误。

## 特别感谢

- [ARC](https://github.com/A-Dawn)
- [A_memorix](https://github.com/A-Dawn/A_memorix/tree/basic)

## 许可证

本项目遵循 [AGPLv3 License](LICENSE)。
