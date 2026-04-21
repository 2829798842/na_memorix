# na_memorix

`na_memorix` 是一个基于 A_memorix 特化为 Nekro Agent 的知识库/记忆插件。它保留了核心检索与图谱处理链路，但将运行方式、存储后端和宿主集成方式改成了更适合 Nekro 插件体系的实现。

## Web 界面入口

- [打开主面板](/plugins/litroenade.na_memorix/)
- [打开导入中心](/plugins/litroenade.na_memorix/import)
- [打开检索调优](/plugins/litroenade.na_memorix/tuning)
- [项目仓库](https://github.com/litroenade/na_memorix)



## 它能做什么

- 导入文本知识、聊天总结、实体与关系数据。
- 执行语义检索、时间检索、实体检索和关系检索。
- 维护记忆图谱，支持边权调整、冻结、恢复、强化、保护等操作。
- 为 Agent 自动注入记忆上下文。
- 按每日固定时间批量总结活跃频道新增聊天记录，并写入长期记忆。
- 在导入中心一键迁移 Nekro-Agent 已存储的原生记忆和历史聊天记录。
- 普通文本上传/粘贴可选启用 LLM 实体关系抽取，将资料写入知识图谱。
- 提供人物画像查询、覆盖和注册表管理。
- 提供 Web 可视化界面，用于浏览图谱、查看来源、管理记忆和触发重建索引。
- 暴露兼容的 `/api/*` 与 `/v1/*` 接口，便于前端和旧调用链继续工作。

## 运行策略

`na_memorix` 默认不启用插件休眠。

原因是它承担的是长期记忆与知识库基础能力：自动记忆注入依赖 prompt inject，`memorix_search`、`memorix_status` 等主动检索工具依赖 sandbox methods prompt。若插件进入休眠状态，宿主只会展示插件简述，不会渲染记忆注入内容和工具说明，模型就无法稳定获得相关记忆或主动调用检索工具。

关闭休眠的代价是每轮对话会多一小段插件提示和工具说明。可以通过 `AUTO_INJECT_TOP_K`、`AUTO_INJECT_MIN_SCORE`、聊天过滤配置和全局开关控制注入规模。

### 存储后端
 
 
- 图快照：持久化到 PostgreSQL 图表；本地旧图文件仅作为兼容迁移输入
- 稀疏检索默认后端：PostgreSQL

### 外部依赖来源

- PostgreSQL 连接由宿主侧数据库配置解析，`MetadataStore` 通过 `resolve_db_url()` 建立连接。
- Qdrant 连接使用 Nekro 的向量库配置。
- Embedding/OpenAI 兼容接口配置来自插件 runtime 配置与环境变量合并结果，兼容：
  - `[embedding.openapi]`
  - `[embedding.openai]`
  - `OPENAPI_*`
  - `OPENAI_*`

### 动态依赖与运行前置条件

- 宿主 `nekro-agent` 已自带 `openai`、`psycopg2-binary`、`qdrant-client` 等核心依赖，`na_memorix` 直接复用，不重复做动态安装。
- 插件额外接入了 Nekro 的 `dynamic_import_pkg(...)` 动态依赖导入机制，当前按需注入的依赖有：
  - `scipy`：图存储首次初始化时尝试安装。
  - `jieba`：中文检索首次进入 `jieba`/`mixed` 分词路径时尝试安装。
  - `sentence-transformers`：本地 embedding 模型首次加载时尝试安装。
- 如果宿主无法访问配置的 PyPI 镜像，或动态安装失败，`/api/ui_capabilities` 与主面板会直接提示依赖缺失原因；硬依赖未就绪时，导入中心和检索调优页会自动退化到兼容说明模式。
- 若需要手动预装，请确保宿主 Python 环境至少可导入 `scipy`、`jieba`、`sentence_transformers`，并允许插件进程写入 Nekro 的动态包目录。

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
- `SUMMARIZATION_TIMEOUT_SECONDS`
  - 单次聊天总结模型请求超时时间，默认 `60` 秒，独立于 Embedding 超时。
- `GRAPH_EXTRACTION_TIMEOUT_SECONDS`
  - 上传或粘贴文本启用 LLM 抽取时，单次实体/关系抽取请求超时时间，默认 `60` 秒。
- `SUMMARIZATION_CONTEXT_LENGTH`
  - 手动总结和定时总结每次读取的聊天消息窗口大小。
- `SCHEDULED_SUMMARY_ENABLED`
  - 是否启用定时批量总结。该功能会调用总结模型并消耗 token，默认关闭。
- `SCHEDULED_SUMMARY_TIMES`
  - 每日触发时间列表，每行一个 `HH:MM`，例如 `04:00`。
- `SCHEDULED_SUMMARY_CHAT_LIMIT`
  - 每次定时总结最多处理多少个有新增文本消息的活跃频道。
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

## 导入与清理说明

- 导入中心支持上传、粘贴、自动迁移记忆和时序回填。
- “启用 LLM 抽取”会对普通文本分块调用聊天模型抽取实体与关系，可能消耗较多 token；若只需要段落检索，可关闭该选项。
- “自动迁移记忆”会复用 Nekro-Agent 已存储的原生记忆和聊天记录；聊天总结会调用总结模型，重复执行会根据游标增量推进。
- 清理误导入资料时，优先按来源删除，例如 `upload:xxx.txt`、`paste:xxx`、`chat_summary:{chat_key}`。这会清理对应段落、向量以及孤立关系/图边，不会删除宿主原始聊天记录。
- 若需要完全重置 na_memorix，需要同时清空插件数据、向量、图谱和自动迁移游标；不要直接删除宿主数据库里的聊天记录或 Nekro-Agent 原生记忆表。

## 前端说明

- 前端静态资源位于 [web](./web) 目录。
- 页面由插件路由挂载，不应假定自己运行在站点根路径。
- 前端请求应始终使用插件相对路径，避免在宿主前缀下出现路径错误。

## 特别感谢

- [ARC](https://github.com/A-Dawn)
- [A_memorix](https://github.com/A-Dawn/A_memorix/tree/basic)

## 许可证

本项目遵循 [AGPLv3 License](LICENSE)。
