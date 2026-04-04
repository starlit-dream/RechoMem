# AGENTS.md — src/storage

三个存储后端，各自对应三层语义漏斗的一个层级。每个文件是独立的异步访问层，通过 `mod.rs` 重导出后挂载到 `AppState`。

## 文件职责

**`sqlite.rs` — L2 精简事实层**

`SqliteStore` 持有 `SqlitePool`，管理两张表：

- `memory_blocks`：完整 block 元数据，含 `raw_offset` / `raw_length` 指针
- `entity_index`：实体反向索引，支持多实体命中查询

公开方法：`connect()` → `init_schema()` → `upsert_block()` / `get_block()` / `search_by_entities()`。  
初始化时通过内联 SQL 建表（无 migrate 文件），`connect()` 内部自动调用 `init_schema()`。

**`jsonl.rs` — L3 原始语境层**

`JsonlStore` 持有文件路径字符串，不保留长期文件句柄（每次操作按需打开）。

- `append_lines()`：追加写入，返回 `(offset, length)` 供 L2 记录
- `read_chunk()`：按 `offset + length` seek 定位后读取字节片段，再解析为 `Vec<RawConversationLine>`

**L3 不可变原则**：JSONL 文件只允许追加，禁止原地修改或截断。

**`vector.rs` — L1 语义索引层**

`VectorStore` 不持有长期连接（每次操作重新 `lancedb::connect()`），避免跨线程 Arc 问题。

- `init(dims)`：若表不存在则建表并建 Auto 索引；已存在则跳过
- `insert(block_id, vector)`：写入一条向量记录
- `search(query_vector, limit)`：ANN 查询，返回命中 `block_id` 列表

向量维度由 `AppConfig.embedding_model` 决定，`text-embedding-3-small` 默认 1536 维。

**`mod.rs`**

仅做 `pub mod` 重导出，不含业务逻辑。

## 关键约束

所有公开方法返回 `crate::error::Result<T>`，禁止在模块边界使用 `unwrap()` 或 `expect()`。

`sqlite.rs` 和 `jsonl.rs` 使用 `tokio::fs` 与 `sqlx` 异步 API，禁止阻塞当前线程。

`vector.rs` 的 Arrow RecordBatch schema 固定为 `(id: Int32, vector: FixedSizeList<Float32>)`，列名不可随意修改，否则 `search()` 的列提取会 panic。

## 与上层的边界

`mcp.rs` 通过 `Arc<AppState>` 持有三个 Store 句柄，不直接操作 SQL 或文件。

`retrieval.rs` 只接受已从存储层查出的 `Vec<SummaryRecord>` 和向量命中列表，不直接依赖 storage 模块。
