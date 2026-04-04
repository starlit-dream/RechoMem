use std::sync::Arc;

use rmcp::handler::server::{router::tool::ToolRouter, wrapper::Parameters, ServerHandler};
use rmcp::model::{Implementation, ServerInfo};
use rmcp::schemars;
use rmcp::service::ServiceExt;
use rmcp::{tool_handler, Json};
use serde::Deserialize;
use tracing::warn;
use uuid::Uuid;

use crate::embedding::EmbeddingService;
use crate::error::RechoMemError;
use crate::ingest::IngestionService;
use crate::storage::jsonl::JsonlStore;
use crate::storage::sqlite::SqliteStore;
use crate::storage::vector::VectorStore;
use crate::types::{
    IngestConversationRequest, IngestMemoryBlockRequest, IngestMemoryBlockResult, RawLogChunk,
    SearchResultsResponse, StructuredRawLogChunk, SummaryRecord,
};

#[derive(Clone)]
pub struct AppState {
    pub sqlite: SqliteStore,
    pub jsonl: JsonlStore,
    pub vector: VectorStore,
    pub embedding: EmbeddingService,
    pub ingestion: IngestionService,
}

#[derive(Clone)]
pub struct RechoMemMcp {
    pub state: Arc<AppState>,
    tool_router: ToolRouter<Self>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SearchArgs {
    pub query: String,
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct BlockArgs {
    pub block_id: String,
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for RechoMemMcp {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::default()
            .with_server_info(
                Implementation::new("rechomem", env!("CARGO_PKG_VERSION"))
                    .with_title("RechoMem"),
            )
            .with_instructions(
                "Three-layer semantic memory MCP server for search, summaries, raw log drill-down, and block ingestion."
                    .to_string(),
            )
    }
}

#[rmcp::tool_router(router = tool_router)]
impl RechoMemMcp {
    pub fn new(state: Arc<AppState>) -> Self {
        Self {
            state,
            tool_router: Self::tool_router(),
        }
    }

    #[rmcp::tool(
        name = "search_memory_index",
        description = "Search memory summaries by text query and entity hints"
    )]
    pub async fn search_memory_index(
        &self,
        Parameters(args): Parameters<SearchArgs>,
    ) -> std::result::Result<Json<SearchResultsResponse>, rmcp::ErrorData> {
        if args.query.trim().is_empty() {
            return Err(invalid_params("query must not be empty"));
        }

        let limit = validated_limit(args.limit)?;
        let entities = extract_entities(&args.query);
        let entity_hits = self
            .state
            .sqlite
            .search_by_entities(&entities, limit)
            .await
            .map_err(internal_error)?;

        let semantic_records = match self.state.embedding.embed_text(&args.query).await {
            Ok(query_vector) => {
                let vector_hit_ids = match self.state.vector.search(query_vector, limit).await {
                    Ok(ids) => ids,
                    Err(error) => {
                        warn!("vector search failed during search_memory_index: {error}");
                        Vec::new()
                    }
                };

                match self.state.sqlite.get_blocks_by_ids(&vector_hit_ids).await {
                    Ok(records) => records
                        .into_iter()
                        .map(|record| {
                            let semantic_score =
                                semantic_score_for_block(&record.block_id, &vector_hit_ids);
                            (record, semantic_score)
                        })
                        .collect::<Vec<_>>(),
                    Err(error) => {
                        warn!("sqlite hydration failed during search_memory_index: {error}");
                        Vec::new()
                    }
                }
            }
            Err(error) => {
                warn!("embedding generation failed during search_memory_index: {error}");
                Vec::new()
            }
        };

        Ok(Json(SearchResultsResponse {
            results: crate::retrieval::rerank_results(semantic_records, entity_hits, limit),
        }))
    }

    #[rmcp::tool(
        name = "read_context_summary",
        description = "Read one memory block summary by block id"
    )]
    pub async fn read_context_summary(
        &self,
        Parameters(args): Parameters<BlockArgs>,
    ) -> std::result::Result<Json<SummaryRecord>, rmcp::ErrorData> {
        let block_id = validate_block_id(&args.block_id)?;
        let block = self
            .state
            .sqlite
            .get_block(&block_id)
            .await
            .map_err(map_domain_error)?;

        Ok(Json(SummaryRecord {
            block_id: block.block_id,
            topic: block.topic,
            summary: block.summary,
            entities: block.entities,
            created_at: block.created_at,
        }))
    }

    #[rmcp::tool(
        name = "drill_down_raw_logs",
        description = "Drill down into raw JSONL logs by block id"
    )]
    pub async fn drill_down_raw_logs(
        &self,
        Parameters(args): Parameters<BlockArgs>,
    ) -> std::result::Result<Json<RawLogChunk>, rmcp::ErrorData> {
        let block_id = validate_block_id(&args.block_id)?;
        let block = self
            .state
            .sqlite
            .get_block(&block_id)
            .await
            .map_err(map_domain_error)?;

        self.state
            .jsonl
            .read_chunk(&block.block_id, block.raw_offset, block.raw_length)
            .await
            .map(Json)
            .map_err(internal_error)
    }

    #[rmcp::tool(
        name = "drill_down_raw_logs_structured",
        description = "Read one raw JSONL block as structured conversation lines"
    )]
    pub async fn drill_down_raw_logs_structured(
        &self,
        Parameters(args): Parameters<BlockArgs>,
    ) -> std::result::Result<Json<StructuredRawLogChunk>, rmcp::ErrorData> {
        let block_id = validate_block_id(&args.block_id)?;
        let block = self
            .state
            .sqlite
            .get_block(&block_id)
            .await
            .map_err(map_domain_error)?;

        self.state
            .jsonl
            .read_structured_chunk(&block.block_id, block.raw_offset, block.raw_length)
            .await
            .map(Json)
            .map_err(internal_error)
    }

    #[rmcp::tool(
        name = "ingest_memory_block",
        description = "Ingest one prepared memory block into JSONL, SQLite, and LanceDB"
    )]
    pub async fn ingest_memory_block(
        &self,
        Parameters(args): Parameters<IngestMemoryBlockRequest>,
    ) -> std::result::Result<Json<IngestMemoryBlockResult>, rmcp::ErrorData> {
        self.state
            .ingestion
            .ingest_block(args)
            .await
            .map(Json)
            .map_err(map_domain_error)
    }

    #[rmcp::tool(
        name = "ingest_conversation",
        description = "Summarize raw conversation lines and ingest the resulting memory block"
    )]
    pub async fn ingest_conversation(
        &self,
        Parameters(args): Parameters<IngestConversationRequest>,
    ) -> std::result::Result<Json<IngestMemoryBlockResult>, rmcp::ErrorData> {
        self.state
            .ingestion
            .ingest_conversation(args)
            .await
            .map(Json)
            .map_err(map_domain_error)
    }
}

pub async fn run_stdio_server(service: RechoMemMcp) -> std::result::Result<(), RechoMemError> {
    let transport = (tokio::io::stdin(), tokio::io::stdout());
    let running = service
        .serve(transport)
        .await
        .map_err(|e| crate::error::RechoMemError::Internal(e.to_string()))?;
    running
        .waiting()
        .await
        .map_err(|e| crate::error::RechoMemError::Internal(e.to_string()))?;
    Ok(())
}

fn extract_entities(query: &str) -> Vec<String> {
    query
        .split_whitespace()
        .filter(|token| token.len() > 1)
        .map(|token| {
            token
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|token| !token.is_empty())
        .collect()
}

fn semantic_score_for_block(block_id: &str, vector_hit_ids: &[String]) -> f32 {
    vector_hit_ids
        .iter()
        .position(|candidate| candidate == block_id)
        .map(|index| 1.0 / (1.0 + index as f32))
        .unwrap_or(0.0)
}

fn internal_error(error: impl std::fmt::Display) -> rmcp::ErrorData {
    rmcp::ErrorData::internal_error(error.to_string(), None)
}

fn invalid_params(message: impl Into<String>) -> rmcp::ErrorData {
    rmcp::ErrorData::invalid_params(message.into(), None)
}

fn map_domain_error(error: RechoMemError) -> rmcp::ErrorData {
    match error {
        RechoMemError::InvalidRequest(message) => rmcp::ErrorData::invalid_params(message, None),
        RechoMemError::NotFound(message) => {
            rmcp::ErrorData::invalid_params(format!("missing record: {message}"), None)
        }
        other => internal_error(other),
    }
}

fn validated_limit(limit: Option<usize>) -> std::result::Result<usize, rmcp::ErrorData> {
    let value = limit.unwrap_or(5);
    if value == 0 || value > 50 {
        return Err(invalid_params("limit must be between 1 and 50"));
    }
    Ok(value)
}

fn validate_block_id(block_id: &str) -> std::result::Result<String, rmcp::ErrorData> {
    let normalized = block_id.trim();
    if normalized.is_empty() {
        return Err(invalid_params("block_id must not be empty"));
    }
    if Uuid::parse_str(normalized).is_err() {
        return Err(invalid_params("block_id must be a valid UUID"));
    }
    Ok(normalized.to_string())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use chrono::Utc;
    use rmcp::handler::server::wrapper::Parameters;
    use rmcp::model::{CallToolRequestParams, ClientInfo};
    use rmcp::{ClientHandler, ServiceExt};

    use super::{AppState, BlockArgs, RechoMemMcp, SearchArgs};
    use crate::embedding::EmbeddingService;
    use crate::error::RechoMemError;
    use crate::ingest::IngestionService;
    use crate::storage::jsonl::JsonlStore;
    use crate::storage::sqlite::SqliteStore;
    use crate::storage::vector::VectorStore;
    use crate::types::{IngestConversationRequest, IngestMemoryBlockRequest, RawConversationLine};

    #[derive(Debug, Clone, Default)]
    struct DummyClientHandler;

    impl ClientHandler for DummyClientHandler {
        fn get_info(&self) -> ClientInfo {
            ClientInfo::default()
        }
    }

    #[tokio::test]
    async fn ingest_search_and_drill_down_round_trip() -> anyhow::Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let sqlite_path = temp_dir.path().join("rechomem.sqlite");
        let lancedb_path = temp_dir.path().join("lancedb");
        let raw_log_path = temp_dir.path().join("raw_logs.jsonl");

        tokio::fs::create_dir_all(&lancedb_path).await?;

        let sqlite_url = SqliteStore::database_url(&sqlite_path);
        let sqlite = SqliteStore::connect(&sqlite_url).await?;
        let jsonl = JsonlStore::new(raw_log_path.to_string_lossy().to_string());
        jsonl.ensure_exists().await?;
        let vector = VectorStore::new(lancedb_path.to_string_lossy().to_string(), "memory_index");
        vector.init(8).await?;
        let embedding = EmbeddingService::fake(8);
        let ingestion = IngestionService::new(
            jsonl.clone(),
            sqlite.clone(),
            vector.clone(),
            embedding.clone(),
        );

        let state = Arc::new(AppState {
            sqlite,
            jsonl,
            vector,
            embedding,
            ingestion,
        });
        let service = RechoMemMcp::new(state);

        let ingest_result = service
            .ingest_memory_block(Parameters(IngestMemoryBlockRequest {
                topic: "OpenAI integration".to_string(),
                summary: "OpenAI embeddings are now stored and searchable".to_string(),
                entities: vec!["OpenAI".to_string(), "Embeddings".to_string()],
                raw_lines: vec![
                    RawConversationLine {
                        timestamp: Utc::now(),
                        role: "user".to_string(),
                        content: "Please store the OpenAI embedding migration notes.".to_string(),
                    },
                    RawConversationLine {
                        timestamp: Utc::now(),
                        role: "assistant".to_string(),
                        content: "Stored the embeddings migration notes for future search."
                            .to_string(),
                    },
                ],
            }))
            .await?
            .0;

        let search_results = service
            .search_memory_index(Parameters(SearchArgs {
                query: "openai embeddings".to_string(),
                limit: Some(5),
            }))
            .await?
            .0
            .results;

        assert!(
            !search_results.is_empty(),
            "search should return at least one result"
        );
        assert_eq!(search_results[0].block_id, ingest_result.block_id);

        let summary = service
            .read_context_summary(Parameters(BlockArgs {
                block_id: ingest_result.block_id.clone(),
            }))
            .await?
            .0;
        assert_eq!(summary.topic, "OpenAI integration");
        assert!(summary.entities.iter().any(|entity| entity == "OpenAI"));

        let raw_chunk = service
            .drill_down_raw_logs(Parameters(BlockArgs {
                block_id: ingest_result.block_id,
            }))
            .await?
            .0;
        assert!(raw_chunk
            .content
            .contains("OpenAI embedding migration notes"));
        assert!(raw_chunk
            .content
            .contains("Stored the embeddings migration notes"));

        Ok(())
    }

    #[tokio::test]
    async fn drill_down_reads_exact_block_after_multiple_ingests() -> anyhow::Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let sqlite_path = temp_dir.path().join("rechomem.sqlite");
        let lancedb_path = temp_dir.path().join("lancedb");
        let raw_log_path = temp_dir.path().join("raw_logs.jsonl");

        tokio::fs::create_dir_all(&lancedb_path).await?;

        let sqlite = SqliteStore::connect(&SqliteStore::database_url(&sqlite_path)).await?;
        let jsonl = JsonlStore::new(raw_log_path.to_string_lossy().to_string());
        jsonl.ensure_exists().await?;
        let vector = VectorStore::new(lancedb_path.to_string_lossy().to_string(), "memory_index");
        vector.init(8).await?;
        let embedding = EmbeddingService::fake(8);
        let ingestion = IngestionService::new(
            jsonl.clone(),
            sqlite.clone(),
            vector.clone(),
            embedding.clone(),
        );

        let state = Arc::new(AppState {
            sqlite,
            jsonl,
            vector,
            embedding,
            ingestion,
        });
        let service = RechoMemMcp::new(state);

        let first = service
            .ingest_memory_block(Parameters(IngestMemoryBlockRequest {
                topic: "First".to_string(),
                summary: "First summary".to_string(),
                entities: vec!["Alpha".to_string()],
                raw_lines: vec![RawConversationLine {
                    timestamp: Utc::now(),
                    role: "user".to_string(),
                    content: "first block unique marker".to_string(),
                }],
            }))
            .await?
            .0;

        let second = service
            .ingest_memory_block(Parameters(IngestMemoryBlockRequest {
                topic: "Second".to_string(),
                summary: "Second summary".to_string(),
                entities: vec!["Beta".to_string()],
                raw_lines: vec![RawConversationLine {
                    timestamp: Utc::now(),
                    role: "assistant".to_string(),
                    content: "second block unique marker".to_string(),
                }],
            }))
            .await?
            .0;

        let first_chunk = service
            .drill_down_raw_logs(Parameters(BlockArgs {
                block_id: first.block_id,
            }))
            .await?
            .0;
        let second_chunk = service
            .drill_down_raw_logs(Parameters(BlockArgs {
                block_id: second.block_id,
            }))
            .await?
            .0;

        assert!(first_chunk.content.contains("first block unique marker"));
        assert!(!first_chunk.content.contains("second block unique marker"));
        assert!(second_chunk.content.contains("second block unique marker"));
        assert!(!second_chunk.content.contains("first block unique marker"));

        Ok(())
    }

    #[tokio::test]
    async fn rmcp_transport_round_trip_ingest_and_search() -> anyhow::Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let sqlite_path = temp_dir.path().join("rechomem.sqlite");
        let lancedb_path = temp_dir.path().join("lancedb");
        let raw_log_path = temp_dir.path().join("raw_logs.jsonl");

        tokio::fs::create_dir_all(&lancedb_path).await?;

        let sqlite = SqliteStore::connect(&SqliteStore::database_url(&sqlite_path)).await?;
        let jsonl = JsonlStore::new(raw_log_path.to_string_lossy().to_string());
        jsonl.ensure_exists().await?;
        let vector = VectorStore::new(lancedb_path.to_string_lossy().to_string(), "memory_index");
        vector.init(8).await?;
        let embedding = EmbeddingService::fake(8);
        let ingestion = IngestionService::new(
            jsonl.clone(),
            sqlite.clone(),
            vector.clone(),
            embedding.clone(),
        );

        let state = Arc::new(AppState {
            sqlite,
            jsonl,
            vector,
            embedding,
            ingestion,
        });
        let server = RechoMemMcp::new(state);

        let (server_transport, client_transport) = tokio::io::duplex(4096);
        let server_handle = tokio::spawn(async move {
            server.serve(server_transport).await?.waiting().await?;
            anyhow::Ok(())
        });

        let client = DummyClientHandler.serve(client_transport).await?;

        let ingest_response = client
            .call_tool(
                CallToolRequestParams::new("ingest_memory_block").with_arguments(
                    serde_json::json!({
                        "topic": "Protocol test",
                        "summary": "Protocol summary",
                        "entities": ["OpenAI,"],
                        "raw_lines": [{
                            "timestamp": Utc::now().to_rfc3339(),
                            "role": "user",
                            "content": "protocol round trip marker"
                        }]
                    })
                    .as_object()
                    .unwrap()
                    .clone(),
                ),
            )
            .await?;
        assert!(ingest_response.structured_content.is_some());

        let search_response = client
            .call_tool(
                CallToolRequestParams::new("search_memory_index").with_arguments(
                    serde_json::json!({
                        "query": "openai",
                        "limit": 5
                    })
                    .as_object()
                    .unwrap()
                    .clone(),
                ),
            )
            .await?;

        let structured = search_response
            .structured_content
            .expect("search should return structured content");
        let results = structured
            .get("results")
            .and_then(|value| value.as_array())
            .expect("results array should exist");
        assert!(
            !results.is_empty(),
            "transport search should return results"
        );

        client.cancel().await?;
        server_handle.await??;
        Ok(())
    }

    #[tokio::test]
    async fn ingest_conversation_generates_summary_and_structured_logs() -> anyhow::Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let sqlite_path = temp_dir.path().join("rechomem.sqlite");
        let lancedb_path = temp_dir.path().join("lancedb");
        let raw_log_path = temp_dir.path().join("raw_logs.jsonl");

        tokio::fs::create_dir_all(&lancedb_path).await?;

        let sqlite = SqliteStore::connect(&SqliteStore::database_url(&sqlite_path)).await?;
        let jsonl = JsonlStore::new(raw_log_path.to_string_lossy().to_string());
        jsonl.ensure_exists().await?;
        let vector = VectorStore::new(lancedb_path.to_string_lossy().to_string(), "memory_index");
        vector.init(8).await?;
        let embedding = EmbeddingService::fake(8);
        let ingestion = IngestionService::new(
            jsonl.clone(),
            sqlite.clone(),
            vector.clone(),
            embedding.clone(),
        );

        let state = Arc::new(AppState {
            sqlite,
            jsonl,
            vector,
            embedding,
            ingestion,
        });
        let service = RechoMemMcp::new(state);

        let ingest_result = service
            .ingest_conversation(Parameters(IngestConversationRequest {
                topic_hint: Some("Release planning".to_string()),
                raw_lines: vec![
                    RawConversationLine {
                        timestamp: Utc::now(),
                        role: "user".to_string(),
                        content: "Please capture the release planning notes for OpenAI migration."
                            .to_string(),
                    },
                    RawConversationLine {
                        timestamp: Utc::now(),
                        role: "assistant".to_string(),
                        content: "Recorded the migration tasks and owners for the next release."
                            .to_string(),
                    },
                ],
            }))
            .await?
            .0;

        let summary = service
            .read_context_summary(Parameters(BlockArgs {
                block_id: ingest_result.block_id.clone(),
            }))
            .await?
            .0;
        assert_eq!(summary.topic, "Release planning");
        assert!(summary.summary.contains("release") || summary.summary.contains("Recorded"));

        let structured = service
            .drill_down_raw_logs_structured(Parameters(BlockArgs {
                block_id: ingest_result.block_id,
            }))
            .await?
            .0;
        assert_eq!(structured.lines.len(), 2);
        assert_eq!(structured.lines[0].role, "user");
        assert!(structured.lines[1].content.contains("migration tasks"));

        Ok(())
    }

    #[tokio::test]
    async fn invalid_params_are_mapped_for_search_and_ingest() -> anyhow::Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let sqlite_path = temp_dir.path().join("rechomem.sqlite");
        let lancedb_path = temp_dir.path().join("lancedb");
        let raw_log_path = temp_dir.path().join("raw_logs.jsonl");

        tokio::fs::create_dir_all(&lancedb_path).await?;

        let sqlite = SqliteStore::connect(&SqliteStore::database_url(&sqlite_path)).await?;
        let jsonl = JsonlStore::new(raw_log_path.to_string_lossy().to_string());
        jsonl.ensure_exists().await?;
        let vector = VectorStore::new(lancedb_path.to_string_lossy().to_string(), "memory_index");
        vector.init(8).await?;
        let embedding = EmbeddingService::fake(8);
        let ingestion = IngestionService::new(
            jsonl.clone(),
            sqlite.clone(),
            vector.clone(),
            embedding.clone(),
        );

        let state = Arc::new(AppState {
            sqlite,
            jsonl,
            vector,
            embedding,
            ingestion,
        });
        let service = RechoMemMcp::new(state);

        let search_error = match service
            .search_memory_index(Parameters(SearchArgs {
                query: " ".to_string(),
                limit: Some(0),
            }))
            .await
        {
            Ok(_) => panic!("empty query should be rejected"),
            Err(error) => error,
        };
        assert_eq!(
            search_error.code,
            rmcp::ErrorData::invalid_params("x", None).code
        );

        let invalid_block_error = match service
            .read_context_summary(Parameters(BlockArgs {
                block_id: "not-a-uuid".to_string(),
            }))
            .await
        {
            Ok(_) => panic!("invalid block id should be rejected"),
            Err(error) => error,
        };
        assert_eq!(
            invalid_block_error.code,
            rmcp::ErrorData::invalid_params("x", None).code
        );
        assert!(invalid_block_error.message.contains("valid UUID"));

        let ingest_error = match service
            .ingest_memory_block(Parameters(IngestMemoryBlockRequest {
                topic: " ".to_string(),
                summary: "summary".to_string(),
                entities: vec![],
                raw_lines: vec![RawConversationLine {
                    timestamp: Utc::now(),
                    role: "user".to_string(),
                    content: "content".to_string(),
                }],
            }))
            .await
        {
            Ok(_) => panic!("empty topic should be rejected"),
            Err(error) => error,
        };
        assert_eq!(
            ingest_error.code,
            rmcp::ErrorData::invalid_params("x", None).code
        );

        let domain_error = service
            .state
            .ingestion
            .ingest_conversation(IngestConversationRequest {
                topic_hint: None,
                raw_lines: vec![],
            })
            .await
            .expect_err("empty conversation should be rejected");
        assert!(matches!(domain_error, RechoMemError::InvalidRequest(_)));

        Ok(())
    }

    #[tokio::test]
    async fn missing_block_maps_to_invalid_params() -> anyhow::Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let sqlite_path = temp_dir.path().join("rechomem.sqlite");
        let lancedb_path = temp_dir.path().join("lancedb");
        let raw_log_path = temp_dir.path().join("raw_logs.jsonl");

        tokio::fs::create_dir_all(&lancedb_path).await?;

        let sqlite = SqliteStore::connect(&SqliteStore::database_url(&sqlite_path)).await?;
        let jsonl = JsonlStore::new(raw_log_path.to_string_lossy().to_string());
        jsonl.ensure_exists().await?;
        let vector = VectorStore::new(lancedb_path.to_string_lossy().to_string(), "memory_index");
        vector.init(8).await?;
        let embedding = EmbeddingService::fake(8);
        let ingestion = IngestionService::new(
            jsonl.clone(),
            sqlite.clone(),
            vector.clone(),
            embedding.clone(),
        );

        let state = Arc::new(AppState {
            sqlite,
            jsonl,
            vector,
            embedding,
            ingestion,
        });
        let service = RechoMemMcp::new(state);
        let missing_id = uuid::Uuid::new_v4().to_string();

        let error = match service
            .read_context_summary(Parameters(BlockArgs {
                block_id: missing_id,
            }))
            .await
        {
            Ok(_) => panic!("missing block should fail"),
            Err(error) => error,
        };

        assert_eq!(error.code, rmcp::ErrorData::invalid_params("x", None).code);
        assert!(error.message.contains("missing record"));

        Ok(())
    }

    #[tokio::test]
    async fn block_id_with_surrounding_spaces_is_accepted() -> anyhow::Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let sqlite_path = temp_dir.path().join("rechomem.sqlite");
        let lancedb_path = temp_dir.path().join("lancedb");
        let raw_log_path = temp_dir.path().join("raw_logs.jsonl");

        tokio::fs::create_dir_all(&lancedb_path).await?;

        let sqlite = SqliteStore::connect(&SqliteStore::database_url(&sqlite_path)).await?;
        let jsonl = JsonlStore::new(raw_log_path.to_string_lossy().to_string());
        jsonl.ensure_exists().await?;
        let vector = VectorStore::new(lancedb_path.to_string_lossy().to_string(), "memory_index");
        vector.init(8).await?;
        let embedding = EmbeddingService::fake(8);
        let ingestion = IngestionService::new(
            jsonl.clone(),
            sqlite.clone(),
            vector.clone(),
            embedding.clone(),
        );

        let state = Arc::new(AppState {
            sqlite,
            jsonl,
            vector,
            embedding,
            ingestion,
        });
        let service = RechoMemMcp::new(state);

        let ingest_result = service
            .ingest_memory_block(Parameters(IngestMemoryBlockRequest {
                topic: "Trimmed block".to_string(),
                summary: "Trimmed block summary".to_string(),
                entities: vec!["Trim".to_string()],
                raw_lines: vec![RawConversationLine {
                    timestamp: Utc::now(),
                    role: "user".to_string(),
                    content: "trim validation marker".to_string(),
                }],
            }))
            .await?
            .0;

        let summary = service
            .read_context_summary(Parameters(BlockArgs {
                block_id: format!("  {}  ", ingest_result.block_id),
            }))
            .await?
            .0;

        assert_eq!(summary.topic, "Trimmed block");
        Ok(())
    }

    #[tokio::test]
    async fn structured_raw_logs_reject_mismatched_block_id() -> anyhow::Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let raw_log_path = temp_dir.path().join("raw_logs.jsonl");
        let jsonl = JsonlStore::new(raw_log_path.to_string_lossy().to_string());
        jsonl.ensure_exists().await?;

        let (offset, length) = jsonl
            .append_lines(
                "expected-block",
                &[RawConversationLine {
                    timestamp: Utc::now(),
                    role: "user".to_string(),
                    content: "raw marker".to_string(),
                }],
            )
            .await?;

        let error = jsonl
            .read_structured_chunk("wrong-block", offset, length)
            .await
            .expect_err("mismatched block id should fail");
        assert!(error.to_string().contains("raw log block_id mismatch"));

        Ok(())
    }
}
