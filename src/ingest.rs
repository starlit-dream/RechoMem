use chrono::Utc;
use uuid::Uuid;

use crate::embedding::EmbeddingService;
use crate::error::{RechoMemError, Result};
use crate::storage::jsonl::JsonlStore;
use crate::storage::sqlite::SqliteStore;
use crate::storage::vector::VectorStore;
use crate::types::{
    IngestConversationRequest, IngestMemoryBlockRequest, IngestMemoryBlockResult, MemoryBlock,
};

#[derive(Clone)]
pub struct IngestionService {
    jsonl: JsonlStore,
    sqlite: SqliteStore,
    vector: VectorStore,
    embedding: EmbeddingService,
}

impl IngestionService {
    pub fn new(
        jsonl: JsonlStore,
        sqlite: SqliteStore,
        vector: VectorStore,
        embedding: EmbeddingService,
    ) -> Self {
        Self {
            jsonl,
            sqlite,
            vector,
            embedding,
        }
    }

    pub async fn ingest_block(
        &self,
        request: IngestMemoryBlockRequest,
    ) -> Result<IngestMemoryBlockResult> {
        validate_prepared_block(&request)?;
        let block_id = Uuid::new_v4().to_string();
        let created_at = Utc::now();
        let embedding_input = format!(
            "{}\n\n{}\n\n{}",
            request.topic,
            request.summary,
            request.entities.join(" ")
        );
        let vector = self.embedding.embed_text(&embedding_input).await?;
        let (raw_offset, raw_length) = self
            .jsonl
            .append_lines(&block_id, &request.raw_lines)
            .await?;

        let block = MemoryBlock {
            block_id: block_id.clone(),
            topic: request.topic,
            summary: request.summary,
            entities: request.entities,
            status: "pending".to_string(),
            raw_offset,
            raw_length,
            created_at,
            updated_at: created_at,
        };

        self.sqlite.upsert_block(&block).await?;
        self.vector.insert(&block_id, &vector).await?;
        let ready_at = Utc::now();
        self.sqlite
            .upsert_block(&MemoryBlock {
                status: "ready".to_string(),
                updated_at: ready_at,
                ..block.clone()
            })
            .await?;

        Ok(IngestMemoryBlockResult {
            block_id,
            raw_offset,
            raw_length,
            created_at,
        })
    }

    pub async fn ingest_conversation(
        &self,
        request: IngestConversationRequest,
    ) -> Result<IngestMemoryBlockResult> {
        if request.raw_lines.is_empty() {
            return Err(RechoMemError::InvalidRequest(
                "raw_lines must contain at least one conversation line".to_string(),
            ));
        }

        let generated = self
            .embedding
            .summarize_lines(&request.raw_lines, request.topic_hint.as_deref())
            .await?;

        self.ingest_block(IngestMemoryBlockRequest {
            topic: generated.topic,
            summary: generated.summary,
            entities: generated.entities,
            raw_lines: request.raw_lines,
        })
        .await
    }
}

fn validate_prepared_block(request: &IngestMemoryBlockRequest) -> Result<()> {
    if request.topic.trim().is_empty() {
        return Err(RechoMemError::InvalidRequest(
            "topic must not be empty".to_string(),
        ));
    }
    if request.summary.trim().is_empty() {
        return Err(RechoMemError::InvalidRequest(
            "summary must not be empty".to_string(),
        ));
    }
    if request.raw_lines.is_empty() {
        return Err(RechoMemError::InvalidRequest(
            "raw_lines must contain at least one conversation line".to_string(),
        ));
    }
    if request
        .raw_lines
        .iter()
        .any(|line| line.role.trim().is_empty() || line.content.trim().is_empty())
    {
        return Err(RechoMemError::InvalidRequest(
            "each raw line must include non-empty role and content".to_string(),
        ));
    }
    Ok(())
}
