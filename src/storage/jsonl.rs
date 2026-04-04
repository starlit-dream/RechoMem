use std::path::Path;
use std::sync::Arc;

use tokio::fs::OpenOptions;
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt, BufWriter, SeekFrom};
use tokio::sync::Mutex;

use crate::error::{RechoMemError, Result};
use crate::types::{RawConversationLine, RawLogChunk, StructuredRawLogChunk};

#[derive(Clone)]
pub struct JsonlStore {
    path: String,
    write_lock: Arc<Mutex<()>>,
}

impl JsonlStore {
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            write_lock: Arc::new(Mutex::new(())),
        }
    }

    pub async fn ensure_exists(&self) -> Result<()> {
        if let Some(parent) = Path::new(&self.path).parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .await?;

        Ok(())
    }

    pub async fn append_lines(
        &self,
        block_id: &str,
        lines: &[RawConversationLine],
    ) -> Result<(u64, u64)> {
        self.ensure_exists().await?;
        let _guard = self.write_lock.lock().await;

        let mut encoded_block = Vec::new();
        for line in lines {
            let value = serde_json::json!({
                "block_id": block_id,
                "timestamp": line.timestamp,
                "role": line.role,
                "content": line.content,
            });
            let mut encoded = serde_json::to_vec(&value)?;
            encoded.push(b'\n');
            encoded_block.extend_from_slice(&encoded);
        }

        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&self.path)
            .await?;

        let start = file.metadata().await?.len();
        let mut writer = BufWriter::new(file);
        writer.write_all(&encoded_block).await?;
        writer.flush().await?;
        let length = encoded_block.len() as u64;
        Ok((start, length))
    }

    pub async fn read_chunk(
        &self,
        block_id: &str,
        offset: u64,
        length: u64,
    ) -> Result<RawLogChunk> {
        self.ensure_exists().await?;
        let mut file = OpenOptions::new().read(true).open(&self.path).await?;
        file.seek(SeekFrom::Start(offset)).await?;

        let mut buffer = vec![0_u8; length as usize];
        file.read_exact(&mut buffer).await?;

        Ok(RawLogChunk {
            block_id: block_id.to_string(),
            content: String::from_utf8_lossy(&buffer).to_string(),
            raw_offset: offset,
            raw_length: length,
        })
    }

    pub async fn read_structured_chunk(
        &self,
        block_id: &str,
        offset: u64,
        length: u64,
    ) -> Result<StructuredRawLogChunk> {
        let chunk = self.read_chunk(block_id, offset, length).await?;
        let mut lines = Vec::new();

        for line in chunk.content.lines().filter(|line| !line.trim().is_empty()) {
            let value: serde_json::Value = serde_json::from_str(line)?;
            let stored_block_id = value
                .get("block_id")
                .and_then(|value| value.as_str())
                .ok_or_else(|| {
                    RechoMemError::Internal("missing block_id field in raw log line".to_string())
                })?;
            if stored_block_id != block_id {
                return Err(RechoMemError::Internal(format!(
                    "raw log block_id mismatch: expected {block_id}, got {stored_block_id}"
                )));
            }
            lines.push(RawConversationLine {
                timestamp: serde_json::from_value(value.get("timestamp").cloned().ok_or_else(
                    || {
                        RechoMemError::Internal(
                            "missing timestamp field in raw log line".to_string(),
                        )
                    },
                )?)?,
                role: value
                    .get("role")
                    .and_then(|value| value.as_str())
                    .ok_or_else(|| {
                        RechoMemError::Internal("missing role field in raw log line".to_string())
                    })?
                    .to_string(),
                content: value
                    .get("content")
                    .and_then(|value| value.as_str())
                    .ok_or_else(|| {
                        RechoMemError::Internal("missing content field in raw log line".to_string())
                    })?
                    .to_string(),
            });
        }

        Ok(StructuredRawLogChunk {
            block_id: chunk.block_id,
            lines,
            raw_offset: chunk.raw_offset,
            raw_length: chunk.raw_length,
        })
    }
}
