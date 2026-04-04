use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;

use chrono::{DateTime, Utc};
use sqlx::{
    sqlite::{SqliteConnectOptions, SqlitePoolOptions},
    Pool, Row, Sqlite,
};

use crate::error::{RechoMemError, Result};
use crate::types::{MemoryBlock, SummaryRecord};

fn normalize_entity(entity: &str) -> String {
    entity
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase()
}

#[derive(Clone)]
pub struct SqliteStore {
    pool: Pool<Sqlite>,
}

impl SqliteStore {
    pub fn database_url(path: impl AsRef<Path>) -> String {
        path.as_ref().to_string_lossy().replace('\\', "/")
    }

    pub async fn connect(database_url: &str) -> Result<Self> {
        let connect_options = if database_url.starts_with("sqlite:") {
            SqliteConnectOptions::from_str(database_url)?
        } else {
            SqliteConnectOptions::new()
                .filename(database_url)
                .create_if_missing(true)
        };

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(connect_options)
            .await?;

        let store = Self { pool };
        store.init().await?;
        Ok(store)
    }

    async fn init(&self) -> Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS memory_blocks (
                block_id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                summary TEXT NOT NULL,
                entities_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                raw_offset INTEGER NOT NULL,
                raw_length INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS entity_index (
                entity TEXT NOT NULL,
                block_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (entity, block_id),
                FOREIGN KEY (block_id) REFERENCES memory_blocks(block_id)
            );
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_entity_index_entity ON entity_index(entity);")
            .execute(&self.pool)
            .await?;

        sqlx::query("ALTER TABLE memory_blocks ADD COLUMN status TEXT NOT NULL DEFAULT 'pending';")
            .execute(&self.pool)
            .await
            .ok();

        Ok(())
    }

    pub async fn upsert_block(&self, block: &MemoryBlock) -> Result<()> {
        let entities_json = serde_json::to_string(&block.entities)?;
        let mut tx = self.pool.begin().await?;

        sqlx::query(
            r#"
            INSERT INTO memory_blocks (
                block_id, topic, summary, entities_json, status, raw_offset, raw_length, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(block_id) DO UPDATE SET
                topic = excluded.topic,
                summary = excluded.summary,
                entities_json = excluded.entities_json,
                status = excluded.status,
                raw_offset = excluded.raw_offset,
                raw_length = excluded.raw_length,
                updated_at = excluded.updated_at
            "#,
        )
        .bind(&block.block_id)
        .bind(&block.topic)
        .bind(&block.summary)
        .bind(&entities_json)
        .bind(&block.status)
        .bind(block.raw_offset as i64)
        .bind(block.raw_length as i64)
        .bind(block.created_at.to_rfc3339())
        .bind(block.updated_at.to_rfc3339())
        .execute(&mut *tx)
        .await?;

        sqlx::query("DELETE FROM entity_index WHERE block_id = ?")
            .bind(&block.block_id)
            .execute(&mut *tx)
            .await?;

        for entity in &block.entities {
            sqlx::query(
                "INSERT OR IGNORE INTO entity_index (entity, block_id, created_at) VALUES (?, ?, ?)",
            )
            .bind(normalize_entity(entity))
            .bind(&block.block_id)
            .bind(Utc::now().to_rfc3339())
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;
        Ok(())
    }

    pub async fn get_block(&self, block_id: &str) -> Result<MemoryBlock> {
        let row = sqlx::query(
            r#"
            SELECT block_id, topic, summary, entities_json, status, raw_offset, raw_length, created_at, updated_at
            FROM memory_blocks
            WHERE block_id = ? AND status = 'ready'
            "#,
        )
        .bind(block_id)
        .fetch_optional(&self.pool)
        .await?
        .ok_or_else(|| RechoMemError::NotFound(block_id.to_string()))?;

        Self::row_to_block(&row)
    }

    pub async fn search_by_entities(
        &self,
        entities: &[String],
        limit: usize,
    ) -> Result<Vec<SummaryRecord>> {
        if entities.is_empty() {
            return Ok(Vec::new());
        }

        let placeholders = vec!["?"; entities.len()].join(", ");
        let sql = format!(
            r#"
            SELECT mb.block_id, mb.topic, mb.summary, mb.entities_json, mb.created_at
            FROM entity_index ei
            JOIN memory_blocks mb ON mb.block_id = ei.block_id
            WHERE ei.entity IN ({})
              AND mb.status = 'ready'
            GROUP BY mb.block_id, mb.topic, mb.summary, mb.entities_json, mb.created_at
            ORDER BY COUNT(*) DESC, mb.updated_at DESC
            LIMIT ?
            "#,
            placeholders
        );

        let mut query = sqlx::query(&sql);
        for entity in entities {
            query = query.bind(normalize_entity(entity));
        }
        query = query.bind(limit as i64);

        let rows = query.fetch_all(&self.pool).await?;
        rows.iter().map(Self::row_to_summary).collect()
    }

    pub async fn get_blocks_by_ids(&self, block_ids: &[String]) -> Result<Vec<SummaryRecord>> {
        if block_ids.is_empty() {
            return Ok(Vec::new());
        }

        let placeholders = vec!["?"; block_ids.len()].join(", ");
        let sql = format!(
            r#"
            SELECT block_id, topic, summary, entities_json, created_at
            FROM memory_blocks
            WHERE block_id IN ({}) AND status = 'ready'
            "#,
            placeholders
        );

        let mut query = sqlx::query(&sql);
        for block_id in block_ids {
            query = query.bind(block_id);
        }

        let rows = query.fetch_all(&self.pool).await?;
        let by_id: HashMap<String, SummaryRecord> = rows
            .iter()
            .map(Self::row_to_summary)
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .map(|record| (record.block_id.clone(), record))
            .collect();

        Ok(block_ids
            .iter()
            .filter_map(|block_id| by_id.get(block_id).cloned())
            .collect())
    }

    fn row_to_block(row: &sqlx::sqlite::SqliteRow) -> Result<MemoryBlock> {
        let entities_json: String = row.try_get("entities_json")?;
        let created_at: String = row.try_get("created_at")?;
        let updated_at: String = row.try_get("updated_at")?;

        Ok(MemoryBlock {
            block_id: row.try_get("block_id")?,
            topic: row.try_get("topic")?,
            summary: row.try_get("summary")?,
            entities: serde_json::from_str(&entities_json)?,
            status: row.try_get("status")?,
            raw_offset: row.try_get::<i64, _>("raw_offset")? as u64,
            raw_length: row.try_get::<i64, _>("raw_length")? as u64,
            created_at: DateTime::<Utc>::from_str(&created_at)
                .map_err(|e| RechoMemError::Internal(e.to_string()))?,
            updated_at: DateTime::<Utc>::from_str(&updated_at)
                .map_err(|e| RechoMemError::Internal(e.to_string()))?,
        })
    }

    fn row_to_summary(row: &sqlx::sqlite::SqliteRow) -> Result<SummaryRecord> {
        let entities_json: String = row.try_get("entities_json")?;
        let created_at: String = row.try_get("created_at")?;

        Ok(SummaryRecord {
            block_id: row.try_get("block_id")?,
            topic: row.try_get("topic")?,
            summary: row.try_get("summary")?,
            entities: serde_json::from_str(&entities_json)?,
            created_at: DateTime::<Utc>::from_str(&created_at)
                .map_err(|e| RechoMemError::Internal(e.to_string()))?,
        })
    }
}
