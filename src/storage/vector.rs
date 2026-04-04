use std::sync::Arc;

use arrow_array::types::Float32Type;
use arrow_array::{Array, FixedSizeListArray, RecordBatch, StringArray};
use arrow_schema::DataType;
use futures::TryStreamExt;
use lancedb::index::Index;
use lancedb::query::{ExecutableQuery, QueryBase};

use crate::error::{RechoMemError, Result};

#[derive(Clone)]
pub struct VectorStore {
    db_path: String,
    table_name: String,
}

impl VectorStore {
    pub fn new(db_path: impl Into<String>, table_name: impl Into<String>) -> Self {
        Self {
            db_path: db_path.into(),
            table_name: table_name.into(),
        }
    }

    async fn connect(&self) -> Result<lancedb::Connection> {
        Ok(lancedb::connect(&self.db_path).execute().await?)
    }

    pub async fn init(&self, dims: i32) -> Result<()> {
        let db = self.connect().await?;
        if db
            .table_names()
            .execute()
            .await?
            .iter()
            .any(|name| name == &self.table_name)
        {
            let table = db.open_table(&self.table_name).execute().await?;
            let schema = table.schema().await?;
            let vector_field = schema
                .field_with_name("vector")
                .map_err(|e| RechoMemError::Internal(e.to_string()))?;

            match vector_field.data_type() {
                DataType::FixedSizeList(_, existing_dims) if *existing_dims == dims => {
                    return Ok(())
                }
                DataType::FixedSizeList(_, existing_dims) => {
                    return Err(RechoMemError::Config(format!(
                        "vector dimension mismatch for existing LanceDB table: expected {}, got {}",
                        dims, existing_dims
                    )));
                }
                other => {
                    return Err(RechoMemError::Internal(format!(
                        "unexpected vector column type: {other:?}"
                    )));
                }
            }
        }

        let batch = RecordBatch::try_from_iter(vec![
            (
                "block_id",
                Arc::new(StringArray::from(vec!["__init__"])) as Arc<dyn arrow_array::Array>,
            ),
            (
                "vector",
                Arc::new(
                    FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                        vec![Some(vec![Some(0.0_f32); dims as usize])],
                        dims,
                    ),
                ) as Arc<dyn arrow_array::Array>,
            ),
        ])
        .map_err(|e| RechoMemError::Internal(e.to_string()))?;

        let table = db.create_table(&self.table_name, batch).execute().await?;
        if dims >= 256 {
            table
                .create_index(&["vector"], Index::Auto)
                .execute()
                .await?;
        }
        Ok(())
    }

    pub async fn insert(&self, block_id: &str, vector: &[f32]) -> Result<()> {
        let db = self.connect().await?;
        let table = db.open_table(&self.table_name).execute().await?;
        let dims = i32::try_from(vector.len())
            .map_err(|_| RechoMemError::Internal("vector dimension overflow".to_string()))?;

        let batch = RecordBatch::try_from_iter(vec![
            (
                "block_id",
                Arc::new(StringArray::from(vec![block_id])) as Arc<dyn arrow_array::Array>,
            ),
            (
                "vector",
                Arc::new(
                    FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                        vec![Some(vector.iter().copied().map(Some).collect::<Vec<_>>())],
                        dims,
                    ),
                ) as Arc<dyn arrow_array::Array>,
            ),
        ])
        .map_err(|e| RechoMemError::Internal(e.to_string()))?;

        table.add(batch).execute().await?;
        Ok(())
    }

    pub async fn search(&self, query_vector: Vec<f32>, limit: usize) -> Result<Vec<String>> {
        let db = self.connect().await?;
        let table = db.open_table(&self.table_name).execute().await?;
        let query = table.query();
        let vector_query = query.nearest_to(query_vector.as_slice())?.limit(limit);
        let stream = vector_query.execute().await?;
        let batches: Vec<RecordBatch> = stream.try_collect().await?;

        let mut block_ids = Vec::new();
        for batch in batches {
            if let Some(array) = batch.column_by_name("block_id") {
                let strings = array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| {
                        RechoMemError::Internal("block_id column type mismatch".to_string())
                    })?;
                for index in 0..strings.len() {
                    let value = strings.value(index);
                    if value != "__init__" {
                        block_ids.push(value.to_string());
                    }
                }
            }
        }
        Ok(block_ids)
    }
}
