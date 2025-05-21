# Homemade Lightweight Vector DB

Lightweight but production-ready with multiple indexing options, designed for quick protyping and understanding. The below readme is written by an LLM so please don't be mad!

## Features

- **Multiple Index Types**:
  - BM25 for traditional text search
  - HNSW (Hierarchical Navigable Small World) for fast approximate nearest neighbors search
  - FAISS Flat for exact vector search
  - FAISS IVF-PQ for quantized search with lower memory footprint

- **Flexible Search Capabilities**:
  - Text-based search (BM25)
  - Vector similarity search
  - Metadata filtering
  - Hybrid search (combined text + vector)

- **Production-Ready Features**:
  - Persistence (save and load indexes)
  - Automatic index selection based on dataset size
  - Graceful fallback for very small datasets
  - Metadata storage and filtering
  - Comprehensive documentation

## Why LightVectorDB?

Most projects don't need a complex distributed vector database with high operational overhead. LightVectorDB provides:

- **Simplicity**: Single Python file with no complex setup or infrastructure
- **Speed**: Optimized for typical workloads (thousands to millions of vectors)
- **Flexibility**: Multiple index types and search methods in one unified API
- **Persistence**: Easy save/load functionality for your vector data
- **Low Dependencies**: Minimal external library requirements

## Requirements

```
numpy
rank_bm25
hnswlib
faiss-cpu  # or faiss-gpu for GPU acceleration
```

## Installation

```bash
pip install numpy rank_bm25 hnswlib faiss-cpu
```

Then simply copy the `local_db.py` file to your project.

## Quick Start

```python
import numpy as np
from local_db import VectorDatabase

# Create a vector database with 384-dimensional vectors using HNSW index
db = VectorDatabase(dim=384, index_type='hnsw')

# Add documents with their vector embeddings and metadata
docs = ["This is a sample document", "Another example text"]
vectors = np.random.rand(2, 384).astype('float32')  # Your embedding vectors
metadata = [{"category": "sample", "id": 1}, {"category": "example", "id": 2}]

db.add(docs=docs, vectors=vectors, metas=metadata)

# Vector similarity search
query_vector = np.random.rand(384).astype('float32')
results = db.query_vector(query_vector, top_k=5)

# Text search
text_results = db.query_text("sample document", top_k=5)

# Metadata filtering
filtered_results = db.query_metadata(conditions={"category": "sample"})

# Hybrid search (text + vector)
hybrid_results = db.hybrid_search(
    query_text="sample",
    query_vector=query_vector,
    vector_weight=0.7,
    top_k=5
)

# Save the database
db.save("my_vector_db")

# Load the database later
loaded_db = VectorDatabase.load("my_vector_db")
```

## Index Types

- **HNSW** (default): Best for general-purpose vector search with good balance of speed and accuracy
- **Flat**: Exact search, highest accuracy but slower for large collections
- **IVFPQ**: Quantized search for larger datasets with lower memory usage

## Advanced Usage

### Metadata Filtering

```python
# Pre-filter (reduces search space before vector search)
results = db.query_vector(
    vector=query_vector,
    pre_filter=lambda meta: meta.get("date") > "2023-01-01"
)

# Post-filter (filters results after vector search)
results = db.query_vector(
    vector=query_vector,
    post_filter=lambda meta: "important" in meta.get("tags", [])
)
```

### Hybrid Search with Custom Weights

```python
# Prioritize vector similarity (80%) over text matching (20%)
results = db.hybrid_search(
    query_text="machine learning",
    query_vector=embedding,
    vector_weight=0.8,
    top_k=10
)
```

## Performance Considerations

- HNSW works best for most use cases up to a few million vectors
- For very small datasets (<100 vectors), Flat index may outperform other options
- For large datasets (>1M vectors), consider IVF-PQ to reduce memory usage

## Limitations

LightVectorDB has some important limitations to be aware of:

1. **No Update or Delete Operations**: The database doesn't support modifying or removing existing items after insertion. If you need to update or delete data, you would need to rebuild the database.

2. **Memory-bound Operation**: All vectors and documents are held in memory, limiting scalability for very large datasets.

3. **Basic Tokenization**: BM25 text search uses simple whitespace tokenization (`doc.split()`), which may not be optimal for all languages or specialized text.

4. **No Automatic Scaling**: Unlike distributed databases, there's no automatic sharding or clustering capabilities.

5. **Single-thread Operation**: Operations are not parallelized, which may impact performance for large batch operations.

6. **Limited Metadata Query Capabilities**: Metadata queries support exact matching only, with no range queries, complex aggregations, or JSON path queries.

7. **No Incremental BM25 Updates**: Adding documents rebuilds the entire BM25 index rather than incrementally updating it.

8. **No Transaction Support**: There are no ACID guarantees or transactional operations.

9. **Manual Index Parameters**: You must manually select appropriate parameters like `ef_construction`, `M`, `ivf_clusters`, etc., based on your data.

10. **Limited Persistence Options**: Saving and loading can only be done to local filesystem with a simple pickle-based approach.

For production systems with high scalability requirements or needs for advanced features like updates/deletes, consider more complex solutions like Pinecone, Weaviate, or Elasticsearch with vector search capabilities.

## License
I don't know but it is where anyone can use it without my permission. I swear I won't say anything.

`MIT`

## Contributing

Contributions, comments and Bugs resolution are welcome! Please feel free to submit a Pull Request.
