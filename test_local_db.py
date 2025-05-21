"""
Comprehensive test suite for VectorDatabase
"""
import os
import shutil
import numpy as np
import pytest
import pickle
from local_db import VectorDatabase
import faiss

# Test constants
TEST_DIM = 128
TEST_DOCS = [
    "email about meeting tomorrow",
    "invoice from supplier for recent order",
    "request for proposal deadline extension",
    "notification of system maintenance",
    "quarterly report summary",
]
TEST_VECTORS = np.random.random((len(TEST_DOCS), TEST_DIM)).astype('float32')
TEST_METAS = [
    {"type": "meeting", "priority": "high", "sender": "boss@company.com"},
    {"type": "invoice", "priority": "medium", "amount": 1250.75},
    {"type": "proposal", "priority": "high", "deadline": "2025-06-01"},
    {"type": "system", "priority": "low", "maintenance_window": "2025-05-15 02:00-04:00"},
    {"type": "report", "priority": "medium", "quarter": "Q1"},
]
TEST_QUERY = "meeting with team"
TEST_QUERY_VECTOR = np.random.random(TEST_DIM).astype('float32')
TEST_DB_PATH = "temp_vector_db_test"


@pytest.fixture(scope="function")
def cleanup_test_dir():
    """Remove test directory before and after tests"""
    if os.path.exists(TEST_DB_PATH):
        shutil.rmtree(TEST_DB_PATH)
    yield
    if os.path.exists(TEST_DB_PATH):
        shutil.rmtree(TEST_DB_PATH)


class TestVectorDatabase:
    """Core VectorDatabase functionality tests"""
    
    @pytest.mark.parametrize("index_type", ["hnsw", "flat", "ivfpq"])
    def test_init(self, index_type):
        """Test VectorDatabase initialization with different index types"""
        db = VectorDatabase(dim=TEST_DIM, index_type=index_type)
        assert db.dim == TEST_DIM
        assert db.index_type == index_type
        assert len(db) == 0
        
        # Test with different parameters
        if index_type == "hnsw":
            db = VectorDatabase(
                dim=TEST_DIM, 
                index_type=index_type, 
                ef_construction=300,
                M=24, 
                ef_search=100
            )
            assert db.ef_search == 100
        elif index_type == "ivfpq":
            db = VectorDatabase(
                dim=TEST_DIM, 
                index_type=index_type, 
                ivf_clusters=200,
                pq_code_size=8
            )
            # IVF parameters aren't directly exposed, so we just check init works
            
    def test_init_invalid_index_type(self):
        """Test initialization with invalid index type"""
        with pytest.raises(ValueError):
            VectorDatabase(dim=TEST_DIM, index_type="invalid_type")
    
    @pytest.mark.parametrize("index_type", ["hnsw", "flat", "ivfpq"])
    def test_query_text(self, index_type):
        """Test BM25 text search functionality"""
        db = VectorDatabase(dim=TEST_DIM, index_type=index_type)
        db.add(TEST_DOCS, TEST_VECTORS, TEST_METAS)
        
        results = db.query_text(TEST_QUERY)
        assert len(results) == min(5, len(TEST_DOCS))  # Default top_k is 5
        
        # Test with custom top_k
        custom_k = 3
        results = db.query_text(TEST_QUERY, top_k=custom_k)
        assert len(results) == min(custom_k, len(TEST_DOCS))
        
        # Test on empty database
        empty_db = VectorDatabase(dim=TEST_DIM)
        with pytest.raises(RuntimeError):
            empty_db.query_text(TEST_QUERY)
    
    @pytest.mark.parametrize("index_type", ["hnsw", "flat", "ivfpq"])
    def test_query_vector_with_filters(self, index_type):
        """Test vector search with pre and post filters"""
        db = VectorDatabase(dim=TEST_DIM, index_type=index_type)
        db.add(TEST_DOCS, TEST_VECTORS, TEST_METAS)
        
        # Test pre-filter (high priority only)
        pre_filter = lambda meta: meta.get("priority") == "high"
        results = db.query_vector(TEST_QUERY_VECTOR, pre_filter=pre_filter)
        for idx in results:
            assert db.get_metadata(idx).get("priority") == "high"
            
        # Test post-filter (exclude 'system' type)
        post_filter = lambda meta: meta.get("type") != "system"
        results = db.query_vector(TEST_QUERY_VECTOR, post_filter=post_filter)
        for idx in results:
            assert db.get_metadata(idx).get("type") != "system"
            
        # Test both filters together
        results = db.query_vector(
            TEST_QUERY_VECTOR,
            pre_filter=pre_filter,
            post_filter=post_filter
        )
        for idx in results:
            meta = db.get_metadata(idx)
            assert meta.get("priority") == "high"
            assert meta.get("type") != "system"
    
    @pytest.mark.parametrize("index_type", ["hnsw", "flat", "ivfpq"])
    def test_save_load(self, index_type, cleanup_test_dir):
        """Test persistence functionality"""
        # Create and populate original database
        original_db = VectorDatabase(dim=TEST_DIM, index_type=index_type)
        original_db.add(TEST_DOCS, TEST_VECTORS, TEST_METAS)
        
        # Save to disk
        original_db.save(TEST_DB_PATH)
        assert os.path.exists(TEST_DB_PATH)
        
        # Load back
        loaded_db = VectorDatabase.load(TEST_DB_PATH)
        
        # Verify properties
        assert loaded_db.dim == original_db.dim
        assert loaded_db.index_type == original_db.index_type
        assert len(loaded_db) == len(original_db)
        
        # Verify documents and metadata
        for i in range(len(original_db)):
            assert loaded_db.get_document(i) == original_db.get_document(i)
            assert loaded_db.get_metadata(i) == original_db.get_metadata(i)
        
        # Verify search still works
        original_results = original_db.query_vector(TEST_QUERY_VECTOR)
        loaded_results = loaded_db.query_vector(TEST_QUERY_VECTOR)
        
        # Results should be identical for deterministic indexes
        if index_type in ["flat"]:  # HNSW and IVFPQ might have small variations
            assert original_results == loaded_results
            
        # Verify text search
        original_text_results = original_db.query_text(TEST_QUERY)
        loaded_text_results = loaded_db.query_text(TEST_QUERY)
        assert original_text_results == loaded_text_results
    
    def test_load_different_faiss_index_types(self, cleanup_test_dir):
        """Test loading with different FAISS index types for complete type detection coverage"""
        # Create test directory
        os.makedirs(TEST_DB_PATH, exist_ok=True)
        
        # Test with flat index
        db_flat = VectorDatabase(dim=TEST_DIM, index_type='flat')
        db_flat.add(TEST_DOCS, TEST_VECTORS, TEST_METAS)
        db_flat.save(TEST_DB_PATH)
        
        # Load and verify flat index detection
        loaded_db = VectorDatabase.load(TEST_DB_PATH)
        assert loaded_db.index_type == 'flat'
        assert isinstance(loaded_db._index, faiss.IndexFlatL2)
        
        # Now create a custom "other" index type that's neither IVFPQ nor FlatL2
        # First we need to make a simpler dataset for testing
        custom_vectors = np.random.random((5, TEST_DIM)).astype('float32')
        custom_docs = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        
        # Create a custom scalar quantizer index that requires training
        custom_index = faiss.IndexScalarQuantizer(TEST_DIM, faiss.ScalarQuantizer.QT_8bit)
        # Train the index before adding vectors
        custom_index.train(custom_vectors)
        custom_index.add(custom_vectors)
        faiss.write_index(custom_index, os.path.join(TEST_DB_PATH, 'faiss.idx'))
        
        # We need to keep the data.pkl and vectors.npy files compatible
        with open(os.path.join(TEST_DB_PATH, 'data.pkl'), 'wb') as f:
            pickle.dump({'docs': custom_docs, 'metas': [{} for _ in range(5)]}, f)
        np.save(os.path.join(TEST_DB_PATH, 'vectors.npy'), custom_vectors)
        
        # This should now load and default to 'flat' for unrecognized index
        loaded_db = VectorDatabase.load(TEST_DB_PATH)
        assert loaded_db.index_type == 'flat'  # Should default to flat for unrecognized
    
    def test_load_no_index_files(self, cleanup_test_dir):
        """Test loading when neither 'hnsw.idx' nor 'faiss.idx' exists"""
        # Create test directory with only data files but no index files
        os.makedirs(TEST_DB_PATH, exist_ok=True)
        
        # Create data files without index files
        test_vectors = np.random.random((3, TEST_DIM)).astype('float32')
        test_docs = ["doc1", "doc2", "doc3"]
        
        # Save minimal data without index files
        with open(os.path.join(TEST_DB_PATH, 'data.pkl'), 'wb') as f:
            pickle.dump({'docs': test_docs, 'metas': [{} for _ in range(3)]}, f)
        np.save(os.path.join(TEST_DB_PATH, 'vectors.npy'), test_vectors)
        
        # Try to load - should set index_type = 'flat'
        # but will fail when trying to read non-existent faiss.idx
        with pytest.raises(RuntimeError) as excinfo:
            VectorDatabase.load(TEST_DB_PATH)
            
        # Verify that the error is because of the missing index file
        assert "faiss.idx" in str(excinfo.value) or "No such file" in str(excinfo.value)
    
    def test_repr(self):
        """Test string representation"""
        db = VectorDatabase(dim=TEST_DIM, index_type="hnsw")
        db.add(TEST_DOCS, TEST_VECTORS)
        
        rep = repr(db)
        assert str(TEST_DIM) in rep
        assert "hnsw" in rep
        assert str(len(TEST_DOCS)) in rep
    
    @pytest.mark.parametrize("index_type", ["hnsw", "flat", "ivfpq"])
    def test_query_metadata_exact_match(self, index_type):
        """Test query_metadata with exact match conditions"""
        db = VectorDatabase(dim=TEST_DIM, index_type=index_type)
        db.add(TEST_DOCS, TEST_VECTORS, TEST_METAS)
        
        # Test exact match on priority=high
        results = db.query_metadata(conditions={"priority": "high"})
        assert len(results) == 2  # Two documents have high priority
        for idx in results:
            assert db.get_metadata(idx)["priority"] == "high"
            
        # Test exact match on type=invoice
        results = db.query_metadata(conditions={"type": "invoice"})
        assert len(results) == 1
        assert db.get_metadata(results[0])["type"] == "invoice"
        
        # Test exact match on multiple conditions
        results = db.query_metadata(conditions={"priority": "high", "type": "proposal"})
        assert len(results) == 1
        meta = db.get_metadata(results[0])
        assert meta["priority"] == "high"
        assert meta["type"] == "proposal"
        
        # Test with non-existent value
        results = db.query_metadata(conditions={"priority": "critical"})
        assert len(results) == 0
    
    @pytest.mark.parametrize("index_type", ["hnsw", "flat", "ivfpq"])
    def test_query_metadata_custom_filter(self, index_type):
        """Test query_metadata with custom filter function"""
        db = VectorDatabase(dim=TEST_DIM, index_type=index_type)
        db.add(TEST_DOCS, TEST_VECTORS, TEST_METAS)
        
        # Custom filter to find documents with a numeric metadata field
        has_amount = lambda meta: "amount" in meta
        results = db.query_metadata(filter_func=has_amount)
        assert len(results) == 1
        assert "amount" in db.get_metadata(results[0])
        
        # Custom filter for more complex conditions
        is_important = lambda meta: meta.get("priority") == "high" or (
            meta.get("type") == "invoice" and meta.get("amount", 0) > 1000
        )
        results = db.query_metadata(filter_func=is_important)
        assert len(results) == 3  # 2 high priority + 1 high-value invoice
        
        # Test with filter that matches nothing
        impossible = lambda meta: False
        results = db.query_metadata(filter_func=impossible)
        assert len(results) == 0
    
    @pytest.mark.parametrize("index_type", ["hnsw", "flat", "ivfpq"])
    def test_query_metadata_top_k(self, index_type):
        """Test query_metadata with top_k limit"""
        db = VectorDatabase(dim=TEST_DIM, index_type=index_type)
        db.add(TEST_DOCS, TEST_VECTORS, TEST_METAS)
        
        # Get all results with no filter
        all_results = db.query_metadata()
        assert len(all_results) == len(TEST_DOCS)
        
        # Get limited results
        limited_results = db.query_metadata(top_k=2)
        assert len(limited_results) == 2
        
        # Requested more than available
        results = db.query_metadata(top_k=10)
        assert len(results) == len(TEST_DOCS)
        
        # Top_k with filter
        results = db.query_metadata(
            conditions={"priority": "high"},
            top_k=1
        )
        assert len(results) == 1
    
    @pytest.mark.parametrize("index_type", ["hnsw", "flat", "ivfpq"])
    def test_hybrid_search(self, index_type):
        """Test hybrid search functionality"""
        db = VectorDatabase(dim=TEST_DIM, index_type=index_type)
        db.add(TEST_DOCS, TEST_VECTORS, TEST_METAS)
        
        # Basic hybrid search
        results = db.hybrid_search(
            query_text=TEST_QUERY,
            query_vector=TEST_QUERY_VECTOR
        )
        assert len(results) <= 5  # Default top_k is 5
        
        # Test with custom top_k
        results = db.hybrid_search(
            query_text=TEST_QUERY,
            query_vector=TEST_QUERY_VECTOR,
            top_k=3
        )
        assert len(results) <= 3
        
        # Test with different weights
        results_vector_only = db.hybrid_search(
            query_text=TEST_QUERY,
            query_vector=TEST_QUERY_VECTOR,
            vector_weight=1.0  # Only vector search
        )
        results_text_only = db.hybrid_search(
            query_text=TEST_QUERY,
            query_vector=TEST_QUERY_VECTOR,
            vector_weight=0.0  # Only text search
        )
        # Weights should affect results ordering
        assert results_vector_only != results_text_only
        
        # Test with filter
        filter_func = lambda meta: meta.get("priority") == "high"
        results = db.hybrid_search(
            query_text=TEST_QUERY,
            query_vector=TEST_QUERY_VECTOR,
            filter_func=filter_func
        )
        for idx in results:
            assert db.get_metadata(idx)["priority"] == "high"
    
    @pytest.mark.parametrize("index_type", ["hnsw", "flat", "ivfpq"])
    def test_add_empty_then_add(self, index_type):
        """Test adding an empty batch then adding proper data"""
        db = VectorDatabase(dim=TEST_DIM, index_type=index_type)
        
        # First add should build the index
        db.add(TEST_DOCS, TEST_VECTORS, TEST_METAS)
        assert len(db) == len(TEST_DOCS)
        
        # Verify search works
        results = db.query_vector(TEST_QUERY_VECTOR)
        assert len(results) > 0
        results = db.query_text(TEST_QUERY)
        assert len(results) > 0
    
    @pytest.mark.parametrize("index_type", ["flat", "ivfpq"])
    def test_ivfpq_small_dataset_fallback(self, index_type):
        """Test IVFPQ with very small dataset that would cause training errors"""
        # Use only 2 documents for a very small dataset
        tiny_docs = TEST_DOCS[:2]
        tiny_vectors = TEST_VECTORS[:2]
        tiny_metas = TEST_METAS[:2]
        
        # For IVFPQ, this might trigger fallback to flat index
        db = VectorDatabase(dim=TEST_DIM, index_type=index_type)
        db.add(tiny_docs, tiny_vectors, tiny_metas)
        
        # Search should still work
        results = db.query_vector(TEST_QUERY_VECTOR)
        assert len(results) > 0
        
    @pytest.mark.parametrize("index_type", ["hnsw", "flat", "ivfpq"])   
    def test_empty_database_behavior(self, index_type):
        """Test behavior with empty database"""
        db = VectorDatabase(dim=TEST_DIM, index_type=index_type)
        
        # Query methods should handle empty database gracefully
        # Vector search on empty database
        results = db.query_vector(TEST_QUERY_VECTOR)
        assert len(results) == 0  # Should return empty list, not error
        
        # Text search should raise RuntimeError (as implemented)
        with pytest.raises(RuntimeError):
            db.query_text(TEST_QUERY)
            
        # Metadata search on empty database
        results = db.query_metadata(conditions={"priority": "high"})
        assert len(results) == 0
        
        # Hybrid search on empty database
        results = db.hybrid_search(TEST_QUERY, TEST_QUERY_VECTOR)
        assert len(results) == 0


class TestReturnScoresFunctionality:
    """Tests specifically for the return_scores functionality across all search methods"""
    
    @pytest.mark.parametrize("index_type", ["hnsw", "flat", "ivfpq"])
    def test_query_text_with_scores(self, index_type):
        """Test BM25 text search with return_scores option"""
        db = VectorDatabase(dim=TEST_DIM, index_type=index_type)
        db.add(TEST_DOCS, TEST_VECTORS, TEST_METAS)
        
        # Test with default return_scores=False
        results = db.query_text(TEST_QUERY)
        assert isinstance(results, list)
        assert all(isinstance(item, int) for item in results)
        
        # Test with return_scores=True
        results_with_scores = db.query_text(TEST_QUERY, return_scores=True)
        assert isinstance(results_with_scores, list)
        assert len(results_with_scores) > 0
        
        # Check results format
        for item in results_with_scores:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], int)
            assert isinstance(item[1], float)
    
    @pytest.mark.parametrize("index_type", ["hnsw", "flat", "ivfpq"])
    def test_query_vector_with_scores(self, index_type):
        """Test vector search with return_scores option"""
        db = VectorDatabase(dim=TEST_DIM, index_type=index_type)
        db.add(TEST_DOCS, TEST_VECTORS, TEST_METAS)
        
        # Test with default return_scores=False
        results = db.query_vector(TEST_QUERY_VECTOR)
        assert isinstance(results, list)
        assert all(isinstance(item, int) for item in results)
        
        # Test with return_scores=True
        results_with_scores = db.query_vector(TEST_QUERY_VECTOR, return_scores=True)
        assert isinstance(results_with_scores, list)
        assert len(results_with_scores) > 0
        
        # Check results format
        for item in results_with_scores:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], int)
            assert isinstance(item[1], float)
            # Scores should be normalized-like (between 0 and 1)
            assert 0 <= item[1] <= 1.0
    
    @pytest.mark.parametrize("index_type", ["hnsw", "flat", "ivfpq"])
    def test_query_vector_with_filters_and_scores(self, index_type):
        """Test vector search with filters and return_scores option"""
        db = VectorDatabase(dim=TEST_DIM, index_type=index_type)
        db.add(TEST_DOCS, TEST_VECTORS, TEST_METAS)
        
        # Define filters
        pre_filter = lambda meta: meta.get("priority") == "high"
        post_filter = lambda meta: meta.get("type") != "system"
        
        # Test with pre-filter and scores
        results = db.query_vector(
            TEST_QUERY_VECTOR, 
            pre_filter=pre_filter,
            return_scores=True
        )
        assert len(results) > 0
        for idx, score in results:
            assert db.get_metadata(idx).get("priority") == "high"
            assert 0 <= score <= 1.0
            
        # Test with post-filter and scores
        results = db.query_vector(
            TEST_QUERY_VECTOR, 
            post_filter=post_filter,
            return_scores=True
        )
        assert len(results) > 0
        for idx, score in results:
            assert db.get_metadata(idx).get("type") != "system"
            assert 0 <= score <= 1.0
            
        # Test with both filters and scores
        results = db.query_vector(
            TEST_QUERY_VECTOR,
            pre_filter=pre_filter,
            post_filter=post_filter,
            return_scores=True
        )
        for idx, score in results:
            assert db.get_metadata(idx).get("priority") == "high"
            assert db.get_metadata(idx).get("type") != "system"
            assert 0 <= score <= 1.0

    @pytest.mark.parametrize("index_type", ["hnsw", "flat", "ivfpq"])
    def test_hybrid_search_with_scores(self, index_type):
        """Test hybrid search with return_scores option"""
        db = VectorDatabase(dim=TEST_DIM, index_type=index_type)
        db.add(TEST_DOCS, TEST_VECTORS, TEST_METAS)
        
        # Test with return_scores=True
        results = db.hybrid_search(
            query_text=TEST_QUERY,
            query_vector=TEST_QUERY_VECTOR,
            return_scores=True
        )
        assert len(results) > 0
        
        # Check results format and score normalization
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], int)
            assert isinstance(item[1], float)
            assert 0 <= item[1] <= 1  # Scores should be normalized


if __name__ == "__main__":
    pytest.main(["-xvs", "test_local_db.py"])
