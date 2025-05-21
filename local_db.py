import os
import pickle
from typing import List, Optional, Dict, Any, Callable, Union, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
import hnswlib
import faiss


class VectorDatabase:
    """
    A lightweight, production-ready vector database supporting multiple index types:
      - BM25 for text search
      - HNSW (hnswlib) for approximate nearest neighbors
      - FAISS Flat (exact search)
      - FAISS IVF-PQ (quantized search)
    Includes metadata storage, persistence, and simple filtering.
    """

    def __init__(
        self,
        dim: int,
        index_type: str = 'hnsw',  # options: 'hnsw', 'flat', 'ivfpq'
        ef_construction: int = 200,
        M: int = 16,
        ivf_clusters: int = 100,
        pq_code_size: int = 16,
        ef_search: int = 50,
        init_index: bool = True,
    ):
        self.dim = dim
        self.index_type = index_type
        self.ef_search = ef_search
        self._docs: List[str] = []
        self._metas: List[Dict[str, Any]] = []
        self._vectors: np.ndarray = np.zeros((0, dim), dtype='float32')
        self._bm25: Optional[BM25Okapi] = None

        # initialize chosen index
        if index_type == 'hnsw':
            self._index = hnswlib.Index(space='l2', dim=dim)
            # only init if not loading a saved index
            if init_index:
                self._index.init_index(max_elements=1_000_000, ef_construction=ef_construction, M=M)
                self._index.set_ef(ef_search)
        elif index_type == 'flat':
            self._index = faiss.IndexFlatL2(dim)
        elif index_type == 'ivfpq':
            quantizer = faiss.IndexFlatL2(dim)
            self._index = faiss.IndexIVFPQ(quantizer, dim, ivf_clusters, pq_code_size, 8)
            self._index.nprobe = max(1, ivf_clusters // 10)
        else:
            raise ValueError(f"Unsupported index_type '{index_type}'")

    def add(
        self,
        docs: List[str],
        vectors: np.ndarray,
        metas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add new documents with embeddings and optional metadata.
        """
        n = len(docs)
        assert vectors.shape == (n, self.dim), "Shape mismatch"

        start_id = len(self._docs)
        self._docs.extend(docs)
        self._metas.extend(metas or [{} for _ in range(n)])
        self._vectors = np.vstack([self._vectors, vectors.astype('float32')])

        # BM25 rebuild (fast for small corpora; replace with incremental if needed)
        tokenized = [doc.split() for doc in self._docs]
        self._bm25 = BM25Okapi(tokenized)

        # index insert
        if self.index_type == 'hnsw':
            ids = np.arange(start_id, start_id + n)
            self._index.add_items(vectors.astype('float32'), ids)
        else:
            if self.index_type == 'ivfpq' and not self._index.is_trained:
                # For IVFPQ, ensure number of clusters is less than number of vectors
                if isinstance(self._index, faiss.IndexIVFPQ):
                    total_vectors = len(self._vectors)
                    # FAISS requires number of training points >= number of clusters
                    if total_vectors < self._index.nlist:
                        # Set clusters to at most half the dataset size to ensure enough data points per cluster
                        # Use at least 1 cluster but no more than half the data points
                        new_clusters = max(1, min(4, int(total_vectors / 2)))
                        # Recreate the index with fewer clusters
                        quantizer = faiss.IndexFlatL2(self.dim)
                        self._index = faiss.IndexIVFPQ(
                            quantizer, self.dim, new_clusters, 
                            self._index.pq.M, self._index.pq.nbits
                        )
                        self._index.nprobe = max(1, new_clusters // 2)
                
                # For extremely small datasets, FAISS IndexIVFPQ might still fail
                # In that case, fall back to flat index (exact search)
                try:
                    self._index.train(self._vectors)
                except RuntimeError as e:
                    if "Number of training points" in str(e):
                        # Fall back to flat index for very small datasets
                        print(f"Warning: Not enough training data for IVFPQ, falling back to flat index")
                        self._index = faiss.IndexFlatL2(self.dim)
                        self.index_type = 'flat'  # Update index type to reflect change
                    else:
                        # Re-raise if it's a different error
                        raise
            # Add vectors to the index
            self._index.add(vectors.astype('float32'))

    def query_text(self, query: str, top_k: int = 5, return_scores: bool = False) -> Union[List[int], List[Tuple[int, float]]]:
        """
        BM25 text search.
        
        Args:
            query: Text query for search
            top_k: Number of results to return
            return_scores: Whether to return (id, score) tuples instead of just ids
            
        Returns:
            List of document ids or (id, score) tuples if return_scores=True
        """
        if not self._bm25:
            raise RuntimeError("No documents indexed yet.")
        scores = self._bm25.get_scores(query.split())
        sorted_indices = np.argsort(scores)[-top_k:][::-1]
        
        if return_scores:
            return [(int(idx), float(scores[idx])) for idx in sorted_indices]
        else:
            return [int(idx) for idx in sorted_indices]

    def query_vector(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        pre_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
        post_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
        return_scores: bool = False
    ) -> Union[List[int], List[Tuple[int, float]]]:
        """
        Vector similarity search with optional metadata filters.
        
        Args:
            vector: Embedding vector for similarity search
            top_k: Number of results to return
            pre_filter: Function to filter candidate documents before vector search
            post_filter: Function to filter results after vector search
            return_scores: Whether to return (id, score) tuples instead of just ids
            
        Returns:
            List of document ids or (id, score) tuples if return_scores=True
        """
        # Return empty list for empty database
        if len(self._docs) == 0:
            return []
            
        v = vector.reshape(1, -1).astype('float32')
        # optional pre-filter to reduce search space
        if pre_filter:
            valid_ids = [i for i, m in enumerate(self._metas) if pre_filter(m)]
        else:
            valid_ids = None

        if self.index_type == 'hnsw':
            ids, distances = self._index.knn_query(v, k=top_k)
            result_ids = ids[0].tolist()
            scores = [1.0 / (1.0 + d) for d in distances[0].tolist()]  # Convert distance to similarity score
        else:
            distances, indices = self._index.search(v, top_k)
            result_ids = indices[0].tolist()
            scores = [1.0 / (1.0 + d) for d in distances[0].tolist()]  # Convert distance to similarity score

        # filter results using pre-filter list
        if valid_ids is not None:
            if return_scores:
                filtered_results = [(id, score) for id, score in zip(result_ids, scores) if id in valid_ids]
                result_ids = [id for id, _ in filtered_results]
                scores = [score for _, score in filtered_results]
            else:
                result_ids = [i for i in result_ids if i in valid_ids]
                
        # apply post-filter
        if post_filter:
            if return_scores:
                filtered_results = [(id, score) for id, score in zip(result_ids, scores) 
                                   if id < len(self._metas) and post_filter(self._metas[id])]
                result_ids = [id for id, _ in filtered_results]
                scores = [score for _, score in filtered_results]
            else:
                result_ids = [i for i in result_ids if i < len(self._metas) and post_filter(self._metas[i])]
                
        if return_scores:
            return list(zip(result_ids, scores))
        else:
            return result_ids

    def get_document(self, idx: int) -> str:
        """Retrieve stored document text."""
        return self._docs[idx]

    def get_metadata(self, idx: int) -> Dict[str, Any]:
        """Retrieve stored metadata."""
        return self._metas[idx]

    def save(self, folder_path: str) -> None:
        """Persist documents, metadata, vectors, and index files."""
        os.makedirs(folder_path, exist_ok=True)
        # data
        with open(os.path.join(folder_path, 'data.pkl'), 'wb') as f:
            pickle.dump({'docs': self._docs, 'metas': self._metas}, f)
        np.save(os.path.join(folder_path, 'vectors.npy'), self._vectors)

        # index
        if self.index_type == 'hnsw':
            self._index.save_index(os.path.join(folder_path, 'hnsw.idx'))
        else:
            faiss.write_index(self._index, os.path.join(folder_path, 'faiss.idx'))

    @classmethod
    def load(cls, folder_path: str) -> 'VectorDatabase':
        """Load a persisted VectorDatabase from disk."""
        # load data
        with open(os.path.join(folder_path, 'data.pkl'), 'rb') as f:
            data = pickle.load(f)
        vectors = np.load(os.path.join(folder_path, 'vectors.npy'))
        dim = vectors.shape[1]
        
        # detect index type
        idx_files = os.listdir(folder_path)
        if 'hnsw.idx' in idx_files:
            index_type = 'hnsw'
        elif 'faiss.idx' in idx_files:
            # Correctly detect index type from FAISS index file
            faiss_index = faiss.read_index(os.path.join(folder_path, 'faiss.idx'))
            if isinstance(faiss_index, faiss.IndexIVFPQ):
                index_type = 'ivfpq'
            elif isinstance(faiss_index, faiss.IndexFlatL2):
                index_type = 'flat'
            else:
                # Default to flat for any unrecognized FAISS index
                index_type = 'flat'
        else:
            index_type = 'flat'

        # instantiate without initializing HNSW
        inst = cls(dim, index_type=index_type, init_index=False)
        inst._docs = data['docs']
        inst._metas = data['metas']
        inst._vectors = vectors

        # rebuild BM25
        inst._bm25 = BM25Okapi([d.split() for d in inst._docs])
        # load index
        if index_type == 'hnsw':
            # pass max_elements for proper capacity
            inst._index.load_index(os.path.join(folder_path, 'hnsw.idx'), max_elements=len(inst._docs))
            inst._index.set_ef(inst.ef_search)
        else:
            inst._index = faiss.read_index(os.path.join(folder_path, 'faiss.idx'))
        return inst

    def __len__(self) -> int:
        return len(self._docs)

    def __repr__(self) -> str:
        return (
            f"<VectorDatabase dim={self.dim} docs={len(self)} "
            f"index_type={self.index_type}>"
        )
        
    def query_metadata(
        self,
        filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None,
        conditions: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None
    ) -> List[int]:
        """
        Search documents by metadata only without using vector embeddings.

        Args:
            filter_func: Custom function that takes a metadata dict and returns True/False
            conditions: Dict of key-value pairs for exact match filtering
            top_k: Optional limit on number of results returned

        Returns:
            List of document IDs matching the criteria

        Examples:
            # Find documents with specific category
            db.query_metadata(conditions={"category": "finance"})
            
            # Find documents with custom filter function
            db.query_metadata(filter_func=lambda meta: meta.get("priority", 0) > 3)
        """
        if len(self._metas) == 0:
            return []
            
        results = []
        
        # Apply exact match conditions
        if conditions:
            for idx, meta in enumerate(self._metas):
                if all(meta.get(k) == v for k, v in conditions.items()):
                    results.append(idx)
        # Apply custom filter function
        elif filter_func:
            results = [idx for idx, meta in enumerate(self._metas) if filter_func(meta)]
        # Return all if no filters
        else:
            results = list(range(len(self._metas)))
            
        # Apply top_k limit if specified
        if top_k is not None and top_k > 0 and top_k < len(results):
            results = results[:top_k]
            
        return results

    def hybrid_search(
        self,
        query_text: str,
        query_vector: np.ndarray,
        top_k: int = 5,
        vector_weight: float = 0.5,
        filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None,
        return_scores: bool = False
    ) -> List[Union[int, Tuple[int, float]]]:
        """
        Perform a hybrid search combining BM25 text and vector similarity scores.
        
        Args:
            query_text: Text query for BM25 search
            query_vector: Embedding vector for similarity search
            top_k: Number of results to return
            vector_weight: Weight for vector search (0-1), text weight will be (1-vector_weight)
            filter_func: Optional function to filter results by metadata
            return_scores: Whether to return (id, score) tuples instead of just ids
            
        Returns:
            List of document ids or (id, normalized_score) tuples if return_scores=True
        """
        if not self._bm25 or len(self._docs) == 0:
            return []
            
        # Get BM25 scores for all documents
        text_weight = 1.0 - vector_weight
        text_scores = np.array(self._bm25.get_scores(query_text.split()))
        
        # Get vector scores for all documents
        v = query_vector.reshape(1, -1).astype('float32')
        all_vectors = self._vectors
        vector_scores = np.zeros(len(self._docs))
        
        # For large datasets, approximate vector scores using the index
        if len(self._docs) > 1000:
            # Get more results than top_k to ensure good coverage
            k = min(len(self._docs), max(top_k * 10, 100))
            
            if self.index_type == 'hnsw':
                ids, distances = self._index.knn_query(v, k=k)
                for i, idx in enumerate(ids[0]):
                    if idx < len(vector_scores):
                        # Convert distances to scores (similarity = 1/distance)
                        vector_scores[idx] = 1.0 / (1.0 + distances[0][i])
            else:
                # For FAISS indices
                distances, indices = self._index.search(v, k)
                for i, idx in enumerate(indices[0]):
                    if idx >= 0 and idx < len(vector_scores):  # FAISS can return -1 for not enough results
                        vector_scores[idx] = 1.0 / (1.0 + distances[0][i])
        else:
            # For small datasets, compute exact distances for all vectors
            distances = np.linalg.norm(all_vectors - v, axis=1)
            vector_scores = 1.0 / (1.0 + distances)  # Convert to similarity
        
        # Normalize scores
        if len(text_scores) > 0 and text_scores.max() > 0:
            text_scores = text_scores / text_scores.max()
        if len(vector_scores) > 0 and vector_scores.max() > 0:
            vector_scores = vector_scores / vector_scores.max()
        
        # Combine scores
        combined_scores = (vector_weight * vector_scores) + (text_weight * text_scores)
        
        # Apply metadata filter if provided
        if filter_func:
            mask = np.array([filter_func(meta) for meta in self._metas])
            combined_scores = combined_scores * mask
            
        # Get top-k indices
        top_indices = np.argsort(-combined_scores)[:top_k]
        
        # Return with or without scores
        if return_scores:
            return [(int(idx), float(combined_scores[idx])) for idx in top_indices if combined_scores[idx] > 0]
        else:
            return [int(idx) for idx in top_indices if combined_scores[idx] > 0]
