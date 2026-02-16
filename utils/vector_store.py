import numpy as np
import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Singleton: load the embedding model once, reuse forever
_embedding_model: SentenceTransformer | None = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model: all-MiniLM-L6-v2 (first time)")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    else:
        logger.info("Reusing cached embedding model")
    return _embedding_model


class SchemaVectorStore:
    """
    Lightweight in-memory vector store for schema search.
    Uses sentence-transformers directly with cosine similarity.
    No external vector DB needed for a small schema.
    """

    def __init__(self):
        self.model = _get_embedding_model()
        self.documents: List[Document] = []
        self.embeddings: np.ndarray | None = None

    def initialize_from_schema(self, schema_info):
        """
        schema_info: { 'table_name': {'columns': ['col TYPE', ...], 'foreign_keys': [...]}, ... }
        Also accepts legacy format: { 'table_name': ['col1', ...], ... }
        """
        self.documents = []
        for table, info in schema_info.items():
            if isinstance(info, dict):
                cols = info.get("columns", [])
                fks = info.get("foreign_keys", [])
                content = f"Table: {table}\nColumns: {', '.join(cols)}"
                if fks:
                    content += f"\nForeign Keys: {', '.join(fks)}"
            else:
                content = f"Table: {table}\nColumns: {', '.join(info)}"
            self.documents.append(Document(page_content=content, metadata={"table": table}))

        self._encode_documents()

    def initialize_from_schema_text(self, schema_text: Dict[str, str]):
        """
        schema_text: { 'table_name': 'pre-formatted description string', ... }
        Use this when you have rich, human-authored table descriptions.
        """
        self.documents = []
        for table, text in schema_text.items():
            self.documents.append(Document(page_content=text, metadata={"table": table}))

        self._encode_documents()

    def _encode_documents(self):
        """Encode all documents into embeddings."""
        texts = [doc.page_content for doc in self.documents]
        self.embeddings = self.model.encode(texts, normalize_embeddings=True)
        logger.info(f"Indexed {len(self.documents)} schema documents in memory")

    def search(self, query: str, k: int = 3) -> List[Document]:
        """
        Search for relevant schema components using vector similarity.
        
        Args:
            query: User's natural language query
            k: Number of top results to return
            
        Returns:
            List of Document objects with relevant schema components
        """
        if not self.documents or self.embeddings is None:
            logger.warning("Vector store is empty — run initialize_from_schema first")
            return []

        query_embedding = self.model.encode([query], normalize_embeddings=True)
        # Cosine similarity (embeddings are already L2-normalized)
        scores = np.dot(self.embeddings, query_embedding.T).flatten()
        top_k = min(k, len(self.documents))
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = [self.documents[i] for i in top_indices]
        
        # Log retrieval results
        table_names = [doc.metadata.get("table", "unknown") for doc in results]
        logger.info(f"Vector search for '{query[:50]}...' returned {len(results)} tables: {table_names}")
        
        return results

    def search_tables(self, query: str, k: int = 3) -> List[str]:
        """
        Search for relevant table names only.
        
        Args:
            query: User's natural language query
            k: Number of top results to return
            
        Returns:
            List of table names
        """
        results = self.search(query, k=k)
        return [doc.metadata.get("table") for doc in results if doc.metadata.get("table")]
