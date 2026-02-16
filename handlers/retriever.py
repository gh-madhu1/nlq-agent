from typing import List, Dict, Any
from utils.vector_store import SchemaVectorStore
import logging

logger = logging.getLogger(__name__)


class SchemaRetriever:
    """
    Schema retriever using vector search to find relevant tables and columns.
    Returns filtered schema information for the planning stage.
    """

    def __init__(self, vector_store: SchemaVectorStore, schema_info: Dict[str, Dict[str, Any]]):
        self.vector_store = vector_store
        self.schema_info = schema_info

    def retrieve_relevant_schema(self, query: str, k: int = 5) -> str:
        """
        Retrieve relevant schema as formatted text (backward compatibility).
        
        Args:
            query: User's natural language query
            k: Number of schema components to retrieve
            
        Returns:
            Formatted string with relevant schema components
        """
        results = self.vector_store.search(query, k=k)
        schema_context = "Relevant Schema Components:\n"
        for doc in results:
            schema_context += f"- {doc.page_content}\n"
        return schema_context

    def get_relevant_schema_dict(self, query: str, k: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve relevant schema as a filtered dictionary for the planner.
        
        Args:
            query: User's natural language query
            k: Number of schema components to retrieve
            
        Returns:
            Filtered schema_info dict containing only relevant tables
        """
        # Get relevant tables from vector search
        results = self.vector_store.search(query, k=k)
        relevant_tables = set()
        
        for doc in results:
            table_name = doc.metadata.get("table")
            if table_name:
                relevant_tables.add(table_name)
        
        # Filter schema_info to only include relevant tables
        filtered_schema = {
            table: info 
            for table, info in self.schema_info.items() 
            if table in relevant_tables
        }
        
        logger.info(f"Retrieved {len(filtered_schema)} relevant tables: {list(filtered_schema.keys())}")
        
        # Fallback: if no tables found, return all tables
        if not filtered_schema:
            logger.warning("No relevant tables found via vector search, using full schema")
            return self.schema_info
        
        return filtered_schema

    def get_relevant_tables(self, query: str, k: int = 5) -> List[str]:
        """
        Get list of relevant table names.
        
        Args:
            query: User's natural language query
            k: Number of schema components to retrieve
            
        Returns:
            List of relevant table names
        """
        results = self.vector_store.search(query, k=k)
        tables = []
        
        for doc in results:
            table_name = doc.metadata.get("table")
            if table_name and table_name not in tables:
                tables.append(table_name)
        
        return tables
